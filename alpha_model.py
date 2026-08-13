#!/usr/bin/env python3
"""Progressive, validated effective-Townsend models for automatic scans.

The module deliberately separates three ideas:

* every independent ROOT is retained;
* repeated runs at the same physical point are statistically combined before
  fitting, so a refinement accident cannot overweight one electric field;
* the simplest Townsend model supported by the available data is selected and
  validated before it is allowed to seed a new scan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
import json
import math

import numpy as np
from scipy.optimize import brentq, least_squares


# The progressive hierarchy. More complex models are considered only after the
# data contain enough unique fields to constrain them.
MODEL_SPECS = {
    "townsend_2p": {
        "label": "Townsend 2p",
        "free": ("logA", "logB"),
        "min_points": 4,
        "min_field_span": 1.08,
    },
    "townsend_3p": {
        "label": "Townsend 3p",
        "free": ("logA", "logB", "m"),
        "min_points": 7,
        "min_field_span": 1.15,
    },
    "townsend_4p": {
        "label": "Townsend 4p",
        "free": ("logA", "logB", "m", "logn"),
        "min_points": 11,
        "min_field_span": 1.25,
    },
}


@dataclass
class AlphaPoint:
    mixture: str
    fraction: float
    pressure_bar: float
    gap_mm: float
    field_v_cm: float
    gain: float
    gain_error: float
    alpha_effective: float
    alpha_error: float
    npe: int
    root: str
    composition: str = ""
    components: list[dict[str, float]] = field(default_factory=list)
    n_runs: int = 1
    roots: list[str] = field(default_factory=list)

    @property
    def field_kv_cm(self) -> float:
        return self.field_v_cm / 1000.0

    @property
    def reduced_field(self) -> float:
        return self.field_kv_cm / self.pressure_bar

    @property
    def reduced_alpha(self) -> float:
        return self.alpha_effective / self.pressure_bar

    @property
    def relative_gain_error(self) -> float:
        if self.gain <= 0.0 or not math.isfinite(self.gain_error):
            return math.inf
        return abs(self.gain_error / self.gain)


@dataclass
class AlphaFit:
    # A/B/m/n are always present for backward compatibility. Fixed parameters
    # take m=0 and/or n=1 in the simpler models.
    A: float
    B: float
    m: float
    n: float
    covariance: list[list[float]]
    valid_reduced_field: list[float]
    n_points: int
    relative_rmse: float

    model_name: str = "townsend_4p"
    model_label: str = "Townsend 4p"
    n_parameters: int = 4
    n_runs: int = 0
    n_unique_points: int = 0
    n_pressures: int = 0
    field_span: float = math.nan
    gain_span: float = math.nan
    log_rmse: float = math.nan
    cv_median_factor: float = math.inf
    cv_max_factor: float = math.inf
    aicc: float = math.inf
    bic: float = math.inf
    covariance_condition: float = math.inf
    parameters_at_bounds: list[str] = field(default_factory=list)
    usable_for_prediction: bool = False
    fit_status: str = "unvalidated"
    rejection_reasons: list[str] = field(default_factory=list)
    candidate_scores: list[dict] = field(default_factory=list)

    def as_json(self) -> dict:
        return asdict(self)


@dataclass
class _Candidate:
    fit: AlphaFit
    residual_log: np.ndarray


def model_reduced_alpha(x, A, B, m=0.0, n=1.0):
    """Generalised effective Townsend relation.

        alpha_eff / p = A (E / p)^m exp[-(B / (E / p))^n]

    Units used by the campaign:
      E / p       -> kV cm^-1 bar^-1
      alpha_eff/p -> cm^-1 bar^-1
    """
    x = np.asarray(x, dtype=float)
    safe_x = np.clip(x, 1.0e-12, None)
    exponent = -np.power(np.clip(B / safe_x, 0.0, 1.0e6), n)
    return A * np.power(safe_x, m) * np.exp(np.clip(exponent, -745.0, 0.0))


def gain_to_alpha(gain: float, gap_mm: float) -> float:
    if gain <= 1.0 or gap_mm <= 0.0:
        return math.nan
    return math.log(gain) / (0.1 * gap_mm)


def alpha_to_gain(alpha_effective: float, gap_mm: float) -> float:
    exponent = float(alpha_effective) * 0.1 * gap_mm
    if exponent >= 709.0:
        return math.inf
    return math.exp(exponent)


def _valid_raw_point(point: AlphaPoint) -> bool:
    return (
        point.pressure_bar > 0.0
        and point.gap_mm > 0.0
        and point.field_v_cm > 0.0
        and point.gain > 0.0
        and all(math.isfinite(value) for value in (
            point.pressure_bar, point.gap_mm, point.field_v_cm, point.gain,
        ))
    )


def combine_duplicate_points(
    points: Iterable[AlphaPoint],
    *,
    field_abs_tolerance_v_cm: float = 2.0,
    field_relative_tolerance: float = 5.0e-5,
) -> list[AlphaPoint]:
    """Pool independent runs at the same physical ``(p, gap, E)`` point.

    The tolerance is intentionally much smaller than a normal refinement step,
    but large enough to merge fields that differ only through filename/JSON
    rounding. The pooled uncertainty includes both within-run and between-run
    variance.
    """
    ordered = sorted(
        (point for point in points if _valid_raw_point(point)),
        key=lambda point: (point.pressure_bar, point.gap_mm, point.field_v_cm),
    )

    groups: list[list[AlphaPoint]] = []
    for point in ordered:
        if not groups:
            groups.append([point])
            continue
        reference = groups[-1][0]
        field_tolerance = max(
            field_abs_tolerance_v_cm,
            field_relative_tolerance
            * max(abs(reference.field_v_cm), abs(point.field_v_cm), 1.0),
        )
        same_condition = (
            math.isclose(point.pressure_bar, reference.pressure_bar,
                         rel_tol=1.0e-10, abs_tol=1.0e-10)
            and math.isclose(point.gap_mm, reference.gap_mm,
                             rel_tol=1.0e-10, abs_tol=1.0e-10)
            and abs(point.field_v_cm - reference.field_v_cm) <= field_tolerance
        )
        if same_condition:
            groups[-1].append(point)
        else:
            groups.append([point])

    combined: list[AlphaPoint] = []
    for group in groups:
        counts = np.asarray([max(1, int(point.npe)) for point in group], dtype=float)
        gains = np.asarray([point.gain for point in group], dtype=float)
        fields = np.asarray([point.field_v_cm for point in group], dtype=float)
        total_n = int(np.sum(counts))
        gain = float(np.average(gains, weights=counts))
        field_v_cm = float(np.average(fields, weights=counts))

        total_ss = 0.0
        for point, count in zip(group, counts):
            # gain_error is the standard error of the run mean.
            if point.gain_error > 0.0 and math.isfinite(point.gain_error):
                sample_variance = point.gain_error ** 2 * count
                total_ss += max(count - 1.0, 0.0) * sample_variance
            total_ss += count * (point.gain - gain) ** 2
        gain_error = (
            math.sqrt(max(total_ss, 0.0) / (total_n - 1) / total_n)
            if total_n > 1 else 0.0
        )

        reference = group[0]
        alpha = gain_to_alpha(gain, reference.gap_mm)
        alpha_error = (
            gain_error / (gain * 0.1 * reference.gap_mm)
            if gain > 1.0 and gain_error >= 0.0 else math.nan
        )
        roots: list[str] = []
        for point in group:
            roots.extend(point.roots or [point.root])

        combined.append(AlphaPoint(
            mixture=reference.mixture,
            fraction=reference.fraction,
            pressure_bar=reference.pressure_bar,
            gap_mm=reference.gap_mm,
            field_v_cm=field_v_cm,
            gain=gain,
            gain_error=gain_error,
            alpha_effective=alpha,
            alpha_error=alpha_error,
            npe=total_n,
            root=reference.root,
            composition=reference.composition,
            components=reference.components,
            n_runs=sum(max(1, int(getattr(point, "n_runs", 1))) for point in group),
            roots=roots,
        ))
    return combined


def _clean_fit_points(points: Iterable[AlphaPoint]) -> list[AlphaPoint]:
    return [
        point for point in combine_duplicate_points(points)
        if point.gain > 1.0
        and point.reduced_field > 0.0
        and point.reduced_alpha > 0.0
        and math.isfinite(point.reduced_alpha)
    ]


def _parameter_vector_to_values(model_name: str, theta: Sequence[float]):
    if model_name == "townsend_2p":
        logA, logB = theta
        return math.exp(logA), math.exp(logB), 0.0, 1.0
    if model_name == "townsend_3p":
        logA, logB, m = theta
        return math.exp(logA), math.exp(logB), float(m), 1.0
    logA, logB, m, logn = theta
    return math.exp(logA), math.exp(logB), float(m), math.exp(logn)


def _initial_and_bounds(model_name: str, x: np.ndarray, y: np.ndarray):
    xmin = max(float(np.min(x)), 1.0e-6)
    xmax = max(float(np.max(x)), xmin * 1.001)
    B0 = float(np.sqrt(xmin * xmax))
    m0 = 0.8
    n0 = 1.0
    shape = np.power(x, m0 if model_name != "townsend_2p" else 0.0)
    shape *= np.exp(-np.power(B0 / x, n0))
    A0 = max(float(np.median(y / np.clip(shape, 1.0e-30, None))), 1.0e-12)

    log_b_low = math.log(max(1.0e-4 * xmin, 1.0e-9))
    log_b_high = math.log(max(20.0 * xmax, xmin * 2.0))
    if model_name == "townsend_2p":
        theta0 = np.asarray([math.log(A0), math.log(B0)])
        lower = np.asarray([-40.0, log_b_low])
        upper = np.asarray([40.0, log_b_high])
    elif model_name == "townsend_3p":
        theta0 = np.asarray([math.log(A0), math.log(B0), m0])
        lower = np.asarray([-40.0, log_b_low, 0.0])
        upper = np.asarray([40.0, log_b_high, 5.0])
    else:
        theta0 = np.asarray([math.log(A0), math.log(B0), m0, math.log(n0)])
        lower = np.asarray([-40.0, log_b_low, 0.0, math.log(0.25)])
        upper = np.asarray([40.0, log_b_high, 5.0, math.log(4.0)])
    return theta0, lower, upper


def _fit_candidate(
    clean: list[AlphaPoint], model_name: str, *, calculate_cv: bool,
) -> _Candidate | None:
    spec = MODEL_SPECS[model_name]
    if len(clean) < int(spec["min_points"]):
        return None

    x = np.asarray([point.reduced_field for point in clean], dtype=float)
    y = np.asarray([point.reduced_alpha for point in clean], dtype=float)
    relative_error = np.asarray([
        point.alpha_error / point.alpha_effective
        if point.alpha_error > 0.0 and math.isfinite(point.alpha_error)
        else 0.12
        for point in clean
    ], dtype=float)
    relative_error = np.clip(relative_error, 0.03, 0.50)

    theta0, lower, upper = _initial_and_bounds(model_name, x, y)

    def residuals(theta):
        A, B, m, n = _parameter_vector_to_values(model_name, theta)
        prediction = model_reduced_alpha(x, A, B, m, n)
        return (np.log(np.clip(prediction, 1.0e-300, None)) - np.log(y)) / relative_error

    try:
        result = least_squares(
            residuals,
            theta0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=10000,
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
        )
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if not result.success or not np.all(np.isfinite(result.x)):
        return None

    A, B, m, n = _parameter_vector_to_values(model_name, result.x)
    prediction = model_reduced_alpha(x, A, B, m, n)
    if np.any(~np.isfinite(prediction)) or np.any(prediction <= 0.0):
        return None

    residual_log = np.log(prediction) - np.log(y)
    n_obs = len(clean)
    k = len(result.x)
    rss = max(float(np.sum(np.square(residual_log))), 1.0e-30)
    log_rmse = float(np.sqrt(np.mean(np.square(residual_log))))
    relative_rmse = float(np.sqrt(np.mean(np.square((prediction - y) / y))))
    aic = n_obs * math.log(rss / n_obs) + 2.0 * k
    aicc = (
        aic + 2.0 * k * (k + 1.0) / (n_obs - k - 1.0)
        if n_obs > k + 1 else math.inf
    )
    bic = n_obs * math.log(rss / n_obs) + k * math.log(n_obs)

    covariance_theta = np.full((k, k), np.nan)
    covariance_condition = math.inf
    if result.jac.shape[0] > result.jac.shape[1]:
        try:
            information = result.jac.T @ result.jac
            covariance_condition = float(np.linalg.cond(information))
            dof = result.jac.shape[0] - result.jac.shape[1]
            variance = 2.0 * result.cost / max(dof, 1)
            covariance_theta = np.linalg.pinv(information) * variance
        except np.linalg.LinAlgError:
            pass

    # Embed covariance into the common (A, B, m, n) basis.
    common_covariance = np.full((4, 4), np.nan)
    if model_name == "townsend_2p":
        jac_transform = np.asarray([[A, 0.0], [0.0, B], [0.0, 0.0], [0.0, 0.0]])
    elif model_name == "townsend_3p":
        jac_transform = np.asarray([
            [A, 0.0, 0.0], [0.0, B, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
        ])
    else:
        jac_transform = np.diag([A, B, 1.0, n])
    if np.all(np.isfinite(covariance_theta)):
        common_covariance = jac_transform @ covariance_theta @ jac_transform.T

    parameters_at_bounds: list[str] = []
    span = np.maximum(upper - lower, 1.0)
    normalized_distance = np.minimum(result.x - lower, upper - result.x) / span
    for name, distance in zip(spec["free"], normalized_distance):
        if distance < 0.01:
            parameters_at_bounds.append(name.replace("log", ""))

    field_span = float(np.max(x) / np.min(x))
    gains = np.asarray([point.gain for point in clean], dtype=float)
    gain_span = float(np.max(gains) / np.min(gains))
    n_pressures = len({round(point.pressure_bar, 10) for point in clean})
    n_runs = sum(max(1, int(getattr(point, "n_runs", 1))) for point in clean)

    fit = AlphaFit(
        A=A,
        B=B,
        m=m,
        n=n,
        covariance=common_covariance.tolist(),
        valid_reduced_field=[float(np.min(x)), float(np.max(x))],
        n_points=n_obs,
        relative_rmse=relative_rmse,
        model_name=model_name,
        model_label=str(spec["label"]),
        n_parameters=k,
        n_runs=n_runs,
        n_unique_points=n_obs,
        n_pressures=n_pressures,
        field_span=field_span,
        gain_span=gain_span,
        log_rmse=log_rmse,
        aicc=float(aicc),
        bic=float(bic),
        covariance_condition=covariance_condition,
        parameters_at_bounds=parameters_at_bounds,
    )

    if calculate_cv:
        factors = _cross_validation_factors(clean, model_name)
        if factors:
            fit.cv_median_factor = float(np.median(factors))
            fit.cv_max_factor = float(np.max(factors))

    reasons: list[str] = []
    if field_span < float(spec["min_field_span"]):
        reasons.append("insufficient_reduced_field_span")
    if gain_span < 1.5:
        reasons.append("insufficient_gain_span")
    if parameters_at_bounds:
        reasons.append("parameter_at_bound")
    if not math.isfinite(covariance_condition) or covariance_condition > 1.0e14:
        reasons.append("ill_conditioned_covariance")
    if calculate_cv and (
        not math.isfinite(fit.cv_median_factor)
        or fit.cv_median_factor > 1.40
    ):
        reasons.append("poor_cross_validation")
    if calculate_cv and (
        not math.isfinite(fit.cv_max_factor)
        or fit.cv_max_factor > 3.0
    ):
        reasons.append("large_cross_validation_outlier")
    if log_rmse > math.log(1.35):
        reasons.append("large_training_residuals")

    fit.rejection_reasons = reasons
    fit.usable_for_prediction = not reasons
    fit.fit_status = "accepted" if fit.usable_for_prediction else "rejected_" + reasons[0]
    return _Candidate(fit=fit, residual_log=residual_log)


def _cross_validation_factors(
    clean: list[AlphaPoint], model_name: str,
) -> list[float]:
    """Deterministic leave-one-out/blocked cross-validation in gain space."""
    n_points = len(clean)
    minimum = int(MODEL_SPECS[model_name]["min_points"])
    if n_points <= minimum:
        return []

    # Full leave-one-out for the normal campaign sizes. For very dense scans,
    # evaluate a deterministic set of 20 representatives to keep updates quick.
    if n_points <= 20:
        held_out_indices = list(range(n_points))
    else:
        held_out_indices = sorted(set(
            int(round(value))
            for value in np.linspace(0, n_points - 1, 20)
        ))

    factors: list[float] = []
    for index in held_out_indices:
        training = [point for i, point in enumerate(clean) if i != index]
        candidate = _fit_candidate(training, model_name, calculate_cv=False)
        if candidate is None:
            continue
        held = clean[index]
        predicted = predict_gain(
            candidate.fit, held.pressure_bar, held.gap_mm, held.field_v_cm
        )
        if predicted > 0.0 and math.isfinite(predicted):
            factors.append(max(predicted / held.gain, held.gain / predicted))
    return factors


def fit_alpha(points: Iterable[AlphaPoint]) -> AlphaFit | None:
    """Select the simplest well-supported model and validate it.

    All candidates are fitted whenever enough unique fields exist. Selection is
    based primarily on gain-space cross-validation and secondarily on AICc,
    with a small explicit complexity penalty. A higher-order model is therefore
    adopted only when new points genuinely improve predictive performance.
    """
    raw_points = list(points)
    clean = _clean_fit_points(raw_points)
    if len(clean) < int(MODEL_SPECS["townsend_2p"]["min_points"]):
        return None

    candidates: list[_Candidate] = []
    for model_name in MODEL_SPECS:
        candidate = _fit_candidate(clean, model_name, calculate_cv=True)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None

    summaries = []
    for candidate in candidates:
        fit = candidate.fit
        complexity_penalty = 0.015 * fit.n_parameters
        cv_score = (
            math.log(max(fit.cv_median_factor, 1.0))
            if math.isfinite(fit.cv_median_factor)
            else 5.0
        )
        tail_penalty = (
            0.15 * math.log(max(fit.cv_max_factor, 1.0))
            if math.isfinite(fit.cv_max_factor)
            else 1.0
        )
        rejection_penalty = 2.0 if not fit.usable_for_prediction else 0.0
        score = cv_score + tail_penalty + complexity_penalty + rejection_penalty
        summaries.append({
            "model_name": fit.model_name,
            "model_label": fit.model_label,
            "n_parameters": fit.n_parameters,
            "aicc": fit.aicc,
            "bic": fit.bic,
            "log_rmse": fit.log_rmse,
            "cv_median_factor": fit.cv_median_factor,
            "cv_max_factor": fit.cv_max_factor,
            "usable_for_prediction": fit.usable_for_prediction,
            "rejection_reasons": fit.rejection_reasons,
            "selection_score": score,
        })

    accepted = [candidate for candidate in candidates if candidate.fit.usable_for_prediction]
    pool = accepted or candidates
    chosen = min(
        pool,
        key=lambda candidate: next(
            item["selection_score"] for item in summaries
            if item["model_name"] == candidate.fit.model_name
        ),
    )

    # Prefer the simpler model when its score is effectively tied. This avoids
    # model-order flicker after one noisy new ROOT.
    chosen_score = next(item["selection_score"] for item in summaries
                        if item["model_name"] == chosen.fit.model_name)
    tied = [
        candidate for candidate in pool
        if next(item["selection_score"] for item in summaries
                if item["model_name"] == candidate.fit.model_name)
        <= chosen_score + 0.025
    ]
    chosen = min(tied, key=lambda candidate: candidate.fit.n_parameters)
    chosen.fit.candidate_scores = summaries
    return chosen.fit


def predict_gain(
    fit: AlphaFit, pressure_bar: float, gap_mm: float, field_v_cm: float,
) -> float:
    if pressure_bar <= 0.0 or gap_mm <= 0.0 or field_v_cm <= 0.0:
        return math.nan
    reduced_field = (field_v_cm / 1000.0) / pressure_bar
    reduced_alpha = model_reduced_alpha(
        reduced_field, fit.A, fit.B, fit.m, fit.n
    )
    alpha_effective = pressure_bar * float(reduced_alpha)
    return alpha_to_gain(alpha_effective, gap_mm)


def field_for_gain(
    fit: AlphaFit,
    pressure_bar: float,
    gap_mm: float,
    target_gain: float,
    *,
    maximum_expansion: float = 128.0,
) -> float:
    """Invert a monotonic fitted curve and return electric field in V/cm."""
    if pressure_bar <= 0.0 or gap_mm <= 0.0:
        raise ValueError("Pressure and gap must be positive")
    if target_gain <= 1.0:
        # G=1 corresponds to alpha=0 and is reached only asymptotically by the
        # Townsend form. Use an infinitesimal positive avalanche for plotting.
        target_gain = 1.0 + 1.0e-9

    target_alpha = gain_to_alpha(target_gain, gap_mm)
    target_reduced_alpha = target_alpha / pressure_bar

    def equation(reduced_field):
        prediction = model_reduced_alpha(
            reduced_field, fit.A, fit.B, fit.m, fit.n
        )
        return float(prediction - target_reduced_alpha)

    measured_low = max(1.0e-5, fit.valid_reduced_field[0])
    measured_high = max(fit.valid_reduced_field[1], measured_low * 1.001)
    low = measured_low / 4.0
    high = measured_high * 2.0
    minimum_low = measured_low / maximum_expansion
    maximum_high = measured_high * maximum_expansion

    for _ in range(80):
        value_low = equation(low)
        value_high = equation(high)
        if value_low <= 0.0 <= value_high:
            reduced_field = brentq(equation, low, high, maxiter=300)
            return 1000.0 * pressure_bar * reduced_field
        if value_low > 0.0 and low > minimum_low:
            low = max(minimum_low, low / 1.8)
        elif value_high < 0.0 and high < maximum_high:
            high = min(maximum_high, high * 1.8)
        else:
            break
    raise ValueError("The fitted alpha curve could not bracket the target gain")


def fit_extrapolation_fraction(
    fit: AlphaFit, pressure_bar: float, field_v_cm: float,
) -> float:
    """Distance outside the measured E/p interval, normalized to its span."""
    reduced_field = field_v_cm / 1000.0 / pressure_bar
    low, high = fit.valid_reduced_field
    span = max(high - low, 1.0e-12)
    if low <= reduced_field <= high:
        return 0.0
    if reduced_field < low:
        return (low - reduced_field) / span
    return (reduced_field - high) / span


def _point_composition_key(point: AlphaPoint) -> str:
    return point.composition or f"fraction_{point.fraction:g}"


def write_alpha_file(
    path: Path, mixture: str, gap_mm: float, points: list[AlphaPoint],
) -> dict[str, AlphaFit | None]:
    path.parent.mkdir(parents=True, exist_ok=True)

    compositions: dict[str, dict] = {}
    fits: dict[str, AlphaFit | None] = {}
    for key in sorted({_point_composition_key(point) for point in points}):
        composition_points = [
            point for point in points if _point_composition_key(point) == key
        ]
        combined = combine_duplicate_points(composition_points)
        fit = fit_alpha(composition_points)
        fits[key] = fit
        components = composition_points[0].components if composition_points else []
        compositions[key] = {
            "components": components,
            "parameters": fit.as_json() if fit is not None else None,
            "summary": {
                "root_runs": len(composition_points),
                "unique_physical_points": len(combined),
                "pressures": sorted({point.pressure_bar for point in combined}),
            },
            "combined_points": [asdict(point) for point in combined],
            "points": [asdict(point) for point in sorted(
                composition_points,
                key=lambda p: (p.pressure_bar, p.field_v_cm, p.npe, p.root),
            )],
        }

    payload = {
        "mixture": mixture,
        "gap_mm": gap_mm,
        "model_hierarchy": [MODEL_SPECS[name]["label"] for name in MODEL_SPECS],
        "model": "alpha_eff/p = A*(E/p)^m*exp(-(B/(E/p))^n)",
        "units": {
            "E_over_p": "kV cm^-1 bar^-1",
            "alpha_over_p": "cm^-1 bar^-1",
            "field": "V cm^-1",
            "alpha_effective": "cm^-1",
        },
        "compositions": compositions,
    }

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return fits


def read_fit(path: Path, composition: str | float) -> AlphaFit | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = str(composition)
    entry = payload.get("compositions", {}).get(key, {})
    if not entry and isinstance(composition, (int, float)):
        entry = payload.get("fractions", {}).get(f"{float(composition):g}", {})
    parameters = entry.get("parameters")
    if not parameters:
        return None

    # Ignore future/unknown fields while accepting older JSONs with only the
    # original eight AlphaFit members.
    field_names = set(AlphaFit.__dataclass_fields__)
    filtered = {key: value for key, value in parameters.items() if key in field_names}
    return AlphaFit(**filtered)
