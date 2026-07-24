#!/usr/bin/env python3
"""Four-parameter effective-Townsend model used by the automatic scan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable
import json
import math

import numpy as np
from scipy.optimize import brentq, least_squares


MIN_FIT_POINTS = 5


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

    @property
    def field_kv_cm(self) -> float:
        return self.field_v_cm / 1000.0

    @property
    def reduced_field(self) -> float:
        return self.field_kv_cm / self.pressure_bar

    @property
    def reduced_alpha(self) -> float:
        return self.alpha_effective / self.pressure_bar


@dataclass
class AlphaFit:
    A: float
    B: float
    m: float
    n: float
    covariance: list[list[float]]
    valid_reduced_field: list[float]
    n_points: int
    relative_rmse: float

    def as_json(self) -> dict:
        return asdict(self)


def model_reduced_alpha(x, A, B, m, n):
    """
    Generalised four-parameter model.

        alpha_eff / p = A (E / p)^m exp[-(B / (E / p))^n]

    Units used by the campaign:
      E / p       -> kV cm^-1 bar^-1
      alpha_eff/p -> cm^-1 bar^-1
    """
    x = np.asarray(x, dtype=float)
    return A * np.power(x, m) * np.exp(-np.power(B / x, n))


def gain_to_alpha(gain: float, gap_mm: float) -> float:
    if gain <= 1.0 or gap_mm <= 0.0:
        return math.nan
    return math.log(gain) / (0.1 * gap_mm)


def alpha_to_gain(alpha_effective: float, gap_mm: float) -> float:
    return math.exp(alpha_effective * 0.1 * gap_mm)


def fit_alpha(points: Iterable[AlphaPoint]) -> AlphaFit | None:
    clean = [
        point
        for point in points
        if point.pressure_bar > 0.0
        and point.reduced_field > 0.0
        and point.reduced_alpha > 0.0
        and math.isfinite(point.reduced_alpha)
    ]
    if len(clean) < MIN_FIT_POINTS:
        return None

    x = np.asarray([point.reduced_field for point in clean], dtype=float)
    y = np.asarray([point.reduced_alpha for point in clean], dtype=float)

    # Relative alpha errors are used as fit weights. A small floor prevents a
    # single very precise point from dominating the whole curve.
    relative_error = []
    for point in clean:
        if point.alpha_error > 0.0 and math.isfinite(point.alpha_error):
            relative_error.append(point.alpha_error / point.alpha_effective)
        else:
            relative_error.append(0.10)
    relative_error = np.clip(np.asarray(relative_error), 0.02, 0.50)

    B0 = float(np.median(x))
    m0 = 1.0
    n0 = 1.0
    shape = np.power(x, m0) * np.exp(-np.power(B0 / x, n0))
    A0 = float(np.median(y / np.clip(shape, 1.0e-30, None)))
    A0 = max(A0, 1.0e-12)

    theta0 = np.asarray([math.log(A0), math.log(B0), m0, math.log(n0)])
    lower = np.asarray([-40.0, -20.0, 0.0, math.log(0.15)])
    upper = np.asarray([40.0, 20.0, 5.0, math.log(6.0)])

    def residuals(theta):
        logA, logB, m, logn = theta
        prediction = model_reduced_alpha(
            x, math.exp(logA), math.exp(logB), m, math.exp(logn)
        )
        return (np.log(prediction) - np.log(y)) / relative_error

    result = least_squares(
        residuals,
        theta0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=20000,
    )

    logA, logB, m, logn = result.x
    A = math.exp(logA)
    B = math.exp(logB)
    n = math.exp(logn)

    prediction = model_reduced_alpha(x, A, B, m, n)
    relative_rmse = float(np.sqrt(np.mean(np.square((prediction - y) / y))))

    covariance_theta = np.full((4, 4), np.nan)
    if result.jac.shape[0] > result.jac.shape[1]:
        try:
            degrees_of_freedom = result.jac.shape[0] - result.jac.shape[1]
            variance = 2.0 * result.cost / degrees_of_freedom
            covariance_theta = np.linalg.inv(result.jac.T @ result.jac) * variance
        except np.linalg.LinAlgError:
            pass

    # Convert the covariance from (log A, log B, m, log n) to (A, B, m, n).
    transform = np.diag([A, B, 1.0, n])
    covariance = transform @ covariance_theta @ transform.T

    return AlphaFit(
        A=A,
        B=B,
        m=float(m),
        n=n,
        covariance=covariance.tolist(),
        valid_reduced_field=[float(np.min(x)), float(np.max(x))],
        n_points=len(clean),
        relative_rmse=relative_rmse,
    )


def predict_gain(fit: AlphaFit, pressure_bar: float, gap_mm: float,
                 field_v_cm: float) -> float:
    reduced_field = (field_v_cm / 1000.0) / pressure_bar
    reduced_alpha = model_reduced_alpha(
        reduced_field, fit.A, fit.B, fit.m, fit.n
    )
    alpha_effective = pressure_bar * float(reduced_alpha)
    return alpha_to_gain(alpha_effective, gap_mm)


def field_for_gain(fit: AlphaFit, pressure_bar: float, gap_mm: float,
                   target_gain: float) -> float:
    """Invert the fitted curve and return the electric field in V/cm."""
    target_alpha = gain_to_alpha(target_gain, gap_mm)
    target_reduced_alpha = target_alpha / pressure_bar

    def equation(reduced_field):
        prediction = model_reduced_alpha(
            reduced_field, fit.A, fit.B, fit.m, fit.n
        )
        return float(prediction - target_reduced_alpha)

    low = max(1.0e-3, 0.25 * fit.valid_reduced_field[0])
    high = max(2.0 * fit.valid_reduced_field[1], low * 2.0)

    for _ in range(30):
        if equation(low) <= 0.0 <= equation(high):
            reduced_field = brentq(equation, low, high, maxiter=200)
            return 1000.0 * pressure_bar * reduced_field
        if equation(low) > 0.0:
            low *= 0.5
        else:
            high *= 1.6

    raise ValueError("The fitted alpha curve could not bracket the target gain")


def _point_composition_key(point: AlphaPoint) -> str:
    if point.composition:
        return point.composition
    # Backward compatibility for alpha JSON files produced before explicit
    # compositions were introduced.
    return f"fraction_{point.fraction:g}"


def write_alpha_file(path: Path, mixture: str, gap_mm: float,
                     points: list[AlphaPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    compositions: dict[str, dict] = {}
    for key in sorted({_point_composition_key(point) for point in points}):
        composition_points = [
            point for point in points if _point_composition_key(point) == key
        ]
        fit = fit_alpha(composition_points)
        components = composition_points[0].components if composition_points else []
        compositions[key] = {
            "components": components,
            "parameters": fit.as_json() if fit is not None else None,
            "points": [asdict(point) for point in sorted(
                composition_points,
                key=lambda p: (p.pressure_bar, p.field_v_cm, p.npe),
            )],
        }

    payload = {
        "mixture": mixture,
        "gap_mm": gap_mm,
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


def read_fit(path: Path, composition: str | float) -> AlphaFit | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = str(composition)
    entry = payload.get("compositions", {}).get(key, {})
    if not entry and isinstance(composition, (int, float)):
        # Backward compatibility with the former binary-fraction layout.
        entry = payload.get("fractions", {}).get(f"{float(composition):g}", {})
    parameters = entry.get("parameters")
    if not parameters:
        return None
    return AlphaFit(**parameters)
