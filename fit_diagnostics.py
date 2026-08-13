#!/usr/bin/env python3
"""Generate auditable PDF diagnostics for every alpha/gain fit.

PDFs are written under ``fits/`` only. No PNG files are created.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable
import argparse
import math
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from alpha_model import (
    AlphaFit,
    AlphaPoint,
    combine_duplicate_points,
    field_for_gain,
    fit_alpha,
    fit_extrapolation_fraction,
    model_reduced_alpha,
    predict_gain,
)

ROOT = Path(__file__).resolve().parent
FITS_ROOT = ROOT / "fits"


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return token or "composition"


def fit_pdf_path(
    mixture: str, gap_mm: float, composition: str, *, space_charge: bool = False,
) -> Path:
    suffix = "_spacecharge" if space_charge else ""
    return (
        FITS_ROOT
        / safe_token(mixture)
        / f"gap_{gap_mm:.3f}mm{suffix}"
        / f"{safe_token(composition)}.pdf"
    )


def _composition_label(points: list[AlphaPoint], fallback: str) -> str:
    if not points or not points[0].components:
        return fallback
    parts = []
    for item in points[0].components:
        gas = str(item.get("gas", ""))
        fraction = float(item.get("fraction_pct", 0.0))
        display = {
            "ar": "Ar", "he": "He", "cf4": "CF₄", "n2": "N₂",
            "co2": "CO₂", "ch4": "CH₄", "ic4h10": "Iso",
            "c2h2f4": "C₂H₂F₄",
        }.get(gas, gas)
        parts.append(f"{display} {fraction:g} %")
    return " / ".join(parts)


def _gain_grid() -> np.ndarray:
    # The first value is plotted/labeled as G=1, while internally remaining
    # infinitesimally above one so the model inversion is well defined.
    return np.concatenate(([1.0 + 1.0e-9], np.logspace(0.001, 7.0, 240)))


def _field_curve_for_gains(
    fit: AlphaFit,
    pressure_bar: float,
    gap_mm: float,
    gains: np.ndarray,
    *,
    max_upper_extrapolation_fraction: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the monotonic locally invertible part of the fitted curve.

    The requested display range is capped at ``gain_max`` by the caller, but
    the function stops at the first gain that cannot be inverted within the
    model's local expansion window. It never skips a failed gain and resumes at
    a larger one, because that would draw a disconnected and misleading
    extrapolation.
    """
    fields: list[float] = []
    valid_gains: list[float] = []
    extrapolated: list[bool] = []
    for gain in np.sort(np.unique(np.asarray(gains, dtype=float))):
        if gain <= 1.0 + 1.0e-8:
            fields.append(0.0)
            valid_gains.append(1.0)
            extrapolated.append(True)
            continue
        try:
            field = field_for_gain(fit, pressure_bar, gap_mm, float(gain))
        except ValueError:
            break
        if not math.isfinite(field) or field <= 0.0:
            break
        extrapolation_fraction = fit_extrapolation_fraction(
            fit, pressure_bar, field
        )
        reduced_field = field / 1000.0 / pressure_bar
        # The low-field branch is allowed to reach G=1 as explicitly requested.
        # At high field, stop at the same local extrapolation boundary used by
        # the campaign controller instead of drawing the model to absurd fields.
        if (
            reduced_field > fit.valid_reduced_field[1]
            and extrapolation_fraction > max_upper_extrapolation_fraction
        ):
            break
        fields.append(field / 1000.0)
        valid_gains.append(float(gain))
        extrapolated.append(extrapolation_fraction > 0.0)
    return (
        np.asarray(fields),
        np.asarray(valid_gains),
        np.asarray(extrapolated, dtype=bool),
    )


def _alpha_curve_from_gains(gains: np.ndarray, gap_mm: float) -> np.ndarray:
    """Convert the displayed gain curve to alpha_eff, preserving G=1 -> alpha=0."""
    gain_values = np.asarray(gains, dtype=float)
    values = np.zeros_like(gain_values)
    mask = gain_values > 1.0
    values[mask] = np.log(gain_values[mask]) / (0.1 * gap_mm)
    return values


def _plot_upper_limit(values: Iterable[float], lower: float, cap: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.0]
    if not finite:
        return min(cap, max(lower * 10.0, lower + 1.0))
    maximum = max(finite)
    return min(cap, max(lower * 10.0, maximum * 1.15))


def _summary_lines(
    fit: AlphaFit | None,
    raw_points: list[AlphaPoint],
    combined: list[AlphaPoint],
) -> list[str]:
    pressures = sorted({point.pressure_bar for point in combined})
    lines = [
        f"ROOT runs: {len(raw_points)}",
        f"Unique physical points: {len(combined)}",
        f"Pressures: {', '.join(f'{value:g}' for value in pressures) or 'none'} bar",
    ]
    if fit is None:
        lines.extend([
            "Selected model: none",
            "Status: insufficient unique gain points",
        ])
        return lines
    lines.extend([
        f"Selected model: {fit.model_label}",
        f"Status: {fit.fit_status}",
        f"Training RMSE: ×{math.exp(fit.log_rmse):.3f}" if math.isfinite(fit.log_rmse) else "Training RMSE: unavailable",
        f"Cross-validation median: ×{fit.cv_median_factor:.3f}" if math.isfinite(fit.cv_median_factor) else "Cross-validation median: unavailable",
        f"Cross-validation worst: ×{fit.cv_max_factor:.3f}" if math.isfinite(fit.cv_max_factor) else "Cross-validation worst: unavailable",
        f"Reduced-field range: {fit.valid_reduced_field[0]:.4g}–{fit.valid_reduced_field[1]:.4g} kV cm⁻¹ bar⁻¹",
        f"A={fit.A:.5g}, B={fit.B:.5g}, m={fit.m:.5g}, n={fit.n:.5g}",
        f"Covariance condition: {fit.covariance_condition:.3g}",
    ])
    if fit.rejection_reasons:
        lines.append("Warnings: " + ", ".join(fit.rejection_reasons))
    return lines


def _plot_segmented_curve(ax, x, y, extrapolated, *, label: str):
    if len(x) < 2:
        return
    inside = ~extrapolated
    # Plot the full relation dashed first, then overwrite the measured-domain
    # part with a solid line. This makes extrapolation impossible to mistake.
    ax.plot(x, y, linestyle="--", alpha=0.65, label=f"{label} · extrapolation")
    if np.any(inside):
        solid_y = np.where(inside, y, np.nan)
        ax.plot(x, solid_y, linewidth=2.0, label=f"{label} · measured-domain segment")


def generate_fit_pdf(
    path: Path,
    mixture: str,
    gap_mm: float,
    composition: str,
    raw_points: Iterable[AlphaPoint],
    *,
    fit: AlphaFit | None = None,
    gain_min: float = 1.0,
    gain_max: float = 1.0e7,
) -> Path:
    raw_points = list(raw_points)
    combined = combine_duplicate_points(raw_points)
    if fit is None:
        fit = fit_alpha(raw_points)
    path.parent.mkdir(parents=True, exist_ok=True)

    label = _composition_label(raw_points, composition)
    pressures = sorted({point.pressure_bar for point in combined})
    gains_for_curve = _gain_grid()
    gains_for_curve = gains_for_curve[
        (gains_for_curve >= max(gain_min, 1.0)) & (gains_for_curve <= gain_max)
    ]

    with PdfPages(path) as pdf:
        # Page 1: auditable diagnostics in the exact variables used by the
        # campaign operator: gain(E) and alpha_eff(E). Reduced quantities are
        # still used internally by the global fit, but are no longer the main
        # visual output.
        figure = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.72))
        ax_gain = figure.add_subplot(grid[0, 0])
        ax_alpha = figure.add_subplot(grid[0, 1])
        ax_residual = figure.add_subplot(grid[1, 0])
        ax_summary = figure.add_subplot(grid[1, 1])

        reachable_gain: dict[float, float] = {}
        gain_values_for_limits: list[float] = [
            point.gain for point in combined if point.gain > 0.0
        ]

        for pressure in pressures:
            raw_pressure = [
                point for point in raw_points
                if math.isclose(point.pressure_bar, pressure)
            ]
            combined_pressure = [
                point for point in combined
                if math.isclose(point.pressure_bar, pressure)
            ]
            ax_gain.scatter(
                [point.field_kv_cm for point in raw_pressure],
                [point.gain for point in raw_pressure],
                s=13,
                alpha=0.28,
                label=f"{pressure:g} bar · individual ROOTs",
            )
            ax_gain.errorbar(
                [point.field_kv_cm for point in combined_pressure],
                [point.gain for point in combined_pressure],
                yerr=[point.gain_error for point in combined_pressure],
                fmt="o",
                capsize=2,
                label=f"{pressure:g} bar · combined fields",
            )

            alpha_points = [
                point for point in combined_pressure
                if point.gain > 1.0 and math.isfinite(point.alpha_effective)
            ]
            ax_alpha.errorbar(
                [point.field_kv_cm for point in alpha_points],
                [point.alpha_effective for point in alpha_points],
                yerr=[point.alpha_error for point in alpha_points],
                fmt="o",
                capsize=2,
                label=f"{pressure:g} bar · combined fields",
            )

            if fit is not None:
                fields, predicted_gains, extrapolated = _field_curve_for_gains(
                    fit, pressure, gap_mm, gains_for_curve
                )
                if len(predicted_gains):
                    reachable_gain[pressure] = float(predicted_gains[-1])
                    gain_values_for_limits.extend(predicted_gains.tolist())
                _plot_segmented_curve(
                    ax_gain, fields, predicted_gains, extrapolated,
                    label=f"{pressure:g} bar · {fit.model_label}",
                )
                _plot_segmented_curve(
                    ax_alpha, fields,
                    _alpha_curve_from_gains(predicted_gains, gap_mm),
                    extrapolated,
                    label=f"{pressure:g} bar · {fit.model_label}",
                )

                residual_x = []
                residual_y = []
                for point in combined_pressure:
                    predicted = predict_gain(
                        fit, point.pressure_bar, point.gap_mm, point.field_v_cm
                    )
                    if predicted > 0.0 and math.isfinite(predicted):
                        residual_x.append(point.field_kv_cm)
                        residual_y.append(math.log(predicted / point.gain))
                ax_residual.scatter(residual_x, residual_y, label=f"{pressure:g} bar")

        gain_lower = max(gain_min, 1.0)
        gain_upper = _plot_upper_limit(gain_values_for_limits, gain_lower, gain_max)
        ax_gain.set_yscale("log")
        ax_gain.set_ylim(gain_lower, gain_upper)
        ax_gain.set_xlabel("Electric field [kV/cm]")
        ax_gain.set_ylabel("Gain")
        ax_gain.set_title("Gain vs electric field")
        ax_gain.grid(True, which="both", alpha=0.25)
        ax_gain.legend(fontsize=7, ncol=2)

        ax_alpha.set_xlabel("Electric field [kV/cm]")
        ax_alpha.set_ylabel("αeff [cm⁻¹]")
        ax_alpha.set_title("Effective Townsend coefficient vs electric field")
        ax_alpha.grid(True, alpha=0.25)
        ax_alpha.legend(fontsize=7, ncol=2)

        ax_residual.axhline(0.0, linewidth=1.0)
        ax_residual.axhline(math.log(1.2), linestyle="--", alpha=0.5)
        ax_residual.axhline(-math.log(1.2), linestyle="--", alpha=0.5)
        ax_residual.set_xlabel("Electric field [kV/cm]")
        ax_residual.set_ylabel("ln(Gpred/Gmeasured)")
        ax_residual.set_title("Gain-space residuals (±ln 1.2 guides)")
        ax_residual.grid(True, alpha=0.25)
        if pressures:
            ax_residual.legend(fontsize=7)

        ax_summary.axis("off")
        summary_lines = _summary_lines(fit, raw_points, combined)
        if fit is not None:
            for pressure in pressures or [1.0]:
                local_max = reachable_gain.get(pressure)
                if local_max is None:
                    summary_lines.append(
                        f"{pressure:g} bar display limit: no invertible extrapolation"
                    )
                elif local_max >= gain_max * (1.0 - 1.0e-6):
                    summary_lines.append(
                        f"{pressure:g} bar display limit: G={gain_max:.3g} (requested cap)"
                    )
                else:
                    summary_lines.append(
                        f"{pressure:g} bar display limit: G={local_max:.3g} (local inversion limit)"
                    )
        ax_summary.text(
            0.0, 1.0, "\n".join(summary_lines),
            va="top", ha="left", family="monospace", fontsize=9,
            transform=ax_summary.transAxes,
        )
        figure.suptitle(f"{mixture} · {label} · gap {gap_mm:g} mm", fontsize=14)
        pdf.savefig(figure)
        plt.close(figure)

        # Page 2: large, clean extrapolation views. Each pressure stops at the
        # first gain that the local model inversion cannot reach, with an upper
        # cap of 10^7 (or the configured fit_gain_max).
        figure, (ax_gain_large, ax_alpha_large) = plt.subplots(
            1, 2, figsize=(11.69, 8.27), constrained_layout=True
        )
        page_gain_values: list[float] = []
        plotted_any = False
        if fit is not None:
            for pressure in pressures or [1.0]:
                fields, predicted_gains, extrapolated = _field_curve_for_gains(
                    fit, pressure, gap_mm, gains_for_curve
                )
                if len(predicted_gains) < 2:
                    continue
                plotted_any = True
                page_gain_values.extend(predicted_gains.tolist())
                _plot_segmented_curve(
                    ax_gain_large, fields, predicted_gains, extrapolated,
                    label=f"{pressure:g} bar",
                )
                _plot_segmented_curve(
                    ax_alpha_large, fields,
                    _alpha_curve_from_gains(predicted_gains, gap_mm),
                    extrapolated,
                    label=f"{pressure:g} bar",
                )

        for pressure in pressures:
            combined_pressure = [
                point for point in combined
                if math.isclose(point.pressure_bar, pressure)
            ]
            ax_gain_large.errorbar(
                [point.field_kv_cm for point in combined_pressure],
                [point.gain for point in combined_pressure],
                yerr=[point.gain_error for point in combined_pressure],
                fmt="o", capsize=2, label=f"{pressure:g} bar · data",
            )
            alpha_points = [
                point for point in combined_pressure
                if point.gain > 1.0 and math.isfinite(point.alpha_effective)
            ]
            ax_alpha_large.errorbar(
                [point.field_kv_cm for point in alpha_points],
                [point.alpha_effective for point in alpha_points],
                yerr=[point.alpha_error for point in alpha_points],
                fmt="o", capsize=2, label=f"{pressure:g} bar · data",
            )

        page_gain_values.extend(
            point.gain for point in combined if point.gain > 0.0
        )
        page_gain_upper = _plot_upper_limit(
            page_gain_values, gain_lower, gain_max
        )
        ax_gain_large.set_yscale("log")
        ax_gain_large.set_ylim(gain_lower, page_gain_upper)
        ax_gain_large.set_xlabel("Electric field [kV/cm]")
        ax_gain_large.set_ylabel("Gain")
        ax_gain_large.set_title("Gain vs electric field")
        ax_gain_large.grid(True, which="both", alpha=0.25)

        ax_alpha_large.set_xlabel("Electric field [kV/cm]")
        ax_alpha_large.set_ylabel("αeff [cm⁻¹]")
        ax_alpha_large.set_title("αeff vs electric field")
        ax_alpha_large.grid(True, alpha=0.25)

        if plotted_any or combined:
            ax_gain_large.legend(fontsize=7, ncol=2)
            ax_alpha_large.legend(fontsize=7, ncol=2)
        else:
            ax_gain_large.text(
                0.5, 0.5, "No validated fit is available yet",
                ha="center", va="center", transform=ax_gain_large.transAxes,
            )
            ax_alpha_large.text(
                0.5, 0.5, "No validated fit is available yet",
                ha="center", va="center", transform=ax_alpha_large.transAxes,
            )
        figure.suptitle(
            f"Extrapolation capped at G={gain_max:.3g} or at the local model limit",
            fontsize=14,
        )
        pdf.savefig(figure)
        plt.close(figure)

        # Page(s) 3+: exact combined data used by the fit.
        rows = []
        for point in sorted(combined, key=lambda p: (p.pressure_bar, p.field_v_cm)):
            rows.append([
                f"{point.pressure_bar:g}",
                f"{point.field_kv_cm:.6g}",
                f"{point.reduced_field:.6g}",
                f"{point.gain:.7g}",
                f"{point.gain_error:.3g}",
                f"{point.alpha_effective:.6g}" if math.isfinite(point.alpha_effective) else "—",
                str(point.npe),
                str(point.n_runs),
            ])

        chunk_size = 30
        chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)] or [[]]
        for page_index, chunk in enumerate(chunks, start=1):
            figure, ax = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
            ax.axis("off")
            table = ax.table(
                cellText=chunk,
                colLabels=["p [bar]", "E [kV/cm]", "E/p", "Gain", "σGain", "αeff [cm⁻¹]", "npe", "runs"],
                loc="upper center",
                cellLoc="right",
                colLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.35)
            ax.set_title(
                f"Exact combined points used by the fit · page {page_index}/{len(chunks)}\n"
                "Repeated ROOTs at the same field are pooled once; all source ROOTs remain on disk.",
                pad=18,
            )
            pdf.savefig(figure)
            plt.close(figure)

        # Final page: model-order comparison, including rejected candidates.
        figure, ax = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
        ax.axis("off")
        if fit is not None and fit.candidate_scores:
            candidate_rows = []
            for item in fit.candidate_scores:
                candidate_rows.append([
                    item["model_label"],
                    str(item["n_parameters"]),
                    f"{item['aicc']:.3g}",
                    f"×{item['cv_median_factor']:.3g}" if math.isfinite(item["cv_median_factor"]) else "—",
                    f"×{item['cv_max_factor']:.3g}" if math.isfinite(item["cv_max_factor"]) else "—",
                    "yes" if item["usable_for_prediction"] else "no",
                    "\n".join(
                        textwrap.wrap(
                            ", ".join(reason.replace("_", " ") for reason in item["rejection_reasons"]) or "—",
                            width=34,
                        )
                    ),
                ])
            table = ax.table(
                cellText=candidate_rows,
                colLabels=["Candidate", "k", "AICc", "CV median", "CV worst", "usable", "warnings"],
                loc="upper center",
                cellLoc="center",
                colLoc="center",
                colWidths=[0.14, 0.05, 0.08, 0.10, 0.10, 0.07, 0.46],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1.0, 1.8)
            ax.set_title(
                "Progressive model selection\n"
                "The higher-order model is selected only when new points improve out-of-sample prediction.",
                pad=18,
            )
        else:
            ax.text(0.5, 0.5, "Not enough points for model comparison", ha="center", va="center")
        pdf.savefig(figure)
        plt.close(figure)

    return path


def generate_family_fit_pdfs(
    mixture: str,
    gap_mm: float,
    points: Iterable[AlphaPoint],
    *,
    space_charge: bool = False,
    gain_min: float = 1.0,
    gain_max: float = 1.0e7,
    only_composition: str | None = None,
) -> dict[str, dict]:
    selected = [
        point for point in points
        if point.mixture == mixture and math.isclose(point.gap_mm, gap_mm, abs_tol=1.0e-10)
    ]
    groups: dict[str, list[AlphaPoint]] = {}
    for point in selected:
        composition = point.composition or f"fraction_{point.fraction:g}"
        groups.setdefault(composition, []).append(point)

    artifacts: dict[str, dict] = {}
    for composition, composition_points in groups.items():
        if only_composition is not None and composition != only_composition:
            continue
        fit = fit_alpha(composition_points)
        path = fit_pdf_path(
            mixture, gap_mm, composition, space_charge=space_charge
        )
        generate_fit_pdf(
            path,
            mixture,
            gap_mm,
            composition,
            composition_points,
            fit=fit,
            gain_min=gain_min,
            gain_max=gain_max,
        )
        artifacts[composition] = {
            "path": str(path.relative_to(ROOT)),
            "fit": fit,
        }
    return artifacts


def regenerate_all() -> int:
    # Lazy import avoids a module cycle when run_campaign imports the PDF writer.
    from run_campaign import point_flag, scan_roots

    points = scan_roots()
    families = sorted({
        (point.mixture, point.gap_mm, point_flag(point, "space_charge_enabled"))
        for point in points
    })
    count = 0
    for mixture, gap_mm, space_charge in families:
        family_points = [
            point for point in points
            if point.mixture == mixture
            and math.isclose(point.gap_mm, gap_mm, abs_tol=1.0e-10)
            and point_flag(point, "space_charge_enabled") == space_charge
        ]
        artifacts = generate_family_fit_pdfs(
            mixture, gap_mm, family_points, space_charge=space_charge
        )
        count += len(artifacts)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Regenerate every fit PDF from outputs/roots")
    args = parser.parse_args()
    if not args.all:
        parser.error("Use --all to regenerate all fit PDFs")
    count = regenerate_all()
    print(f"Generated {count} fit PDF(s) under {FITS_ROOT}")


if __name__ == "__main__":
    main()
