#!/usr/bin/env python3
"""Automatic gain scan for electricUniform."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable
import argparse
import json
import math
import os
import re
import shutil
import signal
import shlex
import subprocess
import sys
import uuid

import numpy as np
import uproot
import yaml

from alpha_model import (
    AlphaPoint,
    field_for_gain,
    fit_alpha,
    gain_to_alpha,
    read_fit,
    write_alpha_file,
)


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
EXECUTABLE = BUILD / "uniformE"
OUTPUTS = ROOT / "outputs"
ROOT_OUTPUT = OUTPUTS / "roots"
ALPHA_OUTPUT = OUTPUTS / "alpha"
GIF_OUTPUT = OUTPUTS / "gifs"
LEGACY_ROOT_OUTPUT = OUTPUTS / "legacy_roots"

EXPECTED_GAS_DATA_BRANCHES = {
    "gas1",
    "composition1_pct",
    "gas2",
    "composition2_pct",
    "pressure_bar",
    "temperature_K",
    "electricField_V_cm",
    "gap_mm",
    "height_mm",
    "spaceCharge",
    "npe",
    "townsendAlpha_cm_inv",
    "attachmentEta_cm_inv",
    "alphaEffective_cm_inv",
    "driftVelocityZ_cm_ns",
    "longitudinalDiffusion_sqrt_cm",
    "transverseDiffusion_sqrt_cm",
}

FIELD_FROM_NAME = re.compile(r"_(?P<field>[0-9]+(?:\.[0-9]+)?)kVcm_")


class LegacyRootError(ValueError):
    """The ROOT belongs to an older output schema and must not be reused."""


MIXTURE_COMPONENTS = {
    "ArCF4": ("ar", "cf4"),
    "ArN2": ("ar", "n2"),
    "HeCF4": ("he", "cf4"),
    "ArCO2": ("ar", "co2"),
    "ArCH4": ("ar", "ch4"),
    "ArIso": ("ar", "ic4h10"),
    "ArC2H2F4": ("ar", "c2h2f4"),
}

EVENT_LOCK = Lock()
PROCESS_LOCK = Lock()
ACTIVE_PROCESSES: set[subprocess.Popen] = set()
STOP_REQUESTED = False


@dataclass(frozen=True)
class Family:
    mixture: str
    fraction: float
    gap_mm: float


@dataclass(frozen=True)
class Target:
    pressure_bar: float
    gain: float


@dataclass(frozen=True)
class FieldTarget:
    pressure_bar: float
    field_v_cm: float


@dataclass(frozen=True)
class CampaignOptions:
    space_charge: bool = False
    record_excitation_positions: bool = True
    measure_gas_transport: bool = True


@dataclass
class Job:
    family: Family
    target: Target | FieldTarget
    field_v_cm: float
    min_npe: int
    max_npe: int
    target_relative_error: float
    height_factor: float
    options: CampaignOptions
    job_id: int
    scan_mode: str = "gain"


def request_stop(*_) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    with PROCESS_LOCK:
        for process in list(ACTIVE_PROCESSES):
            if process.poll() is None:
                process.terminate()


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)


def emit(event_type: str, **payload) -> None:
    message = {"type": event_type, **payload}
    with EVENT_LOCK:
        print("CAMPAIGN_EVENT " + json.dumps(message, separators=(",", ":")), flush=True)


def build_project(jobs: int | None = None) -> None:
    """Configure and build before every run, with useful errors for the GUI."""
    # Always configure from a clean build directory. This guarantees that the
    # GUI runs the uniformE.cxx currently present in the project, never an old
    # executable left by a previous patch or CMake configuration.
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True, exist_ok=True)
    parallel = max(1, min(int(jobs or 1), os.cpu_count() or 1))
    commands = [
        ["cmake", "-S", str(ROOT), "-B", str(BUILD)],
        ["cmake", "--build", str(BUILD), "-j", str(parallel)],
    ]
    emit("build_started", jobs=parallel)
    output: list[str] = []
    try:
        for command in commands:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if result.stdout:
                output.extend(result.stdout.splitlines())
            if result.returncode != 0:
                rendered = " ".join(shlex.quote(part) for part in command)
                tail = "\n".join(output[-80:])
                raise RuntimeError(f"Build command failed: {rendered}\n{tail}")
    except Exception as error:
        emit("build_failed", error=str(error))
        raise
    emit("build_finished", executable=str(EXECUTABLE))


def mixture_components(mixture: str, fraction: float) -> tuple[str, float, str, float]:
    if mixture not in MIXTURE_COMPONENTS:
        raise ValueError(f"Unknown mixture: {mixture}")
    if not 0.0 <= fraction <= 100.0:
        raise ValueError(f"Invalid fraction for {mixture}: {fraction}")
    gas1, gas2 = MIXTURE_COMPONENTS[mixture]
    return gas1, 100.0 - fraction, gas2, fraction


def scalar(tree, names: Iterable[str], default=None):
    for name in names:
        if name not in tree:
            continue
        value = tree[name].array(library="np")
        if len(value) == 0:
            continue
        item = value[0]
        if isinstance(item, bytes):
            return item.decode("utf-8")
        if isinstance(item, np.generic):
            return item.item()
        return item
    return default


def _mixture_from_gases(gas1: str, gas2: str) -> str:
    pair = (gas1.strip().lower(), gas2.strip().lower())
    for mixture, components in MIXTURE_COMPONENTS.items():
        if pair == tuple(component.lower() for component in components):
            return mixture
    return ""


def _field_from_root_name(path: Path) -> float:
    match = FIELD_FROM_NAME.search(path.name)
    if match is None:
        raise LegacyRootError(
            f"cannot recover electric field from current ROOT name: {path.name}"
        )
    return 1000.0 * float(match.group("field"))


def _top_level_names(root_file) -> set[str]:
    return {str(name).split(";", 1)[0] for name in root_file.keys()}


def read_root(path: Path, *, field_v_cm: float | None = None) -> AlphaPoint:
    with uproot.open(path) as root_file:
        names = _top_level_names(root_file)
        if "levelMap" in names:
            raise LegacyRootError("contains removed levelMap tree")
        required_objects = {"gasData", "dataPerPrimaryElectron", "hLevels"}
        missing_objects = required_objects - names
        if missing_objects:
            raise LegacyRootError(
                "missing current objects: " + ", ".join(sorted(missing_objects))
            )

        tree = root_file["gasData"]
        branches = set(tree.keys())
        missing = EXPECTED_GAS_DATA_BRANCHES - branches
        if missing:
            raise LegacyRootError(
                "gasData is missing required branches: "
                + ",".join(sorted(missing))
            )

        gas1 = str(scalar(tree, ["gas1"], ""))
        gas2 = str(scalar(tree, ["gas2"], ""))
        mixture = _mixture_from_gases(gas1, gas2)
        fraction = float(scalar(tree, ["composition2_pct"], 0.0))
        pressure_bar = float(scalar(tree, ["pressure_bar"], math.nan))
        gap_mm = float(scalar(tree, ["gap_mm"], math.nan))
        npe = int(scalar(tree, ["npe"], 0))
        space_charge_enabled = bool(scalar(tree, ["spaceCharge"], False))

        primary = root_file["dataPerPrimaryElectron"]
        if "ne" not in primary:
            raise LegacyRootError("dataPerPrimaryElectron has no ne branch")
        ne = primary["ne"].array(library="np").astype(float)
        if len(ne) == 0:
            raise ValueError("dataPerPrimaryElectron/ne is empty")
        gain = float(np.mean(ne))
        gain_error = (
            float(np.std(ne, ddof=1) / math.sqrt(len(ne)))
            if len(ne) > 1 else 0.0
        )
        if npe <= 0:
            npe = int(len(ne))

        if field_v_cm is None:
            field_v_cm = float(
                scalar(tree, ["electricField_V_cm"], math.nan)
            )
            if not math.isfinite(field_v_cm):
                field_v_cm = _field_from_root_name(path)

        excitation_positions_enabled = "hExcXY" in names and "hExcZT" in names
        transport_values = [
            float(scalar(tree, [name], math.nan))
            for name in (
                "townsendAlpha_cm_inv",
                "attachmentEta_cm_inv",
                "alphaEffective_cm_inv",
                "driftVelocityZ_cm_ns",
                "longitudinalDiffusion_sqrt_cm",
                "transverseDiffusion_sqrt_cm",
            )
        ]
        gas_transport_enabled = any(math.isfinite(value) for value in transport_values)

    if not mixture:
        raise ValueError(f"Unknown gas pair in ROOT: {gas1}/{gas2}")
    if not all(math.isfinite(value) for value in
               (pressure_bar, gap_mm, field_v_cm, gain)):
        raise ValueError("ROOT is missing pressure, gap, electric field or gain")

    alpha = gain_to_alpha(gain, gap_mm)
    alpha_error = (
        gain_error / (gain * 0.1 * gap_mm)
        if gain > 0.0 and math.isfinite(gain_error) else math.nan
    )

    try:
        stored_path = str(path.relative_to(ROOT))
    except ValueError:
        stored_path = str(path)

    point = AlphaPoint(
        mixture=mixture,
        fraction=fraction,
        pressure_bar=pressure_bar,
        gap_mm=gap_mm,
        field_v_cm=float(field_v_cm),
        gain=gain,
        gain_error=gain_error,
        alpha_effective=alpha,
        alpha_error=alpha_error,
        npe=npe,
        root=stored_path,
    )
    point.space_charge_enabled = space_charge_enabled
    point.excitation_positions_enabled = excitation_positions_enabled
    point.gas_transport_enabled = gas_transport_enabled
    return point


def read_field_result(path: Path, job: Job) -> AlphaPoint:
    """Read only the measured gain from a freshly produced field-scan ROOT.

    Field mode does not validate or reuse the historical ROOT schema.  The
    mixture, pressure, gap and field are already known from the requested job;
    the only simulation result needed by the campaign controller is ne.
    """
    with uproot.open(path) as root_file:
        names = _top_level_names(root_file)
        if "dataPerPrimaryElectron" not in names:
            raise ValueError("ROOT has no dataPerPrimaryElectron tree")
        primary = root_file["dataPerPrimaryElectron"]
        if "ne" not in primary:
            raise ValueError("dataPerPrimaryElectron has no ne branch")
        ne = primary["ne"].array(library="np").astype(float)

    if len(ne) == 0:
        raise ValueError("dataPerPrimaryElectron/ne is empty")

    gain = float(np.mean(ne))
    gain_error = (
        float(np.std(ne, ddof=1) / math.sqrt(len(ne)))
        if len(ne) > 1 else 0.0
    )
    alpha = gain_to_alpha(gain, job.family.gap_mm)
    alpha_error = (
        gain_error / (gain * 0.1 * job.family.gap_mm)
        if gain > 0.0 and math.isfinite(gain_error) else math.nan
    )

    point = AlphaPoint(
        mixture=job.family.mixture,
        fraction=job.family.fraction,
        pressure_bar=job.target.pressure_bar,
        gap_mm=job.family.gap_mm,
        field_v_cm=job.field_v_cm,
        gain=gain,
        gain_error=gain_error,
        alpha_effective=alpha,
        alpha_error=alpha_error,
        npe=int(len(ne)),
        root=str(path),
    )
    point.space_charge_enabled = job.options.space_charge
    point.excitation_positions_enabled = job.options.record_excitation_positions
    point.gas_transport_enabled = job.options.measure_gas_transport
    return point


def _quarantine_root(path: Path, reason: str) -> None:
    relative = path.relative_to(ROOT_OUTPUT)
    destination = LEGACY_ROOT_OUTPUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        index = 2
        while True:
            candidate = destination.with_name(
                f"{destination.stem}_legacy{index}{destination.suffix}"
            )
            if not candidate.exists():
                destination = candidate
                break
            index += 1
    path.replace(destination)
    print(
        f"[ROOT schema] moved old ROOT to {destination}: {reason}",
        file=sys.stderr,
    )


def scan_roots() -> list[AlphaPoint]:
    points: list[AlphaPoint] = []
    if not ROOT_OUTPUT.exists():
        return points
    for path in sorted(ROOT_OUTPUT.rglob("*.root")):
        try:
            points.append(read_root(path))
        except LegacyRootError as error:
            _quarantine_root(path, str(error))
        except Exception as error:
            print(f"[WARNING] Ignoring unreadable ROOT {path}: {error}", file=sys.stderr)
    return points


def alpha_path(mixture: str, gap_mm: float, space_charge: bool = False) -> Path:
    suffix = "_spacecharge" if space_charge else ""
    return ALPHA_OUTPUT / mixture / f"gap_{gap_mm:.3f}mm{suffix}.json"


def point_flag(point: AlphaPoint, name: str, default: bool = False) -> bool:
    return bool(getattr(point, name, default))


def save_alpha_for(
    mixture: str, gap_mm: float, points: list[AlphaPoint], space_charge: bool
) -> None:
    selected = [
        point for point in points
        if point.mixture == mixture
        and abs(point.gap_mm - gap_mm) < 1.0e-9
        and point_flag(point, "space_charge_enabled") == space_charge
    ]
    write_alpha_file(
        alpha_path(mixture, gap_mm, space_charge), mixture, gap_mm, selected
    )


def family_points(
    points: list[AlphaPoint], family: Family, options: CampaignOptions
) -> list[AlphaPoint]:
    # Only space charge changes the avalanche physics and therefore the alpha
    # family. The two measurement toggles affect output content, not gain.
    return [
        point for point in points
        if point.mixture == family.mixture
        and abs(point.fraction - family.fraction) < 1.0e-9
        and abs(point.gap_mm - family.gap_mm) < 1.0e-9
        and point_flag(point, "space_charge_enabled") == options.space_charge
        and point.gain > 1.0
        and math.isfinite(point.gain)
    ]


def field_family_points(
    points: list[AlphaPoint], family: Family, options: CampaignOptions
) -> list[AlphaPoint]:
    # Direct-field scans must also accept points with gain <= 1. Such a point is
    # still a valid measurement at the requested field, even though it cannot
    # contribute to log(gain)/gap fits.
    return [
        point for point in points
        if point.mixture == family.mixture
        and abs(point.fraction - family.fraction) < 1.0e-9
        and abs(point.gap_mm - family.gap_mm) < 1.0e-9
        and point_flag(point, "space_charge_enabled") == options.space_charge
    ]


def output_compatible(point: AlphaPoint, options: CampaignOptions) -> bool:
    if options.record_excitation_positions and not point_flag(
        point, "excitation_positions_enabled"
    ):
        return False
    if options.measure_gas_transport and not point_flag(
        point, "gas_transport_enabled"
    ):
        return False
    return True


def target_match(
    points: list[AlphaPoint], target: Target, tolerance: float,
    options: CampaignOptions,
) -> AlphaPoint | None:
    candidates = [
        point for point in points
        if abs(point.pressure_bar - target.pressure_bar) < 1.0e-9
        and output_compatible(point, options)
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda point: abs(point.gain - target.gain) / target.gain)
    difference = abs(best.gain - target.gain) / target.gain
    return best if difference <= tolerance else None


def pending_targets(
    points: list[AlphaPoint], targets: list[Target], tolerance: float,
    options: CampaignOptions,
) -> list[Target]:
    return [
        target for target in targets
        if target_match(points, target, tolerance, options) is None
    ]


def _reference_pressure(values: Iterable[float]) -> float:
    """Choose the pressure used to seed a new family.

    One bar is preferred because it is normally present in the campaigns and
    gives a useful field scale without starting at the most expensive pressure.
    """
    pressures = sorted({float(value) for value in values if float(value) > 0.0})
    if not pressures:
        raise ValueError("No positive pressure is available")
    return min(pressures, key=lambda value: abs(math.log(value / 1.0)))


def _usable_alpha_fit(points: list[AlphaPoint]):
    """Return a fit only after the seed points span a useful range.

    The old scheduler placed one point at every pressure before a fit existed.
    That transported an almost constant E/p to 2, 5 and 10 bar and could
    overshoot the target gain by orders of magnitude.  We now first build the
    reduced-alpha curve at one reference pressure.
    """
    fit = fit_alpha(points)
    if fit is None or fit.n_points < 5:
        return None

    clean = [
        point for point in points
        if point.gain > 1.0
        and point.reduced_field > 0.0
        and math.isfinite(point.reduced_field)
    ]
    if len(clean) < 5:
        return None

    reduced_fields = [point.reduced_field for point in clean]
    gains = [point.gain for point in clean]
    field_span = max(reduced_fields) / max(min(reduced_fields), 1.0e-12)
    gain_span = max(gains) / max(min(gains), 1.0e-12)

    # A narrow cluster cannot determine a stable four-parameter curve.
    if field_span < 1.15 or gain_span < 2.0:
        return None
    if not math.isfinite(fit.relative_rmse) or fit.relative_rmse > 0.60:
        return None
    return fit


def select_target(points: list[AlphaPoint], pending: list[Target]) -> Target:
    if not pending:
        raise ValueError("No pending target is available")

    all_pressures = [target.pressure_bar for target in pending]
    all_pressures.extend(point.pressure_bar for point in points)
    anchor_pressure = _reference_pressure(all_pressures)
    anchor_pending = [
        target for target in pending
        if abs(target.pressure_bar - anchor_pressure) < 1.0e-9
    ]
    anchor_points = [
        point for point in points
        if abs(point.pressure_bar - anchor_pressure) < 1.0e-9
        and point.gain > 1.0
        and math.isfinite(point.gain)
    ]

    fit = _usable_alpha_fit(points)

    # Before extrapolating in pressure, populate a broad gain range at the
    # reference pressure.  This is the key protection against E proportional p.
    if fit is None and anchor_pending:
        if not anchor_points:
            gains = sorted(target.gain for target in anchor_pending)
            return min(
                anchor_pending,
                key=lambda target: abs(
                    math.log(target.gain / gains[len(gains) // 2])
                ),
            )

        def coverage_distance(target: Target) -> float:
            return min(
                abs(math.log(target.gain / point.gain))
                for point in anchor_points
            )

        # Select the least-covered gain, not another pressure.
        return max(anchor_pending, key=coverage_distance)

    if fit is None:
        # This only occurs when the reference pressure has no pending target
        # left but the fit is still unusable.  Stay as close as possible to the
        # reference pressure instead of jumping to the highest pressure.
        pressure = min(
            {target.pressure_bar for target in pending},
            key=lambda value: abs(math.log(value / anchor_pressure)),
        )
        candidates = [
            target for target in pending
            if abs(target.pressure_bar - pressure) < 1.0e-9
        ]
        return candidates[len(candidates) // 2]

    # Once a stable reduced-alpha fit exists, seed missing pressures with a
    # middle target.  field_for_gain then uses alpha_target / p, so E/p falls
    # as pressure rises for a fixed gain and gap.
    pressures_with_points = {round(point.pressure_bar, 12) for point in points}
    missing_pressure_targets = [
        target for target in pending
        if round(target.pressure_bar, 12) not in pressures_with_points
    ]
    if missing_pressure_targets:
        pressure = min(
            {target.pressure_bar for target in missing_pressure_targets},
            key=lambda value: abs(math.log(value / anchor_pressure)),
        )
        candidates = sorted(
            (
                target for target in missing_pressure_targets
                if abs(target.pressure_bar - pressure) < 1.0e-9
            ),
            key=lambda target: target.gain,
        )
        return candidates[len(candidates) // 2]

    def distance(target: Target) -> float:
        same_pressure = [
            point for point in points
            if abs(point.pressure_bar - target.pressure_bar) < 1.0e-9
        ]
        reference = same_pressure or points
        return min(abs(math.log(target.gain / point.gain)) for point in reference)

    return min(pending, key=distance)


def interpolate_field(points: list[AlphaPoint], target: Target) -> float | None:
    same_pressure = sorted(
        [point for point in points if abs(point.pressure_bar - target.pressure_bar) < 1.0e-9],
        key=lambda point: point.gain,
    )
    lower = [point for point in same_pressure if point.gain < target.gain]
    upper = [point for point in same_pressure if point.gain > target.gain]
    if not lower or not upper:
        return None

    left = max(lower, key=lambda point: point.gain)
    right = min(upper, key=lambda point: point.gain)
    if right.gain <= left.gain or right.field_v_cm <= left.field_v_cm:
        return None

    fraction = (
        (math.log(target.gain) - math.log(left.gain))
        / (math.log(right.gain) - math.log(left.gain))
    )
    return left.field_v_cm + fraction * (right.field_v_cm - left.field_v_cm)


def propose_field(points: list[AlphaPoint], target: Target, gap_mm: float) -> float:
    # 1. Direct interpolation at the same pressure is the safest prediction.
    interpolated = interpolate_field(points, target)
    if interpolated is not None:
        field = interpolated
        predictor = "same-pressure interpolation"
    else:
        # 2. The reduced-alpha model is used only after a broad reference-pressure
        # seed exists.  Its inversion explicitly targets alpha_eff / p.
        fit = _usable_alpha_fit(points)
        if fit is not None:
            try:
                field = field_for_gain(
                    fit, target.pressure_bar, gap_mm, target.gain
                )
                predictor = "reduced-alpha fit"
            except ValueError:
                field = math.nan
                predictor = "fit inversion failed"
        else:
            field = math.nan
            predictor = "reference-pressure seed"

        same_pressure = [
            point for point in points
            if abs(point.pressure_bar - target.pressure_bar) < 1.0e-9
            and point.gain > 1.0
            and math.isfinite(point.gain)
        ]

        # 3. Without a valid fit, only use measurements at the SAME pressure.
        # Never transport a constant reduced field between pressures.
        if not math.isfinite(field) and same_pressure:
            nearest = min(
                same_pressure,
                key=lambda point: abs(math.log(target.gain / point.gain)),
            )
            target_alpha = gain_to_alpha(target.gain, gap_mm)
            point_alpha = gain_to_alpha(nearest.gain, gap_mm)
            ratio = target_alpha / point_alpha if point_alpha > 0.0 else 1.0
            factor = float(np.clip(ratio ** 0.45, 0.65, 1.55))
            field = nearest.field_v_cm * factor
            predictor = "same-pressure rescaling"

        # 4. If a campaign contains too few gain targets to build the full fit,
        # transport a point using REDUCED alpha, not a constant E/p.  For the
        # same gain, alpha_target is fixed but alpha_target/p decreases as p
        # rises, so the proposed E/p must also decrease.
        if not math.isfinite(field) and points:
            nearest = min(
                (point for point in points if point.gain > 1.0),
                key=lambda point: (
                    abs(math.log(target.gain / point.gain))
                    + 0.25 * abs(math.log(target.pressure_bar / point.pressure_bar))
                ),
                default=None,
            )
            if nearest is not None:
                target_alpha = gain_to_alpha(target.gain, gap_mm)
                point_alpha = gain_to_alpha(nearest.gain, gap_mm)
                target_reduced_alpha = target_alpha / target.pressure_bar
                point_reduced_alpha = point_alpha / nearest.pressure_bar
                ratio = (
                    target_reduced_alpha / point_reduced_alpha
                    if point_reduced_alpha > 0.0 else 1.0
                )
                factor = float(np.clip(ratio ** 0.35, 0.45, 1.80))
                source_reduced_field = (
                    nearest.field_v_cm / 1000.0 / nearest.pressure_bar
                )
                field = (
                    1000.0 * target.pressure_bar
                    * source_reduced_field * factor
                )
                predictor = "reduced-alpha fallback"

        # 5. First seed for a completely new family.  The scheduler calls this
        # at the pressure closest to 1 bar and obtains several seed points there
        # before any pressure extrapolation is attempted.
        if not math.isfinite(field):
            target_alpha = gain_to_alpha(target.gain, gap_mm)
            reduced_field_kv = float(np.clip(
                8.0 + 12.0 * math.log10(1.0 + target_alpha),
                2.0,
                250.0,
            ))
            field = 1000.0 * target.pressure_bar * reduced_field_kv

    same_pressure = [
        point for point in points
        if abs(point.pressure_bar - target.pressure_bar) < 1.0e-9
    ]
    if same_pressure:
        nearest_field = min(
            same_pressure, key=lambda point: abs(point.field_v_cm - field)
        )
        denominator = max(abs(nearest_field.field_v_cm), 1.0)
        if abs(nearest_field.field_v_cm - field) / denominator < 0.005:
            field = nearest_field.field_v_cm * (
                1.06 if nearest_field.gain < target.gain else 0.94
            )

    reduced_field_kv = (field / 1000.0) / target.pressure_bar
    reduced_field_kv = float(np.clip(reduced_field_kv, 0.05, 500.0))
    field = 1000.0 * target.pressure_bar * reduced_field_kv

    emit(
        "prediction",
        pressure_bar=target.pressure_bar,
        target_gain=target.gain,
        field_v_cm=field,
        reduced_field_kv_cm_bar=reduced_field_kv,
        predictor=predictor,
    )
    return field


def adaptive_npe(gap_mm: float, target_gain: float) -> tuple[int, int, float]:
    # The cost roughly follows gain * gap * npe. Low gains and short gaps can
    # therefore accumulate much more statistics than expensive avalanches.
    transport_budget = 3000.0
    max_npe = int(transport_budget / max(gap_mm * target_gain, 1.0e-6))
    max_npe = int(np.clip(max_npe, 20, 5000))
    min_npe = min(20, max_npe)
    return min_npe, max_npe, 0.03


def root_directory(family: Family) -> Path:
    return ROOT_OUTPUT / family.mixture / f"gap_{family.gap_mm:.3f}mm"


def unique_root_name(point: AlphaPoint) -> Path:
    folder = root_directory(Family(point.mixture, point.fraction, point.gap_mm))
    folder.mkdir(parents=True, exist_ok=True)

    gas1, comp1, gas2, comp2 = mixture_components(
        point.mixture, point.fraction
    )
    stem = (
        f"{gas1}_{comp1:.1f}_{gas2}_{comp2:.1f}_"
        f"{point.field_kv_cm:.1f}kVcm_"
        f"{point.pressure_bar:.3f}bar_"
        f"{point.gap_mm:.4f}mm_"
        f"{point.npe}npe"
    )
    candidate = folder / f"{stem}.root"
    # The same physical point is one product. Re-running it replaces the old
    # file instead of creating _r2, _r3, ... copies.
    candidate.unlink(missing_ok=True)
    return candidate


def run_job(job: Job) -> AlphaPoint:
    gas1, comp1, gas2, comp2 = mixture_components(
        job.family.mixture, job.family.fraction
    )
    folder = root_directory(job.family)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".pending_{uuid.uuid4().hex}.root"

    command = [
        str(EXECUTABLE),
        str(temporary),
        job.family.mixture,
        f"{job.field_v_cm:.12g}",
        f"{job.family.gap_mm:.12g}",
        f"{job.target.pressure_bar:.12g}",
        str(job.min_npe),
        str(job.max_npe),
        f"{job.target_relative_error:.12g}",
        gas1,
        f"{comp1:.12g}",
        gas2,
        f"{comp2:.12g}",
        f"{job.height_factor:.12g}",
        str(int(job.options.space_charge)),
        "0",  # GIFs are generated separately
        "0",
        "2",
        str(job.job_id),
        "",   # gifFile placeholder
        "0",  # gifMoveIons is irrelevant in campaign mode
        "0",  # gifIonSpeedCmNs is irrelevant in campaign mode
        str(int(job.options.record_excitation_positions)),
        str(int(job.options.measure_gas_transport)),
    ]

    target_gain = job.target.gain if isinstance(job.target, Target) else None
    target_field_kv_cm = (
        job.target.field_v_cm / 1000.0
        if isinstance(job.target, FieldTarget) else None
    )
    emit(
        "started",
        job_id=job.job_id,
        scan_mode=job.scan_mode,
        mixture=job.family.mixture,
        fraction=job.family.fraction,
        gap_mm=job.family.gap_mm,
        pressure_bar=job.target.pressure_bar,
        target_gain=target_gain,
        target_field_kv_cm=target_field_kv_cm,
        field_v_cm=job.field_v_cm,
        min_npe=job.min_npe,
        max_npe=job.max_npe,
    )

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with PROCESS_LOCK:
        ACTIVE_PROCESSES.add(process)
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        if line.startswith("PROGRESS"):
            parts = line.split()
            if len(parts) == 4:
                emit(
                    "progress",
                    job_id=job.job_id,
                    current=int(parts[2]),
                    maximum=int(parts[3]),
                )
    return_code = process.wait()
    with PROCESS_LOCK:
        ACTIVE_PROCESSES.discard(process)

    if return_code != 0 or not temporary.exists():
        temporary.unlink(missing_ok=True)
        rendered = " ".join(shlex.quote(part) for part in command)
        message = "".join(output_lines[-40:]).strip()
        raise RuntimeError(
            f"uniformE exited with code {return_code}\n"
            f"Command: {rendered}\n{message}"
        )

    try:
        point = (
            read_field_result(temporary, job)
            if job.scan_mode == "field"
            else read_root(temporary, field_v_cm=job.field_v_cm)
        )
    except Exception as error:
        temporary.unlink(missing_ok=True)
        rendered = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(
            f"uniformE produced a ROOT file, but validation failed: {error}\n"
            f"Command: {rendered}"
        ) from error
    final_path = unique_root_name(point)
    temporary.replace(final_path)
    try:
        point.root = str(final_path.relative_to(ROOT))
    except ValueError:
        point.root = str(final_path)
    return point


def campaign_targets(config: dict) -> tuple[list[Family], dict[Family, list[Target]]]:
    pressures = [float(value) for value in config["pressures_bar"]]
    gaps = {float(gap): [float(gain) for gain in gains]
            for gap, gains in config["gaps_mm"].items()}

    families: list[Family] = []
    targets: dict[Family, list[Target]] = {}
    for mixture, fractions in config["mixtures"].items():
        if mixture not in MIXTURE_COMPONENTS:
            raise ValueError(f"Unknown mixture in YAML: {mixture}")
        for fraction in [float(value) for value in fractions]:
            for gap_mm, gains in gaps.items():
                family = Family(mixture, fraction, gap_mm)
                families.append(family)
                targets[family] = [
                    Target(pressure, gain)
                    for pressure in pressures
                    for gain in gains
                    if gain > 1.0
                ]
    return families, targets


def field_campaign_targets(
    config: dict,
) -> tuple[list[Family], dict[Family, list[FieldTarget]]]:
    pressures = [float(value) for value in config["pressures_bar"]]
    raw_fields = config.get("fields_kv_cm")
    if not isinstance(raw_fields, dict) or not raw_fields:
        raise ValueError(
            "field mode requires fields_kv_cm, for example: "
            "fields_kv_cm: {0.05: [35]}"
        )
    fields_by_gap = {
        float(gap): [1000.0 * float(field) for field in fields]
        for gap, fields in raw_fields.items()
    }

    families: list[Family] = []
    targets: dict[Family, list[FieldTarget]] = {}
    for mixture, fractions in config["mixtures"].items():
        if mixture not in MIXTURE_COMPONENTS:
            raise ValueError(f"Unknown mixture in YAML: {mixture}")
        for fraction in [float(value) for value in fractions]:
            for gap_mm, fields_v_cm in fields_by_gap.items():
                family = Family(mixture, fraction, gap_mm)
                families.append(family)
                targets[family] = [
                    FieldTarget(pressure, field_v_cm)
                    for pressure in pressures
                    for field_v_cm in fields_v_cm
                    if field_v_cm > 0.0
                ]
    return families, targets


def field_target_match(
    points: list[AlphaPoint], target: FieldTarget, options: CampaignOptions,
) -> AlphaPoint | None:
    candidates = [
        point for point in points
        if abs(point.pressure_bar - target.pressure_bar) < 1.0e-9
        and abs(point.field_v_cm - target.field_v_cm)
            <= max(1.0e-6, 1.0e-8 * target.field_v_cm)
        and output_compatible(point, options)
    ]
    return candidates[0] if candidates else None


def field_pending_targets(
    points: list[AlphaPoint], targets: list[FieldTarget], options: CampaignOptions,
) -> list[FieldTarget]:
    return [
        target for target in targets
        if field_target_match(points, target, options) is None
    ]


def direct_field_npe(config: dict) -> tuple[int, int, float]:
    if "npe" in config:
        npe = max(1, int(config["npe"]))
        return npe, npe, 0.0
    min_npe = max(1, int(config.get("min_npe", 20)))
    max_npe = max(min_npe, int(config.get("max_npe", 5000)))
    target_error = float(config.get("target_relative_error", 0.03))
    if target_error < 0.0:
        raise ValueError("target_relative_error cannot be negative")
    return min_npe, max_npe, target_error


def worker_count(value) -> int:
    if str(value).lower() == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(value))


def run_field_campaign(
    *,
    config: dict,
    families: list[Family],
    targets_by_family: dict[Family, list[FieldTarget]],
    workers: int,
    height_factor: float,
    options: CampaignOptions,
) -> None:
    """Run every requested field exactly once.

    Field mode deliberately has no alpha fit, no field prediction, no gain
    tolerance and no reuse of old ROOT files.  The electric field is the input;
    the measured gain is only an output.
    """
    min_npe, max_npe, target_error = direct_field_npe(config)
    jobs_to_run = [
        (family, target)
        for family in families
        for target in targets_by_family[family]
    ]

    emit(
        "campaign_started",
        scan_mode="field",
        families=len(families),
        targets=len(jobs_to_run),
        existing_roots=0,
        workers=workers,
        tolerance=None,
        space_charge=options.space_charge,
        record_excitation_positions=options.record_excitation_positions,
        measure_gas_transport=options.measure_gas_transport,
    )

    active: dict[Future, Job] = {}
    queue = list(jobs_to_run)
    job_id = 0
    failures = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while (queue or active) and not STOP_REQUESTED:
            while queue and len(active) < workers:
                family, target = queue.pop(0)
                job = Job(
                    family=family,
                    target=target,
                    field_v_cm=target.field_v_cm,
                    min_npe=min_npe,
                    max_npe=max_npe,
                    target_relative_error=target_error,
                    height_factor=height_factor,
                    options=options,
                    job_id=job_id,
                    scan_mode="field",
                )
                job_id += 1
                active[executor.submit(run_job, job)] = job

            if not active:
                break

            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                family = job.family
                target = job.target
                assert isinstance(target, FieldTarget)
                try:
                    point = future.result()
                except Exception as error:
                    failures += 1
                    emit(
                        "failed",
                        job_id=job.job_id,
                        scan_mode="field",
                        mixture=family.mixture,
                        fraction=family.fraction,
                        gap_mm=family.gap_mm,
                        pressure_bar=target.pressure_bar,
                        target_gain=None,
                        target_field_kv_cm=target.field_v_cm / 1000.0,
                        field_v_cm=target.field_v_cm,
                        attempt=1,
                        will_retry=False,
                        error=str(error),
                    )
                    continue

                completed += 1
                emit(
                    "result",
                    job_id=job.job_id,
                    scan_mode="field",
                    mixture=family.mixture,
                    fraction=family.fraction,
                    gap_mm=family.gap_mm,
                    pressure_bar=point.pressure_bar,
                    target_gain=None,
                    target_field_kv_cm=target.field_v_cm / 1000.0,
                    field_v_cm=point.field_v_cm,
                    gain=point.gain,
                    gain_error=point.gain_error,
                    npe=point.npe,
                    accepted=True,
                    root=point.root,
                )

    remaining = len(jobs_to_run) - completed
    emit(
        "campaign_finished",
        scan_mode="field",
        roots=completed,
        remaining_targets=remaining,
        completed=remaining == 0 and failures == 0 and not STOP_REQUESTED,
        stopped=STOP_REQUESTED,
    )


def run_campaign(
    config_path: Path,
    *,
    space_charge_override: bool | None = None,
    excitation_positions_override: bool | None = None,
    gas_transport_override: bool | None = None,
    skip_build: bool = False,
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Campaign YAML must contain a top-level mapping")
    scan_mode = str(config.get("scan_mode", "gain")).strip().lower()
    if scan_mode not in {"gain", "field"}:
        raise ValueError("scan_mode must be 'gain' or 'field'")
    tolerance = float(config.get("gain_tolerance", 0.05))
    height_factor = float(config.get("height_factor", 1.5))
    workers = worker_count(config.get("workers", "auto"))
    options = CampaignOptions(
        space_charge=(
            bool(config.get("space_charge", False))
            if space_charge_override is None else space_charge_override
        ),
        record_excitation_positions=(
            bool(config.get("record_excitation_positions", True))
            if excitation_positions_override is None
            else excitation_positions_override
        ),
        measure_gas_transport=(
            bool(config.get("measure_gas_transport", True))
            if gas_transport_override is None else gas_transport_override
        ),
    )

    if scan_mode == "gain" and not 0.0 < tolerance < 1.0:
        raise ValueError("gain_tolerance must be between 0 and 1")
    if height_factor < 1.0:
        raise ValueError("height_factor must be at least 1")

    if not skip_build:
        build_project(workers)
    ROOT_OUTPUT.mkdir(parents=True, exist_ok=True)
    ALPHA_OUTPUT.mkdir(parents=True, exist_ok=True)

    if scan_mode == "field":
        families, field_targets_by_family = field_campaign_targets(config)
        run_field_campaign(
            config=config,
            families=families,
            targets_by_family=field_targets_by_family,
            workers=workers,
            height_factor=height_factor,
            options=options,
        )
        return

    families, targets_by_family = campaign_targets(config)
    points = scan_roots()
    for mixture in config["mixtures"]:
        for gap_mm in {family.gap_mm for family in families if family.mixture == mixture}:
            save_alpha_for(mixture, gap_mm, points, options.space_charge)

    emit(
        "campaign_started",
        scan_mode="gain",
        families=len(families),
        targets=sum(len(value) for value in targets_by_family.values()),
        existing_roots=len(points),
        workers=workers,
        tolerance=tolerance,
        space_charge=options.space_charge,
        record_excitation_positions=options.record_excitation_positions,
        measure_gas_transport=options.measure_gas_transport,
    )

    active: dict[Future, Job] = {}
    active_families: set[Family] = set()
    job_id = 0
    failure_count: dict[tuple[Family, Target], int] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while not STOP_REQUESTED:
            for family in families:
                if len(active) >= workers:
                    break
                if family in active_families:
                    continue

                current_points = family_points(points, family, options)
                pending = pending_targets(
                    current_points, targets_by_family[family], tolerance, options
                )
                if not pending:
                    continue

                # Twenty points at one pressure are enough to identify a problem
                # without allowing an accidental infinite autoscan.
                possible = [
                    target for target in pending
                    if sum(
                        abs(point.pressure_bar - target.pressure_bar) < 1.0e-9
                        for point in current_points
                    ) < 20
                    and failure_count.get((family, target), 0) < 3
                ]
                if not possible:
                    continue

                target = select_target(current_points, possible)
                field = propose_field(current_points, target, family.gap_mm)
                min_npe, max_npe, target_error = adaptive_npe(
                    family.gap_mm, target.gain
                )
                job = Job(
                    family=family,
                    target=target,
                    field_v_cm=field,
                    min_npe=min_npe,
                    max_npe=max_npe,
                    target_relative_error=target_error,
                    height_factor=height_factor,
                    options=options,
                    job_id=job_id,
                    scan_mode="gain",
                )
                job_id += 1
                future = executor.submit(run_job, job)
                active[future] = job
                active_families.add(family)

            if not active:
                break

            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                family = job.family
                requested_target = job.target
                active_families.remove(family)
                try:
                    point = future.result()
                except Exception as error:
                    attempt = failure_count.get((family, requested_target), 0) + 1
                    failure_count[(family, requested_target)] = attempt
                    emit(
                        "failed",
                        job_id=job.job_id,
                        scan_mode="gain",
                        mixture=family.mixture,
                        fraction=family.fraction,
                        gap_mm=family.gap_mm,
                        pressure_bar=requested_target.pressure_bar,
                        target_gain=requested_target.gain,
                        attempt=attempt,
                        will_retry=attempt < 3,
                        error=str(error),
                    )
                    continue

                points.append(point)
                save_alpha_for(
                    family.mixture, family.gap_mm, points, options.space_charge
                )
                matched = target_match(
                    family_points(points, family, options), requested_target,
                    tolerance, options,
                )
                emit(
                    "result",
                    job_id=job.job_id,
                    scan_mode="gain",
                    mixture=family.mixture,
                    fraction=family.fraction,
                    gap_mm=family.gap_mm,
                    pressure_bar=point.pressure_bar,
                    target_gain=requested_target.gain,
                    field_v_cm=point.field_v_cm,
                    gain=point.gain,
                    gain_error=point.gain_error,
                    npe=point.npe,
                    accepted=matched is not None,
                    root=point.root,
                )

    remaining = 0
    for family in families:
        remaining += len(pending_targets(
            family_points(points, family, options), targets_by_family[family],
            tolerance, options,
        ))

    emit(
        "campaign_finished",
        scan_mode="gain",
        roots=len(points),
        remaining_targets=remaining,
        completed=remaining == 0 and not STOP_REQUESTED,
        stopped=STOP_REQUESTED,
    )


def run_gif(args) -> None:
    if not args.no_build:
        build_project(1)
    GIF_OUTPUT.mkdir(parents=True, exist_ok=True)

    field_v_cm = args.field_kv_cm * 1000.0 if args.field_kv_cm else None
    if field_v_cm is None:
        fit = read_fit(
            alpha_path(args.mixture, args.gap_mm, bool(args.space_charge)),
            args.fraction,
        )
        if fit is None:
            raise ValueError("No alpha fit is available for this mixture, fraction and gap")
        field_v_cm = field_for_gain(fit, args.pressure_bar, args.gap_mm, args.gain)

    gas1, comp1, gas2, comp2 = mixture_components(args.mixture, args.fraction)
    gif_name = (
        f"{args.mixture}_f{args.fraction:g}_p{args.pressure_bar:g}bar_"
        f"gap{args.gap_mm:g}mm_E{field_v_cm/1000.0:.4g}kVcm.gif"
    )
    gif_path = GIF_OUTPUT / gif_name
    temporary_root = GIF_OUTPUT / f".gif_{uuid.uuid4().hex}.root"

    command = [
        str(EXECUTABLE),
        str(temporary_root),
        args.mixture,
        f"{field_v_cm:.12g}",
        f"{args.gap_mm:.12g}",
        f"{args.pressure_bar:.12g}",
        str(args.npe),
        str(args.npe),
        "0",
        gas1,
        f"{comp1:.12g}",
        gas2,
        f"{comp2:.12g}",
        f"{args.height_factor:.12g}",
        str(int(bool(args.space_charge))),
        "1",
        f"{args.tmax_ns:.12g}",
        str(args.frames),
        "0",
        str(gif_path),
        str(int(args.move_ions)),
        f"{args.ion_speed_cm_ns:.12g}",
        "0",  # no excitation-position histograms in temporary GIF ROOT
        "0",  # no Magboltz transport branches in temporary GIF ROOT
    ]

    try:
        subprocess.run(command, cwd=ROOT, check=True)
    finally:
        temporary_root.unlink(missing_ok=True)

    print(gif_path)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("config", nargs="?", type=Path)
    command.add_argument("--gif", action="store_true")
    command.add_argument("--mixture", choices=sorted(MIXTURE_COMPONENTS))
    command.add_argument("--fraction", type=float, default=1.0)
    command.add_argument("--pressure-bar", type=float, default=1.0)
    command.add_argument("--gap-mm", type=float, default=0.05)
    command.add_argument("--field-kv-cm", type=float)
    command.add_argument("--gain", type=float)
    command.add_argument("--npe", type=int, default=1)
    command.add_argument("--height-factor", type=float, default=1.5)
    command.add_argument("--tmax-ns", type=float, default=10.0)
    command.add_argument("--frames", type=int, default=80)
    command.add_argument(
        "--space-charge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable charged-ring space charge.",
    )
    command.add_argument(
        "--excitation-positions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable hExcXY and hExcZT in campaign ROOT files.",
    )
    command.add_argument(
        "--gas-transport",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Magboltz drift/diffusion/Townsend measurements.",
    )
    command.add_argument(
        "--no-build", action="store_true",
        help="Skip CMake configure/build (mainly for advanced scripting).",
    )
    command.add_argument(
        "--move-ions", action=argparse.BooleanOptionalAction, default=True,
        help="Move positive ions in the GIF with a constant visual speed.",
    )
    command.add_argument(
        "--ion-speed-cm-ns", type=float, default=1.0e-4,
        help="Constant positive-ion speed used only by the GIF [cm/ns].",
    )
    return command


def main() -> None:
    args = parser().parse_args()
    if args.gif:
        if not args.mixture:
            raise SystemExit("--mixture is required in GIF mode")
        if args.field_kv_cm is None and args.gain is None:
            raise SystemExit("Use either --field-kv-cm or --gain in GIF mode")
        run_gif(args)
        return

    if args.config is None:
        raise SystemExit("Usage: python3 run_campaign.py campaign.yaml")
    run_campaign(
        args.config.resolve(),
        space_charge_override=args.space_charge,
        excitation_positions_override=args.excitation_positions,
        gas_transport_override=args.gas_transport,
        skip_build=args.no_build,
    )


if __name__ == "__main__":
    main()
