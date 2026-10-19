#!/usr/bin/env python3
"""Automatic gain scan for electricUniform."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from datetime import datetime, timezone
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
    combine_duplicate_points,
    field_for_gain,
    fit_alpha,
    fit_extrapolation_fraction,
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
ATTEMPT_LOG = OUTPUTS / "metadata" / "campaign_attempts.jsonl"

_FIT_DIAGNOSTICS_IMPORT_ATTEMPTED = False
_FIT_DIAGNOSTICS_GENERATOR = None
_FIT_DIAGNOSTICS_IMPORT_ERROR: str | None = None
_FIT_DIAGNOSTICS_WARNING_EMITTED = False


def _load_fit_diagnostics_generator():
    """Load the optional PDF backend without making campaigns depend on it."""
    global _FIT_DIAGNOSTICS_IMPORT_ATTEMPTED
    global _FIT_DIAGNOSTICS_GENERATOR
    global _FIT_DIAGNOSTICS_IMPORT_ERROR
    global _FIT_DIAGNOSTICS_WARNING_EMITTED

    if not _FIT_DIAGNOSTICS_IMPORT_ATTEMPTED:
        _FIT_DIAGNOSTICS_IMPORT_ATTEMPTED = True
        try:
            from fit_diagnostics import generate_family_fit_pdfs
            _FIT_DIAGNOSTICS_GENERATOR = generate_family_fit_pdfs
        except ModuleNotFoundError as error:
            _FIT_DIAGNOSTICS_IMPORT_ERROR = error.name or str(error)

    if _FIT_DIAGNOSTICS_GENERATOR is None and not _FIT_DIAGNOSTICS_WARNING_EMITTED:
        _FIT_DIAGNOSTICS_WARNING_EMITTED = True
        missing = _FIT_DIAGNOSTICS_IMPORT_ERROR or "plotting backend"
        print(
            "[WARNING] fits/ PDF generation is disabled because the optional "
            f"Python package '{missing}' is missing. The campaign will continue. "
            "Install it inside the project environment with: "
            ".venv/bin/python -m pip install matplotlib",
            file=sys.stderr,
        )
    return _FIT_DIAGNOSTICS_GENERATOR

# Only the information needed to recover a physical (E, p, gap, gain) point
# is mandatory.  Third-gas, transport, histogram and historical branches are
# optional so older binary ROOT files remain reusable.
CORE_GAS_DATA_BRANCHES = {
    "gas1",
    "composition1_pct",
    "pressure_bar",
    "gap_mm",
}

FIELD_FROM_NAME = re.compile(r"_(?P<field>[0-9]+(?:\.[0-9]+)?)kVcm_")


class LegacyRootError(ValueError):
    """The ROOT belongs to an older output schema and must not be reused."""


class FieldLimitReached(RuntimeError):
    """The requested gain cannot be pursued without exceeding field limits."""


GAS_ALIASES = {
    "ar": "ar",
    "argon": "ar",
    "he": "he",
    "helium": "he",
    "xe": "xe",
    "xenon": "xe",
    "cf4": "cf4",
    "n2": "n2",
    "co2": "co2",
    "ch4": "ch4",
    "iso": "ic4h10",
    "isobutane": "ic4h10",
    "ic4h10": "ic4h10",
    "c2h2f4": "c2h2f4",
}

GAS_DISPLAY = {
    "ar": "Ar",
    "he": "He",
    "xe": "Xe",
    "cf4": "CF4",
    "n2": "N2",
    "co2": "CO2",
    "ch4": "CH4",
    "ic4h10": "Iso",
    "c2h2f4": "C2H2F4",
}

GAS_FILE_TOKEN = {
    "ic4h10": "iso",
}


EVENT_LOCK = Lock()
PROCESS_LOCK = Lock()
ACTIVE_PROCESSES: set[subprocess.Popen] = set()
STOP_REQUESTED = False


@dataclass(frozen=True)
class Family:
    mixture: str
    components: tuple[tuple[str, float], ...]
    gap_mm: float

    @property
    def composition(self) -> str:
        return composition_key(self.components)

    @property
    def composition_label(self) -> str:
        return format_composition(self.components)

    @property
    def fraction(self) -> float:
        # Kept only for old AlphaPoint JSON compatibility. New scheduling and
        # fitting use the complete composition string.
        return self.components[1][1] if len(self.components) > 1 else 0.0


@dataclass(frozen=True)
class Target:
    pressure_bar: float
    gain: float


@dataclass(frozen=True)
class FieldTarget:
    pressure_bar: float
    field_v_cm: float
    min_npe: int = 20
    max_npe: int = 5000
    target_relative_error: float = 0.03


@dataclass(frozen=True)
class CampaignOptions:
    space_charge: bool = False
    penning: bool = False
    record_excitation_positions: bool = True
    measure_gas_transport: bool = False
    magboltz_collisions: int = 1


@dataclass(frozen=True)
class RefinementOptions:
    min_field_kv_cm: float = 0.05
    max_field_kv_cm: float = 500.0
    max_reduced_field_kv_cm_bar: float = 500.0
    min_field_step_kv_cm: float = 0.5
    min_field_step_fraction: float = 0.01
    max_field_step_fraction: float = 0.25
    duplicate_field_tolerance: float = 0.001
    max_refinement_attempts: int = 40

    def bounds_v_cm(self, pressure_bar: float) -> tuple[float, float]:
        minimum = 1000.0 * self.min_field_kv_cm
        maximum = min(
            1000.0 * self.max_field_kv_cm,
            1000.0 * pressure_bar * self.max_reduced_field_kv_cm_bar,
        )
        return minimum, maximum


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


def normalize_gas_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in GAS_ALIASES:
        raise ValueError(f"Unknown gas identifier: {name}")
    return GAS_ALIASES[key]


def normalize_components(raw: dict) -> tuple[tuple[str, float], ...]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "Each mixture composition must be a mapping, for example "
            "{ar: 99, cf4: 1}"
        )
    if len(raw) > 3:
        raise ValueError("This project currently supports at most three gases")

    components: list[tuple[str, float]] = []
    seen: set[str] = set()
    for gas_name, fraction in raw.items():
        gas = normalize_gas_name(str(gas_name))
        value = float(fraction)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid fraction for {gas_name}: {fraction}")
        if gas in seen:
            raise ValueError(f"Gas {gas} is repeated in one composition")
        seen.add(gas)
        components.append((gas, value))

    total = sum(value for _, value in components)
    if total <= 0.0:
        raise ValueError("At least one gas fraction must be positive")
    if abs(total - 100.0) > 1.0e-6:
        raise ValueError(
            f"Gas fractions must add to 100 %, obtained {total:.12g} %"
        )
    return tuple(components)


def parse_components_argument(text: str) -> tuple[tuple[str, float], ...]:
    raw: dict[str, float] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "Invalid --components value. Use gas:fraction pairs, "
                "for example ar:99,cf4:1"
            )
        gas, fraction = item.split(":", 1)
        raw[gas.strip()] = float(fraction)
    return normalize_components(raw)


def parse_mixture_families(config: dict) -> list[tuple[str, tuple[tuple[str, float], ...]]]:
    raw_mixtures = config.get("mixtures")
    if not isinstance(raw_mixtures, dict) or not raw_mixtures:
        raise ValueError("mixtures must be a non-empty YAML mapping")

    parsed: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for mixture, entries in raw_mixtures.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Mixture {mixture} must contain a non-empty list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Mixture {mixture} still uses the old fraction-list format. "
                    "Use explicit compositions such as - {ar: 99, cf4: 1}."
                )
            components = normalize_components(entry)
            canonical_name = mixture_name_from_components(components)
            if str(mixture) != canonical_name:
                raise ValueError(
                    f"Mixture key {mixture!r} does not match its components. "
                    f"Use {canonical_name!r}."
                )
            parsed.append((str(mixture), components))
    return parsed


def composition_key(components: tuple[tuple[str, float], ...]) -> str:
    return "__".join(
        f"{gas}_{fraction:.12g}" for gas, fraction in components
    )


def format_composition(components: tuple[tuple[str, float], ...]) -> str:
    return " / ".join(
        f"{GAS_DISPLAY.get(gas, gas)} {fraction:g}%"
        for gas, fraction in components
    )


def mixture_name_from_components(components: tuple[tuple[str, float], ...]) -> str:
    return "".join(GAS_DISPLAY.get(gas, gas) for gas, _ in components)


def component_dicts(components: tuple[tuple[str, float], ...]) -> list[dict[str, float]]:
    return [{"gas": gas, "fraction_pct": float(fraction)}
            for gas, fraction in components]


def padded_components(components: tuple[tuple[str, float], ...]):
    padded = list(components) + [("", 0.0)] * (3 - len(components))
    return padded[0], padded[1], padded[2]


def components_from_gas_data(tree) -> tuple[tuple[str, float], ...]:
    values = []
    for index in (1, 2, 3):
        gas = str(scalar(tree, [f"gas{index}"], "")).strip().lower()
        fraction = float(scalar(tree, [f"composition{index}_pct"], 0.0))
        if gas:
            values.append((normalize_gas_name(gas), fraction))
    return tuple(values)


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
        # levelMap and hLevels describe historical diagnostic schemas.  They
        # neither invalidate nor alter the gain point, so they are ignored.
        required_objects = {"gasData", "dataPerPrimaryElectron"}
        missing_objects = required_objects - names
        if missing_objects:
            raise LegacyRootError(
                "missing gain objects: " + ", ".join(sorted(missing_objects))
            )

        tree = root_file["gasData"]
        branches = set(tree.keys())
        missing = CORE_GAS_DATA_BRANCHES - branches
        if missing:
            raise LegacyRootError(
                "gasData is missing core branches: "
                + ",".join(sorted(missing))
            )

        components = components_from_gas_data(tree)
        mixture = mixture_name_from_components(components)
        composition = composition_key(components)
        fraction = components[1][1] if len(components) > 1 else 0.0
        pressure_bar = float(scalar(tree, ["pressure_bar"], math.nan))
        gap_mm = float(scalar(tree, ["gap_mm"], math.nan))
        npe = int(scalar(tree, ["npe"], 0))
        space_charge_enabled = bool(scalar(tree, ["spaceCharge"], False))
        penning_enabled = bool(scalar(tree, ["penning"], False))

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

        excitation_positions_enabled = (
            ("hExcXYZ" in names or "hExcXY" in names) and "hExcZT" in names
        )
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

    if not mixture or not components:
        raise ValueError("ROOT contains no valid gas composition")
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
        composition=composition,
        components=component_dicts(components),
    )
    point.space_charge_enabled = space_charge_enabled
    point.penning_enabled = penning_enabled
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
        composition=job.family.composition,
        components=component_dicts(job.family.components),
    )
    point.space_charge_enabled = job.options.space_charge
    point.penning_enabled = job.options.penning
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
    """Read reusable gain points without ever moving or deleting a ROOT."""
    points: list[AlphaPoint] = []
    if not ROOT_OUTPUT.exists():
        return points
    for path in sorted(ROOT_OUTPUT.rglob("*.root")):
        try:
            points.append(read_root(path))
        except Exception as error:
            print(
                f"[WARNING] Keeping but not reusing ROOT {path}: {error}",
                file=sys.stderr,
            )
    return points


def alpha_path(
    mixture: str,
    gap_mm: float,
    space_charge: bool = False,
    penning: bool = False,
) -> Path:
    suffix = ""
    if space_charge:
        suffix += "_spacecharge"
    if penning:
        suffix += "_penning"
    return ALPHA_OUTPUT / mixture / f"gap_{gap_mm:.3f}mm{suffix}.json"


def point_flag(point: AlphaPoint, name: str, default: bool = False) -> bool:
    return bool(getattr(point, name, default))


def point_composition(point: AlphaPoint) -> str:
    value = str(getattr(point, "composition", ""))
    if value:
        return value
    return f"fraction_{point.fraction:g}"


def point_components(point: AlphaPoint) -> tuple[tuple[str, float], ...]:
    raw = getattr(point, "components", [])
    values = []
    for item in raw:
        if isinstance(item, dict) and "gas" in item:
            values.append((normalize_gas_name(item["gas"]), float(item["fraction_pct"])))
    if values:
        return tuple(values)
    raise ValueError(f"Point {point.root} has no explicit gas composition")


def save_alpha_for(
    mixture: str,
    gap_mm: float,
    points: list[AlphaPoint],
    space_charge: bool,
    penning: bool = False,
    *,
    generate_diagnostics: bool = True,
    fit_gain_min: float = 1.0,
    fit_gain_max: float = 1.0e7,
    only_composition: str | None = None,
) -> dict[str, dict]:
    """Rebuild the progressive fit and its auditable PDFs.

    This is called after every new ROOT, so the following proposal always sees
    the newest physical point. The PDF generation is deliberately colocated
    with the JSON update: a fit cannot silently change without its diagnostic
    changing too.
    """
    selected = [
        point for point in points
        if point.mixture == mixture
        and abs(point.gap_mm - gap_mm) < 1.0e-9
        and point_flag(point, "space_charge_enabled") == space_charge
        and point_flag(point, "penning_enabled") == penning
    ]
    write_alpha_file(
        alpha_path(mixture, gap_mm, space_charge, penning), mixture, gap_mm, selected
    )
    if not generate_diagnostics:
        return {}
    generator = _load_fit_diagnostics_generator()
    if generator is None:
        return {}
    try:
        return generator(
            mixture,
            gap_mm,
            selected,
            space_charge=space_charge,
            penning=penning,
            gain_min=fit_gain_min,
            gain_max=fit_gain_max,
            only_composition=only_composition,
        )
    except Exception as error:
        print(
            f"[WARNING] Could not update fits/ PDFs; the campaign will continue: {error}",
            file=sys.stderr,
        )
        return {}


def family_points(
    points: list[AlphaPoint], family: Family, options: CampaignOptions
) -> list[AlphaPoint]:
    # Space charge and Penning transfer change the avalanche physics and
    # therefore define separate alpha families. The measurement toggles affect
    # output content only, not gain.
    return [
        point for point in points
        if point.mixture == family.mixture
        and point_composition(point) == family.composition
        and abs(point.gap_mm - family.gap_mm) < 1.0e-9
        and point_flag(point, "space_charge_enabled") == options.space_charge
        and point_flag(point, "penning_enabled") == options.penning
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
        and point_composition(point) == family.composition
        and abs(point.gap_mm - family.gap_mm) < 1.0e-9
        and point_flag(point, "space_charge_enabled") == options.space_charge
        and point_flag(point, "penning_enabled") == options.penning
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

    summaries = combine_duplicate_points(candidates)
    if not summaries:
        return None
    best = min(
        summaries,
        key=lambda summary: abs(summary.gain - target.gain) / target.gain,
    )
    difference = abs(best.gain - target.gain) / target.gain
    if difference > tolerance:
        return None

    # Callers only need a real point to signal that the target is complete.
    return min(
        candidates,
        key=lambda point: abs(point.field_v_cm - best.field_v_cm),
    )


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
    """Return only a progressively selected and validated alpha fit."""
    fit = fit_alpha(points)
    if fit is None:
        return None
    if not fit.usable_for_prediction:
        return None
    if fit.n_unique_points < 4:
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


def _weighted_isotonic_log_gains(summaries: list[AlphaPoint]) -> list[AlphaPoint]:
    """Return field-ordered points with non-decreasing, noise-smoothed gains.

    A weighted pool-adjacent-violators algorithm is used on ln(G). Original
    measurements are never changed on disk; this monotonic copy is used only by
    the local field controller. It prevents one noisy inversion from creating a
    physically impossible bracket.
    """
    ordered = sorted(summaries, key=lambda point: point.field_v_cm)
    if not ordered:
        return []

    blocks: list[dict] = []
    for index, point in enumerate(ordered):
        relative_error = point.relative_gain_error
        if not math.isfinite(relative_error) or relative_error <= 0.0:
            relative_error = 0.15
        statistical_weight = 1.0 / max(relative_error, 0.03) ** 2
        # Cap the weight so a huge accumulated field cannot erase neighbouring
        # physical information completely.
        weight = float(np.clip(statistical_weight, 1.0, 2500.0))
        blocks.append({
            "start": index,
            "stop": index + 1,
            "weight": weight,
            "value": math.log(point.gain),
        })
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left["weight"] + right["weight"]
            blocks.append({
                "start": left["start"],
                "stop": right["stop"],
                "weight": total_weight,
                "value": (
                    left["weight"] * left["value"]
                    + right["weight"] * right["value"]
                ) / total_weight,
            })

    adjusted = [0.0] * len(ordered)
    for block in blocks:
        for index in range(block["start"], block["stop"]):
            adjusted[index] = math.exp(block["value"])
    return [replace(point, gain=gain) for point, gain in zip(ordered, adjusted)]


def _monotonic_summaries(points: list[AlphaPoint]) -> list[AlphaPoint]:
    return _weighted_isotonic_log_gains(combine_duplicate_points(points))


def _gain_bracket(
    summaries: list[AlphaPoint], target_gain: float,
) -> tuple[AlphaPoint, AlphaPoint] | None:
    """Return the narrowest field interval bracketing the requested gain."""
    ordered = sorted(summaries, key=lambda point: point.field_v_cm)
    candidates = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        if left.gain <= target_gain <= right.gain and right.field_v_cm > left.field_v_cm:
            candidates.append((left, right))
    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair[1].field_v_cm - pair[0].field_v_cm)


def interpolate_field(points: list[AlphaPoint], target: Target) -> float | None:
    same_pressure = [
        point for point in points
        if abs(point.pressure_bar - target.pressure_bar) < 1.0e-9
    ]
    summaries = _monotonic_summaries(same_pressure)
    bracket = _gain_bracket(summaries, target.gain)
    if bracket is None:
        return None

    left, right = bracket
    denominator = math.log(right.gain) - math.log(left.gain)
    if denominator <= 1.0e-12:
        return 0.5 * (left.field_v_cm + right.field_v_cm)
    fraction = (
        math.log(target.gain) - math.log(left.gain)
    ) / denominator
    # A safeguarded regula-falsi step. Staying away from the exact bracket edge
    # guarantees that the interval contracts after the next measurement.
    fraction = float(np.clip(fraction, 0.10, 0.90))
    return left.field_v_cm + fraction * (right.field_v_cm - left.field_v_cm)


def _same_pressure_points(
    points: list[AlphaPoint], target: Target,
) -> list[AlphaPoint]:
    return sorted(
        (
            point for point in points
            if abs(point.pressure_bar - target.pressure_bar) < 1.0e-9
            and point.gain > 1.0
            and math.isfinite(point.gain)
            and point.field_v_cm > 0.0
            and math.isfinite(point.field_v_cm)
        ),
        key=lambda point: point.field_v_cm,
    )


def _edge_log_gain_extrapolation(
    summaries: list[AlphaPoint], target_gain: float, *, upwards: bool,
) -> float | None:
    """Robust local extrapolation of ln(G) versus E at a measured edge."""
    ordered = _weighted_isotonic_log_gains(summaries)
    if len(ordered) < 2:
        return None

    edge = ordered[-1] if upwards else ordered[0]
    neighbours = list(reversed(ordered[:-1])) if upwards else ordered[1:]
    slopes = []
    for neighbour in neighbours[:4]:
        if upwards:
            delta_e = edge.field_v_cm - neighbour.field_v_cm
            delta_log_gain = math.log(edge.gain) - math.log(neighbour.gain)
        else:
            delta_e = neighbour.field_v_cm - edge.field_v_cm
            delta_log_gain = math.log(neighbour.gain) - math.log(edge.gain)
        if delta_e > 1.0e-9 and delta_log_gain > 1.0e-6:
            slopes.append(delta_log_gain / delta_e)
    if not slopes:
        return None

    slope = float(np.median(slopes))
    proposal = edge.field_v_cm + math.log(target_gain / edge.gain) / slope
    return proposal if math.isfinite(proposal) else None


def _enforce_refinement_direction(
    field: float,
    same_pressure: list[AlphaPoint],
    target: Target,
    refinement: RefinementOptions,
) -> tuple[float, str | None]:
    """Force an unbracketed target beyond the correct measured field edge."""
    summaries = _monotonic_summaries(same_pressure)
    if not summaries:
        return field, None

    lower = [summary for summary in summaries if summary.gain < target.gain]
    upper = [summary for summary in summaries if summary.gain > target.gain]
    min_step_abs = 1000.0 * refinement.min_field_step_kv_cm

    if lower and not upper:
        edge = summaries[-1]
        extrapolated = _edge_log_gain_extrapolation(
            summaries, target.gain, upwards=True
        )
        field = extrapolated if extrapolated is not None else max(field, edge.field_v_cm)
        minimum_next = edge.field_v_cm + max(
            min_step_abs, refinement.min_field_step_fraction * edge.field_v_cm,
        )
        maximum_next = edge.field_v_cm * (1.0 + refinement.max_field_step_fraction)
        return float(np.clip(field, minimum_next, maximum_next)), (
            "same-pressure monotonic upper-edge extrapolation"
        )

    if upper and not lower:
        edge = summaries[0]
        extrapolated = _edge_log_gain_extrapolation(
            summaries, target.gain, upwards=False
        )
        field = extrapolated if extrapolated is not None else min(field, edge.field_v_cm)
        maximum_next = edge.field_v_cm - max(
            min_step_abs, refinement.min_field_step_fraction * edge.field_v_cm,
        )
        minimum_next = edge.field_v_cm * (1.0 - refinement.max_field_step_fraction)
        return float(np.clip(field, minimum_next, maximum_next)), (
            "same-pressure monotonic lower-edge extrapolation"
        )

    return field, None


def _duplicate_radius(field_v_cm: float, refinement: RefinementOptions) -> float:
    return max(
        1.0,
        refinement.duplicate_field_tolerance * max(abs(field_v_cm), 1.0),
    )


def _field_is_occupied(
    field: float, summaries: list[AlphaPoint], refinement: RefinementOptions,
) -> bool:
    return any(
        abs(summary.field_v_cm - field)
        <= _duplicate_radius(summary.field_v_cm, refinement)
        for summary in summaries
    )


def _free_field_inside_bracket(
    preferred: float,
    left: AlphaPoint,
    right: AlphaPoint,
    summaries: list[AlphaPoint],
    refinement: RefinementOptions,
) -> float | None:
    """Find an unused point inside a measured gain bracket.

    The old duplicate protection could move away from one occupied field and
    land exactly on another one. Here all occupied exclusion zones are merged
    first, and the widest genuinely free interval is selected.
    """
    lower = left.field_v_cm
    upper = right.field_v_cm
    if upper <= lower:
        return None

    forbidden: list[tuple[float, float]] = []
    for summary in summaries:
        radius = _duplicate_radius(summary.field_v_cm, refinement)
        start = max(lower, summary.field_v_cm - radius)
        stop = min(upper, summary.field_v_cm + radius)
        if stop > start:
            forbidden.append((start, stop))
    forbidden.sort()

    merged: list[list[float]] = []
    for start, stop in forbidden:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)

    free: list[tuple[float, float]] = []
    cursor = lower
    for start, stop in merged:
        if start > cursor:
            free.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < upper:
        free.append((cursor, upper))

    usable = [interval for interval in free if interval[1] - interval[0] > 1.0]
    if not usable:
        return None

    def interval_score(interval: tuple[float, float]) -> tuple[float, float]:
        start, stop = interval
        distance = 0.0
        if preferred < start:
            distance = start - preferred
        elif preferred > stop:
            distance = preferred - stop
        return distance, -(stop - start)

    start, stop = min(usable, key=interval_score)
    candidate = float(np.clip(preferred, start, stop))
    margin = min(1.0, 0.25 * (stop - start))
    candidate = float(np.clip(candidate, start + margin, stop - margin))
    return candidate


def _avoid_duplicate_field(
    field: float,
    same_pressure: list[AlphaPoint],
    target: Target,
    refinement: RefinementOptions,
) -> float:
    occupied_summaries = combine_duplicate_points(same_pressure)
    if not occupied_summaries or not _field_is_occupied(
        field, occupied_summaries, refinement
    ):
        return field

    monotonic_summaries = _weighted_isotonic_log_gains(occupied_summaries)
    bracket = _gain_bracket(monotonic_summaries, target.gain)
    if bracket is not None:
        candidate = _free_field_inside_bracket(
            field, bracket[0], bracket[1], occupied_summaries, refinement
        )
        if candidate is not None:
            return candidate

    nearest = min(
        occupied_summaries, key=lambda summary: abs(summary.field_v_cm - field)
    )
    if all(summary.gain < target.gain for summary in occupied_summaries):
        direction = 1.0
    elif all(summary.gain > target.gain for summary in occupied_summaries):
        direction = -1.0
    else:
        direction = 1.0 if nearest.gain < target.gain else -1.0

    candidate = field
    for _ in range(len(occupied_summaries) + 3):
        collisions = [
            summary for summary in occupied_summaries
            if abs(summary.field_v_cm - candidate)
            <= _duplicate_radius(summary.field_v_cm, refinement)
        ]
        if not collisions:
            return candidate
        obstacle = min(
            collisions, key=lambda summary: abs(summary.field_v_cm - candidate)
        )
        step = max(
            1000.0 * refinement.min_field_step_kv_cm,
            refinement.min_field_step_fraction * obstacle.field_v_cm,
        )
        candidate = obstacle.field_v_cm + direction * step

    return candidate

def _enforce_last_attempt_progress(
    field: float,
    same_pressure: list[AlphaPoint],
    last_attempt: AlphaPoint | None,
    target: Target,
    refinement: RefinementOptions,
) -> tuple[float, str | None]:
    """Force the next refinement to move in the direction required by the
    immediately preceding attempt for this exact target.

    Model fits and old noisy points are never allowed to override this rule:
    a gain above target must lower E, and a gain below target must raise E.
    """
    if last_attempt is None or last_attempt.gain <= 0.0:
        return field, None
    if abs(last_attempt.pressure_bar - target.pressure_bar) > 1.0e-9:
        return field, None

    if math.isclose(last_attempt.gain, target.gain, rel_tol=1.0e-12):
        return field, None

    direction = -1.0 if last_attempt.gain > target.gain else 1.0
    step = max(
        1000.0 * refinement.min_field_step_kv_cm,
        refinement.min_field_step_fraction * last_attempt.field_v_cm,
    )
    if direction < 0.0:
        nearest_allowed = last_attempt.field_v_cm - step
        furthest_allowed = last_attempt.field_v_cm * (
            1.0 - refinement.max_field_step_fraction
        )
        field = float(np.clip(field, furthest_allowed, nearest_allowed))
        method = "last-attempt forced downward step"
    else:
        nearest_allowed = last_attempt.field_v_cm + step
        furthest_allowed = last_attempt.field_v_cm * (
            1.0 + refinement.max_field_step_fraction
        )
        field = float(np.clip(field, nearest_allowed, furthest_allowed))
        method = "last-attempt forced upward step"

    summaries = combine_duplicate_points(same_pressure)
    candidate = field
    for _ in range(len(summaries) + 4):
        collisions = [
            summary for summary in summaries
            if abs(summary.field_v_cm - candidate)
            <= _duplicate_radius(summary.field_v_cm, refinement)
        ]
        if not collisions:
            return candidate, method
        obstacle = min(
            collisions, key=lambda summary: abs(summary.field_v_cm - candidate)
        )
        obstacle_step = max(
            1000.0 * refinement.min_field_step_kv_cm,
            refinement.min_field_step_fraction * obstacle.field_v_cm,
        )
        candidate = obstacle.field_v_cm + direction * obstacle_step

    return candidate, method


def propose_field(
    points: list[AlphaPoint],
    target: Target,
    gap_mm: float,
    refinement: RefinementOptions,
    last_attempt: AlphaPoint | None = None,
) -> float:
    # 1. Direct interpolation at the same pressure is the safest prediction.
    interpolated = interpolate_field(points, target)
    if interpolated is not None:
        field = interpolated
        predictor = "same-pressure interpolation"
    else:
        # 2. The reduced-alpha model is used only after a broad reference-pressure
        # seed exists. Its inversion explicitly targets alpha_eff / p.
        fit = _usable_alpha_fit(points)
        if fit is not None:
            try:
                field = field_for_gain(
                    fit, target.pressure_bar, gap_mm, target.gain
                )
                extrapolation = fit_extrapolation_fraction(
                    fit, target.pressure_bar, field
                )
                if extrapolation > 0.75:
                    raise ValueError(
                        "target lies too far outside the measured E/p range"
                    )
                predictor = f"{fit.model_label} reduced-alpha fit"
            except ValueError:
                field = math.nan
                predictor = "fit inversion failed"
        else:
            field = math.nan
            predictor = "reference-pressure seed"

        same_pressure = _same_pressure_points(points, target)

        # 3. Without a valid fit, only use measurements at the SAME pressure.
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

        # 4. For sparse pressure coverage, transport reduced alpha rather than
        # a constant E/p.
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

        # 5. First seed for a completely new family.
        if not math.isfinite(field):
            target_alpha = gain_to_alpha(target.gain, gap_mm)
            reduced_field_kv = float(np.clip(
                8.0 + 12.0 * math.log10(1.0 + target_alpha),
                2.0,
                250.0,
            ))
            field = 1000.0 * target.pressure_bar * reduced_field_kv

    same_pressure = _same_pressure_points(points, target)

    # A fit is never allowed to point back inside the measured range when every
    # point is on the same side of the requested gain.  This is what previously
    # caused scans to hover around ~50 kV/cm indefinitely.
    field, direction_predictor = _enforce_refinement_direction(
        field, same_pressure, target, refinement
    )
    if direction_predictor:
        predictor = direction_predictor

    field = _avoid_duplicate_field(
        field, same_pressure, target, refinement
    )

    # Absolute safety rule: the newest result for this exact target determines
    # the sign of the next field change. A global fit or an old outlier may
    # choose the magnitude, but never the wrong direction.
    field, last_attempt_predictor = _enforce_last_attempt_progress(
        field, same_pressure, last_attempt, target, refinement
    )
    if last_attempt_predictor:
        predictor = last_attempt_predictor

    minimum_field, maximum_field = refinement.bounds_v_cm(target.pressure_bar)
    occupied_summaries = combine_duplicate_points(same_pressure)
    summaries = _weighted_isotonic_log_gains(occupied_summaries)
    below_only = bool(summaries) and all(
        summary.gain < target.gain for summary in summaries
    )
    above_only = bool(summaries) and all(
        summary.gain > target.gain for summary in summaries
    )

    if below_only:
        edge = max(summaries, key=lambda summary: summary.field_v_cm)
        if edge.field_v_cm >= maximum_field * (1.0 - 1.0e-9):
            raise FieldLimitReached(
                f"Target gain {target.gain:g} remains above the measured gain "
                f"{edge.gain:g} at the configured maximum field "
                f"{maximum_field / 1000.0:g} kV/cm"
            )
    if above_only:
        edge = min(summaries, key=lambda summary: summary.field_v_cm)
        if edge.field_v_cm <= minimum_field * (1.0 + 1.0e-9):
            raise FieldLimitReached(
                f"Target gain {target.gain:g} remains below the measured gain "
                f"{edge.gain:g} at the configured minimum field "
                f"{minimum_field / 1000.0:g} kV/cm"
            )

    unclipped_field = field
    field = float(np.clip(field, minimum_field, maximum_field))
    hit_limit = not math.isclose(
        field, unclipped_field, rel_tol=1.0e-12, abs_tol=1.0e-9
    )

    # Never silently rerun the edge point after clipping to a limit.
    if occupied_summaries and _field_is_occupied(
        field, occupied_summaries, refinement
    ):
        if below_only and math.isclose(field, maximum_field, rel_tol=1.0e-9):
            raise FieldLimitReached(
                f"The next refinement for target gain {target.gain:g} would "
                f"exceed max_field_kv_cm={maximum_field / 1000.0:g}"
            )
        if above_only and math.isclose(field, minimum_field, rel_tol=1.0e-9):
            raise FieldLimitReached(
                f"The next refinement for target gain {target.gain:g} would "
                f"fall below min_field_kv_cm={minimum_field / 1000.0:g}"
            )

    reduced_field_kv = (field / 1000.0) / target.pressure_bar
    emit(
        "prediction",
        pressure_bar=target.pressure_bar,
        target_gain=target.gain,
        field_v_cm=field,
        reduced_field_kv_cm_bar=reduced_field_kv,
        predictor=predictor,
        field_limited=hit_limit,
        minimum_field_kv_cm=minimum_field / 1000.0,
        maximum_field_kv_cm=maximum_field / 1000.0,
    )
    return field

def adaptive_npe(gap_mm: float, target_gain: float) -> tuple[int, int, float]:
    """Choose enough statistics for a meaningful gain refinement.

    The former budget frequently capped high-gain runs at exactly 20 primaries,
    making a 5 % gain target statistically impossible. Exploration still starts
    cheaply, but the simulator may now accumulate a useful error estimate.
    """
    transport_budget = 60000.0
    max_npe = int(transport_budget / max(gap_mm * target_gain, 1.0e-6))
    minimum_cap = 30 if target_gain >= 1.0e5 else 50
    max_npe = int(np.clip(max_npe, minimum_cap, 5000))
    min_npe = min(20, max_npe)
    return min_npe, max_npe, 0.04


def log_saved_attempt(
    job: Job,
    point: AlphaPoint,
    *,
    accepted: bool,
) -> None:
    """Append one immutable record for every successfully written ROOT."""
    requested_gain = (
        float(job.target.gain) if isinstance(job.target, Target) else None
    )
    relative_difference = None
    if requested_gain is not None and requested_gain > 0.0:
        relative_difference = abs(point.gain - requested_gain) / requested_gain

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scan_mode": job.scan_mode,
        "job_id": job.job_id,
        "root": point.root,
        "mixture": job.family.mixture,
        "composition": job.family.composition_label,
        "composition_key": job.family.composition,
        "components": component_dicts(job.family.components),
        "pressure_bar": point.pressure_bar,
        "gap_mm": point.gap_mm,
        "field_v_cm": point.field_v_cm,
        "requested_gain": requested_gain,
        "measured_gain": point.gain,
        "gain_error": point.gain_error,
        "relative_difference": relative_difference,
        "accepted": bool(accepted),
        "npe": point.npe,
        "space_charge": job.options.space_charge,
        "penning": job.options.penning,
        "record_excitation_positions": job.options.record_excitation_positions,
        "measure_gas_transport": job.options.measure_gas_transport,
    }
    ATTEMPT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ATTEMPT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def _history_key(
    mixture: str,
    composition: str,
    gap_mm: float,
    pressure_bar: float,
    target_gain: float,
    space_charge: bool,
    penning: bool,
) -> tuple:
    return (
        mixture,
        composition,
        round(float(gap_mm), 9),
        round(float(pressure_bar), 9),
        round(float(target_gain), 9),
        bool(space_charge),
        bool(penning),
    )


def load_last_gain_attempts() -> dict[tuple, tuple[AlphaPoint, int, bool]]:
    """Recover target-specific refinement state across GUI/process restarts.

    The final integer is the JSONL line number and therefore a stable ordering
    even when old records lack or contain malformed timestamps.
    """
    recovered: dict[tuple, tuple[AlphaPoint, int, bool]] = {}
    if not ATTEMPT_LOG.exists():
        return recovered

    for line_number, line in enumerate(
        ATTEMPT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("scan_mode") != "gain":
            continue
        requested_gain = record.get("requested_gain")
        if not isinstance(requested_gain, (int, float)) or requested_gain <= 0.0:
            continue
        try:
            mixture = str(record["mixture"])
            composition = str(record["composition_key"])
            pressure_bar = float(record["pressure_bar"])
            gap_mm = float(record["gap_mm"])
            field_v_cm = float(record["field_v_cm"])
            measured_gain = float(record["measured_gain"])
            gain_error = float(record.get("gain_error", math.nan))
            npe = int(record.get("npe", 1))
            space_charge = bool(record.get("space_charge", False))
            penning = bool(record.get("penning", False))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (
            pressure_bar, gap_mm, field_v_cm, measured_gain,
        )):
            continue

        components = record.get("components", [])
        fraction = 0.0
        if isinstance(components, list) and len(components) > 1:
            try:
                fraction = float(components[1].get("fraction_pct", 0.0))
            except (TypeError, ValueError, AttributeError):
                fraction = 0.0
        alpha = gain_to_alpha(measured_gain, gap_mm)
        alpha_error = (
            gain_error / (measured_gain * 0.1 * gap_mm)
            if measured_gain > 1.0 and gain_error >= 0.0 and math.isfinite(gain_error)
            else math.nan
        )
        point = AlphaPoint(
            mixture=mixture,
            fraction=fraction,
            pressure_bar=pressure_bar,
            gap_mm=gap_mm,
            field_v_cm=field_v_cm,
            gain=measured_gain,
            gain_error=gain_error,
            alpha_effective=alpha,
            alpha_error=alpha_error,
            npe=max(npe, 1),
            root=str(record.get("root", "")),
            composition=composition,
            components=components if isinstance(components, list) else [],
        )
        point.space_charge_enabled = space_charge
        point.penning_enabled = penning
        key = _history_key(
            mixture, composition, gap_mm, pressure_bar,
            float(requested_gain), space_charge, penning,
        )
        recovered[key] = (point, line_number, bool(record.get("accepted", False)))
    return recovered


def last_attempt_for_target(
    recovered: dict[tuple, tuple[AlphaPoint, int, bool]],
    family: Family,
    target: Target,
    options: CampaignOptions,
) -> tuple[AlphaPoint | None, int, bool]:
    value = recovered.get(_history_key(
        family.mixture,
        family.composition,
        family.gap_mm,
        target.pressure_bar,
        target.gain,
        options.space_charge,
        options.penning,
    ))
    if value is None:
        return None, -1, False
    return value


def root_directory(family: Family) -> Path:
    return ROOT_OUTPUT / family.mixture / f"gap_{family.gap_mm:.3f}mm"


def unique_root_name(point: AlphaPoint) -> Path:
    components = point_components(point)
    family = Family(point.mixture, components, point.gap_mm)
    folder = root_directory(family)
    folder.mkdir(parents=True, exist_ok=True)

    gas_part = "_".join(
        f"{GAS_FILE_TOKEN.get(gas, gas)}_{fraction:.1f}"
        for gas, fraction in components
    )
    stem = (
        f"{gas_part}_"
        f"{point.field_kv_cm:.1f}kVcm_"
        f"{point.pressure_bar:.3f}bar_"
        f"{point.gap_mm:.4f}mm_"
        f"{point.npe}npe"
    )
    candidate = folder / f"{stem}.root"
    if not candidate.exists():
        return candidate

    # Never overwrite a previous simulation.  Repeated/refinement attempts at
    # the same rounded field and npe remain independent physical data points.
    run_index = 2
    while True:
        versioned = folder / f"{stem}_run{run_index}.root"
        if not versioned.exists():
            return versioned
        run_index += 1

def run_job(job: Job) -> AlphaPoint:
    (gas1, comp1), (gas2, comp2), (gas3, comp3) = padded_components(
        job.family.components
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
        gas3,
        f"{comp3:.12g}",
        str(job.options.magboltz_collisions),
        str(int(job.options.penning)),
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
        composition=job.family.composition_label,
        composition_key=job.family.composition,
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
        if line.startswith("MAGBOLTZ_START"):
            parts = line.split()
            payload = {"job_id": job.job_id, "field_v_cm": job.field_v_cm}
            if len(parts) >= 3:
                try:
                    payload["field_v_cm"] = float(parts[2])
                except ValueError:
                    pass
            emit("transport_started", **payload)
        elif line.startswith("MAGBOLTZ_DONE"):
            parts = line.split()
            seconds = 0.0
            if len(parts) >= 3:
                try:
                    seconds = float(parts[2])
                except ValueError:
                    pass
            emit(
                "transport_finished",
                job_id=job.job_id,
                field_v_cm=job.field_v_cm,
                seconds=seconds,
            )
        elif line.startswith("PROGRESS"):
            parts = line.split()
            if len(parts) >= 4:
                payload = {
                    "job_id": job.job_id,
                    "current": int(parts[2]),
                    "maximum": int(parts[3]),
                }
                if len(parts) >= 6:
                    try:
                        payload["running_gain"] = float(parts[4])
                        payload["relative_error"] = float(parts[5])
                    except ValueError:
                        pass
                emit("progress", **payload)
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


def _float_values(raw, *, label: str) -> list[float]:
    if isinstance(raw, (int, float)):
        values = [float(raw)]
    elif isinstance(raw, list):
        values = [float(value) for value in raw]
    else:
        raise ValueError(f"{label} must be a number or a non-empty list")
    if not values:
        raise ValueError(f"{label} must not be empty")
    return values


def _set_pressures(config: dict, entry: dict, *, label: str) -> list[float]:
    raw = entry.get("pressures_bar", config.get("pressures_bar"))
    if raw is None:
        raise ValueError(f"{label} requires pressures_bar")
    pressures = _float_values(raw, label=f"{label}.pressures_bar")
    if any(value <= 0.0 for value in pressures):
        raise ValueError(f"{label}.pressures_bar values must be positive")
    return pressures


def _set_gaps(entry: dict, *, label: str) -> list[float]:
    has_gap = "gap_mm" in entry
    has_gaps = "gaps_mm" in entry
    if has_gap == has_gaps:
        raise ValueError(
            f"{label} must contain exactly one of gap_mm or gaps_mm"
        )
    raw = entry["gap_mm"] if has_gap else entry["gaps_mm"]
    gaps = _float_values(raw, label=f"{label}.gap_mm")
    if any(value <= 0.0 for value in gaps):
        raise ValueError(f"{label} gap values must be positive")
    return gaps


def _set_components(entry: dict, *, label: str):
    raw = entry.get("components")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{label}.components must be a non-empty mapping")
    components = normalize_components(raw)
    return mixture_name_from_components(components), components


def _field_values_kv_cm(entry: dict, *, label: str) -> list[float]:
    direct = entry.get("fields_kv_cm")
    has_reference = "reference_field_kv_cm" in entry
    has_scales = "field_scales" in entry
    if direct is not None and (has_reference or has_scales):
        raise ValueError(
            f"{label} cannot mix fields_kv_cm with "
            "reference_field_kv_cm/field_scales"
        )
    if direct is not None:
        fields = _float_values(direct, label=f"{label}.fields_kv_cm")
    else:
        if not (has_reference and has_scales):
            raise ValueError(
                f"{label} requires fields_kv_cm or both "
                "reference_field_kv_cm and field_scales"
            )
        reference = float(entry["reference_field_kv_cm"])
        scales = _float_values(entry["field_scales"], label=f"{label}.field_scales")
        fields = [reference * scale for scale in scales]
    if any(value <= 0.0 for value in fields):
        raise ValueError(f"{label} electric fields must be positive")
    # Preserve the YAML order while avoiding accidental duplicate jobs.
    return list(dict.fromkeys(round(value, 12) for value in fields))


def _field_statistics(config: dict, entry: dict | None = None) -> tuple[int, int, float]:
    statistics_keys = ("npe", "min_npe", "max_npe", "target_relative_error")
    source = dict(config)
    if entry is not None and any(key in entry for key in statistics_keys):
        for key in statistics_keys:
            source.pop(key, None)
        for key in statistics_keys:
            if key in entry:
                source[key] = entry[key]
    return direct_field_npe(source)


def campaign_targets(config: dict) -> tuple[list[Family], dict[Family, list[Target]]]:
    raw_sets = config.get("gain_sets")
    if raw_sets is not None:
        if not isinstance(raw_sets, list) or not raw_sets:
            raise ValueError("gain_sets must be a non-empty list")
        families: list[Family] = []
        targets: dict[Family, list[Target]] = {}
        for index, entry in enumerate(raw_sets):
            label = f"gain_sets[{index}]"
            if not isinstance(entry, dict):
                raise ValueError(f"{label} must be a mapping")
            mixture, components = _set_components(entry, label=label)
            pressures = _set_pressures(config, entry, label=label)
            gaps = _set_gaps(entry, label=label)
            raw_gains = entry.get("gain_targets", entry.get("gain_target"))
            if raw_gains is None:
                raise ValueError(f"{label} requires gain_targets")
            gains = _float_values(raw_gains, label=f"{label}.gain_targets")
            gains = [gain for gain in gains if gain > 1.0]
            if not gains:
                raise ValueError(f"{label}.gain_targets must contain values above 1")
            for gap_mm in gaps:
                family = Family(mixture, components, gap_mm)
                if family not in targets:
                    families.append(family)
                    targets[family] = []
                targets[family].extend(
                    Target(pressure, gain)
                    for pressure in pressures
                    for gain in gains
                )
        return families, targets

    pressures = [float(value) for value in config["pressures_bar"]]
    gaps = {float(gap): [float(gain) for gain in gains]
            for gap, gains in config["gaps_mm"].items()}

    families: list[Family] = []
    targets: dict[Family, list[Target]] = {}
    for mixture, components in parse_mixture_families(config):
        for gap_mm, gains in gaps.items():
            family = Family(mixture, components, gap_mm)
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
    raw_sets = config.get("field_sets")
    if raw_sets is not None:
        if not isinstance(raw_sets, list) or not raw_sets:
            raise ValueError("field_sets must be a non-empty list")
        families: list[Family] = []
        targets: dict[Family, list[FieldTarget]] = {}
        for index, entry in enumerate(raw_sets):
            label = f"field_sets[{index}]"
            if not isinstance(entry, dict):
                raise ValueError(f"{label} must be a mapping")
            mixture, components = _set_components(entry, label=label)
            pressures = _set_pressures(config, entry, label=label)
            gaps = _set_gaps(entry, label=label)
            fields_kv_cm = _field_values_kv_cm(entry, label=label)
            min_npe, max_npe, target_error = _field_statistics(config, entry)
            for gap_mm in gaps:
                family = Family(mixture, components, gap_mm)
                if family not in targets:
                    families.append(family)
                    targets[family] = []
                targets[family].extend(
                    FieldTarget(
                        pressure, 1000.0 * field_kv_cm,
                        min_npe, max_npe, target_error,
                    )
                    for pressure in pressures
                    for field_kv_cm in fields_kv_cm
                )
        return families, targets

    pressures = [float(value) for value in config["pressures_bar"]]
    raw_fields = config.get("fields_kv_cm")
    if not isinstance(raw_fields, dict) or not raw_fields:
        raise ValueError(
            "field mode requires field_sets or fields_kv_cm, for example: "
            "fields_kv_cm: {0.05: [35]}"
        )
    fields_by_gap = {
        float(gap): [1000.0 * float(field) for field in fields]
        for gap, fields in raw_fields.items()
    }
    min_npe, max_npe, target_error = _field_statistics(config)

    families: list[Family] = []
    targets: dict[Family, list[FieldTarget]] = {}
    for mixture, components in parse_mixture_families(config):
        for gap_mm, fields_v_cm in fields_by_gap.items():
            family = Family(mixture, components, gap_mm)
            families.append(family)
            targets[family] = [
                FieldTarget(
                    pressure, field_v_cm, min_npe, max_npe, target_error
                )
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
    generate_fit_diagnostics: bool = True,
    fit_gain_min: float = 1.0,
    fit_gain_max: float = 1.0e7,
) -> None:
    """Run every requested field exactly once.

    Field mode never changes the requested electric field and does not use a
    gain tolerance or reuse an old ROOT as a completed target.  The measured
    gain is nevertheless added to the shared alpha database and its diagnostic
    PDF is regenerated, so fixed-field scans also improve future predictions.
    """
    jobs_to_run = [
        (family, target)
        for family in families
        for target in targets_by_family[family]
    ]
    diagnostic_points = scan_roots() if generate_fit_diagnostics else []

    emit(
        "campaign_started",
        scan_mode="field",
        families=len(families),
        targets=len(jobs_to_run),
        existing_roots=0,
        workers=workers,
        tolerance=None,
        space_charge=options.space_charge,
        penning=options.penning,
        record_excitation_positions=options.record_excitation_positions,
        measure_gas_transport=options.measure_gas_transport,
        magboltz_collisions=options.magboltz_collisions,
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
                    min_npe=target.min_npe,
                    max_npe=target.max_npe,
                    target_relative_error=target.target_relative_error,
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
                        composition=family.composition_label,
                        composition_key=family.composition,
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
                log_saved_attempt(job, point, accepted=True)
                diagnostic_points.append(point)
                fit_artifacts = save_alpha_for(
                    family.mixture,
                    family.gap_mm,
                    diagnostic_points,
                    options.space_charge,
                    options.penning,
                    generate_diagnostics=generate_fit_diagnostics,
                    fit_gain_min=fit_gain_min,
                    fit_gain_max=fit_gain_max,
                    only_composition=family.composition,
                )
                artifact = fit_artifacts.get(family.composition, {})
                selected_fit = artifact.get("fit")
                emit(
                    "result",
                    job_id=job.job_id,
                    scan_mode="field",
                    mixture=family.mixture,
                    composition=family.composition_label,
                    composition_key=family.composition,
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
                    fit_pdf=artifact.get("path", ""),
                    fit_model=(selected_fit.model_label if selected_fit else "insufficient data"),
                    fit_status=(selected_fit.fit_status if selected_fit else "insufficient_points"),
                    fit_unique_points=(selected_fit.n_unique_points if selected_fit else 0),
                    fit_root_runs=(selected_fit.n_runs if selected_fit else 0),
                    fit_cv_median=(selected_fit.cv_median_factor if selected_fit else None),
                    fit_cv_max=(selected_fit.cv_max_factor if selected_fit else None),
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
    penning_override: bool | None = None,
    skip_build: bool = False,
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Campaign YAML must contain a top-level mapping")
    scan_mode = str(config.get("scan_mode", "gain")).strip().lower()
    if scan_mode not in {"gain", "field"}:
        raise ValueError("scan_mode must be 'gain' or 'field'")
    if scan_mode == "field" and "gain_sets" in config:
        raise ValueError("gain_sets requires scan_mode: gain")
    if scan_mode == "gain" and "field_sets" in config:
        raise ValueError("field_sets requires scan_mode: field")
    tolerance = float(config.get("gain_tolerance", 0.05))
    height_factor = float(config.get("height_factor", 1.5))
    workers = worker_count(config.get("workers", "auto"))
    generate_fit_diagnostics = bool(config.get("generate_fit_diagnostics", True))
    fit_gain_min = float(config.get("fit_gain_min", 1.0))
    fit_gain_max = float(config.get("fit_gain_max", 1.0e7))
    if fit_gain_min < 1.0 or fit_gain_max <= fit_gain_min:
        raise ValueError("fit_gain_min/max must satisfy 1 <= min < max")
    refinement = RefinementOptions(
        min_field_kv_cm=float(config.get("min_field_kv_cm", 0.05)),
        max_field_kv_cm=float(config.get("max_field_kv_cm", 500.0)),
        max_reduced_field_kv_cm_bar=float(
            config.get("max_reduced_field_kv_cm_bar", 500.0)
        ),
        min_field_step_kv_cm=float(config.get("min_field_step_kv_cm", 0.5)),
        min_field_step_fraction=float(
            config.get("min_field_step_fraction", 0.01)
        ),
        max_field_step_fraction=float(
            config.get("max_field_step_fraction", 0.25)
        ),
        duplicate_field_tolerance=float(
            config.get("duplicate_field_tolerance", 0.001)
        ),
        max_refinement_attempts=int(
            config.get("max_refinement_attempts", 40)
        ),
    )
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
            bool(config.get("measure_gas_transport", False))
            if gas_transport_override is None else gas_transport_override
        ),
        magboltz_collisions=int(config.get("magboltz_collisions", 1)),
        penning=(
            bool(config.get("penning", False))
            if penning_override is None else penning_override
        ),
    )

    if scan_mode == "gain" and not 0.0 < tolerance < 1.0:
        raise ValueError("gain_tolerance must be between 0 and 1")
    if height_factor < 1.0:
        raise ValueError("height_factor must be at least 1")
    if options.magboltz_collisions < 1:
        raise ValueError("magboltz_collisions must be at least 1")
    if refinement.min_field_kv_cm <= 0.0:
        raise ValueError("min_field_kv_cm must be positive")
    if refinement.max_field_kv_cm <= refinement.min_field_kv_cm:
        raise ValueError("max_field_kv_cm must exceed min_field_kv_cm")
    if refinement.max_reduced_field_kv_cm_bar <= 0.0:
        raise ValueError("max_reduced_field_kv_cm_bar must be positive")
    if refinement.min_field_step_kv_cm <= 0.0:
        raise ValueError("min_field_step_kv_cm must be positive")
    if not 0.0 < refinement.min_field_step_fraction < 1.0:
        raise ValueError("min_field_step_fraction must be between 0 and 1")
    if not 0.0 < refinement.max_field_step_fraction < 1.0:
        raise ValueError("max_field_step_fraction must be between 0 and 1")
    if refinement.max_field_step_fraction < refinement.min_field_step_fraction:
        raise ValueError(
            "max_field_step_fraction must not be smaller than "
            "min_field_step_fraction"
        )
    if refinement.max_refinement_attempts < 1:
        raise ValueError("max_refinement_attempts must be at least 1")

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
            generate_fit_diagnostics=generate_fit_diagnostics,
            fit_gain_min=fit_gain_min,
            fit_gain_max=fit_gain_max,
        )
        return

    families, targets_by_family = campaign_targets(config)
    points = scan_roots()
    for mixture in dict.fromkeys(family.mixture for family in families):
        for gap_mm in {family.gap_mm for family in families if family.mixture == mixture}:
            save_alpha_for(
                mixture, gap_mm, points, options.space_charge, options.penning,
                generate_diagnostics=generate_fit_diagnostics,
                fit_gain_min=fit_gain_min,
                fit_gain_max=fit_gain_max,
            )

    emit(
        "campaign_started",
        scan_mode="gain",
        families=len(families),
        targets=sum(len(value) for value in targets_by_family.values()),
        existing_roots=len(points),
        workers=workers,
        tolerance=tolerance,
        space_charge=options.space_charge,
        penning=options.penning,
        record_excitation_positions=options.record_excitation_positions,
        measure_gas_transport=options.measure_gas_transport,
        magboltz_collisions=options.magboltz_collisions,
        fits_directory="fits",
        fit_gain_min=fit_gain_min,
        fit_gain_max=fit_gain_max,
    )

    active: dict[Future, Job] = {}
    active_families: set[Family] = set()
    job_id = 0
    failure_count: dict[tuple[Family, Target], int] = {}
    # Keep a missed target at the front of its family queue so the field is
    # corrected immediately instead of opening unrelated pressures first.
    priority_target: dict[Family, Target] = {}
    refinement_attempts: dict[tuple[Family, Target], int] = {}
    last_attempt_point: dict[tuple[Family, Target], AlphaPoint] = {}
    recovered_attempts = load_last_gain_attempts()
    for family in families:
        newest: tuple[int, Target] | None = None
        for target in targets_by_family[family]:
            point, order, accepted = last_attempt_for_target(
                recovered_attempts, family, target, options
            )
            if point is not None and not accepted:
                last_attempt_point[(family, target)] = point
                if newest is None or order > newest[0]:
                    newest = (order, target)
        if newest is not None:
            priority_target[family] = newest[1]
    blocked_targets: set[tuple[Family, Target]] = set()
    reported_exhausted: set[tuple[Family, Target]] = set()
    diagnostic_job_id = -1

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

                for exhausted_target in pending:
                    key = (family, exhausted_target)
                    if (
                        refinement_attempts.get(key, 0)
                        >= refinement.max_refinement_attempts
                        and key not in reported_exhausted
                    ):
                        reported_exhausted.add(key)
                        emit(
                            "failed",
                            job_id=diagnostic_job_id,
                            scan_mode="gain",
                            mixture=family.mixture,
                            composition=family.composition_label,
                            composition_key=family.composition,
                            gap_mm=family.gap_mm,
                            pressure_bar=exhausted_target.pressure_bar,
                            target_gain=exhausted_target.gain,
                            attempt=refinement_attempts.get(key, 0),
                            will_retry=False,
                            error=(
                                "Maximum refinement attempts reached: "
                                f"{refinement.max_refinement_attempts}"
                            ),
                        )
                        diagnostic_job_id -= 1

                possible = [
                    target for target in pending
                    if (family, target) not in blocked_targets
                    and refinement_attempts.get((family, target), 0)
                        < refinement.max_refinement_attempts
                    and failure_count.get((family, target), 0) < 3
                ]
                if not possible:
                    continue

                preferred = priority_target.get(family)
                if preferred is not None and preferred in possible:
                    target = preferred
                else:
                    priority_target.pop(family, None)
                    target = select_target(current_points, possible)
                try:
                    field = propose_field(
                        current_points, target, family.gap_mm, refinement,
                        last_attempt=last_attempt_point.get((family, target)),
                    )
                except FieldLimitReached as error:
                    blocked_targets.add((family, target))
                    priority_target.pop(family, None)
                    emit(
                        "failed",
                        job_id=diagnostic_job_id,
                        scan_mode="gain",
                        mixture=family.mixture,
                        composition=family.composition_label,
                        composition_key=family.composition,
                        gap_mm=family.gap_mm,
                        pressure_bar=target.pressure_bar,
                        target_gain=target.gain,
                        attempt=refinement_attempts.get((family, target), 0),
                        will_retry=False,
                        error=str(error),
                    )
                    diagnostic_job_id -= 1
                    continue
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
                    will_retry = attempt < 3
                    if will_retry:
                        priority_target[family] = requested_target
                    else:
                        priority_target.pop(family, None)
                    emit(
                        "failed",
                        job_id=job.job_id,
                        scan_mode="gain",
                        mixture=family.mixture,
                        composition=family.composition_label,
                        composition_key=family.composition,
                        gap_mm=family.gap_mm,
                        pressure_bar=requested_target.pressure_bar,
                        target_gain=requested_target.gain,
                        attempt=attempt,
                        will_retry=will_retry,
                        error=str(error),
                    )
                    continue

                refinement_attempts[(family, requested_target)] = (
                    refinement_attempts.get((family, requested_target), 0) + 1
                )
                last_attempt_point[(family, requested_target)] = point
                points.append(point)
                fit_artifacts = save_alpha_for(
                    family.mixture,
                    family.gap_mm,
                    points,
                    options.space_charge,
                    options.penning,
                    generate_diagnostics=generate_fit_diagnostics,
                    fit_gain_min=fit_gain_min,
                    fit_gain_max=fit_gain_max,
                    only_composition=family.composition,
                )
                artifact = fit_artifacts.get(family.composition, {})
                selected_fit = artifact.get("fit")
                matched = target_match(
                    family_points(points, family, options), requested_target,
                    tolerance, options,
                )
                if matched is None:
                    priority_target[family] = requested_target
                else:
                    priority_target.pop(family, None)
                    last_attempt_point.pop((family, requested_target), None)
                log_saved_attempt(job, point, accepted=matched is not None)
                emit(
                    "result",
                    job_id=job.job_id,
                    scan_mode="gain",
                    mixture=family.mixture,
                    composition=family.composition_label,
                    composition_key=family.composition,
                    gap_mm=family.gap_mm,
                    pressure_bar=point.pressure_bar,
                    target_gain=requested_target.gain,
                    field_v_cm=point.field_v_cm,
                    gain=point.gain,
                    gain_error=point.gain_error,
                    npe=point.npe,
                    accepted=matched is not None,
                    root=point.root,
                    fit_pdf=artifact.get("path", ""),
                    fit_model=(selected_fit.model_label if selected_fit else "insufficient data"),
                    fit_status=(selected_fit.fit_status if selected_fit else "insufficient_points"),
                    fit_unique_points=(selected_fit.n_unique_points if selected_fit else 0),
                    fit_root_runs=(selected_fit.n_runs if selected_fit else 0),
                    fit_cv_median=(selected_fit.cv_median_factor if selected_fit else None),
                    fit_cv_max=(selected_fit.cv_max_factor if selected_fit else None),
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

    components = parse_components_argument(args.components)
    mixture = mixture_name_from_components(components)
    composition = composition_key(components)
    (gas1, comp1), (gas2, comp2), (gas3, comp3) = padded_components(components)

    field_v_cm = args.field_kv_cm * 1000.0 if args.field_kv_cm else None
    if field_v_cm is None:
        fit = read_fit(
            alpha_path(mixture, args.gap_mm, bool(args.space_charge), bool(args.penning)),
            composition,
        )
        if fit is None:
            raise ValueError(
                "No alpha fit is available for this composition and gap"
            )
        field_v_cm = field_for_gain(
            fit, args.pressure_bar, args.gap_mm, args.gain
        )

    gas_part = "_".join(
        f"{GAS_FILE_TOKEN.get(gas, gas)}_{fraction:.1f}"
        for gas, fraction in components
    )
    gif_name = (
        f"{gas_part}_{field_v_cm/1000.0:.1f}kVcm_"
        f"{args.pressure_bar:.3f}bar_{args.gap_mm:.4f}mm.gif"
    )
    gif_path = GIF_OUTPUT / gif_name
    temporary_root = GIF_OUTPUT / f".gif_{uuid.uuid4().hex}.root"

    command = [
        str(EXECUTABLE),
        str(temporary_root),
        mixture,
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
        "0",
        "0",
        gas3,
        f"{comp3:.12g}",
        "1",  # magboltzCollisions (unused unless gas transport is enabled)
        str(int(bool(args.penning))),
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
    command.add_argument(
        "--components",
        help="Explicit gas composition, for example ar:99,cf4:1",
    )
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
        "--penning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Garfield++ Penning transfer.",
    )
    command.add_argument(
        "--excitation-positions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable hExcXYZ and hExcZT in campaign ROOT files.",
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
        if not args.components:
            raise SystemExit("--components is required in GIF mode")
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
        penning_override=args.penning,
        skip_build=args.no_build,
    )


if __name__ == "__main__":
    main()
