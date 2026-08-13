# electricFieldUniform_multithread

Microscopic electron-avalanche simulations in a **uniform electric field** using
[Garfield++](https://garfieldpp.docs.cern.ch/), Magboltz and ROOT.

The project provides:

- a C++ microscopic avalanche simulator (`uniformE.cxx`);
- automatic campaigns driven by YAML files;
- two campaign modes: **target gain** and **fixed electric field**;
- adaptive statistics and parallel execution;
- optional charged-ring space charge;
- optional Magboltz transport quantities;
- compact excitation-position histograms;
- live avalanche GIFs with optional ion motion and space-charge maps;
- a PySide6 graphical interface for campaigns and GIF generation.

---

## Contents

1. [Physics and geometry](#physics-and-geometry)
2. [Repository structure](#repository-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Graphical interface](#graphical-interface)
7. [Campaigns](#campaigns)
8. [Target-gain campaigns](#target-gain-campaigns)
9. [Fixed-field campaigns](#fixed-field-campaigns)
10. [Runtime options](#runtime-options)
11. [GIF generation](#gif-generation)
12. [Supported mixtures](#supported-mixtures)
13. [Adding a new mixture](#adding-a-new-mixture)
14. [Extending the project to new cases](#extending-the-project-to-new-cases)
15. [Output files](#output-files)
16. [ROOT contents](#root-contents)
17. [Magboltz transport quantities](#magboltz-transport-quantities)
18. [Performance recommendations](#performance-recommendations)
19. [Troubleshooting](#troubleshooting)

---

## Physics and geometry

The simulation uses `Garfield::AvalancheMicroscopic` with a
`Garfield::MediumMagboltz` gas in a uniform electric field.

The geometry convention is:

- anode at `z = 0`;
- cathode/top plane at `z = gap`;
- primary electrons launched at `z = gap`;
- multiplication distance equal to the physical gap;
- simulation height equal to `height_factor × gap`;
- transverse simulation and histogram limits:

  ```text
  x, y in [-2 × gap, +2 × gap]
  ```

The initial transverse position of each primary electron is sampled uniformly in
`[-gap, +gap]`.

The simulated gain is the mean number of final electrons per primary electron:

```math
G = \langle n_e \rangle.
```

The effective Townsend coefficient inferred from the avalanche is

```math
\alpha_{\mathrm{eff}} = \frac{\ln G}{d},
```

where `d` is the physical gap in centimetres.

For gain prediction across pressure, the campaign runner fits the reduced model

```math
\frac{\alpha_{\mathrm{eff}}}{p}
=
A\left(\frac{E}{p}\right)^m
\exp\left[-\left(\frac{B}{E/p}\right)^n\right].
```

Each fit family is defined by:

```text
mixture + additive fraction + gap + space-charge state
```

Pressures are combined through the reduced variables `E/p` and
`alpha_eff/p`.

### Space-charge approximation

When space charge is enabled, positive ions produced by previous primary
electrons are accumulated as charged rings using
`Garfield::ComponentChargedRing`. These rings modify the field seen by later
primary avalanches in the same simulation.

This is a low-cost charged-ring approximation. It is not a full microscopic
positive-ion transport simulation.

In GIF mode, positive ions may additionally be moved with a user-defined
constant velocity. That velocity is a **visualisation parameter only** and must
not be interpreted as a measured or predicted ion mobility.

---

## Repository structure

A typical repository should contain:

```text
.
├── CMakeLists.txt
├── uniformE.cxx               # Garfield++ microscopic simulation
├── run_campaign.py            # YAML campaigns and GIF command-line runner
├── alpha_model.py             # progressive alpha/p models and validation
├── fit_diagnostics.py         # auditable PDF diagnostics
├── gui.py                     # PySide6 interface
├── run_gui.sh                 # stable GUI launcher
├── campaign.yaml              # user campaign configuration
├── .venv/                     # local Python environment
├── build/                     # automatically regenerated CMake build
├── fits/                      # fit PDFs only (no PNG diagnostics)
└── outputs/
    ├── roots/                 # production ROOT files
    ├── alpha/                 # alpha-model JSON files
    ├── gifs/                  # generated GIFs
    └── legacy_roots/          # incompatible/old ROOT schemas, when applicable
```

Use `run_gui.sh` as the normal entry point. The campaign runner recreates the
CMake build directory before running, ensuring that the executable corresponds
to the current `uniformE.cxx`.

---

## Requirements

### C++ and scientific software

- Linux is the primary supported platform.
- A C++17-capable compiler.
- A Fortran compiler for Magboltz/Garfield++ builds.
- CMake 3.12 or newer.
- ROOT.
- Garfield++ with Magboltz.
- GNU Scientific Library (GSL).

The official Garfield++ installation documentation lists a C++ compiler, a
Fortran compiler, ROOT, GSL and CMake as prerequisites:

- https://garfieldpp.docs.cern.ch/install/

ROOT installation instructions:

- https://root.cern/install/
- https://root.cern/install/dependencies/

### Python

Recommended: Python 3.10 or newer in a virtual environment.

Required Python packages:

```text
numpy
scipy
uproot
PyYAML
PySide6
matplotlib
```

Qt recommends using PySide6 inside a virtual environment rather than installing
it globally:

- https://doc.qt.io/qtforpython-6/quickstart.html

---

## Installation

### 1. Install basic system tools

Example for Debian/Ubuntu-based systems:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  gfortran \
  cmake \
  libgsl-dev \
  python3 \
  python3-venv \
  python3-pip \
  git
```

Install ROOT and Garfield++ following their official instructions. Binary ROOT
packages are often the simplest option, while Garfield++ can be built and
installed with CMake.

### 2. Configure ROOT

Before configuring this project, ROOT must be available in the shell:

```bash
source /path/to/root/bin/thisroot.sh
```

Verify it:

```bash
root-config --version
```

`run_gui.sh` also tries the common paths:

```text
$HOME/root-install/bin/thisroot.sh
/opt/root/bin/thisroot.sh
```

If ROOT is installed elsewhere, source its `thisroot.sh` before launching the
GUI or edit the launcher.

### 3. Make Garfield++ discoverable by CMake

If Garfield++ is not installed in a standard prefix, add its installation
prefix to `CMAKE_PREFIX_PATH`:

```bash
export CMAKE_PREFIX_PATH="/path/to/garfield-install:${CMAKE_PREFIX_PATH:-}"
```

If the Garfield++ shared library directory is not already known to the dynamic
linker, also use:

```bash
export LD_LIBRARY_PATH="/path/to/garfield-install/lib:${LD_LIBRARY_PATH:-}"
```

The project uses:

```cmake
find_package(Garfield REQUIRED)
target_link_libraries(uniformE Garfield::Garfield ROOT::Tree)
```

### 4. Create the Python environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy scipy uproot PyYAML PySide6
```

### 5. Make the launcher executable

```bash
chmod +x run_gui.sh
```

### 6. Test a manual build

The GUI builds automatically, but a manual test is useful after installation:

```bash
cmake -S . -B build
cmake --build build -j2
```

If CMake cannot locate Garfield++, confirm `CMAKE_PREFIX_PATH` and the Garfield++
installation.

---

## Quick start

Always open the application from the repository root with:

```bash
./run_gui.sh
```

Do not normally start it with `python3 gui.py`. The launcher deliberately:

1. activates `.venv`;
2. loads ROOT when needed;
3. selects the Qt libraries bundled with PySide6;
4. preserves the existing ROOT/Garfield++ library paths;
5. opens the GUI from the correct project directory.

This avoids common Qt conflicts such as incompatible system and PySide6
`libQt6DBus` libraries.

For a first test:

1. Open the **Campaign** tab.
2. Load or paste a small fixed-field YAML campaign.
3. Keep **Space charge** disabled.
4. Keep **Magboltz transport** disabled for the quickest test.
5. Keep **Excitation positions** enabled if `hExcXYZ` and `hExcZT` are needed.
6. Click **Run campaign**.

---

## Graphical interface

The GUI has two tabs.

### Campaign tab

The Campaign tab provides:

- YAML editor;
- open/save YAML buttons;
- runtime switches;
- automatic CMake configuration and build;
- one table row per active or completed simulation;
- measured field and gain;
- number of primary electrons;
- per-job progress;
- status and error details;
- stop button.

The runtime switches are intentionally controlled by checkboxes instead of being
duplicated in the YAML editor:

- **Space charge**
- **Excitation positions**
- **Magboltz transport**

### GIF tab

The GIF tab allows selection of:

- mixture;
- additive fraction;
- pressure;
- gap;
- geometry height factor;
- number of primary electrons;
- electric field or target gain;
- maximum animation time;
- number of frames;
- ion movement;
- constant visual ion speed;
- space charge.

---

## Campaigns

Campaigns are described with YAML files. YAML comments begin with `#` and are
ignored:

```yaml
# This line is a comment.
workers: 2  # Inline comments are also valid.
```

Two scan modes are available:

```yaml
scan_mode: gain
```

or

```yaml
scan_mode: field
```

The two modes have deliberately different behaviour.

| Mode | Input | Result | Automatic refinement |
|---|---|---|---|
| `gain` | target gains | electric fields producing those gains | yes |
| `field` | fixed electric fields | measured gains | no |

Run a campaign without the GUI with:

```bash
source .venv/bin/activate
python3 run_campaign.py campaign.yaml
```

The command-line runtime flags can override the YAML/default state:

```bash
python3 run_campaign.py campaign.yaml \
  --no-space-charge \
  --excitation-positions \
  --no-gas-transport
```

---

## Target-gain campaigns

A target-gain campaign requests gains and lets the runner determine the required
electric fields.

Example:

```yaml
scan_mode: gain

mixtures:
  ArCF4: [0, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 80, 100]
  # ArN2: [0, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 80, 100]
  # ArCO2: [0, 1, 10, 30, 100]
  # ArCH4: [0, 1, 10, 30, 100]
  # ArIso: [0, 1, 10, 30, 100]
  # ArC2H2F4: [0, 1, 10, 30, 100]

pressures_bar: [0.05, 0.5, 1, 2, 5, 10]

gaps_mm:
  0.05: [1.2, 2, 5, 10, 20, 50, 100, 200]
  # 0.50: [1.2, 2, 5, 10, 20, 50, 100]
  # 1.00: [1.2, 2, 5, 7, 10, 20, 30]

gain_tolerance: 0.05
workers: 2
```

Interpretation:

```yaml
gaps_mm:
  0.05: [2, 10, 100]
```

means:

- physical gap: `0.05 mm`;
- target gains: `2`, `10` and `100`.

A result is accepted when

```math
\frac{|G_{\mathrm{measured}}-G_{\mathrm{target}}|}
{G_{\mathrm{target}}}
\leq \texttt{gain\_tolerance}.
```

The runner:

1. loads every compatible existing ROOT point;
2. statistically combines repeated runs at the same physical `(p, gap, E)` point;
3. selects the simplest supported model in the hierarchy Townsend 2p → 3p → 4p;
4. validates candidate models in gain space before allowing them to seed a scan;
5. uses a monotonic same-pressure controller for the actual refinement;
6. runs `uniformE`, measures the gain and immediately rebuilds the fit;
7. regenerates the corresponding PDF under `fits/`;
8. refines until the target is accepted.

The global alpha model is only a seed/predictor. Once a pressure has measured
points, the local controller has priority. The newest attempt for an exact
target imposes the sign of the next step: an overshoot can never increase the
field and an undershoot can never decrease it. This state is reconstructed from
`outputs/metadata/campaign_attempts.jsonl` after a restart.

The fit parameters are stored under:

```text
outputs/alpha/<mixture>/gap_<gap>mm.json
```

Space-charge and non-space-charge families are kept separate.

### Fit PDFs

Every fit is regenerated after each successfully saved ROOT:

```text
fits/<mixture>/gap_<gap>mm/<composition>.pdf
```

Each PDF contains:

- every individual ROOT used by the family;
- the statistically combined unique `(p, E)` points and their uncertainties;
- measured gain versus field on a logarithmic gain axis;
- `alpha_eff/p` versus `E/p`;
- gain-space residuals `ln(G_pred/G_measured)`;
- the electric field predicted continuously from gain `1` to gain `10^7`;
- solid curves inside the measured reduced-field interval and dashed
  extrapolations outside it;
- the complete numerical table of points entering the fit;
- the 2p/3p/4p model comparison, cross-validation factors and rejection reasons.

Regenerate every PDF without running a campaign with:

```bash
python3 fit_diagnostics.py --all
```

The GUI provides **Open fits/** and **Open selected fit** buttons.

### Per-composition gain sets

The classic `mixtures + pressures_bar + gaps_mm` format remains supported. For
campaigns where each composition needs different pressures, gaps or gain
targets, use `gain_sets`:

```yaml
scan_mode: gain

gain_sets:
  - components: {ar: 90, cf4: 10}
    pressures_bar: [1]
    gap_mm: 0.150
    gain_targets: [10, 100, 1000]

  - components: {ar: 95, cf4: 3, iso: 2}
    pressures_bar: [0.5, 1]
    gaps_mm: [0.050, 0.150]
    gain_targets: [10, 100]

gain_tolerance: 0.05
workers: 2
```

`gap_mm` accepts one gap and `gaps_mm` accepts a list. `gain_target` is also
accepted as an alias of `gain_targets`.

### Adaptive statistics

Gain campaigns use adaptive primary-electron statistics. Low-gain, short-gap
points can use more primaries than expensive high-gain points. A simulation may
stop before its maximum `npe` when the requested statistical precision is
reached.

The GUI progress value is relative to the maximum allowed number of primaries,
not an exact estimate of the remaining wall time.

### Recommended campaign strategy

Do not begin with every mixture, pressure and gap at once. A safer sequence is:

1. one mixture;
2. one gap, preferably `0.05 mm`;
3. a reduced concentration grid;
4. `workers: 1` or `workers: 2`;
5. verify the field predictor and ROOT schema;
6. expand the grid.

---

## Fixed-field campaigns

A fixed-field campaign runs exactly the fields requested. It does not invoke the
gain predictor and does not refine the field.

Minimal example:

```yaml
scan_mode: field

mixtures:
  ArCF4: [1]

pressures_bar: [1]

fields_kv_cm:
  0.05: [35]

npe: 100
workers: 1
```

This simulates:

```text
Ar 99% + CF4 1%
p = 1 bar
gap = 0.05 mm
E = 35 kV/cm
100 primary electrons
```

A larger scan can be written as:

```yaml
scan_mode: field

mixtures:
  ArCF4: [0, 1, 10, 50, 100]
  ArC2H2F4: [1, 10, 50, 100]

pressures_bar: [0.5, 1, 2]

fields_kv_cm:
  0.05: [20, 25, 30, 35, 40, 50]
  0.50: [5, 10, 15, 20]

npe: 100
workers: 2
```

The keys under `fields_kv_cm` are gaps in millimetres. The lists contain fields
in `kV/cm`. This classic product-grid format remains fully supported.

When each composition needs its own fields, use `field_sets`:

```yaml
scan_mode: field

field_sets:
  - components: {ar: 90, cf4: 10}
    pressures_bar: [1]
    gap_mm: 0.150
    reference_field_kv_cm: 39.8
    field_scales: [1.0, 0.9, 0.75, 0.5]

  - components: {ar: 95, cf4: 3, iso: 2}
    pressures_bar: [1]
    gap_mm: 0.150
    fields_kv_cm: [29, 26.1, 21.75, 14.5]

npe: 100
workers: 2
```

Each set accepts either:

- `fields_kv_cm`, containing the explicit fields; or
- `reference_field_kv_cm` together with `field_scales`.

`gap_mm` accepts one gap and `gaps_mm` accepts a list. Top-level `npe` applies
to every set, while a set may override it with its own `npe`.

Instead of a fixed `npe`, field mode may use:

```yaml
min_npe: 20
max_npe: 1000
target_relative_error: 0.03
```

The simulation then stops once the relative statistical error of the mean gain
reaches the requested value, after at least `min_npe` primaries.

---

## Runtime options

### Space charge

GUI checkbox:

```text
Space charge
```

CLI:

```bash
--space-charge
--no-space-charge
```

When enabled, ions from earlier primary avalanches are retained as charged rings
and modify subsequent avalanches. This changes the gain physics, so the campaign
stores a separate alpha-fit family.

Space charge is significantly more expensive and should normally be disabled
while developing or validating a campaign.

### Excitation positions

GUI checkbox:

```text
Excitation positions
```

CLI:

```bash
--excitation-positions
--no-excitation-positions
```

When enabled, the ROOT contains `hExcXYZ` and `hExcZT`. Disabling it reduces
output size and some filling overhead, while retaining `hLevels`.

### Magboltz transport

GUI checkbox:

```text
Magboltz transport
```

CLI:

```bash
--gas-transport
--no-gas-transport
```

When enabled, a Magboltz transport table is generated at the simulation field and
the macroscopic quantities in `gasData` are filled.

This can be much slower than the avalanche-only calculation. For ordinary gain
searches, keep it **disabled** unless the transport quantities are required in
every ROOT. It is often more efficient to enable it only for final selected
points or for a dedicated fixed-field campaign.

---

## GIF generation

GIFs animate the first primary avalanche and use two panels:

- left: electron avalanche and optional positive-ion markers;
- right: base potential or the space-charge potential perturbation.

The transverse display uses:

```text
x in [-2 × gap, +2 × gap]
```

and the vertical axis covers the physical gap.

GIFs are written to:

```text
outputs/gifs/
```

### Generate a GIF from the GUI

1. Open `./run_gui.sh`.
2. Select the **GIF** tab.
3. Select a mixture and additive fraction.
4. Set pressure and gap.
5. Choose **Electric field** or **Target gain**.
6. Set `t max` and the number of frames.
7. Enable or disable ion movement.
8. Set the constant visual ion speed.
9. Enable space charge if required.
10. Click **Generate GIF**.

### Generate a GIF at a fixed field from the command line

```bash
python3 run_campaign.py --gif \
  --mixture ArCF4 \
  --fraction 1 \
  --pressure-bar 1 \
  --gap-mm 0.05 \
  --field-kv-cm 35 \
  --npe 1 \
  --height-factor 1.5 \
  --tmax-ns 2 \
  --frames 100 \
  --space-charge \
  --move-ions \
  --ion-speed-cm-ns 1e-4
```

### Generate a GIF from a target gain

```bash
python3 run_campaign.py --gif \
  --mixture ArCF4 \
  --fraction 1 \
  --pressure-bar 1 \
  --gap-mm 0.05 \
  --gain 20 \
  --npe 1 \
  --tmax-ns 2 \
  --frames 100
```

Target-gain GIF generation requires an existing alpha fit for the selected
mixture, fraction, gap and space-charge state.

### Ion motion in GIFs

The GIF controller moves positive ions along `+z` using

```math
z(t) = z_0 + v_{\mathrm{ion}}(t-t_0),
```

with the user-selected constant `v_ion`.

Important: this velocity is used only to make the ion motion visible in the GIF.
It is not written as a measured gas-transport result and does not affect the
production gain campaign.

Disable it with:

```bash
--no-move-ions
```

---

## Supported mixtures

The runner currently defines these binary mixture families:

| Campaign name | First gas | Additive/second gas |
|---|---:|---:|
| `ArCF4` | Ar | CF4 |
| `ArN2` | Ar | N2 |
| `HeCF4` | He | CF4 |
| `ArCO2` | Ar | CO2 |
| `ArCH4` | Ar | CH4 |
| `ArIso` | Ar | iC4H10 |
| `ArC2H2F4` | Ar | C2H2F4 |

The number in the YAML is always the percentage of the **second gas**.

Examples:

```yaml
mixtures:
  ArCF4: [1]
```

means:

```text
Ar 99% + CF4 1%
```

while

```yaml
mixtures:
  ArC2H2F4: [100]
```

means pure `C2H2F4`.

A fraction of `0` gives pure first gas, and a fraction of `100` gives pure second
gas.

---

## Adding a new mixture

The current design supports binary mixtures. For another binary mixture, the C++
usually does not need to change because gas names and percentages are passed to
`MediumMagboltz::SetComposition`.

### 1. Add the mixture to `run_campaign.py`

Edit `MIXTURE_COMPONENTS`:

```python
MIXTURE_COMPONENTS = {
    "ArCF4": ("ar", "cf4"),
    "ArC2H2F4": ("ar", "c2h2f4"),
    "XeCO2": ("xe", "co2"),  # example
}
```

The key is the user-facing campaign name. The tuple contains the Magboltz gas
identifiers.

Confirm that the gas identifier is accepted by your Garfield++/Magboltz version.

### 2. Add it to the GIF selector in `gui.py`

Add the same campaign name to the mixture combo box:

```python
self.mixture.addItems([
    "ArCF4",
    "ArC2H2F4",
    "XeCO2",
])
```

Campaign YAML editing does not require a hard-coded GUI entry, but the GIF combo
box does.

### 3. Test one small point

```yaml
scan_mode: field

mixtures:
  XeCO2: [10]

pressures_bar: [1]

fields_kv_cm:
  0.05: [20]

npe: 20
workers: 1
```

### 4. Check the generated ROOT

Verify:

- `gasData/gas1` and `gasData/gas2`;
- percentages;
- measured gain;
- `hLevels` entries;
- optional transport fields;
- absence of warnings from `MediumMagboltz::SetComposition`.

---

## Extending the project to new cases

### More fractions, pressures, fields or gains

No code changes are needed. Extend the YAML lists.

### More gaps

Add another gap key under `gaps_mm` or `fields_kv_cm`.

### Pure gases

Use `0` for pure first gas or `100` for pure second gas.

### Ternary mixtures

Ternary mixtures require code changes because the current configuration and
command-line interface carry two gases only. At minimum, extend:

- the C++ `Config` structure;
- argument parsing;
- the `SetComposition` call;
- ROOT naming;
- `gasData` branches;
- YAML mixture mapping;
- GUI controls;
- ROOT validation and alpha-family keys.

Do not encode a ternary mixture as a fake binary campaign name without updating
these layers.

### Non-uniform electric fields

The current project uses `ComponentUser` for a uniform field. A mesh, analytic
geometry or imported field map would require replacing the field component and
reviewing:

- sensor area;
- launch position;
- gain-distance definition;
- field interpolation;
- alpha-model assumptions;
- GIF field map.

### Magnetic fields

The current campaign model assumes no magnetic field. Adding one requires C++
configuration and likely additional YAML/GUI parameters. Transverse drift then
becomes physically relevant.

### Different ion models

Campaign space charge currently uses static charged rings. GIF ion motion uses a
constant visual velocity. A physical ion-drift implementation would require ion
mobility data and `AvalancheMC` or another appropriate Garfield++ transport
model.

### Adding a new ROOT quantity

When adding an output:

1. create/fill it in `uniformE.cxx`;
2. write it before closing the ROOT file;
3. detach ROOT-owned histograms when they also have C++ owners;
4. update `run_campaign.py` if the ROOT validator requires a specific schema;
5. regenerate old ROOTs when the schema changes;
6. document its units in the branch name.

---

## Output files

### ROOT files

```text
outputs/roots/<mixture>/gap_<gap>mm/
```

Example:

```text
outputs/roots/ArCF4/gap_0.050mm/
```

ROOT file names follow the physical configuration:

```text
ar_99.0_cf4_1.0_35.0kVcm_1.000bar_0.0500mm_100npe.root
```

The name contains:

- gas percentages;
- field in `kV/cm`;
- pressure in `bar`;
- gap in `mm`;
- actual number of primary electrons.

### Alpha fits

Machine-readable fit state:

```text
outputs/alpha/<mixture>/gap_<gap>mm.json
```

Human-auditable diagnostics and gain-1-to-10^7 extrapolations:

```text
fits/<mixture>/gap_<gap>mm/<composition>.pdf
```

Space-charge fits use distinct JSON and PDF families. No diagnostic PNG files
are generated.

### GIFs

```text
outputs/gifs/
```

### Legacy ROOT files

When the runner detects an incompatible old schema, it may move the file to:

```text
outputs/legacy_roots/
```

Old ROOT files are not rewritten automatically. Regenerate them with the current
code when the schema or histogram definitions change.

---

## ROOT contents

Current production ROOT files contain:

```text
hElectronEnergyDistribution
hLevels
hExcXYZ                   # optional
hExcZT                    # optional
dataPerPrimaryElectron
dataPerElectron
gasData
```

There is no `levelMap` tree.

### `hElectronEnergyDistribution`

Electron-energy distribution sampled from microscopic null-collision steps.
The internal energy reservoir is capped to avoid unbounded memory growth.

### `hLevels`

`hLevels` deliberately preserves the historical project definition:

```text
all real non-elastic Magboltz collision terms with a valid level index
```

It includes collision types 1–5:

- ionisation;
- attachment;
- generic inelastic channels;
- excitation channels;
- superelastic channels.

Despite the historical plot title `Excitation Distribution`, it is not a pure
`type == excitation` histogram. This convention is retained for compatibility
with the downstream photon/scintillation pipeline.

### `hExcXYZ`

Optional `128 × 128 × 128` integer histogram of the joint spatial positions for
generic inelastic and excitation channels (`type 3 + type 4`).

Limits:

```text
x, y in [-2 × gap, +2 × gap]
z in [0, gap]
```

The three coordinates are filled together, so the complete spatial correlation
`P(x, y, z)` is retained at the histogram resolution.

### `hExcZT`

Optional `128 × 256` integer histogram of `z` versus `t` for generic inelastic
and excitation channels. Its `z` bins match `hExcXYZ`; the time axis can extend
when necessary.

The compact histogram approach avoids storing one tree entry per excitation.
Combining `hExcXYZ` with the conditional time distribution `P(t | z)` preserves
the joint spatial distribution and the longitudinal-time correlation, but not
residual `x-t` or `y-t` correlations.

### `dataPerPrimaryElectron`

One entry per primary electron:

| Branch | Meaning |
|---|---|
| `ne` | final electrons from that primary avalanche |
| `ni` | ions produced by that primary avalanche |
| `npe` | number of represented primaries; currently `1` per entry |

The campaign runner computes the gain and its statistical uncertainty from this
tree.

### `dataPerElectron`

One entry per final electron endpoint:

| Branch | Meaning |
|---|---|
| `status` | Garfield++ electron endpoint status |

At high gain, this tree can become one of the largest objects in the ROOT file.

### `gasData`

One entry per ROOT file:

| Branch | Unit | Meaning |
|---|---:|---|
| `gas1` | — | first Magboltz gas identifier |
| `composition1_pct` | `%` | first-gas fraction |
| `gas2` | — | second Magboltz gas identifier |
| `composition2_pct` | `%` | second-gas fraction |
| `pressure_bar` | `bar` | gas pressure |
| `temperature_K` | `K` | gas temperature |
| `electricField_V_cm` | `V/cm` | uniform electric field magnitude |
| `gap_mm` | `mm` | physical multiplication gap |
| `height_mm` | `mm` | total computational height |
| `spaceCharge` | boolean | charged-ring space charge enabled |
| `npe` | — | actual number of simulated primaries |
| `townsendAlpha_cm_inv` | `cm^-1` | Magboltz Townsend coefficient |
| `attachmentEta_cm_inv` | `cm^-1` | Magboltz attachment coefficient |
| `alphaEffective_cm_inv` | `cm^-1` | `alpha - eta` from Magboltz |
| `driftVelocityZ_cm_ns` | `cm/ns` | electron drift velocity along `z` |
| `longitudinalDiffusion_sqrt_cm` | `sqrt(cm)` | longitudinal diffusion coefficient |
| `transverseDiffusion_sqrt_cm` | `sqrt(cm)` | transverse diffusion coefficient |

When Magboltz transport is disabled, the transport branches exist but contain
`NaN`.

The drift velocity may be negative. In this geometry a negative `z` velocity is
expected when electrons drift from `z = gap` towards the anode at `z = 0`.

To display a negative value correctly in ROOT/JSROOT, select an axis range that
includes negative numbers, for example:

```cpp
gasData->Draw("driftVelocityZ_cm_ns>>h(100,-0.02,0.0)");
```

---

## Magboltz transport quantities

When **Magboltz transport** is enabled, the simulation generates a gas transport
table at the requested field before reading:

- electron drift velocity;
- longitudinal diffusion;
- transverse diffusion;
- Townsend coefficient `alpha`;
- attachment coefficient `eta`;
- effective coefficient `alpha - eta`.

These are macroscopic Magboltz transport quantities. They are distinct from the
simulation-derived value

```math
\ln(G)/d.
```

The transport calculation can dominate the runtime of small avalanche tests.
Recommended workflow:

```text
Field/gain exploration       → Magboltz transport OFF
Final selected field points  → Magboltz transport ON
Dedicated transport scan     → Magboltz transport ON
```

---

## Performance recommendations

### Worker count

Garfield++/Magboltz processes can consume substantial memory. `workers: auto`
may start too many simultaneous jobs and cause swapping.

Recommended starting point:

```yaml
workers: 2
```

Increase it only after monitoring RAM usage.

### Split large scans

Prefer campaigns grouped by mixture and gap:

```text
ArCF4 / 0.05 mm
ArCF4 / 0.50 mm
ArCF4 / 1.00 mm
ArN2  / 0.05 mm
...
```

Start with `0.05 mm`, which is generally the least expensive gap and is useful
for validating the predictor.

### Disable unnecessary outputs

For fast gain refinement:

```text
Space charge          OFF
Magboltz transport    OFF
Excitation positions  OFF if not needed
```

Enable expensive measurements only for final production points when possible.

### Keep GIFs out of campaigns

GIF creation is a diagnostic operation. Do not generate GIFs during large
production campaigns.

---

## Troubleshooting

### PySide6 / Qt6DBus symbol error

Example:

```text
ImportError: libQt6DBus.so.6: undefined symbol ... Qt_6_PRIVATE_API
```

Use:

```bash
./run_gui.sh
```

The launcher prepends the Qt libraries bundled with PySide6 while preserving the
ROOT/Garfield++ library paths.

### Exit code 127

Usually indicates a missing shared library at runtime.

Check:

```bash
ldd build/uniformE | grep "not found"
```

Then verify ROOT and Garfield++ environment variables.

### CMake cannot find Garfield++

```bash
export CMAKE_PREFIX_PATH="/path/to/garfield-install:${CMAKE_PREFIX_PATH:-}"
cmake -S . -B build
```

### Old objects such as `levelMap` still appear

You are opening an old ROOT file or running an old executable.

The current production schema does not contain `levelMap`. Delete/regenerate the
specific ROOT or move old outputs away, then run through `./run_gui.sh`, which
rebuilds the executable.

### `.pending_*.root` files

These are temporary files used while a job is being completed and validated.
After an interrupted campaign, they can be removed safely when no campaign is
running:

```bash
find outputs/roots -type f -name '.pending_*.root' -delete
```

### The system starts swapping

Stop the campaign and reduce:

```yaml
workers: 1
```

or

```yaml
workers: 2
```

Split the campaign by mixture and gap.

### GIF target gain cannot be resolved

A target-gain GIF needs an existing compatible alpha fit. Either run a gain
campaign first or specify the electric field directly with `--field-kv-cm`.

### Magboltz transport is slow

This is expected. Disable it during field/gain exploration and enable it only
for selected final points.

---

## Recommended first tests

### Fixed-field test

```yaml
scan_mode: field

mixtures:
  ArCF4: [1]

pressures_bar: [1]

fields_kv_cm:
  0.05: [35]

npe: 20
workers: 1
```

### Small target-gain test

```yaml
scan_mode: gain

mixtures:
  ArCF4: [0, 1]

pressures_bar: [1]

gaps_mm:
  0.05: [2, 10, 20]

gain_tolerance: 0.05
workers: 1
```

Validate these before launching a large multibar campaign.

---

## Citation and software documentation

This project depends on:

- Garfield++: https://garfieldpp.docs.cern.ch/
- Garfield++ installation: https://garfieldpp.docs.cern.ch/install/
- ROOT: https://root.cern/
- ROOT installation: https://root.cern/install/
- Qt for Python / PySide6: https://doc.qt.io/qtforpython-6/
- CMake: https://cmake.org/

When publishing results produced with Garfield++, Magboltz or ROOT, follow the
citation recommendations of the corresponding projects and document the exact
software versions, gas composition, pressure, temperature, gap, field and runtime
options used in the simulation.
