# Physical calibration and local data

The `physical` commands create an **opt-in model variant before simulation**.
They can change native muscle strength and operating ranges, reviewed muscle
paths, primitive collision shapes, segment mass, centre of mass and inertia.
They operate on private copies of the source model. The existing SOMA visual
export still leaves native mechanics alone unless a calibrated model is supplied.

This supplies calibration software, rather than a validated individualized
human. Population architecture tables are priors from different people. A
synthetic recovery test or finite rollout does not establish physiological accuracy.

## Data that can be used

Run `myosuite-soma physical datasets` for the current source inventory.

| Source | Inputs and useful scope | Local adapter |
|---|---|---|
| [Charles et al. 2019](https://doi.org/10.1371/journal.pone.0223531) | Supporting tables for 20 lower-limb muscles in 10 adults; MRI/DTI architecture, PCSA and estimated forces; CC BY | Published subject DOCX tables S1–S10, or unit-labelled CSV |
| [Mirakhorlo et al. hand/wrist](https://doi.org/10.6084/m9.figshare.4434791) | One specimen; hand/wrist morphology and attachments; CC BY 4.0 | Main-paper Table 2 architecture CSV; separate `Suppl2.docx` attachment parser |
| [Nordez et al. 3D ultrasound](https://doi.org/10.57745/TBCTLO) | Muscle–tendon unit paths/lengths across joint configurations; Etalab Open License 2.0 | Local Slicer markup control polylines and explicitly reviewed native poses |
| [Bizet et al. calf mechanics](https://doi.org/10.57745/SFBCL7) | Ultrasound/fascicles, dynamometer signals and EMG; Etalab Open License 2.0 | Optional C3D analog extraction, explicit dynamometer conversion, portable observation CSV/NPZ |
| [AddBiomechanics](https://www.addbiomechanics.org/download_data.html) | Ground reaction forces and processed motion, joint torque and segment properties; CC BY 4.0 | Processed OpenSim segment importer with explicit body registration; independently reviewed native-coordinate observations for dynamics validation |

Obtain files from their authors and retain attribution and the accompanying
terms. No downloader logs in or accepts a license. New physical-pipeline inputs
and outputs remain local: source recordings, extracted measurements, generated
models and calibration results are excluded from Git, Python distributions and
release assets. Existing project media and earlier derived results are separate
from this new pipeline.

### Import architecture and bind exact actuator names

```sh
myosuite-soma physical import-architecture --source charles --input local-assets/subject-table.docx --subject subject01 --output local-assets/architecture.json --mapping local-muscle-map.json --profile-output local-assets/architecture-profile.json
```

An explicit mapping object binds a published muscle to an exact native actuator:

```json
{
  "SOURCE MUSCLE NAME": {
    "actuator": "NATIVE ACTUATOR NAME",
    "force_source": "architecture",
    "specific_tension_pa": 300000
  }
}
```

The tension above is an illustrative prior, not a calibrated constant. Choose
and record it for the intended population or fit strength to torque data. The
alternative `force_source: "reported"` uses the table's force column, retaining
its estimated provenance. Multiple source heads are never matched by proximity.
If a source muscle is split across actuators, supply a list of mappings with
explicit `fraction` values whose sum is at most one; no volume/force is duplicated.

Imports normalize volume to m³, length to m, area to m² and force to N. They keep
source cell strings, including reported variability, and hash the downloaded
file. Charles' PCSA already contains `cos(pennation)`: projected PCSA multiplies
specific tension directly. Mirakhorlo's PCSA is perpendicular; its pennation
projection is applied once. Missing angle/convention information is not guessed.
Charles' optimal fibre lengths use an external sarcomere-length prior and its
maximum forces are calculated estimates, rather than measured individual forces.

The hand's `Suppl2.docx` contains joint/landmark and muscle attachment geometry;
the fibre/PCSA architecture values are in Table 2 of the [main paper](https://doi.org/10.1080/23335432.2016.1191373).
The architecture importer accepts a local CSV export of that table. It does not
mislabel the attachment supplement as a fibre-architecture table.

### Import and register attachment points

```sh
myosuite-soma physical import-attachments --input local-assets/Suppl2.docx --output local-assets/hand-points.json
myosuite-soma physical map-attachments --model source.xml --anatomy local-assets/hand-points.json --mapping reviewed-point-to-site.json --registration specimen-registration.json --output local-assets/site-profile.json
```

Point IDs identify the original table/row. Their origin/insertion/via labels,
source segments and specimen-rest coordinates are retained. A reviewer supplies
an exact point-ID to native-site map and a proper rotation, translation in metres
and optional positive scale to register the specimen frame to the native world
at `qpos0`. The adapter converts those coordinates to each site's owning body
frame. Reusing the published segment name as a native body name is not automatic.
The builder recomputes transmission lengths after path edits.

## Build a separate native model

```sh
myosuite-soma physical build --model source.xml --profile local-assets/architecture-profile.json --output local-assets/physical-model
```

`--env` can replace `--model` when MyoSuite is installed. Model output is a fresh
directory containing a self-contained native `model.mjb`, a report and, when an
editable source spec is available, MJCF/assets for that spec. MJB files are tied
to the MuJoCo version that produced them. Existing output is never overwritten.

When `--env` exposes an editable MuJoCo spec, the CLI copies live parameter
edits into a private spec and recompiles a **canonical baseline snapshot**.
All physical commands use that snapshot rather than mixing the live model's
cached constants with newly compiled constants. If no compatible editable spec
is available, the compiled model is used and structural-edit restrictions apply.
For reproducibility, freeze one baseline and use its exported XML for subsequent
mapping, fitting and building:

```sh
myosuite-soma physical build --env YOUR_ENV_ID --profile local-assets/baseline-profile.json --output local-assets/native-baseline
myosuite-soma physical build --model local-assets/native-baseline/model.xml --profile local-assets/reviewed-profile.json --output local-assets/physical-model
```

Here `baseline-profile.json` is an explicit version-1 profile with provenance
and no edits, for example `{"version": 1, "provenance": {"status": "estimated", "source": "Unmodified canonical native baseline"}}`.
The export resolves native automatic strength to equivalent explicit forces.
Create later model-bound profiles against this exported baseline, not a
different environment instance or an earlier model hash. The XML/assets are
available only when an editable spec was recovered. A compiled-only MJB remains
usable for supported parameter edits and validation; COM/full-tensor edits to
optimized simple bodies require the original XML/MjSpec for recompilation.

A version-1 profile contains `provenance` with `status` (`measured`, `estimated`
or `synthetic`) and `source`. Its explicit named edits include:

- `muscles`: `force_N` or an `architecture` object, optional normalized `range`,
  `passive_force_max` and `lengthrange_m`. Active/passive strength is updated
  consistently. Auto-sized baseline force is resolved explicitly to avoid mass
  changes accidentally changing strength.
- `bodies`: geometry scale/position and mass/COM/inertia decisions. Native mass
  includes muscle mass already; the builder replaces segment mass rather than
  adding inferred muscle volumes a second time.
- `sites`: body-local `pos_m` for reviewed attachment/path edits.
- `geoms`: supported primitive positions/sizes or a local point-cloud capsule fit.

`physical allometry` prepares an explicit segment prior from scale factors,
with mass preserved, constant-density scaling, or a measured target total mass.
These assumptions initialize a model; they do not establish that fibre length
or muscle strength scales with body size. Primitive segment geometry scaling
does not resize an arbitrary source mesh or rescale every descendant implicitly.

`physical fit-capsules` can fit selected anatomy vertices from a source-checked
SOMA animation to an explicitly assigned native collision geom. Cache provenance
and the qpos trajectory must agree. This fits reference collision proxies; it
does not turn a per-frame rendered surface into a soft-tissue contact solver.

## Torque observation contract

The fitter needs synchronized native `time`, `qpos`, `qvel`, named activation
columns, selected named hinge-joint torque columns and `recording_ids`. Every
actuator must currently be a supported native muscle with a spatial tendon and
one activation state; models mixing muscles with motors/general actuators are
rejected. Whole recordings must remain in one split. CSV columns have this form:

```text
time,recording_id,split,qpos:0,qvel:0,activation:CALF,torque:ANKLE
```

Declare ordered `actuator_names` and `dof_names`, provenance, activation source
and units in a sidecar JSON. `examples/physical_observations.metadata.json` is an
invented template. Supply all `nq` position columns, including unit quaternions
where the model uses them, and all `nv` tangent-velocity columns. Quaternion
derivatives are not native `qvel`. Other units are seconds, metres/radians, N/Nm
and normalized activation. Signs and coordinate conventions must already match
the native model.

```sh
myosuite-soma physical import-observations --input local-recordings.csv --metadata local-recordings.metadata.json --output local-assets/observations.npz
```

The equivalent NPZ contains numeric arrays and scalar text `metadata_json`;
loading uses `allow_pickle=False` plus bounded header/allocation checks. Per-file
inputs are limited to 64 MiB; split long recordings into manageable files first.

The torque target is explicit:

- `actuator`: the target corresponds to native `qfrc_actuator`.
- `net`: the target corresponds to `qfrc_actuator + qfrc_passive`.

Neither convention includes gravity, contact or external-load torque. Raw
inverse-dynamics torque from AddBiomechanics or a dynamometer is not silently
accepted as either. Resolve those terms and the force-frame/moment-arm mapping
before calibration. AddBiomechanics does not provide measured individual muscle
activation/force, so its data alone cannot identify muscle strength.
This package does not automatically retarget a DART/B3D skeleton and ground
forces to MyoSuite. That coordinate/body mapping is an external preprocessing
step; the portable importer checks the resulting native observation schema.

### Import processed segment mass, COM and inertia

Use the processed OpenSim model accompanying a locally obtained motion dataset:

```sh
myosuite-soma physical import-segments --input local-assets/processed-model.osim --output local-assets/segments.json
myosuite-soma physical map-segments --model local-assets/native-source.xml --segments local-assets/segments.json --mapping reviewed-segment-map.json --output local-assets/segment-profile.json
myosuite-soma physical build --model local-assets/native-source.xml --profile local-assets/segment-profile.json --output local-assets/segment-variant
```

The reviewed mapping explicitly names each source and target body, for example:

```json
{
  "SOURCE BODY": {
    "body": "NATIVE BODY",
    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "translation_m": [0, 0, 0]
  }
}
```

These identity values are illustrative. Supply the rigid transformation
`x_native_body = rotation @ x_source_body + translation_m` for the selected
body frames. Both frame decisions are mandatory; matching a body name alone does
not establish that its axes, origin or anatomical extent agree. Registrations
are one-to-one and do not scale geometry or retarget source joint coordinates.

The importer accepts OpenSim 3's explicit inertia components and OpenSim 4's
`[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]` vector. Values are SI: kg, m and kg·m². As defined
by [OpenSim's body properties](https://github.com/opensim-org/opensim-core/blob/main/OpenSim/Simulation/SimbodyEngine/Body.h)
and [matrix conversion](https://github.com/opensim-org/opensim-core/blob/main/OpenSim/Simulation/SimbodyEngine/Body.cpp),
off-diagonal values are already tensor entries and inertia is about COM in body
axes. Registration applies `COM_native = R COM_source + t` and `I_native = R I_source R.T`.
It adds no parallel-axis term to COM inertia. Invalid positive-mass inertias
are rejected; zero-mass helper bodies are listed and excluded.

Use the original XML/MjSpec when building structural COM/tensor edits. Some
compiled MJB models have an optimized mass-matrix layout that cannot support
such edits without recompilation; those cases produce an actionable error.
The adapter records the original model fingerprint and raw OpenSim file hash.
These processed properties remain **estimates from the external participant**,
not measurements of the corresponding SOMA subject. No B3D/DART retargeting,
OpenSim joint conversion or automatic native mass assignment is implied.

### Import 3D-ultrasound MTU paths

The Nordez dataset includes local 3D Slicer `.mrk.json` paths. Harden any Slicer
scene transforms before exporting. A file map explicitly binds each reviewed
path to a native muscle actuator. Paths are relative to that map's directory:

```json
{"NATIVE ACTUATOR": "pose-001.mrk.json"}
```

```sh
myosuite-soma physical import-mtu --files local-assets/path-files.json --units mm --coordinate-system LPS --output local-assets/path-001.json
myosuite-soma physical prepare-paths --markups local-assets/pose-imports.json --trajectory local-assets/reviewed-native-poses.npz --output local-assets/path-observations.json
myosuite-soma physical fit-paths --model local-assets/native-source.xml --observations local-assets/path-observations.json --config reviewed-path-fit.json --output local-assets/path-variant
```

Run `import-mtu` separately for each observed pose. The `pose-imports.json`
manifest is `{"poses": ["path-001.json", "path-002.json"]}`, ordered identically
to rows of the trajectory NPZ. That NPZ contains numeric `qpos` `(T, native_nq)`,
numeric `time` in seconds and text `recording_ids` `(T,)`. Supply the full native
coordinate ordering with reviewed radians, positions and unit quaternions.
Ultrasound posture labels and OpenSim angles are not converted automatically.
Times may restart between recordings and must increase within each recording.

Explicit `mm`/`m` and `LPS`/`RAS` arguments must agree with declarations when
present. The importer supports Slicer's string or UCUM-coded units, normalizes
points to RAS metres, rejects undefined points and requires one unambiguous
open path per file. It follows the
[Slicer markup schema](https://github.com/Slicer/Slicer/blob/main/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json).

The result is the **ordered control polygon length**. For a Slicer `Curve`, this
is an approximation to the interpolated curve's arc length and can understate
it. It is not a new measurement of the continuous tissue path. Stored Slicer
length values and implicit spline interpolation are deliberately not substituted
for that calculation. These observations are marked estimated and preserve
point coordinates, source conventions and per-file hashes. MTU total length
does not independently identify optimal fibre length or tendon slack length.

`fit-paths` adjusts only explicitly selected sites within reviewed body-local
bounds and reports performance on whole held-out recordings. Registration,
native joint limits, path ordering and moment-arm signs need anatomical review.
The Python `mtu_path_observations` API can additionally bind source model and
kinematic hashes under `conventions` before fitting. No scan or path geometry is
included in the package.

A path-fit config explicitly supplies whole held-out recording IDs and absolute
body-local site bounds. This invented example fixes two coordinates and fits
one; replace its names and bounds after anatomical review:

```json
{
  "version": 1,
  "holdout_recordings": ["UNTOUCHED RECORDING"],
  "sites": {
    "REVIEWED NATIVE SITE": {
      "lower_m": [-0.005, 0, 0],
      "upper_m": [0.005, 0, 0]
    }
  },
  "max_nfev": 300,
  "regularization": 0
}
```

Equal bounds fix a coordinate. Optional regularization penalizes displacement
from the bounded source site; held-out targets never enter optimization.
The output contains the fitted profile and baseline/fitted train/validation
length errors in metres, per-actuator errors, parameter sensitivity rank and
the physical rebuild report. Feasible native muscle operating lengths are
recomputed and source strength is preserved. Unit tendon gear and a spatial
path without fixed-joint or pulley elements are required. A successful optimizer
or a low length error does not verify the anatomical mapping or muscle mechanics.

### Local dynamometer/EMG conversion

The optional `ezc3d` package reads raw C3D analog signals:

```python
from myosuite_soma.physical_data import (
    read_c3d_signals, dynamometer_observations, save_observations,
)

signals = read_c3d_signals("local-assets/recording.c3d")
# Inspect signals['channels'] and signals['units'], then supply reviewed config.
observations = dynamometer_observations(signals, reviewed_config)
save_observations(observations, "local-assets/recording-observations.npz")
```

Install `ezc3d` separately when needed. `examples/physical_dynamometer.json` is an
invented single-axis template: it requires native reference vectors/indices,
explicit angle/torque channel gains and offsets, the corrected torque convention,
and a normalized channel or explicit constant for every native muscle actuator.
EMG high/low-pass filtering, rectification and clipping are explicit options;
normalization gain must come from a reviewed calibration such as an MVC, rather
than the maximum value of the held-out recording. Source hashes and processing
choices remain in `conventions` metadata. The default label is `emg_proxy` and
EMG-derived inputs must retain that label; an explicit source override is only
appropriate for independently measured activation or synthetic inputs. Passing
through this adapter does not turn EMG into measured muscle activation.

The same explicit conversion is available from the CLI:

```sh
myosuite-soma physical import-dynamometer --input local-assets/recording.c3d --config reviewed-dynamometer.json --output local-assets/recording-observations.npz
```

## Fit and validate on held-out recordings

```sh
myosuite-soma physical fit --model local-assets/physical-model/model.mjb --observations local-assets/observations.npz --config fit-config.json --output local-assets/fitted-model
myosuite-soma physical validate --model local-assets/fitted-model/model.mjb --observations local-assets/observations.npz --config fit-config.json --output local-assets/validation.json
```

The config supplies exact actuator names and bounded `force_N`, `range0`,
`range1` and/or `passive_force_scale` (MuJoCo's dimensionless `fpmax`). Use whole
`holdout_recordings` or explicit per-recording `split` labels, never both.
`allow_emg_proxy` must be explicitly enabled for EMG-derived inputs. Synthetic
profiles/observations require `--allow-synthetic` and remain labelled as such.

Reports compare baseline/fitted torque on training and held-out recordings,
report parameter sensitivity/rank and retain provenance. They also check native
moment arms and passive/active torque over joint range and finite forward
rollouts. Low joint-torque error does not uniquely identify individual muscles;
co-contraction can leave muscle-specific parameters underdetermined. Independent
fibre, strength or load measurements remain necessary for physiological claims.

## Muscle-model assumptions and the experimental compliant runtime

Native MuJoCo derives effective fibre/tendon lengths from feasible transmission
lengths and normalized operating ranges; it assumes an inelastic biological
tendon and zero pennation. See [the official muscle-model documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#muscles).
Writing a fibre-length value into user metadata does not change native physics.

`physical simulate-compliant` is a separate research runtime for selected muscle
actuators. It supplies an elastic quadratic tendon law, constant-thickness
pennation, native FLV curves and an implicit damped fibre step. It applies its
forces through native moment arms and leaves unselected muscles native. This
operator-split model is not equivalent to OpenSim and has no measured physiological
validation by default. Fibre bounds, tendon-strain bounds and failed force-balance
solves are explicit; inspect its residuals and refine timestep/parameters.

```sh
myosuite-soma physical simulate-compliant --model local-assets/physical-model/model.mjb --parameters local-compliant-parameters.json --controls local-controls.npz --output local-assets/compliant-rollout
```

The version-1 parameter JSON contains provenance and exact muscle names, each
with `force_N`, `optimal_fibre_length_m`, `tendon_slack_length_m` and optionally
`pennation_deg`, `strain_at_force_max`, damping and numerical bounds. The controls
NPZ supplies `(steps, model.nu)` `ctrl` and optional native `initial_qpos`,
`initial_qvel`, `initial_activation`. Use a fresh output directory. Selected
muscles need supported spatial tendon transmissions in metres with unit gear,
native muscle activation dynamics and no unsupported force clamp. RK4 is
rejected because the separate fibre state is not integrated at its internal
stages. This research runtime must not be applied to a joint-only transmission
whose numerical lengths are angles.

Replacement forces are reported separately from native `actuator_force`; the
runtime does not fabricate native force/length measurements for its new states.
Generated outputs include fibre length, pennation and tendon force under their
own names. Use calibrated effective fibre/slack/stiffness data to evaluate this
extension before research conclusions or reusing a policy trained on native
muscles. The physical variant's changed dynamics also require policy re-evaluation.

## Validate rigid-body dynamics with measured external loads

This separate check uses native motion and measured ground reaction forces or
other external wrenches. **It needs no activation or EMG data.** It evaluates
rigid-body force balance and can compare mass/inertia variants without claiming
that individual muscle forces have been identified:

```sh
myosuite-soma physical validate-dynamics --model local-assets/native-baseline/model.xml --observations local-assets/native-dynamics.json --output local-assets/baseline-dynamics.json
myosuite-soma physical validate-dynamics --model local-assets/physical-model/model.mjb --observations local-assets/native-dynamics.json --output local-assets/variant-dynamics.json
```

The JSON observations contain provenance and the following native-coordinate
arrays. Coordinates and external loads must be synchronized and registered
before using this command; raw AddBiomechanics B3D coordinates are not retargeted.

| Field | Shape and convention |
|---|---|
| `time` | `(T,)`, seconds; increasing within each recording |
| `qpos` | `(T, nq)`, complete native coordinate ordering, including unit quaternions |
| `qvel`, `qacc` | `(T, nv)`, native tangent velocities and accelerations; not raw quaternion derivatives |
| `recording_ids` | Optional `(T,)` text IDs; strongly recommended for separate recordings |
| `external_forces` | Explicit list of measured loads, or `[]` to declare their absence |
| `dof_names`, `reference_generalized_force` | Optional independent hinge/slide joint names and `(T, D)` reference forces, in Nm/N respectively |

Each external-load entry contains exactly `body` (the native body receiving the
load), `point_world_m`, `force_world_N` and `torque_world_Nm`. Each numeric array
has shape `(T, 3)` in world axes. Torque is the free moment at the supplied force
application point; the lever-arm moment is added by the native Jacobian mapping.
Review force-plate COP, force signs, moment origins and body assignments first.
Missing loads are not inferred from simulated contacts. If supplied, `units`
must be exactly:

```json
{
  "time": "s",
  "qpos": "SI",
  "qvel": "SI/s",
  "qacc": "SI/s2",
  "external_forces": "world_SI",
  "reference_generalized_force": "native_SI"
}
```

The report evaluates `M*qacc + bias - passive - J_external.T*wrench`. It omits
native contact constraint forces to avoid counting measured GRFs twice, and
reports how many native contacts were ignored. Free-base force/moment residuals,
unactuated-coordinate residuals and optional reference errors expose consistency
problems. Free-base translations use world axes and rotations the native local
basis; their reported units remain N and Nm separately. These residuals can
reflect motion differentiation, load registration, missing forces or inaccurate
mass properties, rather than only the muscle model.

Active scalar polynomial joint equalities use virtual-work projection; other
active equality types are rejected. Source positions must satisfy the scalar
coupling within `1e-5` in native coordinate units. Velocity and acceleration
must also satisfy its first and second derivatives. Default absolute tolerances
are `1e-3` rad/s or m/s and `1e-2` rad/s² or m/s², respectively. Explicit finite,
nonnegative overrides belong in observation metadata:

```json
{
  "conventions": {
    "equality_kinematics_tolerances": {
      "velocity_abs": 0.001,
      "acceleration_abs": 0.01
    }
  }
}
```

Reports retain the declared tolerances and maximum residual for each coupled
joint under `joint_equality_kinematics`; inconsistent observations are rejected
rather than silently projected. Torque calibration performs the corresponding
velocity check and retains it in provenance. These are numerical consistency
tolerances, not physiological acceptance thresholds.

Record `conventions.source_kinematic_sha256` when available to reject a changed
coordinate/geometry model. An optional `source_model_sha256` records the original
physical baseline; the report compares it with the candidate and exposes
`same_input_physics`. A changed mass/inertia variant can deliberately differ
physically while retaining compatible kinematics. Review those hashes alongside
the errors before comparing models. This validation does not fit muscle
strength and requires no fabricated muscle activation.
