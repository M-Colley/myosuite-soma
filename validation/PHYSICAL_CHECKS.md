# Physical pipeline checks

Executed locally on 2026-10-25 for version `0.2.0a1`. These checks establish
software behavior and numerical consistency, not individualized physiological
accuracy or anatomically reviewed muscle correspondences.

- Full package and pinned MyoSuite adapter suite: **459 passed, 1 skipped**.
  The skip requires Windows symbolic-link privilege. Ruff checks and formatting
  pass. The existing renderer emits its upstream empty-prefix attachment warning.
- The physical builder checks independent source copies, explicit and automatic
  muscle forces, PCSA/pennation conventions, native length-range recomputation,
  site/wrapping changes, mass/COM/full inertia tensors, and capsule fitting.
  Simple-body COM/inertia changes require editable-source recompilation; a
  compiled-only input is rejected when its mass-matrix layout cannot support
  the requested change. Free-joint placement is supplied through initial state,
  rather than a compiled body-position edit.
- MuJoCo 3.15 needs an explicit primitive-bound and body-BVH refresh after
  geometry/inertial-frame edits; 3.17 provides the native refresh. Regressions
  compare all five supported primitive bounds and actual multi-geom contacts
  with independent fresh compilation. Capsule conversion of tendon-wrapping
  spheres/cylinders is rejected.
- Native MJB exports are reloaded and fingerprinted. Optional portable MJCF
  exports include their local source assets, retain tendon spring-length
  intervals, and are independently compiled with bounded XML-writer rounding
  checks. An unavailable optional editable export is reported without claiming
  success or discarding a verified MJB. Generated files remain private.
- Torque and attachment-path fitting use whole-recording holdouts. Tests alter
  held-out targets to verify that they cannot affect fitted parameters or
  training sensitivity rank. Independent recovery cases check force and path
  parameters, activation conventions, scalar joint-coupling derivatives,
  positive effective fibre/slack targets, and underdetermined parameter reports.
- External-load inverse dynamics is checked against analytical pendulum and
  free-base cases, including off-centre forces and rotated coordinate bases.
  Measured external loads do not require invented muscle activations. Model
  contacts are not added again when measured ground forces are supplied.
- The separate compliant-muscle runtime has equilibrium, multi-DOF moment-arm,
  native activation, warning/failure and step-refinement checks. It uses a
  quadratic elastic tendon, constant-thickness pennation, native active/passive
  curves and implicit fibre steps with damping. This is an operator-split
  research model; it has no measured physiological validation, hysteresis or
  continuum tissue-contact claim.
- Local published-source imports read all **20 muscles** in a Charles subject
  DOCX and **269 coordinate points** across 51 source-label groups in the hand
  attachment supplement. These are parser checks, not 51 verified muscle heads.
  Source formatting, units, provenance and hashes are retained locally.
  Additional synthetic cases cover OpenSim segment properties, Slicer markup
  units/frames, C3D signal extraction and dynamometer conversion. Actual C3D
  write/read is tested when the optional `ezc3d` dependency is installed.
- A private audit used actual native elbow (**1 qpos / 1 DOF / 6 muscles**), leg
  (**35 / 34 / 80**) and full-body (**89 / 88 / 354**) models. Explicit segment
  mass changes, source immutability, nine-angle passive/active curves, 40-step
  finite native rollouts, and in-memory MJB/portable-MJCF reloads passed. A
  separate native-elbow synthetic recovery identified two planted strengths
  with held-out torque RMSE below **2e-13 Nm**. This near-zero recovery is a
  software consistency result, not an error estimate for measured human torque.
- A local S1 foot-surface selection exercised cache-to-native capsule fitting.
  The candidate remains **unreviewed and unapplied**; successful fitting does
  not demonstrate improved contact or anatomical membership.
- The built wheel is checked without Torch, MyoSuite or USD: **204 physical and
  compliant tests pass on MuJoCo 3.17**; **198 pass and 6 skip on 3.15**. Those
  six skips require the optional editable-export API added in 3.17. Both new
  distributions pass metadata checks and a model/data/media member scan.

New datasets, source assets, model variants, observations and research rollouts
are not committed or uploaded. New wheel/source distributions are checked for
dataset, model and media members before CI artifacts are uploaded. Historical
repository previews and earlier releases are outside this new distribution
check; it is not a retrospective license clearance.

Remaining evidence includes anatomist-approved head/tendon mappings, measured
subject-specific strength and fibre/slack constraints, synchronized external
loads and motion, contact validation, and matched-pose geometry under different
loads. Population architecture and geometric allometry are explicit priors,
not measurements of the SOMA subject. See the
[physical guide](../docs/PHYSICAL_CALIBRATION.md) for workflows and limitations.
