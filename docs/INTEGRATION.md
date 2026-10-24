# Optional MyoSuite adapter

`myosuite-soma` is a standalone geometry backend. It does not bundle a MyoSuite
fork, Blender, simulation assets, learned policies or SOMA weights. Direct use
with a `mujoco.MjModel` and numeric arrays does not require MyoSuite.

The optional adapter adds SOMA configuration, native physiology capture, cache
selection and export wiring to MyoSuite's existing Blender renderer. It delegates
geometry to this package. Ordinary MyoSuite rendering and simulation remain in
MyoSuite, and the adapter does not register environments or modify native dynamics.

## Install into a supported source checkout

The initial adapter targets MyoSuite `dev` commit
`c0af9c656de2aa1807924e63d0d5320d77aa1ebd`. Use a clean, editable checkout of
that version and the backend's `inference`, `myosuite` and `render` extras.

```shell
myosuite-soma install-adapter --myosuite-source /path/to/myosuite
python -m pip install -e /path/to/myosuite
myosuite-soma doctor
```

`install-adapter` first checks source compatibility and whether the patch can
apply, then applies it. It is an explicit source modification. Keep the checkout
under version control and review the patch. A different release may require an
updated adapter even when its package version satisfies the optional dependency.
Package installation and source-adapter compatibility are distinct checks.

`myosuite-soma render ...` delegates to this adapted renderer. It provides an
actionable error if MyoSuite or the expected integration is unavailable. The
backend's cache and conditioning commands remain available without that adapter.
The retarget CLI can use MyoSuite to construct a named environment; its numerical
Python API can also work with a supplied MuJoCo model.

## Data exchanged at the boundary

| Input | Meaning |
|---|---|
| Model and `qpos` | Exact source kinematic model and recorded trajectory |
| Times | Actual, strictly increasing seconds |
| Named activation | Native actuator activation observations or explicit rest values |
| Named physiology | Optional native activation, musculotendon length/velocity and signed transmission force |
| JSON calibration | Subject-matched release paths, named body correspondences and world placement |

The geometry backend returns fully posed world-space meshes and provenance.
MyoSuite handles USD/Blender export. Source force, optimal fibre length, tendon
slack length, pennation, wrapping, mass/inertia, collision geometry and controller
states remain native. Generated anatomy is an offline visual observation and
does not feed back into dynamics.

Cache provenance includes trajectory/model and optional conditioning fingerprints.
Standalone viewing validates numeric storage, without requiring the current
checkpoint files. This does not establish that those current files match a past
run; use the stored asset/source hashes and regenerate after changes.

## Recorded movement and training scope

Inverse kinematics reconstructs bounded source kinematics from an explicit
recording layout. It does not infer muscle recruitment or contact forces. Keep
replay colors uniform unless synchronized activity is supplied. Quaternion-aware
velocity, source equality polynomials and joint limits are retained.

This integration is an offline replay/export pipeline. Chunked storage supports
bounded-memory iteration; the USD/Blender consumer may still materialize a full
animation. Neither storage streaming nor the optional adapter implements live
parallel training observations, differentiable physics or coupled soft tissue.
