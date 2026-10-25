# Run SOMA geometry with MyoSuite

The Python backend is `myosuite_soma`. It accepts a MuJoCo model and recorded
arrays directly. MyoSuite is optional. MyoSuite's Blender renderer additionally
needs the small, version-checked adapter described in [INTEGRATION.md](INTEGRATION.md).

## Install

Use a virtual environment and install from this repository:

```shell
python -m pip install -e ".[inference,myosuite,render]"
myosuite-soma doctor
```

The core has NumPy, SciPy, MuJoCo and OSQP dependencies. `inference` adds Torch;
`myosuite` adds the simulation library; `render` adds the MuJoCo USD support used
by the external renderer. Blender is a separate installation. These dependencies
do not provide release geometry, recordings or pretrained SOMA weights.

## Supply release assets

Obtain SKIM v2 and `SOMA_checkpoints.tar` through the
[SOMA release](https://github.com/edualvarado/SOMA#data-preparation) and retain
their accompanying terms. This repository includes source code, configuration
and numeric validation summaries. It does not redistribute those assets.

For one subject you need the dataset metadata, its canonical/layer/individual
geometry and rig files, its main checkpoint, the adjacent `_rigid.pth` checkpoint,
and its static canonical array. Preserve the release directory structure.

Copy [soma_s1_elbow_calibrated.json](../examples/soma_s1_elbow_calibrated.json) to
a local configuration and edit `data_root` and `checkpoint`. All JSON paths
resolve relative to that JSON, not the shell's working directory. The examples
point to a placeholder `local-assets` directory alongside `examples`.

```shell
python -m myosuite_soma.soma --inspect-config examples/soma_s1_elbow_calibrated.json
```

Inspect the actual subject's joint and object names. S2-S5 have their own rigs,
weights and labels; changing only `subject` in an S1 calibration is insufficient.

## Use the backend directly

The caller supplies a matching source model, valid recorded poses and named
activation columns. A normal muscle recording has `(T, nq)` positions, strictly
increasing `(T,)` seconds and `(T, A)` activation. It can also supply recorded
actuator length, velocity and force through `physiology`.

```python
from myosuite_soma.soma import generate_animation, save_animation

animation = generate_animation(
    "local-soma.json", model, qpos, time, actuator_names, activation,
    physiology=recorded_physiology,
)
save_animation("animation.npz", animation)
```

With the `render` extra installed, uniformly sampled geometry can be exported
directly to USD without the MyoSuite adapter:

```shell
myosuite-soma export-usd animation.npz anatomy.usda
```

GLB export uses the core dependencies and needs neither USD nor Blender:

```shell
myosuite-soma export-glb animation animation.npz local-assets/anatomy-motion.glb
```

This bakes the recorded vertex deformations for playback in a glTF viewer. To
create a reference skin for MyoSuite's native GLB importer instead, run:

```shell
myosuite-soma export-glb rigged --config local-soma.json --env myoMimicFullbody-v0 local-assets/anatomy-skin.glb
```

Use a calibration for that exact source model. A standalone XML can be supplied
with `--model` instead of `--env`. Rigged export needs local subject assets and
the `inference` extra; loading an existing numeric animation does not. Follow
the [GLB guide](GLB.md) for joint coverage, native skin import and the distinction
between ordinary skinning and baked SOMA deformations. Keep converted anatomy
files local under the accompanying asset terms.

Here `model` is an existing `mujoco.MjModel`; the arrays come from your
simulation or validated kinematic replay. No environment or asset download is
created implicitly. Unsupported names, dimensions, quaternions and calibration
fields fail with an error. A kinematic recording does not supply muscle activity:
use rest activation and uniform colors unless synchronized activity is known.

## Render through MyoSuite

Install the [thin adapter](INTEGRATION.md) into the supported editable MyoSuite
source first. Then run an explicitly random, short pipeline demonstration:

```shell
myosuite-soma render --env myoElbowPose1D6MFixed-v0 --random --seed 0 --seconds 0.4 --soma-config examples/soma_s1_elbow_calibrated.json --output renders/elbow --export-only
```

Replace `--random` with a compatible policy or motion recording for meaningful
motion. Remove `--export-only` and pass `--blender /path/to/blender` to render.
Use a new output directory for each run. This command delegates to the installed
MyoSuite renderer; installing the backend alone does not add renderer arguments.

Full-body and leg starting profiles are in [examples](../examples/README.md).
The full-body calibration requires `musclemimic_models==1.0.6` and its exact
89/88/354 source model. Changed source models need recalibration. Registration,
contact morphology and experimental conditioning remain explicit opt-ins;
see [ADVANCED.md](ADVANCED.md) and the measured [limitations](../validation/REPORT.md).
