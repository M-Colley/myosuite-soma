# Optional MyoSuite adapter

The standalone `myosuite_soma` backend uses NumPy, SciPy, MuJoCo and optional Torch. It does not import MyoSuite during package initialization, inference, cache operations or standalone XML retargeting.

MyoSuite's existing Blender pipeline needs a small integration to record synchronized native physiology, call the external backend, and preserve the generated world-space surfaces. All SOMA inference, calibration, conditioning, retargeting and geometry implementations stay in this repository.

## Tested source version

The supplied patch targets MyoSuite dev commit:

```text
c0af9c656de2aa1807924e63d0d5320d77aa1ebd
```

It touches three files: `myosuite/viz/blender_render.py`, `myosuite/core/trajectory_io.py` and `myosuite/utils/video_io.py`. The latter changes preserve explicit trajectory safety and exact encoded dimensions. It adds no environments and changes no source physics.

## Install explicitly

Use a virtual environment, then follow these commands from this repository:

```sh
python -m pip install -e ".[inference,myosuite,render]"
git clone https://github.com/MyoHub/myosuite.git ../myosuite
git -C ../myosuite checkout c0af9c656de2aa1807924e63d0d5320d77aa1ebd
python -m pip install -e ../myosuite
myosuite-soma install-adapter --myosuite-source ../myosuite --check
myosuite-soma install-adapter --myosuite-source ../myosuite
myosuite-soma render --help
```

The command uses the patch packaged with the wheel. `--check` makes no source changes; applying requires a clean patch check, and repeated application fails with an explanation. No source modification happens at import or render time. The same patch is available as [`myosuite-adapter.patch`](myosuite-adapter.patch) for review.

Newer upstream revisions may differ. The tested commit is explicit; patch applicability alone is not validation of another revision. Re-run the adapter tests and a real export after an upstream update.

## Render

After installing the release assets and configuring their paths:

```sh
myosuite-soma render --env myoElbowPose1D6MFixed-v0 --random --seed 0 --seconds 0.4 --soma-config local-soma.json --output renders/elbow --export-only --muscle-color uniform
```

The adapter retains MyoSuite's flags, policy loading, source-state recording, USD export and Blender scene builder. Source checkpoints require their original environment/model versions. Export ends at episode termination. Use a new output directory for every run.

To render an existing numeric animation without MyoSuite:

```sh
myosuite-soma export-usd anatomy.npz anatomy.usdc
```

Standalone USD output preserves fixed topology, world-space coordinates, uniform sample spacing and the original starting timestamp in layer metadata. It does not create a studio camera or replay a native policy.
