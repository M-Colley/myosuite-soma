# Reproduce the checks without redistributing release assets

The [report](REPORT.md) summarizes a historical integration run. Its source
hashes are retained in [source-manifest.json](source-manifest.json). A public
checkout supplies numerical software tests, portable calibrated profiles and
measurement protocols; release recordings, networks and geometry must be
obtained separately. A new run must record its own executed package commit and
hashes rather than reusing the historical implementation fingerprint.

## Software checks

Install the test and inference extras into a virtual environment, then run:

```shell
python -m pip install -e ".[test,inference]"
python -m pytest
myosuite-soma doctor
```

These tests exercise numeric geometry, conditioning, archive and retargeting
contracts using small fixtures. Asset-free tests cannot establish parity with
a private local release or measured biological validity. The optional MyoSuite
renderer additionally needs the [adapter](../docs/INTEGRATION.md).

## Asset and trajectory setup

Obtain SKIM v2 and matching release checkpoints and verify their release
checksums. Edit the placeholder paths in a local copy of the selected
[profile](../examples/README.md). Preserve the original JSON plus its hash,
checkpoint/rig/canonical file hashes and exact source model. Each subject has
its own order and assets; do not substitute S1 parameters into S2-S5.

```shell
python -m myosuite_soma.soma --inspect-config examples/subjects/S2/fullbody-raw.json
```

Reproduce a recorded kinematic slice with explicit layout and sampling:

```shell
myosuite-soma retarget --env myoMimicFullbody-v0 --profile examples/soma_s1_fullbody_calibrated.json --ik-config examples/soma_fullbody_ik.json --recording /data/SKIM_dataset/S1/shot_006.npz --packing interleaved --root-convention native --frequency 25 --start 650 --frames 76 --output recorded-006.npz
```

Use the exact `musclemimic_models==1.0.6` full-body source. Keep kinematic replay
uniform-colored with explicit rest activation unless synchronized activity is
available. The independent historical solver protocol used 17 uniformly spaced
keyframes from each complete shot 003/006; shot 015 was development data. A
selected continuous slice is a separate temporal test, not an independent
whole-shot keyframe study.

## Geometry and contact protocol

Generate raw and explicitly registered profiles against the **same** source
model, poses, times and physiology using `generate_animation`. The supplied
S2-S5 `*-raw.json` and `*-registered.json` retain the historical numeric
calibrations/settings; only their asset paths were made portable. Their original
and portable fingerprints are in [calibration-provenance.json](calibration-provenance.json).

Retain every diagnostic state (97 at 30 Hz) and every independent policy-walking
state (201 at 100 Hz), including first and last. Intermediate diagnostic
calibration holdouts are 84 states after the 13 evenly spaced training states.
Do not refit on the walking trajectory or choose favorable frames. The
historical fitting/generation orchestration was part of the original local
audit bundle rather than a bundled source-data downloader.

Evaluate all named anatomical objects and skin for finite coordinates and
unchanged topology/native physical-parameter hashes. Report withheld knees and
elbows separately from fitted controls. Measure whole-skin floor minima, extra
foot-support penetration relative to each frame's source primitive clearance,
correction-displacement finite differences, sampled edge/area ratios and all
solver/bound diagnostics. A positive sampled registration Jacobian does not
prove the combined contact field is injective.

The original cases retained their full trajectories but sampled surface
edge/area quality at four diagnostic frames. State that sampling scope in any
reproduction. Keep worsening errors and residual penetration in the result;
changing source collision geometry or relaxing bounds changes the experiment.

## Conditional training protocol

Follow [CONDITIONING.md](../docs/CONDITIONING.md) for numeric paired data,
collection and training/evaluation commands. The local controlled example used
fabricated shared-field targets on real S1 topology, 32 training frames and one
whole held-out synthetic recording of 8 frames. It verifies exact-zero reference
behavior and differing output at identical pose with differing feature input.
It supplies no physiological validation. Do not claim that reproducing its
small analytic error validates measured muscle deformation.
