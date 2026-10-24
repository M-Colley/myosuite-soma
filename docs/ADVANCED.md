# Registered anatomy, contact fitting and conditional geometry

These optional stages extend offline visualization. Source strength, optimal
fibre length, tendon slack length, pennation, joints, masses and collisions
remain unchanged.

```text
Recorded pose and native muscle states
  -> released SOMA prediction
  -> optional canonical physiology corrective
  -> source-world placement
  -> optional shared registration and local contact morphology
  -> numeric cache -> USD -> Blender
```

## Geometry configuration

Add `geometry` to an existing SOMA JSON. Omit it for raw released geometry.
For example, this section identifies exact source body points:

```json
{
  "geometry": {
    "landmarks": {
      "RightFoot": {"body": "talus_r"},
      "RightToeBase": {"body": "toes_r"},
      "LeftFoot": {"body": "talus_l"},
      "LeftToeBase": {"body": "toes_l"}
    },
    "radius_m": 0.35,
    "max_displacement_m": 0.12,
    "min_segment_scale": 0.6,
    "max_segment_scale": 1.5,
    "min_jacobian": 0.05,
    "regularization": 0.001,
    "max_gradient_norm": 0.6,
    "contacts": []
  }
}
```

This is a geometry section, not a complete asset configuration. Names depend
on the exact source model. Body targets may specify `offset` in local metres;
`joint` targets use true joint anchors. `heldout_landmarks` identifies targets
excluded from fitting for independent geometric checks.

A shared regularized smooth field moves skin and all underlying surfaces
together. Segment, displacement, gradient and sampled Jacobian limits bound
the fit; close limbs can retain residual alignment error. Agreement with fitted
controls is not anatomical ground truth. Report omitted landmarks separately.

Each `FootContactConfig` names the rig ankle/toe pair, source collision
primitives and an explicit static source plane. Contact ownership comes from
the subject's native foot/toe skinning weights, including the individual muscle
bindings; nearby contralateral limbs and hands receive no correction without
that membership. Production generation supplies these masks automatically.
Standalone `deform_animation` callers must supply `ContactMembership`, normally
through `contact_membership_from_predictor(predictor, config.contacts)`.
The local morph preserves the ankle/toe axis, with smooth stance/release and
time-aware regularization. Its full three-dimensional segment projection stays
continuous for nearly vertical feet, and contact fitting leaves rig centers fixed. It
uses source collision clearance, not inferred contact force. Unsupported terrain
and unrecorded mocap dependencies are rejected.

Compare absolute skin clearance and added penetration relative to the source
collision sole, which can itself penetrate a soft contact plane. Report sliding,
temporal variation and mesh distortion. Contact fitting can substantially alter
foot thickness; it does not establish tissue incompressibility, contact pressure
or personalized anatomy.

Version 3 contact diagnostics distinguish local scale checks from a full spatial
Jacobian, because anatomical membership varies over a surface. Mesh distortion
checks remain sampled evidence, not a global injectivity guarantee. Historical
version 2 caches can be viewed for comparison; regenerate them with current
source before using corrected contact behavior.

## Recorded full-body motion

`myosuite_soma.soma_retarget` solves bounded kinematic inverse kinematics using
an explicit source profile and `IKConfig`. It applies native equality
polynomials, retains joint limits, and writes quaternion-aware velocities.
For an original SKIM recording:

```shell
python -m myosuite_soma.soma_retarget --env myoMimicFullbody-v0 --profile fullbody-soma.json --ik-config recorded-ik-config.json --recording /data/SKIM_dataset/S1/shot_006.npz --packing interleaved --root-convention native --frequency 25 --start 650 --frames 76 --output recorded-006.npz
```

Original and training files use different six-dimensional rotation layouts
and root conventions. The packing and root convention are separate explicit
options. Training poses require `--packing concatenated` and
`--root-convention skim-training-ry90`; `--roots-recording` can supply original
native roots when packed training files omit them. Do not apply coordinate
changes twice. An optional constant `--world-translation X Y Z` is recorded in
provenance; no per-frame floor lift is applied.

Replay is kinematic, not controller-generated motion. Keep activation at rest
and use uniform anatomy colors unless genuine synchronized native activation
is supplied. Report IK error separately from subsequent anatomy registration.

## Physiology-conditioned geometry

[CONDITIONING.md](CONDITIONING.md) documents the numeric dataset,
training CLI, whole-group holdouts and measurement requirements. An optional
`conditioning_model` selects a trained artifact. The renderer supplies native
activation, musculotendon length/velocity and signed actuator transmission
force. These are not labelled as fibre measurements. True fibre features need
genuine measurements or a separately validated fibre model.

The shared canonical corrective uses the existing skinning transport and is
zero at its reference physiology. Required features, units, actuator order and
artifact hashes participate in cache validation, even without activation colors.

The supplied demonstration uses real S1 topology with fabricated targets. It
verifies software behavior only and requires `allow_synthetic_conditioning`.
Physiological validation needs independent matched-pose geometry under different
loads. Skin observations alone do not establish internal muscle shape.

## Chunked caches

Default NPZ export remains available. Choose chunked storage with:

```shell
myosuite-soma render --env myoLegWalk-v0 --random --seconds 0.4 --soma-config local-soma.json --output renders/new-run --export-only --soma-cache-format chunks
```

Add `--soma-cache-error-mm 0.1` to opt into displacement quantization. The bound
is maximum Euclidean vertex error measured against input coordinates. Exact
mode requires float32-representable coordinates. Both modes use numeric arrays,
member checksums and source provenance. Conversion needs no Torch/assets:

```shell
python -m myosuite_soma.soma_cache renders/old-run/soma.npz renders/walking.chunks --chunk-frames 16 --max-error-mm 0.1
```

`AnimationArchive(path).iter_chunks()` provides bounded-memory playback and mesh
selection. `.load()` materializes geometry with an explicit allocation guard.
The existing USD/Blender consumer also materializes the animation; streaming
storage does not make it or parallel training a live pipeline.

```shell
myosuite-soma benchmark renders/old-run/soma.npz benchmark-results --chunk-frames 16 --max-error-mm 0.1
```

Add `--config local-soma.json --motion recording.npz --env myoLegWalk-v0` to time
asset loading, retargeting and native inference separately. Reports include
versions, sizes, errors, payload and sampled RSS. Timings exclude rendering and
depend on hardware/background activity.

## Adapter boundary and provenance

The `myosuite_soma` backend handles geometry and accepts MuJoCo models directly.
The renderer commands above additionally require MyoSuite's optional thin SOMA
adapter; installing this backend alone does not add arguments to an existing
MyoSuite installation. See [integration](INTEGRATION.md).

Keep the exact source model, calibration, deformer and conditioning artifact with
each output. Regenerate caches after changing them. Retain raw geometry and
before/after metrics; do not edit fingerprints to accept mismatched inputs.
The historical validation in [REPORT.md](../validation/REPORT.md) predates the
package extraction; its executed source hashes identify that implementation.
