# Geometry workflow and muscle-state meaning

The backend produces offline anatomy from a source MuJoCo trajectory. The
released SOMA predictor supplies pose-conditioned shape; the source simulator
continues to determine motion, muscle activation and force.

```text
Source model + recorded qpos/time + named native physiology
  -> explicit named source-to-subject local rotation calibration
  -> subject-matched released main and rigid networks
  -> canonical offsets on skin and individual anatomical objects
  -> optional separately trained physiology residual in canonical metres
  -> original linear blend skinning and world placement
  -> optional shared world-space registration field
  -> optional per-foot native-membership contact morphology
  -> validated numeric geometry cache
  -> standalone USD or baked GLB animation
  -> optional external MyoSuite USD/Blender renderer
```

## What changes at each stage

| Quantity | Stage and behavior |
|---|---|
| Maximum force, optimal fibre length, tendon slack length, pennation and force/velocity parameters | Remain native source-model parameters throughout |
| Joint limits, equality polynomials, masses, inertia, tendons and collision geometry | Remain source-model mechanics; visualization does not feed back |
| Native activation, actuator length/velocity/force | Supplied recording inputs; length/velocity represent musculotendon transmission for these muscle models |
| Canonical skin and individual object vertices | Main and rigid released networks infer pose-conditioned corrections |
| Optional conditioned vertices | Separate learned shared-field residual added before skinning; exactly zero at its stored reference physiology |
| Posed vertices | Original subject skinning transports offsets; source-world placement applies calibrated scale/rotation/root motion |
| Optional registered skin/internal surfaces | One shared bounded field after world placement; named source anchors constrain it |
| Optional foot morphology | Source primitive-plane clearance drives local compression with static native Foot+Toe membership and temporal regularization |
| Colors | Explicit mesh-to-actuator weighted averages only; uniform for unmapped objects |

Canonical additive vectors use the same skinning transport as the released
geometry. Root translation is not applied to those vectors twice. Geometry is
fully posed before USD export, so the renderer must not apply another armature.

The [GLB exporter](GLB.md) has two explicit contracts. Baked animation stores
the same fully posed world-space samples as morph targets without an armature.
Reference rig export evaluates geometry at the source model's `qpos0`, binds
it to named source bodies and exports their hierarchy, inverse bind matrices
and remapped subject weights. Subsequent native skinning is an approximation;
it does not run SOMA's predictor or recompute registration/contact. Neither
export changes source physics.

The contact field is a visual correction, not a material or pressure solver.
Source collision soles can penetrate a soft-contact plane; report absolute
skin height and extra visual penetration relative to the matching native
support separately. Nearby hands or the opposite leg receive no contact
correction without that subject's native material membership. Rig centers
remain fixed during contact morphology.

## Explicit coordinate and naming contracts

`basis` maps source-parent axes to SOMA-parent axes. Each `body_map` entry names
source child/parent bodies and a calibrated rest rotation. `rest_qpos` selects
the native reference, otherwise `qpos0` is used. Root offset is expressed in
the source root body's local metres and rotates with it; output translation
is a constant world displacement. Every subject uses its own joint order,
parents, canonical layers, individual objects and matching checkpoints.

Recorded six-dimensional rotations have explicit packing and root conventions.
Original SKIM files use `interleaved`/`native`; the released training pose
layout uses `concatenated`/`skim-training-ry90`. Do not infer one from the other
or apply its root transform twice. Kinematic IK uses source joint limits and
equality couplings and derives velocity with MuJoCo's quaternion-aware operation.

SOMA object labels and native actuator labels are not one-to-one. Some released
objects are groups, tendons or unresolved regions. Display aggregation is
explicit; numbered surfaces do not establish muscle-head identity. A shared
color or deformation field is not a validated internal anatomical measurement.

## Optional research and storage layers

Conditioning requires a separately fitted artifact and complete named features.
The included validation fixture uses fabricated targets and is synthetic only.
Pose-only release data cannot identify different shape at the same pose under
different loads. True fibre features must be supplied explicitly; actuator
transmission states are not silently renamed as fibre observations.

NPZ and chunked archives use numeric arrays, bounded allocation checks and
checksums. Quantization is optional and reports maximum Euclidean vertex error.
Provenance checks distinguish standalone viewing from source-checked reuse.
Streaming archive iteration does not make a consumer that materializes an entire
animation stream automatically. This implementation is not a live training or
coupled soft-tissue simulation backend.

See [advanced options](ADVANCED.md), [conditioning](CONDITIONING.md) and the
[measured failure cases](../validation/REPORT.md).
