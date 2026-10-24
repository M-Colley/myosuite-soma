# Scientific validation and known limits

This is a portable summary of the completed local SOMA/MyoSuite integration audit before extraction into `myosuite_soma`. It is **historical evidence**, not a claim that the new package has rerun every asset-backed experiment. Current package tests/CI are separate. [measurements.json](measurements.json) contains the selected unrounded results; [source-manifest.json](source-manifest.json) identifies exact original report hashes. Raw recordings, anatomy meshes and pretrained weights are not bundled. Rendered demonstrations are credited in `assets/` and distributed with the prerelease.

## Models and reproducibility boundary

| Use | Source environment | qpos / qvel / actuators |
|---|---|---:|
| Elbow | `myoElbowPose1D6MFixed-v0` | 1 / 1 / 6 |
| Full body | `myoMimicFullbody-v0`, `musclemimic_models==1.0.6` | 89 / 88 / 354 |
| Walking | `myoLegWalk-v0` | 35 / 34 / 80 |

The original MyoSuite base was `c0af9c656de2aa1807924e63d0d5320d77aa1ebd`. The executed geometry implementation was `joint-rbf-contact-3`:

```text
71e1ea266506ca6f983224b80052c0f191d69ac766e5b846ecd2306c510f9cb4
```

This is the original file hash. Namespace/import changes during packaging produce different file bytes. It does not identify the new packaged module. MuJoCo model/checkpoint/calibration/trajectory fingerprints remain necessary to reproduce an experiment. The original integrated validation used MuJoCo 3.15 and Python 3.13, with focused geometry checks also executed under Python 3.10 and 3.14. Asset-dependent results require the matching release assets and source models, beyond a successful package installation.

## All five released subjects: implementation parity

| Subject | Real poses / shots | Individual objects + skin | Maximum skin coordinate difference (mm) | Maximum object coordinate difference (mm) |
|---|---:|---:|---:|---:|
| S1 | 180 / 20 | 95 + skin | 0.000596 | 0.000715 |
| S2 | 51 / 13 | 97 + skin | 0.000477 | 0.000596 |
| S3 | 51 / 13 | 101 + skin | 0.000477 | 0.000596 |
| S4 | 54 / 14 | 99 + skin | 0.000596 | 0.000715 |
| S5 | 54 / 14 | 97 + skin | 0.000477 | 0.000596 |

**390 real poses across all 74 recorded shots** agree with the released equations within 0.000596 mm for skin and 0.000715 mm for individual objects. Subjects use their own rigs and checkpoints; joint names resolve S2's different order. This establishes numerical implementation parity. It does not measure subject anatomy, transferred skeletal alignment or activation-dependent deformation. Some released objects represent groups, tendons or unresolved labels; object counts are not independent physical muscle counts.

## Fresh calibration and independent walking

S2-S5 each received independently optimized full-body and leg calibration. No S1 fit was relabeled as another subject. Each fit uses 13 synthetic diagnostic states and holds out 84 intermediate states. Such a holdout measures interpolation within that diagnostic, not a new motion family. Walking uses all 201 native policy states and was not used to refit these profiles. Registered held-out controls exclude the two knees (walking) or knees and elbows (full body) from the deformation fit. These targets are source-model anchors, not measured tissue.

| Subject / case | States | Held-out anchor RMSE raw → registered (mm) | Whole-skin minimum z raw → registered (mm) | Max correction velocity (m/s) |
|---|---:|---:|---:|---:|
| S2 / fullbody | 97 | 15.55 → 16.59 **worse** | -22.81 → 6.00 | 0.088 |
| S2 / walking | 201 | 56.61 → 35.63 | -92.39 → -48.99 | 4.409 |
| S3 / fullbody | 97 | 10.66 → 15.69 **worse** | 6.54 → 10.83 | 0.061 |
| S3 / walking | 201 | 39.67 → 26.26 | -77.66 → -18.04 | 2.401 |
| S4 / fullbody | 97 | 16.83 → 20.84 **worse** | -87.76 → -39.95 | 0.080 |
| S4 / walking | 201 | 48.93 → 29.53 | -94.57 → -59.18 | 4.194 |
| S5 / fullbody | 97 | 14.27 → 14.07 | -113.31 → -71.66 | 0.066 |
| S5 / walking | 201 | 53.59 → 28.97 | -145.68 → -88.52 | 4.029 |

All **1,192 source states** generated finite skin and individual surfaces with unchanged native physical parameters and converged bounded temporal solves. Walking knee errors improve for all four. Full-body held-out errors worsen for **S2/S3/S4**. The S5 walking skin still reaches **88.52 mm below the plane**. Registration is therefore opt-in, with retained raw output and explicit failure metrics.

Skin z uses the minimum across all skin vertices and frames, with the explicit floor at z=0. Native walking collision primitives themselves reach -19.50 mm under soft contact. Extra visual penetration relative to the corresponding native support is a separate per-frame metric; its maxima are 39.17/19.11/51.98/83.97 mm for S2/S3/S4/S5 walking. These are not differences of independently selected global minima. Correction velocity differentiates registered-minus-raw displacement, not total skin motion. Walking maxima of 2.40-4.41 m/s remain a temporal limitation despite solver convergence.

Shared bounds were unchanged across subjects: 0.12 m maximum displacement, segment scale 0.6-1.5, minimum sampled registration Jacobian 0.05, field gradient norm 0.6, and contact compression no lower than 0.20. The walking runs approach their lower sampled Jacobian bounds and contact compression bounds. Sampled triangle/edge ratios expose local distortion; positive sampled bounds and fixed-membership compression do not prove combined-field injectivity, tissue volume preservation or freedom from self-intersection. Full unrounded bounds and surface statistics are retained in measurements.json.

For S1, the final registered skin minima are -6.90 mm (full-body diagnostic) and -27.06 mm (policy walking); maximum combined corrections are 50.16 and 54.14 mm. These also leave contact error.

## Independent recorded movement

Thirty-four independent whole-shot keyframes from S1 shots 003/006 give source-anchor IK RMSE **24.67 mm** and raw visual roundtrip RMSE **39.63 mm**. Shot 015 was used for solver development and is not counted as an independent solver holdout. These are bounded per-pose IK reconstruction results, not controller or dynamics validation. The continuous 76-frame shot 006 registration improves withheld knee/elbow error from **37.27 to 23.00 mm** while leaving native physical parameters unchanged.

Joint limits, equality polynomials and quaternion-aware velocities are retained. A recording supplies kinematics, not genuine muscle activation or load. Uniform colors and rest activation are appropriate unless synchronized activity is available.

## Conditional geometry is research tooling

The released predictor takes pose, not activation. An optional separate shared canonical residual model supports named activation/transmission/fibre features, whole-recording/load/subject splits, training, held-out evaluation and opt-in inference. Its local demonstration used real S1 topology with **fabricated target geometry**: 32 training and 8 held-out synthetic frames. At an identical pose, changing the supplied biceps feature changes vertices by approximately **3.05 mm**. Reference physiology gives exactly zero residual. Tiny analytic-fixture recovery errors prove numerical behavior only.

`physiologically_validated` is **false**. No measured paired load-dependent training geometry is supplied. Simulated musculotendon length/velocity and signed actuator transmission force are distinguished from true fibre observations. Skin-only targets do not validate internal muscle shape. Matched-pose measurements under different loads/co-contraction, synchronized state provenance, independent motion/subject/load holdouts and anatomical review are still needed; see [CONDITIONING.md](../docs/CONDITIONING.md).

## Software evidence and acceptance

The historical integration suite collected 194 cases with 1 skip and zero failures/errors; the core regression suite collected 312 with 6 skips and zero failures/errors; source-model parity collected 3 with no failures. Focused geometry collected 67 each under Python 3.10 and 3.14. These suites overlap and are not a count of distinct scientific experiments. The standalone package has its own test matrix and compatibility checks. Configured CI platforms must be distinguished from jobs actually executed.

Scientific acceptance is still limited by independent anatomical correspondence, residual contact/shape/temporal errors and missing paired physiology geometry. Native strength, optimal fibre length, tendon slack length, pennation, mass/inertia, collision geometry and controller parameters were not changed to conceal those errors. No numerical or rendering success is treated as proof of physiological tissue mechanics.
