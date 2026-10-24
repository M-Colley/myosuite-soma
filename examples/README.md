# Portable configuration examples

All asset paths are placeholders relative to each JSON. Obtain release assets
separately and edit the paths in a local copy. No checkpoint, mesh, recording,
policy, token or local machine path is included here.

| File | Scope |
|---|---|
| `soma_elbow.json` | Generic `myoElbow1D6` identity-calibration starting point |
| `soma_s1_elbow_calibrated.json` | Historical S1 / `myoElbowPose1D6MFixed-v0` geometric calibration |
| `soma_s1_fullbody_calibrated.json` | S1 / `myoMimicFullbody-v0`, exact `musclemimic_models==1.0.6` |
| `soma_s1_legs_calibrated.json` | S1 / `myoLegWalk-v0` source skeleton |
| `soma_fullbody_ik.json` | IK landmark/objective configuration; not a SOMA asset config |
| `soma_conditioning_dataset.schema.json` | Numeric paired-data metadata schema; not measured training data |

The S1 numeric fits are model-specific approximations. Preserve subject-specific
rig order and checkpoint pairing. Validate neutral poses and movement sweeps
after any model, subject, asset or calibration change. Full-body and leg profiles
are distinct. Subject fits are not transferable by replacing an asset filename.

`muscle_map` defines a display-color aggregation, not an anatomical muscle-head
identity or a learned activation. Unresolved surfaces retain uniform colors.
Numbered triceps and gastrocnemius surfaces do not establish head assignments.
The historical numerical results and unresolved errors are summarized in
[validation](../validation/REPORT.md).

## Independently fitted S2-S5 profiles

`subjects/S2` through `subjects/S5` each provide `fullbody-raw.json`,
`fullbody-registered.json`, `walking-raw.json` and `walking-registered.json`.
The raw files preserve each subject's independently fitted numeric calibration;
registered files additionally enable the exact experimental field/contact settings
used in the historical eight-case comparison. These are geometry calibrations,
not pretrained networks. Release weights and surfaces are still required.
Their paths point to the repository-root placeholder `local-assets` directory.

Only `data_root` and `checkpoint` were changed from the historical configurations.
[Calibration provenance](../validation/calibration-provenance.json) records
original and portable-file hashes. Registered output worsens some held-out
full-body anchors and retains substantial contact error; select it explicitly
and retain raw comparisons. These profiles are restricted to their named source
models and release subjects, not a general personalized-body fit.
