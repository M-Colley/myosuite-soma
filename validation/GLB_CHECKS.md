# GLB compatibility checks

Executed locally on 2026-10-25 for version `0.1.0a2`. These checks establish file
and importer compatibility; they do not validate anatomical correspondences or
make native linear blend skinning equivalent to SOMA inference.

- Full package and pinned MyoSuite adapter suite: **254 passed, 1 skipped**.
  The skip requires Windows symbolic-link privilege. Ruff checks and formatting
  pass. The existing renderer emits its upstream empty-prefix attachment warning.
- The new tests independently decode the GLB buffers and reconstruct sampled
  positions, normals, interpolation, bind transforms and weights. Synthetic
  inference tests distinguish the source model's `qpos0` from calibration rest.
- MyoSuite's actual `read_skin_glb`, `load_skin`, `.skn` round trip and `SkinPose`
  are exercised using invented articulated geometry with six influences per
  vertex, nontrivial body transforms and three native poses.
- Local subject assets S1–S5 each produced a full-body reference rig and a
  three-sample nonuniform animation: **96/98/102/100/98 surfaces**, respectively,
  including skin. Independent native-skinning comparisons across three poses
  had maximum coordinate error below **1.11e-7 m**; baked sample reconstruction
  error was at most **7.46e-9 m**. Recorded native physical parameter digests
  remained unchanged for every subject.
- A separate S1 leg-model reference with all 96 surfaces also loaded through
  the native skin path. This uses inherited bindings for regions absent from
  that reduced source model; it is not evidence of independent upper-body control.
- The official Khronos `gltf-validator@2.0.0-dev.3.10` reported **zero errors and
  zero warnings** for all eleven full-body/leg files. The separate GLB CI job
  uses only invented synthetic meshes, rejects errors and warnings, and uploads
  no generated GLBs.
- Blender **4.5.3 LTS** imported S1's 96 anatomy surfaces and 46 named bones.
  All native vertex influences were retained, including vertices with 13
  influences. Maximum evaluated bind-pose coordinate error was below
  **7.21e-7 m**. Imported animated positions at the three nonuniform keys and
  intermediate times differed by at most **1.24e-7 m**.
- A native full-body forward rollout used the generated GLB through the
  existing `--skin` path, without SOMA generation. All five exported USD skin
  samples matched `SkinPose` recomputed from the recorded native states exactly
  (maximum coordinate difference **0 m**); source parameter digests remained
  unchanged.
- The built wheel was installed in a separate environment without Torch,
  MyoSuite or USD: **36 passed, 13 skipped** for the GLB/CLI tests. The skips
  need Torch inference. Both numeric cache formats exported there. Distribution
  metadata checks passed; all 25 wheel and 91 source-archive members were
  inspected, with no geometry, weights, numeric datasets, previews or videos.

The real subject GLBs and audit payloads remain local. They are not included in
Git, wheel/source distributions, release uploads or CI artifacts. The wheel
and source distribution contain conversion code and documentation; the source
distribution excludes the historical preview-media directory. Dataset,
checkpoint and converted-model access remains subject to the authors' terms.

See [the GLB guide](../docs/GLB.md) for local conversion commands, explicit mapping
rules, ordinary-skinning versus baked-deformation behavior, and size/viewer limits.
