# GLB export and MyoSuite compatibility

The package provides two local conversion workflows. Choose a reference rigged
skin for MyoSuite's native skin importer, or bake an existing SOMA animation for
playback in a glTF viewer.

| Output | What it preserves | How it is driven |
|---|---|---|
| Reference rigged GLB | Predicted anatomy at the source model's `qpos0`, topology and remapped subject weights | Ordinary linear blend skinning from named MuJoCo bodies |
| Baked animated GLB | Cached skin and internal-surface vertex samples, including any corrections already applied | Standard glTF morph-target animation at the recorded times |

Both use glTF 2.0 binary files, metres, and glTF's Y-up convention. The exporter
converts MuJoCo `(x, y, z)` to glTF `(x, z, -y)` once. Existing output files are
never overwritten. The writer uses the core package dependencies; it does not
need Blender, USD or a patched MyoSuite installation.

## Export a native MyoSuite skin

Install the `inference` extra, obtain the subject's assets separately, and make
a local configuration for the exact source model. For the documented full-body
profile, this additionally requires the matching MyoSuite environment and
`musclemimic_models==1.0.6`.

```sh
myosuite-soma export-glb rigged --config local-soma.json --env myoMimicFullbody-v0 local-assets/anatomy-skin.glb
```

To use a standalone MuJoCo model instead of a MyoSuite environment:

```sh
myosuite-soma export-glb rigged --config local-soma.json --model source-model.xml local-assets/anatomy-skin.glb
```

The converter evaluates the SOMA geometry at `model.qpos0`, binds it to the
source model's named body frames at that pose, and writes a chained skeleton
with inverse bind matrices. The configuration's calibration reference is still
used when retargeting that pose; it does not replace the GLB bind pose.

Reference export requires a pose-only configuration. If `conditioning_model`
is set, export a previously generated conditioned animation cache with the
baked workflow instead; the converter does not invent physiology for a bind pose.

The default output contains the skin only. With `--include-muscles`, separate
internal surfaces share the same skeleton in one GLB. The skin is the first
skinned mesh so MyoSuite's existing importer selects it. That importer reads
the skin mesh and weights; it does not import the other surfaces as physical
muscles or collision geometry.

On a MyoSuite revision with the rigged-GLB skin support from
[PR #554](https://github.com/MyoHub/myosuite/pull/554), use the file with the
existing rendering interface:

```sh
python -m myosuite.viz.blender_render --env myoMimicFullbody-v0 --random --seconds 0.4 --skin local-assets/anatomy-skin.glb --output renders/glb-skin
```

Pass `--blender /path/to/blender` as required by that renderer. The SOMA source
adapter is unnecessary for this reference skin route. Follow upstream's
instructions when painting weights in Blender: preserve the rest pose and body
names, and export all bone influences rather than truncating them.

### Joint coverage

SOMA joints and MuJoCo bodies are different rigs. Conversion uses exact names
from the configuration's `body_map` and, where supplied, a flat
`--joint-map` JSON such as `{"Hips": "Full Body", "LeftUpLeg": "femur_l"}` for
the documented full-body model. Explicit
entries override the configuration's assignments. The root can be supplied by
`root_body` or the elbow preset; an otherwise unmapped anchor joint can use its
configured `anchor_body`. An unmapped joint inherits its nearest mapped SOMA ancestor;
its weights are added to that ancestor's source body. Multiple SOMA joints may
therefore contribute to one source body. Weighted joints without a mapped
ancestor fail with an error rather than silently receiving a guessed body.

Use an explicit map for missing regions or deliberately reduced source models.
The resolved assignments are recorded in the GLB metadata. Review inherited
assignments before relying on the resulting skin. A successful
conversion and preserved weights do not establish anatomical correspondence.
All influence sets are written; the exporter does not silently discard weights
to meet a four-influence limit.

This reference file preserves the anatomy at the bind pose. During subsequent
motion, native MyoSuite skinning does not evaluate SOMA's learned shape
corrections, rigid corrections, registration or contact morphology. Those effects
are available through the existing SOMA pipeline or the baked export below.

## Bake the learned deformation

Start with an existing numeric animation cache created by the backend:

```sh
myosuite-soma export-glb animation anatomy.npz local-assets/anatomy-motion.glb
```

The input may also be a chunked archive directory:

```sh
myosuite-soma export-glb animation anatomy.chunks local-assets/anatomy-motion.glb
```

Each surface retains its topology and name. The first sample is the base mesh;
each later sample supplies a position and normal morph target. Animated weights
reconstruct the cached geometry at keyframes, subject to float32 representation.
Playback between samples linearly interpolates those shapes; it does not run
SOMA inference at new poses. Material colors are uniform skin/muscle colors,
not animated native activation or a new estimate of muscle control.

The GLB timeline starts at zero and retains the source time range as metadata.
Nonuniform sample times are supported. The exporter rejects timelines that
cannot remain strictly increasing in glTF's float32 time representation.
After importing into Blender, set the scene playback/render range to the
imported action's range. Blender can leave its default scene range unchanged,
and short clips can have keys at frame zero and fractional frames.

The default limit is 256 morph targets per surface, corresponding to at most
257 input samples. For a viewer that supports a larger asset, opt in explicitly:

```sh
myosuite-soma export-glb animation anatomy.npz local-assets/anatomy-motion.glb --max-targets 512
```

Morph data grows with vertices times samples, and the animation weights grow
with the number of samples squared. Export loads the full animation, including
when the source is a chunked archive. Use short clips for local inspection;
this format is not a bounded-memory stream or a live simulation interface.
Viewer limits can be stricter than the glTF format limits.

MyoSuite's skin importer reads reference vertices and weights. It does not
evaluate glTF animation channels or morph targets, so a baked clip is intended
for a glTF viewer or Blender rather than `--skin`.

## Python API

```python
from myosuite_soma.glb import export_glb
from myosuite_soma.soma import load_animation

animation = load_animation("anatomy.npz")
export_glb(animation, "local-assets/anatomy-motion.glb", max_targets=256)
```

`myosuite_soma.glb.export_rigged_glb` is the lower-level writer for a one-frame
`SomaAnimation`, explicit body names, parent indices, world-space rigid bind
matrices, and a vertex-weight matrix for each surface. The rigged CLI constructs
those inputs from the local subject and source model; generic callers must
supply a validated correspondence themselves.

For the same source-model conversion as the rigged CLI:

```python
from myosuite_soma.glb_export import export_reference_glb

export_reference_glb(
    "local-soma.json", model, "local-assets/anatomy-skin.glb",
    include_muscles=False,
    joint_map={"Hips": "Full Body"},
)
```

## Local assets and distribution

Obtain the dataset, geometry and checkpoints through the
[SOMA authors' download site](https://gvv-assets.mpi-inf.mpg.de/soma/) and follow
the terms accompanying that download. The supplied download terms restrict
redistribution, including converted models. This project supplies conversion
code; it does not grant permission to distribute generated anatomy.

Keep original and converted assets in a local directory such as `local-assets/`.
Generated `.glb` and `.gltf` files are ignored by Git. No subject geometry,
weights, checkpoints or generated GLB should be added to repository commits,
Python distributions or GitHub release assets. The public source-code license
is separate from the terms for downloaded assets. Tests construct synthetic
meshes instead of shipping subject anatomy.

A separate CI job generates invented meshes and validates both GLB modes with
Khronos `gltf-validator@2.0.0-dev.3.10`; errors and warnings fail the job, and
generated files are never uploaded. Atomic output publication currently needs
a filesystem supporting hard links, such as NTFS; an unsupported destination
fails without publishing a partial file.

Local all-subject, MyoSuite importer and Blender checks are recorded in
[GLB compatibility checks](../validation/GLB_CHECKS.md). These assess conversion
and playback, rather than anatomical mapping accuracy.
