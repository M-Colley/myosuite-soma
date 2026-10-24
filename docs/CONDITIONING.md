# Optional physiology-conditioned geometry

The released SOMA predictor remains pose-only. The optional conditioning module
adds a separate, experimental corrective using recorded physiology. It changes
visual vertices and leaves MuJoCo dynamics, muscle parameters and contact forces
unchanged. **No measured load-conditioned geometry dataset or physiologically
validated corrective is included.** The supplied demonstration uses real S1
anatomy with explicitly fabricated corrective targets.

## What is executable

`myosuite_soma.soma_conditioning` provides a numeric training dataset, whole-group
splits, regularized training, checkpoint loading, held-out evaluation, and an
optional animation/export hook. Training fits one smooth canonical vector field
shared by the skin and individual anatomical surfaces. Gaussian spatial nodes
provide the field; low-rank modes reduce its coefficients. Ridge regression
maps physiology and its interactions with a small pose projection to those
coefficients. The original skinning transports these additive vectors into the
posed frame; root translation is not applied twice.

The model has no intercept. When all selected physiology features equal their
stored reference values, its corrective is exactly zero and the original SOMA
vertices are preserved. A stored reference is a mathematical baseline, not an
automatically identified physiological resting state. Framewise feature bounds
and a maximum corrective magnitude reject unsupported inputs; these bounds do
not certify that a multivariate input is in the training distribution.

## Dataset and collection requirements

Use `ConditioningDataset` and `save_dataset()` to construct a pickle-free `.npz`
file. The schema requires these arrays and explicit JSON metadata:

| Field | Meaning |
|---|---|
| `rest_points`, `mesh_names`, `mesh_counts` | Concatenated canonical geometry in exactly the export mesh order |
| `rotations` | `(frames, joints, 3, 3)` proper local rotations in the released checkpoint's joint order |
| `baseline`, `target` | Corresponded `(frames, vertices, 3)` geometry in **SOMA canonical corrective metres** |
| `states`, `reference` | Named `(frames, actuators)` observations and constant per-actuator zero-corrective references |
| `actuator_names` | Exact native names, with no duplicated or implicit head correspondences |
| `subject`, `motion`, `load` | One stable subject, complete recording, and load-condition ID per frame |
| `observation_weights` | Nonnegative per-vertex observation confidence; zero means unobserved |
| Metadata | `data_kind`, coordinate frame, feature units, geometry/physiology provenance and measurement protocol |

Supported state fields deliberately distinguish transmission and fiber states:

| Field | Required unit string |
|---|---|
| `activation` | `dimensionless` |
| `actuator_length` | `m_musculotendon` |
| `actuator_velocity` | `m_s_musculotendon` |
| `actuator_force` | `N_signed_transmission` |
| `fiber_length` | `m_fiber` |
| `fiber_velocity` | `m_s_fiber` |

For the muscle models used here, MuJoCo's recorded actuator length and velocity
describe the musculotendon transmission. They are not measured fiber length or
velocity. True fiber fields must be collected or estimated by an independently
documented method and supplied explicitly; the pipeline does not substitute
one quantity for the other. General motor-actuator coordinates may have other
units and are outside this muscle-data schema.

Collection must synchronize calibrated multi-view skin geometry, skeletal pose,
load/contact measurements and activation observations. EMG requires a documented
normalization, delay and uncertainty model; simulated activation and transmission
force must retain their simulated provenance. Record repeated poses under low
and high external loads and different co-contraction, including unloaded
references. Preserve whole recordings and load protocols for held-out tests.

Raw world scans cannot be passed as canonical targets. Establish surface
correspondence, remove root placement, and undo the same blended skeletal
transform used by SOMA, rejecting ill-conditioned inverse transforms. Subtract
the released pose-conditioned baseline in that same canonical frame. Document
registration uncertainty so it cannot be mistaken for a load-driven residual.
Geometry targets must match the selected subject, checkpoint and topology.

Surface capture directly supervises skin. Internal muscles require additional
evidence such as suitably registered ultrasound/MRI or an independently
validated mechanical model. If only skin is observed, set internal-surface
weights to zero. Those surfaces inherit the shared field, but their deformation
is an unmeasured extrapolation; low skin error does not validate muscle shapes.
The shared field promotes coherent corrections, without guaranteeing tissue
incompressibility, attachment accuracy, absence of intersections or correct
contact mechanics. Those require independent anatomical and geometric checks.

## Training and held-out evaluation

```python
from myosuite_soma.soma_conditioning import (
    ConditioningFitConfig, evaluate_conditioning, grouped_split, load_dataset,
    save_conditioning_model, train_conditioning, validation_gate,
)

data = load_dataset("paired_geometry.npz")
train, holdout = grouped_split(
    data, holdout_motions=["walking_turn_03"], holdout_loads=["heavy_heldout"]
)
model = train_conditioning(data, train, holdout, ConditioningFitConfig())
save_conditioning_model("research_corrective.npz", model)
print(evaluate_conditioning(model, data, holdout))
print(validation_gate(data))
```

Selected subject, motion or load IDs are held out with an OR rule and cannot
occur in training. Split by entire motion families, subjects and load conditions
in separate evaluations as appropriate; random neighboring-frame splits are
not sufficient. Field modes, feature scaling and pose projections are fitted
using training observations only. Changing held-out geometry cannot alter the
trained coefficients. Confidence-weighted evaluation reports errors only on
observed vertices and explicitly counts that scope.

The equivalent command line is:

```text
python -m myosuite_soma.soma_conditioning train paired_geometry.npz research_corrective.npz --holdout-motion walking_turn_03 --holdout-load heavy_heldout
python -m myosuite_soma.soma_conditioning evaluate paired_geometry.npz research_corrective.npz
```

`data_kind="pose_only"` is rejected for training. Synthetic data can exercise
the code, while `validation_gate()` always reports that physiological validity
has not been established. Independent matched-pose/different-load observations
and a measurement protocol make paired data eligible for external validation;
they do not automatically validate the model. The checkpoint loader rejects
artifacts asserting `physiologically_validated=true`.

## Optional inference and export

Add an explicit model path to an otherwise normal SOMA JSON configuration:

```json
{
  "conditioning_model": "research_corrective.npz",
  "allow_synthetic_conditioning": false
}
```

The usual fields for dataset, checkpoint and retargeting remain required. The
renderer supplies recorded activation, actuator length, velocity and force.
Models using true fiber states require those fields to be supplied through
the Python API. `generate_animation(..., physiology=recorded_features)` calls
the selected corrective before world placement and geometry/contact adaptation.
All features declared by the artifact must exist and align with named source
columns; missing force or fiber measurements cause a clear error.

Synthetic artifacts are blocked unless `allow_synthetic_conditioning=true` is
set explicitly. The animation records the artifact hash, data kind, feature
units, validation status, native actuator order and a hash of every required
physiology field. Source-checked cache loading verifies those fields even if
no activation colors are mapped. Standalone cache viewing remains numeric and
does not require its training data or Torch inference.

The historical controlled fixture is summarized in [validation/REPORT.md](../validation/REPORT.md). Its real S1 topology and fabricated targets are not redistributed. The small synthetic tests in this repository exercise the same numerical contracts without release assets. The fixture's near-zero recovery error demonstrates software behavior only.

Before using measured results to support physiological claims, assess errors
on independent measured subjects/loads/motions, matched-pose sensitivity,
measurement uncertainty, temporal behavior, attachment drift, tissue volume,
self-intersections and contacts. Compare against the unchanged pose-only
baseline. Publish observation scope and failure cases, including internal
surfaces that were never directly observed.
