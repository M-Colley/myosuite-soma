# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under Apache-2.0.
"""Conditional geometry contracts using explicitly synthetic supervision."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from myosuite_soma.soma_conditioning import (
    ConditioningDataset,
    ConditioningFitConfig,
    _basis,
    _centers,
    anatomy_arrays,
    apply_conditioning,
    evaluate_conditioning,
    grouped_split,
    load_conditioning_model,
    load_dataset,
    physiology_digest,
    save_conditioning_model,
    save_dataset,
    train_conditioning,
    validation_gate,
)


def synthetic_dataset(
    points: np.ndarray | None = None,
    mesh_names: tuple[str, ...] = ("muscle", "skin"),
    mesh_counts: np.ndarray | None = None,
    joints: int = 3,
) -> ConditioningDataset:
    """Generate a known shared field, without anatomical or physiological claims."""
    rng = np.random.default_rng(14)
    points = rng.uniform(-0.3, 0.3, (30, 3)) if points is None else points
    counts = np.array([15, 15]) if mesh_counts is None else mesh_counts
    centers = _centers(points, 6)
    field = _basis(points, centers, 0.35)[:, 1, None] * np.array(
        [[0.003, -0.002, 0.001]]
    )
    activation = np.tile(np.linspace(0, 1, 7), 9)[:, None]
    frames = len(activation)
    baseline = np.broadcast_to(points, (frames, *points.shape)).copy()
    return ConditioningDataset(
        points,
        mesh_names,
        counts,
        np.broadcast_to(np.eye(3), (frames, joints, 3, 3)).copy(),
        ("biceps",),
        {
            "activation": activation,
            "actuator_force": -100 * activation,
            "actuator_length": 0.2 + 0.01 * activation,
            "actuator_velocity": 0.01 * activation,
        },
        {
            "activation": np.zeros(1),
            "actuator_force": np.zeros(1),
            "actuator_length": np.full(1, 0.2),
            "actuator_velocity": np.zeros(1),
        },
        baseline,
        baseline + activation[:, :, None] * field,
        ("synthetic_subject",) * frames,
        tuple(np.repeat(["mobility", "reach", "heldout_walk"], 21)),
        tuple(np.tile(np.repeat(["low", "medium", "high"], 7), 3)),
        {
            "schema_version": 1,
            "data_kind": "synthetic",
            "coordinate_frame": "SOMA_canonical_corrective_m",
            "provenance": "Analytic smooth field, synthetic physiology; test fixture only",
            "feature_units": {
                "activation": "dimensionless",
                "actuator_force": "N_signed_transmission",
                "actuator_length": "m_musculotendon",
                "actuator_velocity": "m_s_musculotendon",
            },
        },
    )


def fitted(data: ConditioningDataset):
    train, holdout = grouped_split(data, holdout_motions=["heldout_walk"])
    model = train_conditioning(
        data,
        train,
        holdout,
        ConditioningFitConfig(centers=6, radius_m=0.35, rank=2, pose_rank=0),
    )
    return model, train, holdout


def test_fixed_pose_changed_activation_and_zero_reference() -> None:
    data = synthetic_dataset()
    model, _, holdout = fitted(data)
    report = evaluate_conditioning(model, data, holdout)
    assert report["corrected_rmse_m"] < 1e-7
    assert report["baseline_rmse_m"] > 1e-4
    offsets = model.offsets(
        data.rest_points, data.rotations, data.states, data.actuator_names
    )
    assert np.count_nonzero(offsets[0]) == 0
    assert np.max(np.abs(offsets[6])) > 0.001
    assert not model.metadata["physiologically_validated"]
    assert not validation_gate(data)["can_claim_physiological_validity"]


def test_split_holds_complete_motion_and_load_groups() -> None:
    data = synthetic_dataset()
    train, holdout = grouped_split(
        data, holdout_motions=["heldout_walk"], holdout_loads=["high"]
    )
    assert "heldout_walk" not in {data.motion[i] for i in train}
    assert "high" not in {data.load[i] for i in train}
    assert all(
        data.motion[i] == "heldout_walk" or data.load[i] == "high" for i in holdout
    )
    with pytest.raises(ValueError, match="groups must not cross"):
        train_conditioning(
            data,
            np.arange(0, len(data.rotations), 2),
            np.arange(1, len(data.rotations), 2),
        )
    with pytest.raises(ValueError, match="Unknown holdout"):
        grouped_split(data, holdout_subjects=["unavailable"])


def test_holdout_target_does_not_change_fit() -> None:
    data = synthetic_dataset()
    first, train, holdout = fitted(data)
    target = data.target.copy()
    target[holdout] += 0.05
    second = train_conditioning(
        replace(data, target=target),
        train,
        holdout,
        ConditioningFitConfig(centers=6, radius_m=0.35, rank=2, pose_rank=0),
    )
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.modes, second.modes)


def test_unobserved_internal_targets_do_not_supervise_skin_fit() -> None:
    data = synthetic_dataset()
    confidence = np.concatenate((np.zeros(15), np.ones(15)))
    skin_only = replace(data, observation_weights=confidence)
    first, train, holdout = fitted(skin_only)
    target = skin_only.target.copy()
    target[:, :15] += 10
    second = train_conditioning(
        replace(skin_only, target=target),
        train,
        holdout,
        ConditioningFitConfig(centers=6, radius_m=0.35, rank=2, pose_rank=0),
    )
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.metadata["unobserved_vertices"] == 15
    report = evaluate_conditioning(second, skin_only, holdout)
    assert report["observed_vertices"] == 15
    assert report["corrected_rmse_m"] < 1e-7


def test_numeric_roundtrip_and_required_physiology(tmp_path: Path) -> None:
    data = synthetic_dataset()
    save_dataset(tmp_path / "data.npz", data)
    loaded = load_dataset(tmp_path / "data.npz")
    model, _, holdout = fitted(loaded)
    save_conditioning_model(tmp_path / "model.npz", model)
    restored = load_conditioning_model(tmp_path / "model.npz")
    assert evaluate_conditioning(restored, loaded, holdout)["corrected_rmse_m"] < 1e-7
    physiology = {
        key: value for key, value in loaded.states.items() if key != "actuator_force"
    }
    with pytest.raises(
        ValueError, match="requires recorded physiology.*actuator_force"
    ):
        restored.offsets(
            data.rest_points, data.rotations, physiology, data.actuator_names
        )
    with pytest.raises(ValueError, match="training feature envelope"):
        restored.offsets(
            data.rest_points,
            data.rotations,
            dict(data.states, actuator_force=data.states["actuator_force"] - 200),
            data.actuator_names,
        )


def test_units_pose_only_and_forged_validation_are_rejected(tmp_path: Path) -> None:
    data = synthetic_dataset()
    wrong_units = dict(
        data.metadata,
        feature_units=dict(data.metadata["feature_units"], actuator_length="m_fiber"),
    )
    with pytest.raises(ValueError, match="Incorrect units"):
        replace(data, metadata=wrong_units).validate()
    pose_only = replace(data, metadata=dict(data.metadata, data_kind="pose_only"))
    train, holdout = grouped_split(pose_only, holdout_motions=["heldout_walk"])
    with pytest.raises(ValueError, match="Pose-only SOMA"):
        train_conditioning(pose_only, train, holdout)
    assert not validation_gate(pose_only)["eligible_for_external_validation"]
    model, _, _ = fitted(data)
    forged = replace(
        model, metadata=dict(model.metadata, physiologically_validated=True)
    )
    save_conditioning_model(tmp_path / "forged.npz", forged)
    with pytest.raises(ValueError, match="unsupported validation claim"):
        load_conditioning_model(tmp_path / "forged.npz")


def test_pickle_and_truncated_archives_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npz"
    np.savez(path, metadata=np.array({"schema_version": 1}, dtype=object))
    with pytest.raises(ValueError, match="shape or dtype"):
        load_dataset(path)
    path.write_bytes(path.read_bytes()[:45])
    with pytest.raises(ValueError, match="Invalid numeric"):
        load_dataset(path)


def test_digest_covers_force_length_velocity_without_colors() -> None:
    data = synthetic_dataset()
    before = physiology_digest(data.actuator_names, data.states, sorted(data.states))
    for key in data.states:
        modified = dict(data.states, **{key: data.states[key] + 0.001})
        assert (
            physiology_digest(data.actuator_names, modified, sorted(data.states))
            != before
        )
    with pytest.raises(ValueError, match="Missing conditioning physiology"):
        physiology_digest(data.actuator_names, {}, sorted(data.states))


def test_pose_offsets_matches_independent_blended_rotations(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from ._fixtures import create_synthetic_release
    from myosuite_soma.soma_model import _skinning_transforms, load_soma_model

    root, checkpoint = create_synthetic_release(tmp_path)
    model = load_soma_model(root, "S1", checkpoint, device="cpu")
    rotations = np.broadcast_to(np.eye(3), (2, 3, 3, 3)).copy()
    rotations[1, 1] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    offsets = {
        spec.name: np.full((2, len(spec.rest_vertices), 3), [0.001, 0.002, 0.003])
        for spec in (*model.meshes, model.skin_mesh)
    }
    posed = model.pose_offsets(rotations, offsets)
    with torch.inference_mode():
        pose = torch.tensor(rotations, dtype=torch.float32)
        transforms = _skinning_transforms(
            torch,
            pose,
            torch.zeros((2, 3)),
            model._rest,
            model.parents,
            model._rigid(pose),
        ).numpy()
    weights = [entry[3].numpy() for entry in model._bindings] + [model._weights.numpy()]
    for spec, weight in zip((*model.meshes, model.skin_mesh), weights):
        expected = np.empty_like(posed[spec.name])
        for frame in range(2):
            for vertex in range(len(weight)):
                rotation = sum(
                    weight[vertex, joint] * transforms[frame, joint, :3, :3]
                    for joint in range(3)
                )
                expected[frame, vertex] = rotation @ offsets[spec.name][frame, vertex]
        np.testing.assert_allclose(posed[spec.name], expected, atol=1e-8)


def test_optional_application_keeps_baseline_and_requires_optin(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from ._fixtures import create_synthetic_release
    from myosuite_soma.soma_model import load_soma_model

    root, checkpoint = create_synthetic_release(tmp_path)
    predictor = load_soma_model(root, "S1", checkpoint, device="cpu")
    points, names, counts = anatomy_arrays(predictor)
    data = synthetic_dataset(points, names, counts)
    model, _, _ = fitted(data)
    path = tmp_path / "synthetic.npz"
    save_conditioning_model(path, model)
    before = predictor.predict(data.rotations)
    original = before.skin.copy()
    with pytest.raises(ValueError, match="explicit allow_synthetic"):
        apply_conditioning(
            predictor, before, data.rotations, data.states, data.actuator_names, path
        )
    corrected, metadata = apply_conditioning(
        predictor,
        before,
        data.rotations,
        data.states,
        data.actuator_names,
        path,
        allow_synthetic=True,
    )
    np.testing.assert_array_equal(before.skin, original)
    np.testing.assert_array_equal(corrected.skin[0], original[0])
    assert np.max(np.abs(corrected.skin[6] - original[6])) > 1e-5
    assert metadata["source_physiology_sha256"] == physiology_digest(
        data.actuator_names, data.states, metadata["required_features"]
    )
    assert metadata["data_kind"] == "synthetic"
    json.dumps(metadata, allow_nan=False)
