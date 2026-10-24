# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under Apache-2.0.
"""Synthetic real-format assets; no gated SKIM data or pretrained weights."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.spatial.transform import Rotation

from myosuite_soma.soma_model import _bind, load_soma_model
from ._fixtures import create_synthetic_release

torch = pytest.importorskip("torch")


def _numpy_prediction(
    model: object, rotations: np.ndarray, roots: np.ndarray
) -> tuple[dict, np.ndarray]:
    """Independent functional network + NumPy homogeneous-transform reference."""
    state = {k: v.numpy() for k, v in model._network.state_dict().items()}

    def block(x: np.ndarray, name: str) -> np.ndarray:
        x = x @ state[name + ".0.weight"].T + state[name + ".0.bias"]
        x = (x - state[name + ".1.running_mean"]) / np.sqrt(
            state[name + ".1.running_var"] + 1e-5
        )
        x = x * state[name + ".1.weight"] + state[name + ".1.bias"]
        return x * ndtr(x)

    raw = []
    for prefix in ("m", "s"):
        x1 = block(
            (rotations[:, 1:] - np.eye(3)).reshape(len(rotations), -1), prefix + "_enc1"
        )
        x2 = block(x1, prefix + "_enc2")
        mid = block(x2, prefix + "_bottleneck")
        x = block(np.concatenate((mid, x2), 1), prefix + "_dec1")
        x = block(np.concatenate((x, x1), 1), prefix + "_dec2")
        x = x @ state[prefix + "_out.weight"].T + state[prefix + "_out.bias"]
        raw.append(x.reshape(len(rotations), -1, 3))
    # Fixture has exactly one thick and one flat prism; flat vertices stay pinned.
    mask = np.array([1, 1, 1, 0, 0, 0])[None, :, None]
    dm, ds = raw[0] * mask, (raw[0] + raw[1]) * mask
    state = {k: v.numpy() for k, v in model._rigid.state_dict().items()}
    x = (rotations - np.eye(3)).reshape(len(rotations), -1)
    for index in (0, 2, 4):
        x = x @ state[f"net.{index}.weight"].T + state[f"net.{index}.bias"]
        if index < 4:
            x = np.maximum(x, 0) + 0.2 * np.minimum(x, 0)
    raw_rigid = x.reshape(len(rotations), -1, 6) * state["joint_mask"]
    transforms = np.empty((len(rotations), len(model.parents), 4, 4))
    for frame in range(len(rotations)):
        global_poses = []
        for joint, parent in enumerate(model.parents):
            local = np.eye(4)
            local[:3, :3] = rotations[frame, joint]
            local[:3, 3] = (
                roots[frame]
                if parent < 0
                else model.rest_joints[joint] - model.rest_joints[parent]
            )
            correction = np.eye(4)
            correction[:3, :3] = Rotation.from_rotvec(
                0.1 * raw_rigid[frame, joint, :3]
            ).as_matrix()
            correction[:3, 3] = 0.02 * raw_rigid[frame, joint, 3:]
            local = local @ correction
            global_poses.append(local if parent < 0 else global_poses[parent] @ local)
            inverse_bind = np.eye(4)
            inverse_bind[:3, 3] = -model.rest_joints[joint]
            transforms[frame, joint] = global_poses[-1] @ inverse_bind

    def pose(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        homogeneous = np.concatenate((points, np.ones(points.shape[:-1] + (1,))), -1)
        return np.einsum("vj,bjxy,bvy->bvx", weights, transforms, homogeneous)[..., :3]

    with np.load(Path(model.metadata["rig"]), allow_pickle=False) as rig:
        weights = rig["weights"]
    # The layer is known fixture geometry, deliberately independent of the loader.
    layer = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]])
    muscles = {}
    for mesh in model.meshes:
        distance = np.linalg.norm(mesh.rest_vertices[:, None] - layer, axis=-1)
        blend = np.exp(-((distance / (distance.max(1, keepdims=True) / 2 + 1e-9)) ** 2))
        blend /= blend.sum(1, keepdims=True)
        displacement = np.einsum("nv,bvc->bnc", blend, dm)
        muscles[mesh.name] = pose(mesh.rest_vertices + displacement, blend @ weights)
    return muscles, pose(model.skin_mesh.rest_vertices + ds, weights)


def test_real_format_checkpoint_matches_reference_and_batching(tmp_path: Path) -> None:
    root, checkpoint = create_synthetic_release(tmp_path)
    model = load_soma_model(root, "S1", checkpoint, device="cpu", batch_size=1)
    rotations = np.broadcast_to(np.eye(3), (3, 3, 3, 3)).copy()
    rotations[1, 2] = Rotation.from_euler("z", 0.7).as_matrix()
    rotations[2, 0] = Rotation.from_euler("x", -0.4).as_matrix()
    roots = np.array([[0, 0, 0], [0.2, -0.4, 1], [0, 0, 0]], np.float32)
    original = rotations.copy()
    result = model.predict(rotations, roots)
    reference, skin = _numpy_prediction(model, rotations, roots)
    np.testing.assert_allclose(result.skin, skin, atol=5e-7)
    for name in result.muscles:
        np.testing.assert_allclose(result.muscles[name], reference[name], atol=5e-7)
    np.testing.assert_array_equal(rotations, original)
    batched = load_soma_model(
        root, "S1", checkpoint, device="cpu", batch_size=3
    ).predict(rotations, roots)
    np.testing.assert_allclose(batched.skin, result.skin, atol=5e-7)
    assert not model._network.training and not model._rigid.training
    assert all(
        not p.requires_grad and p.grad is None for p in model._network.parameters()
    )
    assert model.joint_names == ("Hips", "RightArm", "RightForeArm")
    np.testing.assert_array_equal(model._mask.numpy().ravel(), [1, 1, 1, 0, 0, 0])


def test_selection_root_motion_and_input_validation(tmp_path: Path) -> None:
    root, checkpoint = create_synthetic_release(tmp_path)
    model = load_soma_model(
        root, "S1", checkpoint, individual_muscles=["biceps"], device="cpu"
    )
    rotations = np.broadcast_to(np.eye(3), (1, 3, 3, 3)).copy()
    zero = model.predict(rotations)
    shifted = model.predict(rotations, np.array([[1, 2, 3]]))
    np.testing.assert_allclose(
        shifted.muscles["biceps"] - zero.muscles["biceps"], [[[1, 2, 3]] * 3], atol=5e-7
    )
    assert list(zero.muscles) == ["biceps"]
    with pytest.raises(ValueError, match="selection"):
        load_soma_model(root, "S1", checkpoint, individual_muscles=["missing"])
    rotations[0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="proper"):
        model.predict(rotations)
    with pytest.raises(ValueError, match="root_positions"):
        model.predict(np.broadcast_to(np.eye(3), (1, 3, 3, 3)), np.zeros((3,)))


def test_missing_or_incompatible_release_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="SKIM metadata missing"):
        load_soma_model(tmp_path, "S1", tmp_path / "absent.pth")
    root, checkpoint = create_synthetic_release(tmp_path)
    rigid = checkpoint.with_name(checkpoint.stem + "_rigid.pth")
    rigid.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        load_soma_model(root, "S1", checkpoint)
    (root / "metadata.json").write_text('{"dataset_version": 1}')
    with pytest.raises(ValueError, match="dataset_version 2"):
        load_soma_model(root, "S1", checkpoint)


def test_mismatched_joint_order_and_checkpoint_are_rejected(tmp_path: Path) -> None:
    root, checkpoint = create_synthetic_release(tmp_path)
    rig_path = root / "S1" / "canonical_model" / "S1_rig.npz"
    with np.load(rig_path) as original:
        rig = dict(original)
    rig["joint_names"] = rig["joint_names"][::-1]
    np.savez(rig_path, **rig)
    with pytest.raises(ValueError, match="ordering disagree"):
        load_soma_model(root, "S1", checkpoint)
    rig["joint_names"] = rig["joint_names"][::-1]
    np.savez(rig_path, **rig)
    torch.save({"unexpected": torch.ones(1)}, checkpoint)
    with pytest.raises(ValueError, match="architecture"):
        load_soma_model(root, "S1", checkpoint)


def test_binding_uses_sixteen_nearest_vertices_and_normalized_gaussian() -> None:
    layer = np.column_stack((np.arange(20), np.zeros((20, 2)))).astype(np.float32)
    points = np.array([[0.1, 0.01, 0], [19.3, -0.1, 0]], np.float32)
    indices, weights = _bind(points, layer)
    distances = np.linalg.norm(points[:, None] - layer, axis=-1)
    expected_indices = np.argsort(distances, axis=1)[:, :16]
    selected = np.take_along_axis(distances, expected_indices, axis=1)
    expected = np.exp(-((selected / (selected[:, -1:] / 2 + 1e-9)) ** 2))
    expected /= expected.sum(1, keepdims=True)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(weights, expected, atol=1e-7)


def test_import_is_torch_lazy_and_checkpoints_use_restricted_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    import myosuite_soma.soma_model as module

    original_import = builtins.__import__

    def without_torch(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise ImportError("Torch deliberately unavailable")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", without_torch)
        runpy.run_path(module.__file__)
    root, checkpoint = create_synthetic_release(tmp_path)
    original_load = torch.load
    loaded = []

    def checked_load(path: Path, **kwargs: object) -> object:
        assert kwargs.get("weights_only") is True
        loaded.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    load_soma_model(root, "S1", checkpoint, device="cpu")
    assert len(loaded) == 2


def test_reversed_readonly_pose_and_root_sequences_work(tmp_path: Path) -> None:
    root, checkpoint = create_synthetic_release(tmp_path)
    model = load_soma_model(root, "S1", checkpoint, device="cpu")
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3, 3)).copy()
    rotations[1, 2] = Rotation.from_euler("z", 0.7).as_matrix()
    roots = np.array([[0, 0, 0], [1, 2, 3], [-1, 0, 0]], np.float32)
    expected = model.predict(rotations, roots)
    rotations.setflags(write=False)
    roots.setflags(write=False)
    actual = model.predict(rotations[::-1], roots[::-1])
    np.testing.assert_allclose(actual.skin, expected.skin[::-1], atol=5e-7)
    for name in actual.muscles:
        np.testing.assert_allclose(
            actual.muscles[name], expected.muscles[name][::-1], atol=5e-7
        )


def test_geometry_stays_float32_under_global_double_precision(tmp_path: Path) -> None:
    root, checkpoint = create_synthetic_release(tmp_path)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3, 3)).copy()
    expected = load_soma_model(root, "S1", checkpoint, device="cpu").predict(rotations)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = load_soma_model(root, "S1", checkpoint, device="cpu")
        actual = model.predict(rotations)
        assert all(p.dtype == torch.float32 for p in model._network.parameters())
        assert all(p.dtype == torch.float32 for p in model._rigid.parameters())
        assert actual.skin.dtype == np.float32
        np.testing.assert_allclose(actual.skin, expected.skin, atol=5e-7)
    finally:
        torch.set_default_dtype(previous)


def test_geometry_disables_external_autocast_without_changing_caller(
    tmp_path: Path,
) -> None:
    root, checkpoint = create_synthetic_release(tmp_path)
    model = load_soma_model(root, "S1", checkpoint, device="cpu")
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3, 3)).copy()
    expected = model.predict(rotations)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loaded = load_soma_model(root, "S1", checkpoint, device="cpu")
        actual = loaded.predict(rotations)
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.bfloat16
    np.testing.assert_array_equal(actual.skin, expected.skin)
    for name in actual.muscles:
        np.testing.assert_array_equal(actual.muscles[name], expected.muscles[name])


def test_explicit_cpu_ignores_callers_default_device(tmp_path: Path) -> None:
    root, checkpoint = create_synthetic_release(tmp_path)
    rotations = np.broadcast_to(np.eye(3), (1, 3, 3, 3))
    with torch.device("meta"):
        model = load_soma_model(root, "S1", checkpoint, device="cpu")
        actual = model.predict(rotations)
    assert all(p.device.type == "cpu" for p in model._network.parameters())
    assert all(p.device.type == "cpu" for p in model._rigid.parameters())
    assert np.isfinite(actual.skin).all()


def test_fractional_parent_indices_are_not_silently_truncated(tmp_path: Path) -> None:
    root, checkpoint = create_synthetic_release(tmp_path)
    path = root / "S1" / "canonical.npz"
    with np.load(path, allow_pickle=False) as original:
        canonical = dict(original)
    canonical["parents"] = np.array([-1, 0, 0.8])
    np.savez(path, **canonical)
    with pytest.raises(ValueError, match="signed integer indices"):
        load_soma_model(root, "S1", checkpoint, device="cpu")


def test_complex_rotations_are_not_silently_projected_to_real(tmp_path: Path) -> None:
    root, checkpoint = create_synthetic_release(tmp_path)
    model = load_soma_model(root, "S1", checkpoint, device="cpu")
    rotations = np.broadcast_to(np.eye(3), (1, 3, 3, 3)).astype(np.complex64)
    rotations[0, 2, 0, 0] += 0.5j
    with pytest.raises(ValueError, match="real numeric"):
        model.predict(rotations)
