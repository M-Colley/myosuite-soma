"""Recorded-pose conventions, constrained IK and replay regressions."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from myosuite_soma.soma import BodyMapping, SomaConfig
from myosuite_soma.soma_geometry import JointTarget
from myosuite_soma.soma_retarget import (
    IKConfig,
    JointCouplings,
    SomaIKRetargeter,
    decode_recording,
    decode_sixd,
    differentiate_trajectory,
    physical_parameter_digest,
    rig_forward_kinematics,
    save_replay,
)


def test_original_and_training_root_conventions_roundtrip():
    native = Rotation.from_euler(
        "xyz", [[0.2, -0.4, 0.7], [-0.1, 0.3, -0.2]]
    ).as_matrix()[None]
    trained = native.copy()
    trained[:, 0] = (
        Rotation.from_euler("y", 90, degrees=True).as_matrix() @ native[:, 0]
    )
    original_values = native[..., :2].reshape(1, 2, 6)
    training_values = trained[..., :2].swapaxes(-1, -2).reshape(1, 2, 6)
    decoded_original = decode_recording(
        original_values, packing="interleaved", root_convention="native"
    )
    decoded_training = decode_recording(
        training_values, packing="concatenated", root_convention="skim-training-ry90"
    )
    np.testing.assert_allclose(decoded_original, native, atol=1e-14)
    np.testing.assert_allclose(decoded_training, native, atol=1e-14)
    assert not np.allclose(
        decode_recording(
            training_values, packing="concatenated", root_convention="native"
        ),
        native,
    )


def test_explicit_sixd_packing_preserves_columns():
    matrix = Rotation.from_euler("xyz", [0.2, -0.4, 0.7]).as_matrix()
    interleaved = matrix[:, :2].reshape(6)
    concatenated = matrix[:, :2].T.reshape(6)
    np.testing.assert_allclose(decode_sixd(interleaved, packing="interleaved"), matrix)
    np.testing.assert_allclose(
        decode_sixd(concatenated, packing="concatenated"), matrix
    )
    assert not np.allclose(decode_sixd(interleaved, packing="concatenated"), matrix)


@pytest.mark.parametrize("value", [np.zeros(6), [1, 1, 0, 0, 0, 0], [np.nan] * 6])
def test_decoder_rejects_degenerate_values(value):
    with pytest.raises(ValueError):
        decode_sixd(np.asarray(value), packing="interleaved")


def test_rig_fk_uses_parent_rotation_and_native_root_once():
    local = np.tile(np.eye(3), (1, 2, 1, 1))
    local[0, 0] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    _, points = rig_forward_kinematics(
        local,
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([-1, 0]),
    )
    np.testing.assert_allclose(points[0, 1], [1, 3, 3], atol=1e-14)
    with pytest.raises(ValueError):
        rig_forward_kinematics(
            local, np.zeros((1, 3)), np.zeros((2, 3)), np.array([-1, 1])
        )


def coupled_model():
    return mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body><joint name="a" range="-1 1"/><geom size=".1"/>
      <body><joint name="b" range="-.25 .25"/><geom size=".1"/>
        <body><joint name="c" range="-.2 .2"/><geom size=".1"/></body>
      </body></body></worldbody><equality>
      <joint joint1="c" joint2="b" polycoef="0 1 0 0 0"/>
      <joint joint1="b" joint2="a" polycoef="0 2 0 0 0"/>
    </equality></mujoco>""")


def test_chained_equalities_restrict_domain_without_model_edits():
    model = coupled_model()
    before = model.jnt_range.copy()
    domain = JointCouplings(model)
    np.testing.assert_allclose(domain.bounds[:, 0], [-0.1, 0.1])
    pose = np.array([0.08, 0.0, 0.0])
    np.testing.assert_allclose(domain.apply(pose), [0.08, 0.16, 0.16])
    np.testing.assert_array_equal(pose, [0.08, 0.0, 0.0])
    np.testing.assert_array_equal(model.jnt_range, before)


def test_coupling_nonzero_references_matches_actual_constraint_residual():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
      <worldbody><body><joint name="a" ref=".5" range="-1 1"/><geom size=".1"/>
      <body><joint name="b" ref=".3" range="-.2 .6"/><geom size=".1"/></body>
      </body></worldbody><equality><joint joint1="b" joint2="a" polycoef="0 2 0 0 0"/></equality>
      </mujoco>""")
    domain = JointCouplings(model)
    np.testing.assert_allclose(domain.bounds[:, 0], [0.25, 0.65])
    pose = domain.apply(np.array([0.6, 0.3]))
    np.testing.assert_allclose(pose, [0.6, 0.5])
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    equalities = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    assert equalities.any()
    np.testing.assert_allclose(data.efc_pos[equalities], 0, atol=1e-14)


def test_constant_coupling_uses_child_reference():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
      <worldbody><body><joint name="a" range="-1 1"/><geom size=".1"/></body>
      <body><joint name="b" ref=".3" range="-.2 .6"/><geom size=".1"/></body>
      </worldbody><equality><joint joint1="b" polycoef=".1 0 0 0 0"/></equality></mujoco>""")
    np.testing.assert_allclose(JointCouplings(model).apply(model.qpos0), [0, 0.4])


def test_disconnected_domains_intersect_all_constraints_before_branch_selection():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
      <worldbody><body><joint name="a" range="-2 2"/><geom size=".1"/></body>
      <body><joint name="b" range="1 4"/><geom size=".1"/></body>
      <body><joint name="c" range="1 2"/><geom size=".1"/></body>
      </worldbody><equality><joint joint1="b" joint2="a" polycoef="0 0 1 0 0"/>
      <joint joint1="c" joint2="a" polycoef="0 1 0 0 0"/></equality></mujoco>""")
    domain = JointCouplings(model)
    np.testing.assert_allclose(domain.bounds[:, 0], [1, 2])
    np.testing.assert_allclose(
        domain.apply(np.array([1.5, 0.0, 0.0])), [1.5, 2.25, 1.5]
    )


def test_cyclic_joint_equalities_fail():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body><joint name="a" range="-10 10"/><geom size=".1"/></body>
    <body><joint name="b" range="-10 10"/><geom size=".1"/></body>
    </worldbody><equality><joint joint1="a" joint2="b"/><joint joint1="b" joint2="a"/></equality></mujoco>""")
    with pytest.raises(ValueError, match="Cyclic"):
        JointCouplings(model)


def test_quaternion_velocity_roundtrip_handles_sign_and_irregular_times():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    poses = np.tile(model.qpos0, (3, 1))
    for i, yaw in enumerate([0.0, 0.3, 0.8]):
        mujoco.mju_mat2Quat(
            poses[i, 3:7], Rotation.from_euler("z", yaw).as_matrix().reshape(-1)
        )
    poses[1, 3:7] *= -1
    poses[:, 0] = [0.0, 0.1, 0.3]
    original = poses.copy()
    times = np.array([0.0, 0.1, 0.35])
    velocity = differentiate_trajectory(model, poses, times)
    for i in range(2):
        rebuilt = poses[i].copy()
        mujoco.mj_integratePos(model, rebuilt, velocity[i], times[i + 1] - times[i])
        error = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, error, 1.0, rebuilt, poses[i + 1])
        np.testing.assert_allclose(error, 0, atol=1e-14)
    np.testing.assert_array_equal(poses, original)
    with pytest.raises(ValueError):
        differentiate_trajectory(model, poses, [0, 0, 1])


@pytest.mark.parametrize("kind", ["complex", "string", "object"])
def test_numeric_helpers_reject_nonreal_raw_dtypes(kind):
    def wrong(value):
        return np.asarray(value).astype(
            {"complex": complex, "string": str, "object": object}[kind]
        )

    with pytest.raises(ValueError, match="real numeric"):
        decode_sixd(wrong([1, 0, 0, 1, 0, 0]), packing="interleaved")
    local = np.eye(3)[None, None]
    with pytest.raises(ValueError, match="real numeric"):
        rig_forward_kinematics(
            wrong(local), np.zeros((1, 3)), np.zeros((1, 3)), np.array([-1])
        )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="real numeric"):
        differentiate_trajectory(model, np.tile(model.qpos0, (2, 1)), wrong([0, 0.04]))


def test_unnamed_free_joint_replay_loads_without_pickle(tmp_path):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body><freejoint/><geom size=".1"/>
      <body><joint name="bend" range="-10 10"/><geom size=".1"/></body></body></worldbody></mujoco>""")
    poses = np.tile(model.qpos0, (2, 1))
    poses[1, 0] = 0.1
    path = tmp_path / "unnamed.npz"
    save_replay(path, model, poses, np.array([0.0, 0.04]))
    with np.load(path, allow_pickle=False) as cache:
        assert "joint_names" not in cache
        assert all(cache[key].dtype.kind != "O" for key in cache.files)
        np.testing.assert_array_equal(cache["qpos"], poses)
        assert cache["qvel"].shape == (2, model.nv)
        np.testing.assert_allclose(cache["qvel"][:, 0], 2.5)
        np.testing.assert_allclose(cache["qvel"][:, 1:], 0)
    with pytest.raises(FileExistsError):
        save_replay(path, model, poses, np.array([0.0, 0.04]))


@pytest.mark.parametrize("global_basis", [False, True])
def test_bounded_ik_solves_known_rotation_without_mutating_source(
    tmp_path, global_basis
):
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
      <worldbody><body name="root"><freejoint/><geom size=".1"/>
      <body name="arm" pos="0 0 1"><joint name="bend" range="-1 1" axis="0 0 1"/><geom size=".1"/></body>
      </body></worldbody></mujoco>""")
    source = mujoco.MjData(model)
    source.qvel[:] = 0.1
    before = physical_parameter_digest(model)
    basis = Rotation.from_euler("z", 0.3).as_matrix() if global_basis else np.eye(3)
    profile = SomaConfig(
        data_root=tmp_path,
        subject="S1",
        checkpoint=tmp_path / "x.pth",
        body_map={
            "Hips": BodyMapping("root", basis=basis.T),
            "Arm": BodyMapping("arm", "root", basis=basis.T),
        },
        rest_qpos=model.qpos0.copy(),
        anchor_body="root",
        root_body="root",
        anchor_joint="Hips",
        basis=basis,
    )
    predictor = SimpleNamespace(
        joint_names=("Hips", "Arm"),
        parents=np.array([-1, 0]),
        rest_joints=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    config = IKConfig(
        landmarks={"Arm": JointTarget(body="arm")},
        orientation_weights={"Arm": 1.0},
        temporal_weight=0.0,
        neutral_weight=0.0,
    )
    solver = SomaIKRetargeter(model, profile, predictor, config)
    rotations = np.array([np.eye(3), Rotation.from_euler("z", 0.4).as_matrix()])
    pose, metrics = solver.solve(rotations, predictor.rest_joints)
    assert pose[-1] == pytest.approx(0.4, abs=1e-6)
    assert max(metrics["source_anchor_errors_mm"].values()) < 1e-8
    assert physical_parameter_digest(model) == before
    np.testing.assert_array_equal(source.qpos, model.qpos0)
    np.testing.assert_array_equal(source.qvel, np.full(model.nv, 0.1))
