"""Coherent optional visual registration and native-clearance morphology."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite_soma.soma import (
    AnimatedMesh,
    SomaAnimation,
    model_digest,
    validate_animation,
)
from myosuite_soma.soma_geometry import (
    ContactMembership,
    FootContactConfig,
    GeometryConfig,
    JointTarget,
    deform_animation,
    geometry_model_digest,
    target_joint_positions,
    _temporal_scales,
    _contact_region,
    contact_membership_from_predictor,
)


def _case() -> tuple:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>'
        '<body name="ankle" pos="0 0 .1"><joint name="lift" type="slide" axis="0 0 1"/>'
        '<geom name="sole" type="sphere" pos="0 .08 -.082" size=".02"/>'
        '<body name="toe" pos="0 .2 -.05"><geom size=".01" contype="0" conaffinity="0"/>'
        "</body></body></worldbody></mujoco>"
    )
    poses = np.array([[0.0], [0.5]])
    joints = np.array([[[0, 0, 0.1], [0, 0.2, 0.05]], [[0, 0, 0.6], [0, 0.2, 0.55]]])
    points = np.array(
        [[-0.04, 0.05, -0.04], [0.04, 0.05, -0.04], [0, 0.15, -0.04], [0, 0, 0.5]],
        np.float32,
    )
    samples = np.stack([points, points + [0, 0, 0.5]]).astype(np.float32)
    mesh = AnimatedMesh("skin", np.array([[0, 1, 2], [1, 2, 3]]), samples, kind="skin")
    animation = SomaAnimation(
        np.array([0.0, 0.1]), (mesh, replace(mesh, name="muscle", kind="muscle"))
    )
    config = GeometryConfig(
        landmarks={"Foot": JointTarget(body="ankle"), "Toe": JointTarget(body="toe")}
    )
    return model, poses, joints, animation, config


def _apply(model, poses, joints, animation, config, membership=None):
    if membership is None and config.contacts:
        membership = ContactMembership(
            {
                contact.ankle: {
                    mesh.name: np.ones(mesh.points.shape[1])
                    for mesh in animation.meshes
                }
                for contact in config.contacts
            },
            {"kind": "explicit_fixture"},
        )
    return deform_animation(
        animation,
        source_joints=joints,
        joint_names=("Foot", "Toe"),
        parents=np.array([-1, 0]),
        model=model,
        qpos=poses,
        config=config,
        contact_membership=membership,
    )


def test_identity_registration_is_optional_and_does_not_mutate_mechanics() -> None:
    model, poses, joints, animation, config = _case()
    original = [mesh.points.copy() for mesh in animation.meshes]
    before = model_digest(model), geometry_model_digest(model)
    result = _apply(model, poses, joints, animation, config)
    for mesh, initial, out in zip(animation.meshes, original, result.animation.meshes):
        np.testing.assert_array_equal(mesh.points, initial)
        np.testing.assert_array_equal(out.points, initial)
    assert before == (model_digest(model), geometry_model_digest(model))
    assert result.diagnostics["landmark_rmse_after_m"] < 1e-10


def test_shared_field_aligns_landmarks_and_identical_skin_muscle_points() -> None:
    model, poses, joints, animation, config = _case()
    native = joints + [0.02, 0, 0.01]
    result = _apply(model, poses, native, animation, config)
    np.testing.assert_allclose(result.joints, joints, atol=5e-5)
    np.testing.assert_array_equal(
        result.animation.meshes[0].points, result.animation.meshes[1].points
    )
    assert result.diagnostics["landmark_rmse_before_m"] > 0.02
    assert result.diagnostics["minimum_sampled_registration_jacobian"] > 0.5


def test_local_contact_preserves_native_penetration_joint_centers_and_swing() -> None:
    model, poses, joints, animation, config = _case()
    config = replace(
        config,
        contacts=(FootContactConfig("Foot", "Toe", ("sole",), temporal_weight=0),),
    )
    result = _apply(model, poses, joints, animation, config)
    contact = result.diagnostics["contacts"][0]
    assert contact["source_clearance_m"][0] == pytest.approx(-0.002)
    assert contact["visual_after_m"][0] >= -0.00101
    assert (
        contact["maximum_added_penetration_after_m"]
        < contact["maximum_added_penetration_before_m"]
    )
    assert contact["compression_scale"][0] < 1
    assert contact["compression_scale"][1] == 1
    np.testing.assert_array_equal(
        result.animation.meshes[0].points[1], animation.meshes[0].points[1]
    )
    np.testing.assert_array_equal(
        result.animation.meshes[0].points[:, 3], animation.meshes[0].points[:, 3]
    )
    np.testing.assert_allclose(result.joints, joints, atol=1e-9)
    np.testing.assert_array_equal(
        result.animation.meshes[0].points, result.animation.meshes[1].points
    )


def test_collision_provenance_detects_changed_source_sole() -> None:
    model, poses, joints, animation, config = _case()
    result = _apply(model, poses, joints, animation, config)
    result.animation.metadata["source_model_sha256"] = model_digest(model)
    validate_animation(result.animation, model=model)
    model.geom_size[model.geom("sole").id, 0] += 0.01
    with pytest.raises(ValueError, match="collision model"):
        validate_animation(result.animation, model=model)


def test_visual_contact_rejects_noninteracting_source_collision_masks() -> None:
    model, poses, joints, animation, config = _case()
    floor = model.geom("floor").id
    model.geom_contype[floor] = 2
    model.geom_conaffinity[floor] = 0
    config = replace(config, contacts=(FootContactConfig("Foot", "Toe", ("sole",)),))
    with pytest.raises(ValueError, match="collision masks"):
        _apply(model, poses, joints, animation, config)


def test_morphology_bounds_reject_incompatible_segment_proportions() -> None:
    model, poses, joints, animation, config = _case()
    native = joints.copy()
    native[:, 1] = native[:, 0] + (native[:, 1] - native[:, 0]) * 0.1
    with pytest.raises(ValueError, match="bounded segment"):
        _apply(model, poses, native, animation, config)


@pytest.mark.parametrize(
    "change",
    [
        {"radius_m": 0},
        {"max_displacement_m": float("nan")},
        {"min_segment_scale": 1.1},
        {"min_jacobian": 1.2},
    ],
)
def test_config_rejects_invalid_bounds(change: dict) -> None:
    *_, config = _case()
    with pytest.raises(ValueError):
        replace(config, **change).validate()


def test_joint_anchor_targets_use_actual_source_anchor() -> None:
    model, poses, joints, _, config = _case()
    config = replace(
        config,
        landmarks={"Foot": JointTarget(joint="lift"), "Toe": JointTarget(body="toe")},
    )
    targets = target_joint_positions(config, model, poses, ("Foot", "Toe"), joints)
    np.testing.assert_allclose(targets[:, 1], joints[:, 1])
    np.testing.assert_allclose(targets[:, 0], [[0, 0, 0.1], [0, 0, 0.1]])


def test_temporal_scales_use_actual_time_and_respect_clearance_bounds() -> None:
    times = np.linspace(0, 2, 101)
    upper = 1 - 0.4 * np.exp(-(((times - 1) / 0.15) ** 2))
    scale = _temporal_scales(upper, 2, times, 0.06)
    assert np.all(scale <= upper + 1e-9)
    assert np.min(scale) > 0.5
    assert np.max(np.abs(np.diff(scale, n=2))) < 0.03
    coarse = _temporal_scales(upper[::2], 2, times[::2], 0.06)
    np.testing.assert_allclose(scale[::2], coarse, atol=0.015)


def test_temporal_scales_never_relax_zero_or_tiny_actual_bounds() -> None:
    upper = np.array([0.0, 1e-10, 3e-7, 0.2, 1.0, 1e-5, 0.0])
    result = _temporal_scales(upper, 2, np.arange(len(upper)) * 0.03, 0.06)
    assert np.all(result >= 0)
    assert np.all(result <= upper)
    assert result[0] == result[-1] == 0


@pytest.mark.parametrize("count,method", [(41, "dense_bvls"), (1101, "sparse_osqp")])
def test_temporal_solver_fallback_preserves_objective_and_actual_bounds(
    monkeypatch: pytest.MonkeyPatch, count: int, method: str
) -> None:
    import scipy.optimize

    solve = scipy.optimize.lsq_linear

    def sparse_failure(design, values, **kwargs):
        if kwargs.get("method") == "bvls":
            return solve(design, values, **kwargs)
        return SimpleNamespace(success=False, status=0, x=np.ones(count))

    monkeypatch.setattr(scipy.optimize, "lsq_linear", sparse_failure)
    times = np.arange(count) * 0.02
    upper = 1 - 0.4 * np.exp(-(((times - times[count // 2]) / 0.25) ** 2))
    upper[0] = 0
    upper[1] = 1e-10
    diagnostics = {}
    result = _temporal_scales(upper, 2, times, 0.06, diagnostics=diagnostics)
    assert diagnostics["method"] == method
    assert diagnostics["converged"]
    assert np.all(result >= 0)
    assert np.all(result <= upper)
    assert result[0] == 0
    second_before = np.diff(upper, n=2) / (times[1] - times[0]) ** 2
    before = 0.5 * (np.sum((upper - 1) ** 2) + 2 * 0.06**4 * np.sum(second_before**2))
    assert diagnostics["objective"] < before


def test_temporal_solver_does_not_claim_convergence_after_iteration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.optimize
    import osqp

    def failure(*args, **kwargs):
        return SimpleNamespace(success=False, status=0, x=np.ones(1001), nit=2)

    monkeypatch.setattr(scipy.optimize, "lsq_linear", failure)

    class LimitedSolver:
        def setup(self, **kwargs):
            pass

        def warm_start(self, **kwargs):
            pass

        def solve(self, **kwargs):
            return SimpleNamespace(
                x=np.ones(1001), info=SimpleNamespace(status_val=7, iter=2)
            )

    monkeypatch.setattr(osqp, "OSQP", LimitedSolver)
    upper = np.linspace(0, 1, 1001)
    diagnostics = {}
    result = _temporal_scales(
        upper, 2, np.linspace(0, 4, 1001), 0.06, diagnostics=diagnostics
    )
    assert not diagnostics["converged"]
    np.testing.assert_array_equal(result, upper)


def test_contact_requires_explicit_anatomical_membership() -> None:
    model, poses, joints, animation, config = _case()
    config = replace(config, contacts=(FootContactConfig("Foot", "Toe", ("sole",)),))
    with pytest.raises(ValueError, match="explicit static anatomical"):
        deform_animation(
            animation,
            source_joints=joints,
            joint_names=("Foot", "Toe"),
            parents=np.array([-1, 0]),
            model=model,
            qpos=poses,
            config=config,
        )


def test_blended_anatomical_sole_still_drives_contact_constraints() -> None:
    model, poses, joints, animation, config = _case()
    config = replace(
        config,
        contacts=(FootContactConfig("Foot", "Toe", ("sole",), temporal_weight=0),),
    )
    membership = ContactMembership(
        {"Foot": {mesh.name: np.full(4, 0.7) for mesh in animation.meshes}}
    )
    result = _apply(model, poses, joints, animation, config, membership)
    contact = result.diagnostics["contacts"][0]
    assert contact["compression_scale"][0] < 0.9
    assert contact["visual_after_m"][0] >= -0.00101
    assert (
        contact["maximum_added_penetration_after_m"]
        < contact["maximum_added_penetration_before_m"]
    )


def test_anatomical_mask_excludes_contralateral_tissue_at_identical_coordinates() -> (
    None
):
    model, poses, joints, animation, config = _case()
    config = replace(
        config,
        contacts=(FootContactConfig("Foot", "Toe", ("sole",), temporal_weight=0),),
    )
    membership = ContactMembership(
        {"Foot": {"skin": np.ones(4), "muscle": np.zeros(4)}}
    )
    result = _apply(model, poses, joints, animation, config, membership)
    assert not np.array_equal(
        result.animation.meshes[0].points, animation.meshes[0].points
    )
    np.testing.assert_array_equal(
        result.animation.meshes[1].points, animation.meshes[1].points
    )
    registration = _apply(model, poses, joints, animation, replace(config, contacts=()))
    np.testing.assert_array_equal(result.joints, registration.joints)
    assert len(result.diagnostics["contact_membership"]["weights_sha256"]) == 64


def test_vertical_foot_projection_stays_continuous_when_horizontal_axis_reverses() -> (
    None
):
    contact = FootContactConfig("Foot", "Toe", ("sole",))
    point = np.array([[0.05, 0, 0.15]])
    one = _contact_region(
        point,
        np.array([0, 0, 0.1]),
        np.array([1e-5, 0, 0.3]),
        np.array([0, 0, 1]),
        contact,
    )
    two = _contact_region(
        point,
        np.array([0, 0, 0.1]),
        np.array([-1e-5, 0, 0.3]),
        np.array([0, 0, 1]),
        contact,
    )
    for first, second in zip(one, two):
        np.testing.assert_allclose(first, second, atol=1e-5)
    assert abs(one[1][0]) < 1e-5


def test_close_disjoint_anatomical_groups_are_independent_of_contact_order() -> None:
    model, poses, joints, animation, config = _case()
    contacts = (
        FootContactConfig("Foot", "Toe", ("sole",), temporal_weight=0),
        FootContactConfig("Toe", "Foot", ("sole",), temporal_weight=0),
    )
    config = replace(config, contacts=contacts)
    membership = ContactMembership(
        {
            "Foot": {mesh.name: np.array([1, 1, 0, 0]) for mesh in animation.meshes},
            "Toe": {mesh.name: np.array([0, 0, 1, 1]) for mesh in animation.meshes},
        }
    )
    one = _apply(model, poses, joints, animation, config, membership)
    two = _apply(
        model,
        poses,
        joints,
        animation,
        replace(config, contacts=contacts[::-1]),
        membership,
    )
    for first, second in zip(one.animation.meshes, two.animation.meshes):
        np.testing.assert_array_equal(first.points, second.points)
    registration = _apply(model, poses, joints, animation, replace(config, contacts=()))
    np.testing.assert_array_equal(one.joints, registration.joints)
    np.testing.assert_array_equal(two.joints, registration.joints)


def test_native_membership_uses_skin_and_individual_muscle_lbs_bindings() -> None:
    skin = SimpleNamespace(name="skin", rest_vertices=np.zeros((3, 3)))
    muscle = SimpleNamespace(name="muscle", rest_vertices=np.zeros((2, 3)))
    skin_weights = np.array([[0, 1, 0, 0], [0, 0, 0.25, 0.75], [1, 0, 0, 0]])
    muscle_weights = np.array([[0, 1, 0, 0], [0, 0, 0.5, 0.5]])
    predictor = SimpleNamespace(
        meshes=(muscle,),
        skin_mesh=skin,
        joint_names=("Hips", "LeftFoot", "RightFoot", "RightToe"),
        _weights=skin_weights,
        _bindings=[(None, None, None, muscle_weights)],
    )
    membership = contact_membership_from_predictor(
        predictor, (FootContactConfig("RightFoot", "RightToe", ("sole",)),)
    )
    np.testing.assert_array_equal(membership.weights["RightFoot"]["skin"], [0, 1, 0])
    np.testing.assert_array_equal(membership.weights["RightFoot"]["muscle"], [0, 1])
    assert len(membership.metadata["source_lbs_sha256"]) == 64


def test_large_rigid_target_translation_is_damped_within_actual_mesh_bounds() -> None:
    model, _, joints, animation, config = _case()
    model.jnt_axis[0] = [1, 0, 0]
    poses = np.array([[1e4], [1e5], [1e6], [2e5]])
    native = np.repeat(joints[:1], 4, axis=0)
    meshes = tuple(
        replace(mesh, points=np.repeat(mesh.points[:1], 4, axis=0))
        for mesh in animation.meshes
    )
    animation = replace(animation, time=np.arange(4) * 0.03, meshes=meshes)
    result = _apply(model, poses, native, animation, config)
    assert max(result.diagnostics["registration_strength"]) < 0.001
    assert (
        result.diagnostics["maximum_displacement_m"] <= config.max_displacement_m + 1e-7
    )
    for before, after in zip(animation.meshes, result.animation.meshes):
        assert (
            np.linalg.norm(after.points - before.points, axis=2).max()
            <= config.max_displacement_m + 1e-7
        )


def test_geometry_cache_rejects_modified_calibration_provenance() -> None:
    model, poses, joints, animation, config = _case()
    result = _apply(model, poses, joints, animation, config)
    result.animation.metadata["geometry"]["config"]["radius_m"] *= 2
    with pytest.raises(ValueError, match="configuration hash"):
        validate_animation(result.animation)


@pytest.mark.parametrize("bad", [np.zeros((1, 1)), np.array([[0], [np.nan]])])
def test_geometry_rejects_mismatched_or_invalid_source_poses(bad: np.ndarray) -> None:
    model, _, joints, animation, config = _case()
    with pytest.raises(ValueError):
        _apply(model, bad, joints, animation, config)
