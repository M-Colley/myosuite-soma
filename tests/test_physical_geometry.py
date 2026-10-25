"""Physical allometry and explicit cache-to-collision correspondence checks."""

import copy

import mujoco
import numpy as np
import pytest

from myosuite_soma.physical_geometry import (
    anthropometric_profile,
    capsule_profile_from_animation,
)
from myosuite_soma.physical_model import build_physical_model, physical_digest
from myosuite_soma.soma import (
    AnimatedMesh,
    SomaAnimation,
    model_digest,
    qpos_digest,
)

from .test_physical_model import XML


PROVENANCE = {"status": "synthetic", "source": "Invented physical geometry fixture"}


@pytest.mark.parametrize("policy", ["preserve", "constant_density", "target_total"])
def test_segment_prior_mass_com_and_inertia_match_declared_policy(policy):
    source = mujoco.MjModel.from_xml_string(XML)
    source.body_ipos[source.body("lower").id] = [0.02, -0.01, 0.03]
    mujoco.mj_setConst(source, mujoco.MjData(source))
    before = physical_digest(source)
    total = 6.4 if policy == "target_total" else None
    profile = anthropometric_profile(
        source,
        {"lower": 1.2},
        provenance=PROVENANCE,
        mass_policy=policy,
        target_mass_kg=total,
    )
    result = build_physical_model(source, profile, allow_synthetic=True)
    scales = np.ones(source.nbody)
    scales[source.body("lower").id] = 1.2
    expected_mass = source.body_mass.copy()
    if policy != "preserve":
        expected_mass *= scales**3
    if total is not None:
        expected_mass *= total / expected_mass.sum()
    np.testing.assert_allclose(result.model.body_mass, expected_mass)
    for body in range(1, source.nbody):
        scale = scales[body]
        np.testing.assert_allclose(
            result.model.body_ipos[body], source.body_ipos[body] * scale
        )
        np.testing.assert_allclose(
            result.model.body_inertia[body],
            source.body_inertia[body]
            * (expected_mass[body] / source.body_mass[body])
            * scale**2,
        )
        assert np.all(result.model.body_inertia[body] > 0)
    assert result.report["total_mass_kg"] == pytest.approx(expected_mass.sum())
    np.testing.assert_array_equal(
        result.model.actuator_gainprm[:, 2],
        source.actuator_gainprm[:, 3] / source.actuator_acc0,
    )
    assert profile["source_model_sha256"] == before
    assert physical_digest(source) == before


@pytest.mark.parametrize("scale", [0, -1, True, np.nan, np.inf, "1.2", [1.2]])
def test_invalid_segment_scales_fail_cleanly(scale):
    model = mujoco.MjModel.from_xml_string(XML)
    with pytest.raises(ValueError):
        anthropometric_profile(model, {"lower": scale}, provenance=PROVENANCE)


@pytest.mark.parametrize(
    "policy,total",
    [
        ("unknown", None),
        ("preserve", 5),
        ("target_total", None),
        ("target_total", -1),
        ("target_total", True),
    ],
)
def test_mass_policy_rejects_ambiguous_or_impossible_targets(policy, total):
    with pytest.raises(ValueError):
        anthropometric_profile(
            mujoco.MjModel.from_xml_string(XML),
            {},
            provenance=PROVENANCE,
            mass_policy=policy,
            target_mass_kg=total,
        )


def _cache_fixture(*, mocap=False):
    xml = XML
    if mocap:
        xml = xml.replace(
            "</worldbody>",
            '<body name="tracked" mocap="true" pos="1 2 3"><geom name="tracked-geom" size=".1"/></body></worldbody>',
        )
    model = mujoco.MjModel.from_xml_string(xml)
    poses = np.array([[0.0], [0.6]])
    body = model.body("tracked" if mocap else "lower").id
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    local = np.concatenate(
        [
            np.column_stack(
                (0.04 * np.cos(angles), 0.04 * np.sin(angles), np.full(len(angles), z))
            )
            for z in [-0.3, 0.3]
        ]
    )
    data = mujoco.MjData(model)
    points = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        points.append(local @ data.xmat[body].reshape(3, 3).T + data.xpos[body])
    animation = SomaAnimation(
        np.array([2.0, 2.1]),
        (
            AnimatedMesh(
                "Skin",
                np.array([[0, 1, 12], [1, 13, 12]]),
                np.array(points, np.float32),
                kind="skin",
            ),
        ),
        {
            "source_model_sha256": model_digest(model),
            "source_qpos_sha256": qpos_digest(poses),
            "coordinate_system": "mujoco_world",
            "units": "m",
        },
    )
    geom = "tracked-geom" if mocap else "lower-geom"
    assignment = {
        geom: {
            "surface": "Skin",
            "vertex_ids": list(range(len(local))),
            "frame": 1,
            "axis": [0, 0, 1],
        }
    }
    return model, poses, animation, assignment, local


def test_explicit_vertices_are_transformed_into_the_assigned_native_body_frame():
    model, poses, animation, assignments, expected = _cache_fixture()
    before, original = physical_digest(model), copy.deepcopy(assignments)
    profile = capsule_profile_from_animation(
        animation, model, poses, assignments, provenance=PROVENANCE
    )
    entry = profile["geoms"]["lower-geom"]
    np.testing.assert_allclose(entry["capsule_points_m"], expected, atol=6e-8)
    assert entry["capsule_axis"] == [0, 0, 1]
    assert profile["recompute_lengthrange"] is True
    assert profile["source_model_sha256"] == before
    built = build_physical_model(model, profile, allow_synthetic=True)
    np.testing.assert_allclose(
        built.model.geom("lower-geom").size[:2], [0.04, 0.3], atol=6e-8
    )
    assert physical_digest(model) == before
    assert assignments == original


def test_cache_provenance_does_not_allow_wrong_model_or_motion():
    model, poses, animation, assignments, _ = _cache_fixture()
    wrong = copy.copy(model)
    wrong.body_pos[model.body("lower").id, 2] -= 0.1
    with pytest.raises(ValueError, match="source model"):
        capsule_profile_from_animation(
            animation, wrong, poses, assignments, provenance=PROVENANCE
        )
    with pytest.raises(ValueError, match="source qpos"):
        capsule_profile_from_animation(
            animation, model, poses + 0.01, assignments, provenance=PROVENANCE
        )
    with pytest.raises(ValueError, match="source qpos|every animation"):
        capsule_profile_from_animation(
            animation, model, poses[:1], assignments, provenance=PROVENANCE
        )


@pytest.mark.parametrize(
    "change",
    [
        {"frame": 20},
        {"frame": True},
        {"surface": "absent"},
        {"vertex_ids": [0, 1, 1, 2]},
        {"vertex_ids": [0, 1, 2, 100]},
        {"guess_nearest": True},
    ],
)
def test_collision_membership_must_be_explicit_and_valid(change):
    model, poses, animation, assignments, _ = _cache_fixture()
    assignments["lower-geom"].update(change)
    with pytest.raises(ValueError):
        capsule_profile_from_animation(
            animation, model, poses, assignments, provenance=PROVENANCE
        )


def test_noncollision_geoms_and_unrecorded_mocap_frames_are_not_silently_fitted():
    model, poses, animation, assignments, _ = _cache_fixture()
    geom = model.geom("lower-geom").id
    model.geom_contype[geom] = model.geom_conaffinity[geom] = 0
    with pytest.raises(ValueError, match="collision geom"):
        capsule_profile_from_animation(
            animation, model, poses, assignments, provenance=PROVENANCE
        )
    model, poses, animation, assignments, _ = _cache_fixture(mocap=True)
    with pytest.raises(ValueError, match="mocap"):
        capsule_profile_from_animation(
            animation, model, poses, assignments, provenance=PROVENANCE
        )
