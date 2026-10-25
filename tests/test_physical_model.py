"""Independent native dynamics checks for explicit physical personalization."""

import copy
import json
import zipfile

import mujoco
import numpy as np
import pytest

from myosuite_soma.physical_model import (
    architecture_force,
    build_physical_model,
    fit_capsule,
    physical_digest,
    save_physical_model,
)


XML = """<mujoco><compiler angle="radian"/>
<option gravity="0 0 0"/>
<worldbody><body name="upper"><geom name="upper-geom" type="capsule" size=".04 .2" mass="2"/>
<site name="origin" pos=".2 .2 .1"/>
<body name="lower" pos="0 0 -.5"><joint name="hinge" axis="1 0 0" range="-.5 .9"/>
<geom name="lower-geom" type="capsule" size=".03 .2" mass="1"/>
<site name="insertion" pos=".1 .1 -.3"/>
<body name="hand" pos="0 0 -.4"><geom name="hand-geom" type="sphere" size=".05" mass=".2"/></body>
</body></body></worldbody>
<tendon><spatial name="path"><site site="origin"/><site site="insertion"/></spatial></tendon>
<actuator><muscle name="muscle" tendon="path" force="-1" scale="200" lengthrange=".7 1.1"/></actuator>
</mujoco>"""


def _profile(**edits):
    return {
        "version": 1,
        "provenance": {"status": "synthetic", "source": "Independent generated test"},
        **edits,
    }


def test_architecture_estimate_uses_explicit_pcsa_projection_once():
    force, report = architecture_force(
        {
            "volume_m3": 0.00012,
            "fibre_length_m": 0.06,
            "pennation_deg": 30,
            "specific_tension_pa": 300000,
        }
    )
    expected = 300000 * (0.00012 / 0.06) * np.cos(np.pi / 6)
    assert force == pytest.approx(expected)
    assert report["status"] == "estimated"
    projected, _ = architecture_force(
        {
            "pcsa_m2": 0.002 * np.cos(np.pi / 6),
            "pcsa_kind": "projected",
            "pennation_deg": 30,
            "specific_tension_pa": 300000,
        }
    )
    assert projected == pytest.approx(expected)
    with pytest.raises(ValueError, match="pcsa_kind"):
        architecture_force({"pcsa_m2": 0.002, "specific_tension_pa": 300000})


def test_source_and_profile_are_immutable_and_auto_force_does_not_follow_mass():
    spec = mujoco.MjSpec.from_string(XML)
    source = spec.compile()
    source_hash, xml = physical_digest(source), spec.to_xml()
    baseline = float(source.actuator_gainprm[0, 3] / source.actuator_acc0[0])
    profile = _profile(bodies={"lower": {"mass_kg": 3}})
    untouched = copy.deepcopy(profile)
    built = build_physical_model(spec, profile, allow_synthetic=True)
    assert spec.to_xml() == xml
    assert physical_digest(source) == source_hash
    assert profile == untouched
    assert built.model.body("lower").mass == pytest.approx(3)
    assert built.model.actuator_gainprm[0, 2] == pytest.approx(baseline)
    assert built.model.actuator_biasprm[0, 2] == pytest.approx(baseline)
    assert built.model.actuator_acc0[0] != pytest.approx(source.actuator_acc0[0])
    np.testing.assert_allclose(
        built.model.body("lower").inertia, source.body("lower").inertia * 3
    )
    assert built.report["total_mass_kg"] == pytest.approx(source.body_mass.sum() + 2)


def test_explicit_strength_changes_both_active_and_passive_curves():
    source = mujoco.MjModel.from_xml_string(XML)
    profile = _profile(
        muscles={
            "muscle": {"force_N": 425, "range": [0.6, 1.2], "passive_force_max": 1.8}
        }
    )
    built = build_physical_model(source, profile, allow_synthetic=True)
    for field in ("actuator_gainprm", "actuator_biasprm"):
        parameters = getattr(built.model, field)[0]
        np.testing.assert_array_equal(parameters[:3], [0.6, 1.2, 425])
        assert parameters[7] == 1.8
    expected_l0 = 0.4 / 0.6
    assert built.report["effective_lengths"]["muscle"][
        "optimal_fibre_length_m"
    ] == pytest.approx(expected_l0)
    data = mujoco.MjData(built.model)
    data.act[:] = 0.7
    mujoco.mj_forward(built.model, data)
    assert np.isfinite(data.actuator_force).all()
    assert data.actuator_force[0] < 0


def test_segment_scale_has_exact_local_scope_and_refreshes_path_ranges():
    source = mujoco.MjModel.from_xml_string(XML)
    original = physical_digest(source)
    built = build_physical_model(
        source, _profile(bodies={"lower": {"scale": 1.25}}), allow_synthetic=True
    )
    np.testing.assert_allclose(built.model.body("lower").pos, source.body("lower").pos)
    np.testing.assert_allclose(
        built.model.body("hand").pos, source.body("hand").pos * 1.25
    )
    np.testing.assert_allclose(
        built.model.geom("lower-geom").size, source.geom("lower-geom").size * 1.25
    )
    np.testing.assert_allclose(
        built.model.site("insertion").pos, source.site("insertion").pos * 1.25
    )
    np.testing.assert_array_equal(
        built.model.geom("upper-geom").size, source.geom("upper-geom").size
    )
    np.testing.assert_array_equal(
        built.model.geom("hand-geom").size, source.geom("hand-geom").size
    )
    np.testing.assert_array_equal(built.model.body_mass, source.body_mass)
    np.testing.assert_array_equal(built.model.body_inertia, source.body_inertia)
    assert built.report["lengthrange_recomputed"] == ["muscle"]
    assert not np.array_equal(
        built.model.actuator_lengthrange, source.actuator_lengthrange
    )
    assert physical_digest(source) == original
    with pytest.raises(ValueError, match="stale ranges"):
        build_physical_model(
            source,
            _profile(bodies={"lower": {"scale": 1.1}}, recompute_lengthrange=False),
            allow_synthetic=True,
        )


def test_capsule_contains_all_samples_and_updates_native_collision_bounds():
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    points = np.concatenate(
        [
            np.column_stack(
                (0.04 * np.cos(angles), 0.04 * np.sin(angles), np.full(16, z))
            )
            for z in [-0.3, 0.3]
        ]
    )
    fitted = fit_capsule(points, [0, 0, 1])
    assert fitted["size_m"][:2] == pytest.approx([0.04, 0.3])
    source = mujoco.MjModel.from_xml_string(XML)
    built = build_physical_model(
        source,
        _profile(
            geoms={
                "lower-geom": {
                    "capsule_points_m": points.tolist(),
                    "capsule_axis": [0, 0, 1],
                }
            }
        ),
        allow_synthetic=True,
    )
    geom = built.model.geom("lower-geom").id
    assert built.model.geom_rbound[geom] == pytest.approx(0.34)
    np.testing.assert_allclose(built.model.geom_aabb[geom, 3:], [0.04, 0.04, 0.34])
    assert built.report["capsule_fits"]["lower-geom"]["size_m"][:2] == pytest.approx(
        [0.04, 0.3]
    )


def test_capsule_collision_fit_cannot_invalidate_a_native_tendon_wrapping_geom():
    source = mujoco.MjModel.from_xml_string(
        XML.replace(
            '<site site="insertion"/>',
            '<geom geom="hand-geom"/><site site="insertion"/>',
        )
    )
    before = physical_digest(source)
    with pytest.raises(
        ValueError, match="tendon wrapping geom.*cannot become a capsule"
    ):
        build_physical_model(
            source,
            _profile(
                geoms={
                    "hand-geom": {
                        "capsule_points_m": [
                            [0, 0, -0.2],
                            [0, 0, 0.2],
                            [0.03, 0, 0],
                            [0, 0.03, 0],
                        ]
                    }
                }
            ),
            allow_synthetic=True,
        )
    assert physical_digest(source) == before


@pytest.mark.parametrize(
    "kind,initial_size,updated_size",
    [
        ("sphere", ".1", [0.35, 0, 0]),
        ("capsule", ".1 .2", [0.25, 0.4, 0]),
        ("cylinder", ".1 .2", [0.25, 0.4, 0]),
        ("ellipsoid", ".1 .2 .3", [0.25, 0.35, 0.45]),
        ("box", ".1 .2 .3", [0.25, 0.35, 0.45]),
    ],
)
def test_primitive_bounds_match_native_compiler_after_pose_and_size_edit(
    kind, initial_size, updated_size
):
    spec = mujoco.MjSpec.from_string(
        '<mujoco><compiler angle="radian"/><worldbody><body name="segment">'
        '<freejoint/><inertial pos=".1 .2 .3" mass="1" '
        'diaginertia="1 1 1"/><geom name="anchor" type="sphere" size=".1"/>'
        f'<geom name="target" type="{kind}" size="{initial_size}" '
        'pos="2 0 0" euler=".2 .3 .4"/></body></worldbody></mujoco>'
    )
    spec.body("segment").simple = False
    built = build_physical_model(
        spec,
        _profile(geoms={"target": {"pos_m": [3, 0.2, 0.1], "size_m": updated_size}}),
        allow_synthetic=True,
    )
    reference_spec = spec.copy()
    reference_spec.geom("target").pos = [3, 0.2, 0.1]
    reference_spec.geom("target").size = updated_size
    reference = reference_spec.compile()
    np.testing.assert_allclose(built.model.geom_rbound, reference.geom_rbound)
    np.testing.assert_allclose(built.model.geom_aabb, reference.geom_aabb)
    # Three nodes ensure the inertial-frame leaf and parent boxes are covered.
    assert built.model.body_bvhnum[built.model.body("segment").id] == 3
    np.testing.assert_array_equal(built.model.bvh_nodeid, reference.bvh_nodeid)
    np.testing.assert_allclose(built.model.bvh_aabb, reference.bvh_aabb, atol=2e-14)


@pytest.mark.parametrize("operation", ["geom", "scale", "com"])
def test_geometry_and_inertial_edits_preserve_actual_midphase_contacts(operation):
    target_position = "2" if operation == "scale" else "3"
    probe_position = "4.25" if operation == "scale" else "3.25"
    target_radius = ".1" if operation in ("geom", "scale") else ".3"
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="segment" pos="0 0 1"><freejoint/>'
        '<inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>'
        '<geom name="anchor" type="sphere" size=".1"/>'
        f'<geom name="target" type="sphere" size="{target_radius}" '
        f'pos="{target_position} 0 0"/></body>'
        f'<body name="probe" pos="{probe_position} 0 1"><freejoint/>'
        '<geom name="probegeom" type="sphere" size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    spec.body("segment").simple = False
    reference_spec = spec.copy()
    if operation == "geom":
        profile = _profile(geoms={"target": {"size_m": [0.3, 0, 0]}})
        reference_spec.geom("target").size = [0.3, 0, 0]
    elif operation == "scale":
        profile = _profile(bodies={"segment": {"scale": 2}})
        reference_spec.geom("target").pos = [4, 0, 0]
        reference_spec.geom("target").size = [0.2, 0, 0]
        reference_spec.geom("anchor").size = [0.2, 0, 0]
    else:
        profile = _profile(bodies={"segment": {"com_m": [0.6, 0.2, 0.1]}})
        reference_spec.body("segment").ipos = [0.6, 0.2, 0.1]
    built = build_physical_model(spec, profile, allow_synthetic=True)
    reference = reference_spec.compile()
    contacts = []
    for model in (built.model, reference):
        assert not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE))
        assert model.body_bvhnum[model.body("segment").id] == 3
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        target = model.geom("target").id
        probe = model.geom("probegeom").id
        contact = [c for c in data.contact if set(c.geom) == {target, probe}]
        assert len(contact) == 1
        assert contact[0].dist < 0
        contacts.append((contact[0].dist, contact[0].pos.copy()))
    assert contacts[0][0] == pytest.approx(contacts[1][0], abs=1e-12)
    np.testing.assert_allclose(contacts[0][1], contacts[1][1], atol=1e-12)


@pytest.mark.parametrize(
    "edits,match",
    [
        ({"mystery": 1}, "Unknown"),
        ({"bodies": {"lower": {"mass_kg": -1}}}, "positive"),
        ({"bodies": {"lower": {"inertia_kg_m2": [0.1, 0.1, 1]}}}, "triangle"),
        ({"bodies": {"lower": {"inertia_kg_m2": [1e-15, 1e-15, 1e-13]}}}, "triangle"),
        ({"sites": {"absent": {"pos_m": [0, 0, 0]}}}, "missing"),
        ({"muscles": {"muscle": {"force_N": -1}}}, "nonnegative"),
        ({"muscles": {"muscle": {"range": [1, 1]}}}, "increasing"),
        ({"source_model_sha256": "incorrect"}, "does not match"),
    ],
)
def test_profile_rejections_do_not_mutate_source(edits, match):
    model = mujoco.MjModel.from_xml_string(XML)
    original = physical_digest(model)
    with pytest.raises(ValueError, match=match):
        build_physical_model(model, _profile(**edits), allow_synthetic=True)
    assert physical_digest(model) == original


def test_synthetic_opt_in_and_native_motor_rejection():
    model = mujoco.MjModel.from_xml_string(XML)
    with pytest.raises(ValueError, match="Synthetic"):
        build_physical_model(model, _profile())
    motor = mujoco.MjModel.from_xml_string(
        XML.replace(
            '<muscle name="muscle" tendon="path" force="-1" scale="200" lengthrange=".7 1.1"/>',
            '<motor name="muscle" joint="hinge"/>',
        )
    )
    with pytest.raises(ValueError, match="native muscle"):
        build_physical_model(
            motor, _profile(muscles={"muscle": {"force_N": 100}}), allow_synthetic=True
        )


def test_saved_binary_and_editable_spec_preserve_calibrated_dynamics(tmp_path):
    source_path = tmp_path / "source.xml"
    source_path.write_text(XML)
    built = build_physical_model(
        source_path,
        _profile(muscles={"muscle": {"force_N": 432}}),
        allow_synthetic=True,
    )
    out = save_physical_model(built, tmp_path / "private-output")
    loaded = mujoco.MjModel.from_binary_path(str(out / "model.mjb"))
    assert physical_digest(loaded) == physical_digest(built.model)
    report = json.loads((out / "report.json").read_text())
    assert report["target_model_sha256"] == physical_digest(loaded)
    if hasattr(mujoco.MjSpec, "copy_back"):
        assert (out / "model.xml").is_file()
        assert (out / "model.zip").is_file()
        with zipfile.ZipFile(out / "model.zip") as archive:
            assert any(n.endswith(".xml") for n in archive.namelist())
        restored = mujoco.MjSpec.from_zip(str(out / "model.zip")).compile()
        np.testing.assert_array_equal(
            restored.actuator_gainprm, built.model.actuator_gainprm
        )
    with pytest.raises(ValueError, match="new output"):
        save_physical_model(built, out)
    assert physical_digest(
        mujoco.MjModel.from_binary_path(str(out / "model.mjb"))
    ) == physical_digest(loaded)


def test_full_body_local_inertia_and_effective_lengths_are_encoded_physically():
    source = mujoco.MjModel.from_xml_string(XML)
    theta = 0.4
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1.0],
        ]
    )
    tensor = rotation @ np.diag([0.02, 0.03, 0.04]) @ rotation.T
    profile = _profile(
        bodies={
            "lower": {"inertia_kg_m2": tensor.tolist(), "com_m": [0.01, 0.02, -0.03]}
        },
        muscles={
            "muscle": {"optimal_fibre_length_m": 0.2, "tendon_slack_length_m": 0.6}
        },
    )
    built = build_physical_model(source, profile, allow_synthetic=True)
    body = built.model.body("lower").id
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, built.model.body_iquat[body])
    matrix = matrix.reshape(3, 3)
    np.testing.assert_allclose(
        matrix @ np.diag(built.model.body_inertia[body]) @ matrix.T, tensor, atol=1e-12
    )
    np.testing.assert_allclose(built.model.actuator_gainprm[0, :2], [0.5, 2.5])
    np.testing.assert_allclose(built.model.actuator_biasprm[0, :2], [0.5, 2.5])
    effective = built.report["effective_lengths"]["muscle"]
    assert effective["optimal_fibre_length_m"] == pytest.approx(0.2)
    assert effective["inelastic_tendon_length_m"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "muscle",
    [
        {"optimal_fibre_length_m": 0.2},
        {
            "optimal_fibre_length_m": 0.2,
            "tendon_slack_length_m": 0.6,
            "range": [0.5, 2.5],
        },
        {"optimal_fibre_length_m": 0.2, "tendon_slack_length_m": -0.1},
        {"optimal_fibre_length_m": 0.2, "tendon_slack_length_m": 0.8},
    ],
)
def test_impossible_or_ambiguous_fibre_targets_are_rejected(muscle):
    with pytest.raises(ValueError):
        build_physical_model(
            mujoco.MjModel.from_xml_string(XML),
            _profile(muscles={"muscle": muscle}),
            allow_synthetic=True,
        )


def test_segment_scaling_does_not_depend_on_json_body_order():
    model = mujoco.MjModel.from_xml_string(XML)
    edits = {"upper": {"scale": 1.1}, "lower": {"pos_m": [0, 0, -0.7], "scale": 1.2}}
    first = build_physical_model(model, _profile(bodies=edits), allow_synthetic=True)
    second = build_physical_model(
        model,
        _profile(bodies=dict(reversed(list(edits.items())))),
        allow_synthetic=True,
    )
    np.testing.assert_array_equal(first.model.body_pos, second.model.body_pos)
    np.testing.assert_array_equal(first.model.site_pos, second.model.site_pos)
    assert physical_digest(first.model) == physical_digest(second.model)


def test_editable_export_copies_external_assets_and_reloads_without_source(tmp_path):
    if not hasattr(mujoco.MjSpec, "copy_back"):
        pytest.skip("Editable spec export needs MuJoCo 3.17+")
    source = tmp_path / "source"
    source.mkdir()
    (source / "tetra.obj").write_text(
        "v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    path = source / "source.xml"
    path.write_text(
        '<mujoco><asset><mesh name="tetra" file="tetra.obj"/></asset><worldbody><body name="b"><joint/><geom name="g" type="mesh" mesh="tetra"/></body></worldbody></mujoco>'
    )
    built = build_physical_model(
        path, _profile(bodies={"b": {"mass_kg": 0.3}}), allow_synthetic=True
    )
    out = save_physical_model(built, tmp_path / "local-model")
    with zipfile.ZipFile(out / "model.zip") as archive:
        assert any(n.endswith(".obj") for n in archive.namelist())
        assert not any(
            n.startswith(("/", "\\")) or ":" in n for n in archive.namelist()
        )
    # Removing the independently generated source file proves portable loading.
    (source / "tetra.obj").unlink()
    restored = mujoco.MjSpec.from_zip(str(out / "model.zip")).compile()
    xml_restored = mujoco.MjModel.from_xml_path(str(out / "model.xml"))
    np.testing.assert_allclose(restored.body_mass, built.model.body_mass)
    np.testing.assert_array_equal(xml_restored.mesh_vert, built.model.mesh_vert)
    with pytest.raises(ValueError, match="primitive geoms only"):
        build_physical_model(
            restored, _profile(bodies={"b": {"scale": 1.1}}), allow_synthetic=True
        )


def test_editable_writer_precision_is_reported_while_binary_is_exact(tmp_path):
    if not hasattr(mujoco.MjSpec, "copy_back"):
        pytest.skip("Editable spec export needs MuJoCo 3.17+")
    built = build_physical_model(
        mujoco.MjSpec.from_string(XML),
        _profile(bodies={"lower": {"mass_kg": 1.534315}}),
        allow_synthetic=True,
    )
    output = save_physical_model(built, tmp_path / "local-precision")
    restored = mujoco.MjModel.from_binary_path(str(output / "model.mjb"))
    assert physical_digest(restored) == physical_digest(built.model)
    editable = mujoco.MjSpec.from_zip(str(output / "model.zip")).compile()
    assert editable.body("lower").mass == pytest.approx(1.534315, rel=6e-6)
    report = json.loads((output / "report.json").read_text())
    assert report["editable_roundtrip"]["native_xml_significant_digits"] == 6
    assert "body_mass" in report["editable_roundtrip"]["nonzero_roundtrip_differences"]


def test_simple_free_body_com_edits_recompile_inertial_coupling_correctly():
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="floating"><freejoint/><geom type="box" size=".1 .2 .3" mass="2"/></body></worldbody></mujoco>'
    )
    source = spec.compile()
    before = physical_digest(source)
    profile = _profile(
        source_model_sha256=before,
        bodies={
            "floating": {
                "com_m": [0.02, 0.03, 0.04],
                "inertia_kg_m2": [
                    [0.02, 0.001, 0.002],
                    [0.001, 0.03, 0.003],
                    [0.002, 0.003, 0.04],
                ],
            }
        },
    )
    with pytest.raises(ValueError, match="MjSpec or XML"):
        build_physical_model(source, profile, allow_synthetic=True)
    built = build_physical_model(spec, profile, allow_synthetic=True)
    assert built.report["source_model_sha256"] == before
    assert built.report["recompiled_for_inertial_edits"] == ["floating"]
    reference_spec = spec.copy()
    body = reference_spec.body("floating")
    body.simple = False
    body.explicitinertial = True
    body.mass = 2
    body.ipos = [0.02, 0.03, 0.04]
    body.fullinertia = [0.02, 0.03, 0.04, 0.001, 0.002, 0.003]
    reference = reference_spec.compile()
    outputs = []
    for model in (built.model, reference):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mass_matrix = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass_matrix)
        outputs.append((mass_matrix, data.qacc.copy()))
        np.testing.assert_allclose(data.qacc[:3], [0, 0, -9.81], atol=1e-12)
        np.testing.assert_allclose(data.qacc[3:], 0, atol=1e-12)
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-12)
    assert abs(outputs[0][0][0, 4]) > 0.05
    assert physical_digest(source) == before


def test_native_editable_euler_rounding_is_compared_as_rotation(tmp_path):
    if not hasattr(mujoco.MjSpec, "copy_back"):
        pytest.skip("Editable spec export needs MuJoCo 3.17+")
    spec = mujoco.MjSpec.from_string(
        XML.replace(
            'body name="upper"',
            'body name="upper" euler="1.57079613 -.00039817 .00039817"',
        )
    )
    built = build_physical_model(spec, _profile(), allow_synthetic=True)
    output = save_physical_model(built, tmp_path / "local-euler")
    report = json.loads((output / "report.json").read_text())
    differences = report["editable_roundtrip"]["nonzero_roundtrip_differences"]
    assert differences["body_quat"]["maximum_orientation_difference_rad"] <= 2e-5


def test_path_edits_preserve_active_passive_tendon_rest_length_on_export(tmp_path):
    if not hasattr(mujoco.MjSpec, "copy_back"):
        pytest.skip("Editable spec export needs MuJoCo 3.17+")
    spec = mujoco.MjSpec.from_string(
        XML.replace('<spatial name="path">', '<spatial name="path" stiffness="350">')
    )
    original = spec.compile()
    built = build_physical_model(
        spec,
        _profile(sites={"insertion": {"pos_m": [0.1, 0.2, -0.35]}}),
        allow_synthetic=True,
    )
    np.testing.assert_array_equal(
        built.model.tendon_lengthspring, original.tendon_lengthspring
    )
    out = save_physical_model(built, tmp_path / "local-spring")
    restored = mujoco.MjSpec.from_zip(str(out / "model.zip")).compile()
    np.testing.assert_allclose(
        restored.tendon_lengthspring, original.tendon_lengthspring, rtol=6e-6
    )
    states = []
    for model in (built.model, restored):
        data = mujoco.MjData(model)
        data.qpos[0] = 0.4
        mujoco.mj_forward(model, data)
        states.append(data.qfrc_passive.copy())
    # Directly bound force sensitivity to the native XML springlength rounding.
    data = mujoco.MjData(built.model)
    lengths = []
    for angle in [0.4 - 1e-6, 0.4 + 1e-6]:
        data.qpos[0] = angle
        mujoco.mj_forward(built.model, data)
        lengths.append(data.ten_length[0])
    derivative = (lengths[1] - lengths[0]) / 2e-6
    delta = np.abs(restored.tendon_lengthspring - built.model.tendon_lengthspring).max()
    bound = 350 * delta * abs(derivative) + 2e-9
    assert np.abs(states[0] - states[1]).max() <= bound


def test_optional_editable_failure_keeps_the_verified_binary(tmp_path, monkeypatch):
    import myosuite_soma.physical_model as physical

    built = build_physical_model(
        mujoco.MjSpec.from_string(XML), _profile(), allow_synthetic=True
    )
    if built.spec is None:
        pytest.skip("Editable spec export needs MuJoCo 3.17+")

    def unavailable(spec):
        raise ValueError("Deliberately unavailable editable resource")

    monkeypatch.setattr(physical, "_portable_spec", unavailable)
    output = save_physical_model(built, tmp_path / "local-binary-fallback")
    assert physical_digest(
        mujoco.MjModel.from_binary_path(str(output / "model.mjb"))
    ) == physical_digest(built.model)
    report = json.loads((output / "report.json").read_text())
    assert report["editable_spec_available"] is False
    assert "Deliberately unavailable" in report["editable_export_note"]
    assert not (output / "model.zip").exists()
    assert not (output / "model.xml").exists()


def test_programmatic_legacy_size_hints_do_not_make_editable_xml_invalid(tmp_path):
    if not hasattr(mujoco.MjSpec, "copy_back"):
        pytest.skip("Editable spec export needs MuJoCo 3.17+")
    spec = mujoco.MjSpec.from_string(XML)
    spec.memory = 2**20
    spec.njmax = 100
    built = build_physical_model(spec, _profile(), allow_synthetic=True)
    output = save_physical_model(built, tmp_path / "local-size-hints")
    restored = mujoco.MjSpec.from_zip(str(output / "model.zip")).compile()
    np.testing.assert_array_equal(restored.body_mass, built.model.body_mass)
    assert "njmax=" not in (output / "model.xml").read_text()


def test_free_body_position_cannot_silently_ignore_its_reference_qpos():
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="floating" pos="1 2 3"><freejoint/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    source = spec.compile()
    before = physical_digest(source)
    for original in (source, spec):
        with pytest.raises(ValueError, match="free-joint body.*initial qpos"):
            build_physical_model(
                original,
                _profile(bodies={"floating": {"pos_m": [2, 3, 4]}}),
                allow_synthetic=True,
            )
    assert physical_digest(source) == before
    assert physical_digest(spec.compile()) == before
