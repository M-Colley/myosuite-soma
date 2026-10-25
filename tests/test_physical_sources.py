"""Invented XML and point fixtures, without external anatomical data."""

import copy
import json

import mujoco
import numpy as np
import pytest

from myosuite_soma import physical_sources as sources
from myosuite_soma.physical_model import build_physical_model


def _osim(tmp_path, *, legacy=False, inertia=".03 .04 .05 .002 -.003 .004"):
    if legacy:
        fields = "".join(
            f"<inertia_{axis}>{value}</inertia_{axis}>"
            for axis, value in zip(
                ("xx", "yy", "zz", "xy", "xz", "yz"), inertia.split()
            )
        )
    else:
        fields = f"<inertia>{inertia}</inertia>"
    xml = f"""<OpenSimDocument Version="40000"><Model name="invented">
      <BodySet><objects><Body name="imaginary_segment">
      <mass>2</mass><mass_center>.1 .2 .3</mass_center>{fields}
      </Body><Body name="massless_frame"><mass>0</mass></Body></objects></BodySet>
    </Model></OpenSimDocument>"""
    path = tmp_path / "synthetic-model.osim"
    path.write_text(xml, encoding="utf8")
    return path


def _spec():
    return mujoco.MjSpec.from_string("""<mujoco><worldbody>
      <body name="native_segment"><freejoint/>
      <geom type="box" size=".1 .1 .1" mass="1"/></body>
      <body name="another_segment" pos="2 0 0"><joint type="hinge"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")


def _model():
    return _spec().compile()


def _mapping():
    return {
        "imaginary_segment": {
            "body": "native_segment",
            "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            "translation_m": [0.4, -0.2, 0.1],
        }
    }


@pytest.mark.parametrize("legacy", [True, False])
def test_opensim_sign_order_and_rigid_registration_bridge(tmp_path, legacy):
    imported = sources.import_segment_properties(_osim(tmp_path, legacy=legacy))
    segment = imported["segments"]["imaginary_segment"]
    tensor = np.array(
        [[0.03, 0.002, -0.003], [0.002, 0.04, 0.004], [-0.003, 0.004, 0.05]]
    )
    np.testing.assert_allclose(segment["inertia_kg_m2"], tensor)
    assert imported["massless_bodies_ignored"] == ["massless_frame"]
    assert imported["provenance"]["status"] == "estimated"
    assert len(imported["provenance"]["input_sha256"]) == 64
    spec = _spec()
    model = spec.compile()
    mapping = _mapping()
    profile = sources.segment_profile(imported, model, mapping)
    edit = profile["bodies"]["native_segment"]
    rotation = np.array(mapping["imaginary_segment"]["rotation"])
    expected = rotation @ tensor @ rotation.T
    # Translation changes COM but never adds a parallel-axis term to COM inertia.
    np.testing.assert_allclose(edit["com_m"], [0.2, -0.1, 0.4], atol=1e-12)
    np.testing.assert_allclose(edit["inertia_kg_m2"], expected)
    result = build_physical_model(spec, profile).model
    body = result.body("native_segment").id
    np.testing.assert_allclose(result.body_ipos[body], edit["com_m"])
    assert result.body_mass[body] == 2
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, result.body_iquat[body])
    axes = matrix.reshape(3, 3)
    np.testing.assert_allclose(
        axes @ np.diag(result.body_inertia[body]) @ axes.T, expected
    )
    assert model.body_mass[body] == 1


def test_compiled_only_simple_body_requests_recompilable_source(tmp_path):
    imported = sources.import_segment_properties(_osim(tmp_path))
    model = _model()
    profile = sources.segment_profile(imported, model, _mapping())
    # A compiled free body's diagonal sparsity cannot be repaired just by
    # clearing optimization flags. Recompilation must remain an explicit step.
    with pytest.raises(ValueError, match="MjSpec|XML|simple"):
        build_physical_model(model, profile)


@pytest.mark.parametrize(
    "inertia", ["-.1 .2 .2 0 0 0", ".1 .1 1 0 0 0", ".1 .1 .1 1 0 0", "NaN .1 .1 0 0 0"]
)
def test_invalid_source_inertia_rejected(tmp_path, inertia):
    with pytest.raises(ValueError):
        sources.import_segment_properties(_osim(tmp_path, inertia=inertia))


def test_xml_does_not_expand_entities_or_supply_missing_defaults(tmp_path):
    path = _osim(tmp_path)
    text = path.read_text()
    path.write_text('<!DOCTYPE x [<!ENTITY custom "2">]>' + text)
    with pytest.raises(ValueError, match="DOCTYPE"):
        sources.import_segment_properties(path)
    path.write_text(text.replace("<mass_center>.1 .2 .3</mass_center>", ""))
    with pytest.raises(ValueError, match="mass_center"):
        sources.import_segment_properties(path)
    path.write_text(
        text.replace(
            '<Model name="invented">',
            '<Model name="invented"><length_units>millimeters</length_units>',
        )
    )
    with pytest.raises(ValueError, match="SI meters"):
        sources.import_segment_properties(path)
    path.write_text(
        text.replace('<Body name="massless_frame">', '<Body name="imaginary_segment">')
    )
    with pytest.raises(ValueError, match="unique"):
        sources.import_segment_properties(path)


def test_registration_requires_explicit_proper_frames_and_unique_targets(tmp_path):
    data = sources.import_segment_properties(_osim(tmp_path))
    model = _model()
    bad = _mapping()
    del bad["imaginary_segment"]["rotation"]
    with pytest.raises(ValueError, match="requires"):
        sources.segment_profile(data, model, bad)
    bad = _mapping()
    bad["imaginary_segment"]["rotation"] = np.diag([-1, 1, 1]).tolist()
    with pytest.raises(ValueError, match="proper"):
        sources.segment_profile(data, model, bad)
    bad = _mapping()
    bad["imaginary_segment"]["body"] = "world"
    with pytest.raises(ValueError, match="not world"):
        sources.segment_profile(data, model, bad)
    data["segments"]["second"] = copy.deepcopy(data["segments"]["imaginary_segment"])
    bad = _mapping()
    bad["second"] = bad["imaginary_segment"]
    with pytest.raises(ValueError, match="unique"):
        sources.segment_profile(data, model, bad)


def _markups(tmp_path, *, units="mm", coordinate_system="LPS", kind="Curve"):
    # Invented 3-4-5 plus 12-mm segments. Curve length is the control polygon.
    document = {
        "@schema": "https://example.invalid/synthetic-schema.json",
        "markups": [
            {
                "type": kind,
                "coordinateSystem": coordinate_system,
                "coordinateUnits": units,
                "controlPoints": [
                    {"position": [0, 0, 0], "positionStatus": "defined"},
                    {"position": [3, 4, 0], "positionStatus": "defined"},
                    {"position": [3, 4, 12], "positionStatus": "defined"},
                ],
                "measurements": [{"name": "length", "value": 999999}],
            }
        ],
    }
    path = tmp_path / "synthetic-path.mrk.json"
    path.write_text(json.dumps(document))
    return path


def test_markup_units_coord_control_polygon_and_provenance(tmp_path):
    path = _markups(tmp_path)
    result = sources.import_mtu_markups(
        {"native_muscle": path}, units="mm", coordinate_system="LPS"
    )
    assert result["mtu_length_m"] == [[pytest.approx(0.017)]]
    info = result["paths"]["native_muscle"]
    np.testing.assert_allclose(
        info["points_ras_m"], [[0, 0, 0], [-0.003, -0.004, 0], [-0.003, -0.004, 0.012]]
    )
    assert info["source_coordinate_system"] == "LPS"
    assert len(info["input_sha256"]) == 64
    assert result["provenance"]["status"] == "estimated"
    assert "spline" in result["provenance"]["note"]


def test_coded_ucum_units_and_source_metres_are_explicit(tmp_path):
    path = _markups(tmp_path, units="m", coordinate_system="RAS")
    doc = json.loads(path.read_text())
    doc["markups"][0]["coordinateUnits"] = ["m", "UCUM", "meter"]
    path.write_text(json.dumps(doc))
    result = sources.import_mtu_markups({"a": path}, units="m", coordinate_system="RAS")
    assert result["mtu_length_m"] == [[pytest.approx(17)]]
    np.testing.assert_allclose(result["paths"]["a"]["points_ras_m"][1], [3, 4, 0])
    with pytest.raises(ValueError, match="Explicit source units"):
        sources.import_mtu_markups({"a": path}, units="cm", coordinate_system="RAS")


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"coordinateUnits": "cm"}, "units disagree"),
        ({"coordinateSystem": "RAS"}, "coordinate_system disagrees"),
        ({"type": "ClosedCurve"}, "open"),
        ({"controlPoints": [{"position": [0, 0, 0]}]}, "at least two"),
        (
            {
                "controlPoints": [
                    {"position": [0, 0, 0]},
                    {"position": [1, 0, 0], "positionStatus": "undefined"},
                ]
            },
            "defined",
        ),
    ],
)
def test_ambiguous_markup_or_undefined_points_reject(tmp_path, change, expected):
    path = _markups(tmp_path)
    doc = json.loads(path.read_text())
    doc["markups"][0].update(change)
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=expected):
        sources.import_mtu_markups({"a": path}, units="mm", coordinate_system="LPS")


def test_multiple_markups_and_duplicate_keys_reject(tmp_path):
    path = _markups(tmp_path)
    doc = json.loads(path.read_text())
    doc["markups"].append(doc["markups"][0])
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="exactly one"):
        sources.import_mtu_markups({"a": path}, units="mm", coordinate_system="LPS")
    path.write_text('{"markups": [], "markups": []}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        sources.import_mtu_markups({"a": path}, units="mm", coordinate_system="LPS")


def test_reviewed_native_pose_assembly_matches_path_fitter_schema(tmp_path):
    path = _markups(tmp_path)
    pose = sources.import_mtu_markups(
        {"native_muscle": path}, units="mm", coordinate_system="LPS"
    )
    observations = sources.mtu_path_observations(
        [pose, pose],
        [[0, 0.1], [0, 0.2]],
        time=[0, 0],
        recording_ids=["train", "validation"],
        source_model_sha256="a" * 64,
    )
    assert observations["units"] == {"time": "s", "qpos": "SI", "mtu_length_m": "m"}
    assert observations["mtu_length_m"] == [
        [pytest.approx(0.017)],
        [pytest.approx(0.017)],
    ]
    assert observations["conventions"]["source_model_sha256"] == "a" * 64
    json.dumps(observations, allow_nan=False)
    with pytest.raises(ValueError, match="increase"):
        sources.mtu_path_observations(
            [pose, pose], [[0], [0]], time=[1, 1], recording_ids=["same", "same"]
        )
    with pytest.raises(ValueError, match="identical ordered"):
        other = dict(pose, actuator_names=["different"])
        sources.mtu_path_observations(
            [pose, other], [[0], [0]], time=[0, 1], recording_ids=["same", "same"]
        )


def test_cli_import_and_map_segments_preserve_source_and_frames(tmp_path, capsys):
    from myosuite_soma.cli import main

    source = _osim(tmp_path)
    imported = tmp_path / "segments.json"
    main(
        [
            "physical",
            "import-segments",
            "--input",
            str(source),
            "--output",
            str(imported),
        ]
    )
    data = json.loads(imported.read_text())
    native = tmp_path / "native.xml"
    native.write_text(_spec().to_xml())
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps(_mapping()))
    output = tmp_path / "segment-profile.json"
    main(
        [
            "physical",
            "map-segments",
            "--model",
            str(native),
            "--segments",
            str(imported),
            "--mapping",
            str(mapping),
            "--output",
            str(output),
        ]
    )
    profile = json.loads(output.read_text())
    assert profile["provenance"]["input_sha256"] == data["provenance"]["input_sha256"]
    np.testing.assert_allclose(
        profile["bodies"]["native_segment"]["com_m"], [0.2, -0.1, 0.4]
    )
    built = build_physical_model(native, profile)
    assert built.model.body_mass[built.model.body("native_segment").id] == 2
    original = imported.read_bytes()
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "physical",
                "import-segments",
                "--input",
                str(source),
                "--output",
                str(imported),
            ]
        )
    assert stopped.value.code == 2
    assert imported.read_bytes() == original
    assert str(output) in capsys.readouterr().out


def test_cli_import_mtu_and_prepare_native_path_observations(tmp_path):
    from myosuite_soma.cli import main

    path = _markups(tmp_path)
    files = tmp_path / "files.json"
    files.write_text(json.dumps({"native_muscle": path.name}))
    converted = tmp_path / "path-pose.json"
    main(
        [
            "physical",
            "import-mtu",
            "--files",
            str(files),
            "--units",
            "mm",
            "--coordinate-system",
            "LPS",
            "--output",
            str(converted),
        ]
    )
    manifest = tmp_path / "poses.json"
    manifest.write_text(json.dumps({"poses": [converted.name, converted.name]}))
    trajectory = tmp_path / "reviewed-native-pose.npz"
    np.savez_compressed(
        trajectory,
        qpos=np.tile(_model().qpos0, (2, 1)),
        time=[0, 0],
        recording_ids=np.array(["whole_train_record", "whole_holdout_record"]),
    )
    output = tmp_path / "path-observations.json"
    main(
        [
            "physical",
            "prepare-paths",
            "--markups",
            str(manifest),
            "--trajectory",
            str(trajectory),
            "--output",
            str(output),
        ]
    )
    observed = json.loads(output.read_text())
    assert observed["mtu_length_m"] == [[pytest.approx(0.017)], [pytest.approx(0.017)]]
    assert observed["qpos"] == np.tile(_model().qpos0, (2, 1)).tolist()
    assert observed["recording_ids"] == ["whole_train_record", "whole_holdout_record"]
    assert observed["conventions"]["pose_sources"][0]["native_muscle"]["input_sha256"]
    bad_output = tmp_path / "must-not-exist.json"
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "physical",
                "import-mtu",
                "--files",
                str(files),
                "--units",
                "mm",
                "--coordinate-system",
                "RAS",
                "--output",
                str(bad_output),
            ]
        )
    assert stopped.value.code == 2
    assert not bad_output.exists()
