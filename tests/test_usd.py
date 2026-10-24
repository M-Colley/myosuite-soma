"""Numeric USD export keeps world coordinates, fixed topology and source timing."""

from dataclasses import replace

import numpy as np
import pytest

from myosuite_soma.soma import AnimatedMesh, SomaAnimation, save_animation
from myosuite_soma.soma_cache import write_archive
from myosuite_soma import usd


def _animation():
    points = np.array(
        [
            [[0.4, -0.2, 1.1], [0.7, -0.2, 1.1], [0.4, 0.1, 1.1]],
            [[0.4, -0.2, 1.2], [0.7, -0.2, 1.2], [0.4, 0.1, 1.2]],
            [[0.5, -0.2, 1.3], [0.8, -0.2, 1.3], [0.5, 0.1, 1.3]],
        ],
        np.float32,
    )
    faces = np.array([[0, 1, 2]])
    return SomaAnimation(
        np.array([1.25, 1.30, 1.35]),
        (
            AnimatedMesh(
                "biceps-r", faces, points, activation=np.array([0.1, 0.5, 0.9])
            ),
            AnimatedMesh("triceps-r", faces, points + np.float32(0.2)),
            AnimatedMesh(
                "Skin",
                faces,
                points + np.float32(0.02),
                texcoord=np.array([[0, 0], [1, 0], [0, 1]], np.float32),
                kind="skin",
            ),
        ),
    )


def test_exported_fixed_topology_points_are_world_metres_with_source_times(tmp_path):
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    Gf = pytest.importorskip("pxr.Gf")
    animation = _animation()
    path = usd.export_usd(animation, tmp_path / "animation.usda")
    stage = Usd.Stage.Open(str(path))
    assert stage.GetDefaultPrim().GetPath().pathString == "/World"
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert stage.GetFramesPerSecond() == pytest.approx(20)
    assert stage.GetTimeCodesPerSecond() == pytest.approx(20)
    assert stage.GetStartTimeCode() == 0
    assert stage.GetEndTimeCode() == 2
    data = stage.GetRootLayer().customLayerData
    assert data["source_time_start_seconds"] == pytest.approx(1.25)
    assert data["source_time_end_seconds"] == pytest.approx(1.35)
    meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
    assert len(meshes) == len(animation.meshes)
    for expected, mesh in zip(animation.meshes, meshes):
        np.testing.assert_array_equal(mesh.GetFaceVertexCountsAttr().Get(), [3])
        np.testing.assert_array_equal(mesh.GetFaceVertexIndicesAttr().Get(), [0, 1, 2])
        assert mesh.GetFaceVertexIndicesAttr().GetTimeSamples() == []
        assert mesh.GetPointsAttr().GetTimeSamples() == [0.0, 1.0, 2.0]
        for time in range(3):
            np.testing.assert_array_equal(
                np.asarray(mesh.GetPointsAttr().Get(time)), expected.points[time]
            )
            transform = UsdGeom.XformCache(time).GetLocalToWorldTransform(
                mesh.GetPrim()
            )
            assert transform == Gf.Matrix4d(1)
        assert mesh.GetSubdivisionSchemeAttr().Get() == UsdGeom.Tokens.none
        if expected.texcoord is not None:
            st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
            assert st.GetInterpolation() == UsdGeom.Tokens.vertex
            np.testing.assert_array_equal(np.asarray(st.Get()), expected.texcoord)


def test_bridge_preserves_explicit_activation_columns_and_uniform_unmapped_surfaces():
    Usd = pytest.importorskip("pxr.Usd")
    animation = _animation()
    stage = Usd.Stage.CreateInMemory()
    muscles, skins, activation, columns = usd.write_animation(stage, animation)
    assert len(muscles) == 2 and len(skins) == 1
    assert columns == {muscles[0]: 0}
    np.testing.assert_array_equal(
        activation, animation.meshes[0].activation.astype(np.float32)[:, None]
    )


def test_nonuniform_or_invalid_geometry_does_not_create_output(tmp_path):
    pytest.importorskip("pxr.Usd")
    animation = _animation()
    path = tmp_path / "new-parent" / "invalid.usda"
    with pytest.raises(ValueError, match="uniform sample times"):
        usd.export_usd(replace(animation, time=np.array([0.0, 0.04, 0.11])), path)
    assert not path.exists() and not path.parent.exists()
    broken = replace(animation.meshes[0], faces=np.array([[0, 1, 99]]))
    with pytest.raises(ValueError, match="out-of-range"):
        usd.export_usd(replace(animation, meshes=(broken,)), path)
    assert not path.exists()


def test_export_never_overwrites_and_single_frame_uses_documented_default_rate(
    tmp_path,
):
    Usd = pytest.importorskip("pxr.Usd")
    path = tmp_path / "existing.usda"
    path.write_bytes(b"existing content\n")
    with pytest.raises(ValueError, match="Choose a new"):
        usd.export_usd(_animation(), path)
    assert path.read_bytes() == b"existing content\n"
    source = _animation().meshes[0]
    single = SomaAnimation(
        np.array([4.2]),
        (replace(source, points=source.points[:1], activation=source.activation[:1]),),
    )
    output = usd.export_usd(single, tmp_path / "single.usdc")
    stage = Usd.Stage.Open(str(output))
    assert stage.GetFramesPerSecond() == 30
    assert stage.GetEndTimeCode() == 0


@pytest.mark.parametrize("format", ["npz", "chunks"])
def test_usd_cli_loads_both_numeric_cache_formats(tmp_path, capsys, format):
    pytest.importorskip("pxr.Usd")
    animation = _animation()
    cache = tmp_path / ("cache.npz" if format == "npz" else "cache.chunks")
    if format == "npz":
        save_animation(cache, animation)
    else:
        write_archive(cache, animation, chunk_frames=2)
    output = tmp_path / "export.usdc"
    usd.main([str(cache), str(output)])
    assert output.is_file()
    assert str(output.resolve()) in capsys.readouterr().out
