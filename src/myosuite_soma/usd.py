# Copyright (c) MyoSuite Authors. All rights reserved.
# Adapted for standalone SOMA export by Mark Colley, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Standalone animated USD meshes and the optional renderer's geometry bridge."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable


def _write_mesh(
    stage: Any, path: str, faces: Any, frames: Any, texcoord: Any = None
) -> Any:
    import numpy as np
    from pxr import Sdf, UsdGeom, Vt

    UsdGeom.Xform.Define(stage, path.rsplit("/", 1)[0])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(np.asarray(faces).ravel().tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if texcoord is not None and len(texcoord):
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(Vt.Vec2fArray.FromNumpy(np.asarray(texcoord, dtype=np.float32)))
    points = mesh.CreatePointsAttr()
    for frame, vertices in enumerate(frames):
        points.Set(
            Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)), frame
        )
    return mesh


def write_animation(
    stage: Any,
    animation: Any,
    *,
    write_mesh: Callable[..., Any] = _write_mesh,
    muscle_color: tuple[float, ...] = (0.42, 0.11, 0.10),
    skin_color: tuple[float, ...] = (0.72, 0.62, 0.58),
) -> tuple[list[str], list[str], Any, dict[str, int]]:
    """Write fixed topology and world-space samples, preserving color mappings.

    Args:
        stage: Writable USD stage with time codes configured by the caller.
        animation: Validated SomaAnimation.
        write_mesh: Optional upstream primitive writer.
        muscle_color: Uniform muscle RGB.
        skin_color: Skin RGB.

    Returns:
        Muscle names, skin names, visual activation and named activation columns.
    """
    import numpy as np
    from pxr import Gf, Tf, UsdGeom

    from myosuite_soma.soma import validate_animation

    validate_animation(animation)
    UsdGeom.Xform.Define(stage, "/World/SOMA")
    muscles, skins, tracks, columns = [], [], [], {}
    for index, surface in enumerate(animation.meshes):
        prefix = "Muscle" if surface.kind == "muscle" else "Skin"
        label = Tf.MakeValidIdentifier(surface.name)[:32]
        name = f"{prefix}_SOMA_{index:04d}_{label}"
        mesh = write_mesh(
            stage,
            f"/World/SOMA/{name}_Transform/{name}",
            surface.faces,
            surface.points,
            surface.texcoord,
        )
        color = muscle_color if surface.kind == "muscle" else skin_color
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        (muscles if surface.kind == "muscle" else skins).append(name)
        if surface.kind == "muscle" and surface.activation is not None:
            columns[name] = len(tracks)
            tracks.append(surface.activation)
    activation = (
        np.stack(tracks, axis=1).astype(np.float32)
        if tracks
        else np.empty((len(animation.time), 0), dtype=np.float32)
    )
    return muscles, skins, activation, columns


def export_usd(animation: Any, output: Path | str) -> Path:
    """Export a uniformly sampled animation without requiring MyoSuite.

    Args:
        animation: SomaAnimation in world metres.
        output: New .usd, .usda or .usdc file.

    Returns:
        Absolute output path.

    Raises:
        ValueError: Output exists, sampling varies, or data is invalid.
    """
    import numpy as np
    from pxr import Usd, UsdGeom

    from myosuite_soma.soma import validate_animation

    target = Path(output).resolve()
    if target.exists() or target.suffix.lower() not in (".usd", ".usda", ".usdc"):
        raise ValueError("Choose a new .usd, .usda or .usdc file")
    validate_animation(animation)
    intervals = np.diff(animation.time)
    step = float(intervals[0]) if len(intervals) else 1 / 30
    if not np.allclose(intervals, step, rtol=1e-7, atol=1e-10):
        raise ValueError("Standalone USD export requires uniform sample times")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(target))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(1 / step)
    stage.SetFramesPerSecond(1 / step)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(animation.time) - 1)
    stage.GetRootLayer().customLayerData = {
        "source_time_start_seconds": float(animation.time[0]),
        "source_time_end_seconds": float(animation.time[-1]),
    }
    write_animation(stage, animation)
    stage.GetRootLayer().Save()
    return target


def main(argv: list[str] | None = None) -> None:
    """Export a numeric animation cache to USD.

    Args:
        argv: Optional explicit CLI arguments.
    """
    from myosuite_soma.soma import load_animation
    from myosuite_soma.soma_cache import AnimationArchive

    parser = argparse.ArgumentParser(prog="myosuite-soma export-usd")
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    animation = (
        AnimationArchive(args.cache).load()
        if args.cache.is_dir()
        else load_animation(args.cache)
    )
    print(export_usd(animation, args.output))
