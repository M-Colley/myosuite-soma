"""Create invented geometry for opt-in Khronos validation; no SOMA assets used.

Run explicitly with an output directory. This helper is not a pytest module and
does not download, load or redistribute subject geometry or model checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from myosuite_soma.glb import export_glb, export_rigged_glb
from myosuite_soma.soma import AnimatedMesh, SomaAnimation


def make_fixtures(output: Path) -> None:
    """Write a compact morph clip and a rig with six retained influences."""
    points = np.array(
        [
            [-0.1, -0.08, 0.3],
            [0.1, -0.08, 0.3],
            [0.1, 0.08, 0.3],
            [-0.1, 0.08, 0.3],
            [-0.1, -0.08, 0.7],
            [0.1, -0.08, 0.7],
            [0.1, 0.08, 0.7],
            [-0.1, 0.08, 0.7],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    frames = np.stack(
        (
            points,
            points * [1.08, 0.94, 1] + [0.02, 0.01, 0],
            points * [0.97, 1.06, 1.02] + [-0.01, 0, 0.03],
        )
    ).astype(np.float32)
    metadata = {"fixture": "invented box surfaces; no released SOMA assets"}
    clip = SomaAnimation(
        np.array([4.5, 4.54, 4.61]),
        (
            AnimatedMesh("synthetic-skin", faces, frames, kind="skin"),
            AnimatedMesh("synthetic-muscle", faces, frames * np.float32(0.8)),
        ),
        metadata,
    )
    export_glb(clip, output / "animated.glb")

    names = tuple(f"synthetic-link-{index}" for index in range(6))
    bind = np.broadcast_to(np.eye(4), (6, 4, 4)).copy()
    for index in range(6):
        angle = 0.06 * index
        cosine, sine = np.cos(angle), np.sin(angle)
        bind[index, :3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
        bind[index, :3, 3] = [0.025 * index, 0.01 * index, 0.12 * index]
    raw = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [0, 2, 3, 4, 5, 6],
            [0, 0, 3, 4, 5, 6],
            [0, 0, 0, 1, 0, 0],
            [6, 5, 4, 3, 2, 1],
            [6, 5, 4, 3, 2, 0],
            [6, 5, 4, 3, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    rigged = SomaAnimation(
        np.array([0.0]),
        tuple(
            AnimatedMesh(mesh.name, mesh.faces, mesh.points[:1], kind=mesh.kind)
            for mesh in clip.meshes
        ),
        metadata,
    )
    export_rigged_glb(
        rigged,
        output / "rigged.glb",
        body_names=names,
        parents=np.arange(-1, 5),
        bind_matrices=bind,
        weights={"synthetic-skin": raw, "synthetic-muscle": raw[:, ::-1]},
    )
    print("Created synthetic animated.glb and rigged.glb; no licensed assets loaded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Local fixture output directory")
    args = parser.parse_args()
    make_fixtures(args.output)


if __name__ == "__main__":
    main()
