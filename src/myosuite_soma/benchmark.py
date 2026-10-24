# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Measure SOMA inference and cache I/O separately on installed local assets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def _worker(path: Path, mode: str) -> dict[str, Any]:
    import psutil
    from myosuite_soma.soma import load_animation
    from myosuite_soma.soma_cache import AnimationArchive

    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    begin = time.perf_counter()
    frames = vertices = resident_geometry = 0
    bounds = [float("inf"), float("-inf")]
    source = (
        [load_animation(path)]
        if mode == "npz"
        else AnimationArchive(path).iter_chunks()
    )
    try:
        for animation in source:
            frames += len(animation.time)
            resident_geometry = max(
                resident_geometry, sum(mesh.points.nbytes for mesh in animation.meshes)
            )
            for mesh in animation.meshes:
                vertices += mesh.points.shape[0] * mesh.points.shape[1]
                bounds[0] = min(bounds[0], float(mesh.points.min()))
                bounds[1] = max(bounds[1], float(mesh.points.max()))
            peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        elapsed = time.perf_counter() - begin
        stop.set()
        thread.join()
    return {
        "mode": mode,
        "seconds": elapsed,
        "frames": frames,
        "frames_per_second": frames / elapsed,
        "vertices": vertices,
        "geometry_payload_peak_bytes": resident_geometry,
        "sampled_peak_rss_bytes": peak[0],
        "baseline_rss_bytes": baseline,
        "sampled_rss_increase_bytes": max(0, peak[0] - baseline),
        "coordinate_range_m": bounds,
        "memory_note": "Process RSS sampled at 5 ms; includes validation/decompression; OS file cache is uncontrolled.",
    }


def _inference(config_path: Path, motion_path: Path, env_id: str) -> dict[str, Any]:
    import torch
    from myosuite import make_env
    from myosuite_soma.soma import SomaConfig, retarget_rotations
    from myosuite_soma.soma_model import load_soma_model

    torch.set_num_threads(4)
    config = SomaConfig.from_json(config_path)
    with np.load(motion_path, allow_pickle=False) as motion:
        qpos = motion["qpos"]
    env = make_env(env_id)
    try:
        start = time.perf_counter()
        predictor = load_soma_model(
            config.data_root,
            config.subject,
            config.checkpoint,
            device=config.device,
            batch_size=config.batch_size,
            individual_muscles=config.individual_muscles,
        )
        loaded = time.perf_counter() - start
        start = time.perf_counter()
        rotations = retarget_rotations(
            config, env.unwrapped.model, qpos, predictor.joint_names, predictor.parents
        )
        retargeted = time.perf_counter() - start
        roots = np.broadcast_to(predictor.rest_joints[0], (len(qpos), 3)).copy()
        predictor.predict(rotations[:1], root_positions=roots[:1])
        if config.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = predictor.predict(rotations, root_positions=roots)
        elapsed = time.perf_counter() - start
        return {
            "environment": env_id,
            "subject": config.subject,
            "device": config.device,
            "batch_size": config.batch_size,
            "frames": len(qpos),
            "model_load_seconds": loaded,
            "retarget_seconds": retargeted,
            "inference_seconds": elapsed,
            "inference_frames_per_second": len(qpos) / elapsed,
            "geometry_bytes": result.skin.nbytes
            + sum(value.nbytes for value in result.muscles.values()),
            "torch": torch.__version__,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated()
            if config.device.startswith("cuda") and torch.cuda.is_available()
            else None,
        }
    finally:
        env.close()


def main() -> None:
    """Write measured cache and optional inference results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--max-error-mm", type=float, default=0.1)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--env")
    parser.add_argument("--worker", choices=("npz", "chunks"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.cache, args.worker)))
        return
    from myosuite_soma.soma import load_animation
    from myosuite_soma.soma_cache import write_archive

    if any((args.config, args.motion, args.env)) and not all(
        (args.config, args.motion, args.env)
    ):
        parser.error("--config, --motion and --env must be supplied together")
    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
        "cache": str(args.cache.resolve()),
        "archives": [],
    }
    paths = [("npz", args.cache.resolve())]
    animation = load_animation(args.cache)
    for label, tolerance in (
        ("lossless", None),
        ("quantized", args.max_error_mm / 1000),
    ):
        path = args.output / label
        start = time.perf_counter()
        manifest = write_archive(
            path, animation, chunk_frames=args.chunk_frames, max_error_m=tolerance
        )
        results["archives"].append(
            {
                "name": label,
                "write_seconds": time.perf_counter() - start,
                "bytes": sum(file.stat().st_size for file in path.iterdir()),
                "measured_max_error_m": manifest["measured_max_error_m"],
                "max_error_m": tolerance,
                "chunk_frames": args.chunk_frames,
            }
        )
        paths.append(("chunks", path.resolve()))
    results["original_bytes"] = args.cache.stat().st_size
    del animation
    for mode, path in paths:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(path),
                str(args.output),
                "--worker",
                mode,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        results.setdefault("reads", []).append(
            {"path": str(path), **json.loads(completed.stdout.strip().splitlines()[-1])}
        )
    if args.config is not None:
        results["inference"] = _inference(args.config, args.motion, args.env)
    (args.output / "benchmark.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
