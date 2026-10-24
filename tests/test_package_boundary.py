"""Package boundaries are checked in fresh processes without MyoSuite imports."""

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import mujoco
import numpy as np
import pytest

from myosuite_soma import soma


def _run(source: str, *, block_torch: bool = False) -> subprocess.CompletedProcess:
    package_parent = Path(soma.__file__).resolve().parents[1]
    blocked = ("myosuite", "torch") if block_torch else ("myosuite",)
    guard = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(package_parent)!r})
class ForbiddenDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in {blocked!r}):
            raise ImportError('Deliberately unavailable dependency: ' + fullname)
        return None
sys.meta_path.insert(0, ForbiddenDependency())
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(guard) + textwrap.dedent(source)],
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_base_modules_and_numeric_caches_work_without_torch_or_myosuite(tmp_path):
    path = tmp_path / "portable.npz"
    _run(
        f"""
import importlib
import numpy as np
for name in ('soma', 'soma_model', 'soma_geometry', 'soma_conditioning',
             'soma_cache', 'soma_numeric', 'soma_retarget'):
    importlib.import_module('myosuite_soma.' + name)
from myosuite_soma.soma import AnimatedMesh, SomaAnimation, save_animation, load_animation
from myosuite_soma.soma_cache import write_archive, AnimationArchive
points = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                   [[0, 0, .1], [1, 0, .1], [0, 1, .1]]], np.float32)
animation = SomaAnimation(np.array([0., .04]),
    (AnimatedMesh('Skin', np.array([[0, 1, 2]]), points, kind='skin'),))
save_animation({str(path)!r}, animation)
np.testing.assert_array_equal(load_animation({str(path)!r}).meshes[0].points, points)
write_archive({str(tmp_path / "chunks")!r}, animation, chunk_frames=1)
chunks = list(AnimationArchive({str(tmp_path / "chunks")!r}).iter_chunks())
np.testing.assert_array_equal(np.concatenate([c.meshes[0].points for c in chunks]), points)
assert 'torch' not in sys.modules
assert not any(name == 'myosuite' or name.startswith('myosuite.') for name in sys.modules)
""",
        block_torch=True,
    )


def test_inference_reports_missing_optional_torch(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"dataset_version": 2}))
    _run(
        f"""
from myosuite_soma.soma_model import load_soma_model
try:
    load_soma_model({str(tmp_path)!r}, 'S1', {str(tmp_path / "missing.pth")!r})
except ImportError as error:
    assert 'torch' in str(error).lower(), str(error)
else:
    raise AssertionError('Inference must explain its missing optional dependency')
""",
        block_torch=True,
    )


def test_explicit_xml_retarget_cli_never_imports_myosuite(tmp_path):
    pytest.importorskip("torch")
    from ._fixtures import create_synthetic_release

    root, checkpoint = create_synthetic_release(tmp_path)
    model_path = tmp_path / "model.xml"
    model_path.write_text("""<mujoco><compiler angle="radian"/><worldbody>
      <body name="root" pos=".2 .3 .4"><freejoint/><geom size=".03"/>
      <body name="arm" pos="0 1 0"><joint name="shoulder" axis="0 0 1" range="-1 1"/><geom size=".03"/>
      <body name="forearm" pos="1 0 0"><joint name="elbow" axis="0 0 1" range="-1 1"/><geom size=".03"/>
      </body></body></body></worldbody></mujoco>""")
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "data_root": str(root),
                "subject": "S1",
                "checkpoint": str(checkpoint),
                "body_map": {
                    "Hips": {"body": "root", "parent_body": "world"},
                    "RightArm": {"body": "arm", "parent_body": "root"},
                    "RightForeArm": {"body": "forearm", "parent_body": "arm"},
                },
                "root_body": "root",
                "anchor_body": "root",
                "anchor_joint": "Hips",
                "rest_qpos": mujoco.MjModel.from_xml_path(
                    str(model_path)
                ).qpos0.tolist(),
            }
        )
    )
    config = tmp_path / "ik.json"
    config.write_text(
        json.dumps(
            {
                "landmarks": {
                    "RightArm": {"body": "arm"},
                    "RightForeArm": {"body": "forearm"},
                },
                "orientation_weights": {"RightArm": 1.0, "RightForeArm": 1.0},
                "temporal_weight": 0,
                "neutral_weight": 0,
            }
        )
    )
    recording = tmp_path / "recording.npz"
    np.savez(
        recording,
        pose6d=np.tile(np.eye(3)[:, :2].reshape(6), (2, 3, 1)),
        root_position=np.tile([0.2, 0.3, 0.4], (2, 1)),
    )
    replay = tmp_path / "replay.npz"
    arguments = [
        "--model",
        str(model_path),
        "--profile",
        str(profile),
        "--ik-config",
        str(config),
        "--recording",
        str(recording),
        "--packing",
        "interleaved",
        "--root-convention",
        "native",
        "--frequency",
        "25",
        "--frames",
        "2",
        "--output",
        str(replay),
    ]
    _run(f"""
import runpy
sys.argv = ['soma_retarget', *{arguments!r}]
runpy.run_module('myosuite_soma.soma_retarget', run_name='__main__')
assert not any(name == 'myosuite' or name.startswith('myosuite.') for name in sys.modules)
""")
    with np.load(replay, allow_pickle=False) as data:
        assert data["qpos"].shape == (2, 9)
        assert data["qvel"].shape == (2, 8)
        assert np.isfinite(data["qpos"]).all()
        assert np.isfinite(data["qvel"]).all()
        np.testing.assert_allclose(np.linalg.norm(data["qpos"][:, 3:7], axis=1), 1)
    report = json.loads(replay.with_suffix(".json").read_text())
    assert (
        report["source_parameter_sha256_before"]
        == report["source_parameter_sha256_after"]
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_local_torch_rotation_helpers_match_independent_matrices(dtype):
    torch = pytest.importorskip("torch")
    from scipy.spatial.transform import Rotation
    from myosuite_soma._torch_rotations import axis_angle2quat, quat2mat

    axes = np.array([[1.0, 0, 0], [0, 1, 0], [1.0, 2, -3], [0, 0, 1]])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.array([0.0, -0.7, np.pi - 1e-6, 4.2])
    tensor = torch.tensor(axes, dtype=getattr(torch, dtype))
    quaternion = axis_angle2quat(tensor, torch.tensor(angles, dtype=tensor.dtype))
    actual = quat2mat(quaternion).numpy()
    expected = Rotation.from_rotvec(axes * angles[:, None]).as_matrix()
    np.testing.assert_allclose(actual, expected, atol=4e-7)
    np.testing.assert_allclose(quat2mat(-quaternion).numpy(), actual, atol=0)
    np.testing.assert_array_equal(
        quat2mat(torch.zeros((2, 4))).numpy(), np.tile(np.eye(3), (2, 1, 1))
    )
