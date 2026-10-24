"""Tiny generated MuJoCo and release-format assets used by the package tests."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest


def create_elbow_model():
    """Keep preset-required names without depending on MyoSuite assets."""
    names = ("BIClong", "BICshort", "BRA", "TRIlong", "TRIlat", "TRImed")
    actuators = "".join(
        f'<general name="{name}" joint="r_elbow_flex" dyntype="filter" dynprm=".01"/>'
        for name in names
    )
    return mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<body name="r_humerus" pos="0 0 1"><geom size=".03"/>'
        '<body name="r_ulna_radius_hand" pos="0 0 .3">'
        '<joint name="r_elbow_flex" axis="1 0 0" range="0 2"/>'
        '<geom size=".03"/></body></body></worldbody>'
        f"<actuator>{actuators}</actuator></mujoco>"
    )


def create_synthetic_release(directory: Path) -> tuple[Path, Path]:
    """Create tiny independently generated assets in the official release layout."""
    torch = pytest.importorskip("torch")
    from myosuite_soma.soma_model import _make_network, _make_rigid

    root = directory / "SKIM"
    layer_dir = root / "S1" / "layers" / "tpose"
    muscle_dir = layer_dir / "muscle_meshes_tpose"
    muscle_dir.mkdir(parents=True)
    rig_dir = root / "S1" / "canonical_model"
    rig_dir.mkdir()
    (root / "metadata.json").write_text(json.dumps({"dataset_version": 2}))
    rest = np.array([[0.2, 0.3, 0.4], [0.2, 1.3, 0.4], [1.2, 1.3, 0.4]], np.float32)
    parents = np.array([-1, 0, 1])
    names = ["Hips", "RightArm", "RightForeArm"]
    np.savez(
        root / "S1" / "canonical.npz", parents=parents, joint_names=names, j_rest=rest
    )
    weights = np.array([[1, 0, 0], [0, 0.4, 0.6], [0, 0, 1]] * 2, np.float32)
    np.savez(rig_dir / "S1_rig.npz", j_rest=rest, weights=weights, joint_names=names)
    muscle = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]], np.float32
    )
    skin = muscle.copy()
    skin[:3, 2] += 0.1

    def obj(path: Path, vertices: np.ndarray, faces: list[list[int]]) -> None:
        path.write_text(
            "".join("v " + " ".join(map(str, v)) + "\n" for v in vertices)
            + "".join("f " + " ".join(str(i + 1) for i in f) + "\n" for f in faces)
        )

    obj(layer_dir / "musc_layer-S1-TPose.obj", muscle, [[0, 1, 2], [3, 4, 5]])
    obj(layer_dir / "skin_layer-S1-TPose.obj", skin, [[0, 1, 2], [3, 4, 5]])
    belly = np.array(
        [[0.2, 0.2, -0.02], [0.6, 0.2, -0.02], [0.2, 0.6, -0.02]], np.float32
    )
    obj(muscle_dir / "biceps_t-pose.obj", belly, [[0, 1, 2]])
    obj(muscle_dir / "triceps_t-pose.obj", belly + [0.2, 0, -0.05], [[0, 1, 2]])
    checkpoint = directory / "S1_ABLATION_UNET_RELEASE_V2_e40_best_data.pth"
    with torch.random.fork_rng():
        torch.manual_seed(4)
        network = _make_network(torch, 6, 3, (5, 7))
        rigid = _make_rigid(torch, 3, 4)
    with torch.no_grad():
        for prefix in ("m", "s"):
            getattr(network, f"{prefix}_out").weight.mul_(0.02)
            bias = [0.02, -0.01, 0.03] if prefix == "m" else [-0.005, 0.004, 0.002]
            getattr(network, f"{prefix}_out").bias.copy_(torch.tensor(bias).repeat(6))
        rigid.joint_mask[:] = 1
        rigid.joint_mask[-1] = 0
        rigid.net[4].bias.copy_(
            torch.tensor([0.3, -0.1, 0.2, 0.5, -0.2, 0.1]).repeat(3)
        )
    torch.save(network.state_dict(), checkpoint)
    torch.save(rigid.state_dict(), checkpoint.with_name(checkpoint.stem + "_rigid.pth"))
    return root, checkpoint
