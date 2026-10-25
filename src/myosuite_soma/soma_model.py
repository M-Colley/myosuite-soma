# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under Apache-2.0; SOMA adaptations: licenses/SOMA-MIT.txt.
# Extracted from the audited MyoSuite SOMA integration; imports adapted.
# Source attribution: THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Frozen RELEASE_V2 inference from separately obtained SKIM v2 assets.

No training scripts, Blender modules, or simulation objects are imported. Torch
is optional until loading a model. Geometry stays in SOMA's native metre frame;
retargeting and the final change of world coordinates belong to the caller.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeshSpec:
    """Static topology and canonical vertices of one output mesh."""

    name: str
    faces: np.ndarray
    rest_vertices: np.ndarray
    kind: str = "muscle"
    texcoord: np.ndarray | None = None


@dataclass(frozen=True)
class SOMAPrediction:
    """Posed metre-valued vertices, with arrays shaped ``(frames, vertices, 3)``."""

    muscles: dict[str, np.ndarray]
    skin: np.ndarray


def _read_obj(path: Path, name: str, kind: str) -> MeshSpec:
    vertices, faces = [], []
    with path.open(encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                vertices.append([float(x) for x in fields[1:4]])
            elif fields[0] == "f":
                ids = [int(x.split("/")[0]) for x in fields[1:]]
                ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
                if len(ids) < 3:
                    raise ValueError(f"Invalid OBJ face in {path}")
                faces.extend(
                    (ids[0], ids[i], ids[i + 1]) for i in range(1, len(ids) - 1)
                )
    vert = np.asarray(vertices, np.float32)
    face = np.asarray(faces, np.int32)
    if (
        vert.ndim != 2
        or vert.shape[1:] != (3,)
        or not len(vert)
        or not np.isfinite(vert).all()
        or face.ndim != 2
        or face.shape[1:] != (3,)
        or face.min() < 0
        or face.max() >= len(vert)
    ):
        raise ValueError(f"Invalid or empty mesh: {path}")
    return MeshSpec(name, face, vert, kind)


def _make_network(
    torch: Any, vertices: int, joints: int, widths: tuple[int, int]
) -> Any:
    nn = torch.nn

    class Deformation(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            first, second = widths

            def block(n_in: int, n_out: int) -> Any:
                return nn.Sequential(
                    nn.Linear(n_in, n_out, device="cpu", dtype=torch.float32),
                    nn.BatchNorm1d(n_out, device="cpu", dtype=torch.float32),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )

            for prefix in ("m", "s"):
                setattr(self, f"{prefix}_enc1", block(9 * (joints - 1), first))
                setattr(self, f"{prefix}_enc2", block(first, second))
                setattr(self, f"{prefix}_bottleneck", block(second, second))
                setattr(self, f"{prefix}_dec1", block(2 * second, first))
                setattr(self, f"{prefix}_dec2", block(2 * first, first))
                setattr(
                    self,
                    f"{prefix}_out",
                    nn.Linear(first, 3 * vertices, device="cpu", dtype=torch.float32),
                )

        def forward(self, rotations: Any) -> tuple[Any, Any]:
            features = (
                rotations[:, 1:]
                - torch.eye(3, device=rotations.device, dtype=rotations.dtype)
            ).flatten(1)
            outputs = []
            for prefix in ("m", "s"):
                first = getattr(self, f"{prefix}_enc1")(features)
                second = getattr(self, f"{prefix}_enc2")(first)
                middle = getattr(self, f"{prefix}_bottleneck")(second)
                decoded = getattr(self, f"{prefix}_dec1")(
                    torch.cat((middle, second), 1)
                )
                decoded = getattr(self, f"{prefix}_dec2")(
                    torch.cat((decoded, first), 1)
                )
                outputs.append(
                    getattr(self, f"{prefix}_out")(decoded).reshape(-1, vertices, 3)
                )
            return outputs[0], outputs[0] + outputs[1]

    return Deformation()


def _make_rigid(torch: Any, joints: int, width: int) -> Any:
    from myosuite_soma._torch_rotations import axis_angle2quat, quat2mat

    nn = torch.nn

    class RigidCorrection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(joints * 9, width, device="cpu", dtype=torch.float32),
                nn.LeakyReLU(0.2),
                nn.Linear(width, width, device="cpu", dtype=torch.float32),
                nn.LeakyReLU(0.2),
                nn.Linear(width, joints * 6, device="cpu", dtype=torch.float32),
            )
            self.register_buffer(
                "joint_mask", torch.zeros(joints, 6, device="cpu", dtype=torch.float32)
            )

        def forward(self, rotations: Any) -> Any:
            eye = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
            raw = self.net((rotations - eye).flatten(1)).reshape(-1, joints, 6)
            raw = raw * self.joint_mask
            vector = 0.1 * raw[..., :3]
            angle = vector.norm(dim=-1)
            axis = vector / angle.clamp(min=1e-12).unsqueeze(-1)
            corr = torch.eye(4, device=rotations.device, dtype=rotations.dtype).repeat(
                len(rotations), joints, 1, 1
            )
            corr[..., :3, :3] = quat2mat(axis_angle2quat(axis, angle))
            corr[..., :3, 3] = 0.02 * raw[..., 3:]
            return corr

    return RigidCorrection()


def _stability_mask(torch: Any, muscle: Any, skin: Any, faces: Any) -> Any:
    # Match RELEASE_V2's public evaluation/viewer mask, not its training loss.
    points = torch.cat((muscle[faces], skin[faces]), dim=1).transpose(1, 2)
    centered = points - points.mean(dim=2, keepdim=True)
    weight = torch.zeros(6, 6, device=muscle.device, dtype=muscle.dtype)
    weight.fill_diagonal_(4)
    for i, j in (
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 4),
        (4, 5),
        (5, 3),
        (0, 3),
        (1, 4),
        (2, 5),
    ):
        weight[i, j] = weight[j, i] = 2
    shape = centered @ weight @ centered.transpose(1, 2)
    valid = torch.det(shape).abs() > 0
    mask = torch.zeros(len(muscle), 1, device=muscle.device, dtype=muscle.dtype)
    mask[torch.unique(faces[valid])] = 1
    return mask


def _bind(points: np.ndarray, layer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    distances, ids = cKDTree(layer).query(points, k=min(16, len(layer)))
    if distances.ndim == 1:
        distances, ids = distances[:, None], ids[:, None]
    weights = np.exp(-((distances / (distances[:, -1:] / 2 + 1e-9)) ** 2))
    return ids, (weights / weights.sum(1, keepdims=True)).astype(np.float32)


def _skinning_transforms(
    torch: Any,
    rotations: Any,
    roots: Any,
    rest: Any,
    parents: np.ndarray,
    corrections: Any,
) -> Any:
    global_poses = []
    for joint, parent in enumerate(parents):
        local = torch.eye(4, device=rotations.device, dtype=rotations.dtype).repeat(
            len(rotations), 1, 1
        )
        local[:, :3, :3] = rotations[:, joint]
        local[:, :3, 3] = roots if parent < 0 else rest[joint] - rest[parent]
        local = local @ corrections[:, joint]
        global_poses.append(local if parent < 0 else global_poses[parent] @ local)
    transforms = torch.stack(global_poses, dim=1)
    transforms[:, :, :3, 3] -= (transforms[:, :, :3, :3] @ rest[..., None])[..., 0]
    return transforms


def _pose(torch: Any, points: Any, weights: Any, transforms: Any) -> Any:
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    return torch.einsum("vj,bjxy,bvy->bvx", weights, transforms, homogeneous)[..., :3]


class SOMAModel:
    """Loaded immutable visual anatomy and frozen pose-conditioned networks.

    Attributes:
        joint_names: SOMA joint order, root first.
        parents: Topologically ordered parent indices; root is -1.
        rest_joints: Canonical joint positions in metres.
        meshes: Individual-muscle topology in prediction dictionary order.
        skin_mesh: Skin topology.
        metadata: Asset provenance for a generated cache.
    """

    def __init__(
        self,
        torch: Any,
        device: Any,
        network: Any,
        rigid: Any,
        joint_names: tuple[str, ...],
        parents: np.ndarray,
        rest: np.ndarray,
        weights: np.ndarray,
        layer: MeshSpec,
        skin: MeshSpec,
        meshes: tuple[MeshSpec, ...],
        batch_size: int,
        metadata: dict[str, Any],
    ) -> None:
        self.joint_names, self.parents, self.rest_joints = (
            joint_names,
            parents.copy(),
            rest.copy(),
        )
        self.meshes, self.skin_mesh, self.metadata = meshes, skin, metadata
        self._torch, self._device, self._batch_size = torch, device, batch_size
        self._network, self._rigid = network, rigid

        def tensor(array: np.ndarray) -> Any:
            return torch.as_tensor(array, device=device)

        self._rest, self._weights = tensor(rest), tensor(weights)
        self._skin = tensor(skin.rest_vertices)
        with torch.autocast(device_type=device.type, enabled=False):
            self._mask = _stability_mask(
                torch,
                tensor(layer.rest_vertices),
                self._skin,
                tensor(layer.faces).long(),
            )
        self._bindings = []
        for mesh in meshes:
            ids, blend = _bind(mesh.rest_vertices, layer.rest_vertices)
            blended_weights = (blend[..., None] * weights[ids]).sum(1)
            self._bindings.append(
                (
                    tensor(mesh.rest_vertices),
                    tensor(ids).long(),
                    tensor(blend),
                    tensor(blended_weights),
                )
            )

    def skinning_weights(self) -> dict[str, np.ndarray]:
        """Copy native per-vertex joint weights for the loaded output surfaces.

        Columns follow ``joint_names``. Individual objects use the same
        interpolated layer weights as :meth:`predict`; no weights are fitted,
        pruned or guessed from distance to the simulator's skeleton here.
        """
        result = {
            mesh.name: binding[3].detach().cpu().numpy().copy()
            for mesh, binding in zip(self.meshes, self._bindings)
        }
        result[self.skin_mesh.name] = self._weights.detach().cpu().numpy().copy()
        return result

    def predict(
        self, rotations: np.ndarray, root_positions: np.ndarray | None = None
    ) -> SOMAPrediction:
        """Pose a sequence without modifying simulation state or network weights.

        Args:
            rotations: ``(T,J,3,3)`` local rotations in the checkpoint's frame.
            root_positions: ``(T,3)`` Hips positions; omitted means root at zero,
                matching upstream. Use ``rest_joints[0]`` for canonical placement.

        Returns:
            Muscle and skin vertices in the posed SOMA coordinate frame.

        Raises:
            ValueError: Inputs are nonfinite, malformed, or not proper rotations.
        """
        rotations = np.asarray(rotations)
        if rotations.dtype.kind not in "fiu":
            raise ValueError("rotations must contain real numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            rotations = np.array(rotations, dtype=np.float32, order="C", copy=True)
        expected = (len(self.joint_names), 3, 3)
        if rotations.ndim != 4 or rotations.shape[1:] != expected or not len(rotations):
            raise ValueError(f"rotations must have shape (T,{expected[0]},3,3), T > 0")
        if (
            not np.isfinite(rotations).all()
            or not np.allclose(
                rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=1e-4
            )
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4)
        ):
            raise ValueError(
                "rotations must be finite proper orthonormal rotation matrices"
            )
        if root_positions is None:
            roots = np.zeros((len(rotations), 3), np.float32)
        else:
            roots = np.asarray(root_positions)
            if roots.dtype.kind not in "fiu":
                raise ValueError("root_positions must contain real numeric values")
            with np.errstate(over="ignore", invalid="ignore"):
                roots = np.array(roots, dtype=np.float32, order="C", copy=True)
        if roots.shape != (len(rotations), 3) or not np.isfinite(roots).all():
            raise ValueError("root_positions must be finite with shape (T,3)")
        muscles = {
            mesh.name: np.empty(
                (len(rotations), len(mesh.rest_vertices), 3), np.float32
            )
            for mesh in self.meshes
        }
        skin = np.empty(
            (len(rotations), len(self.skin_mesh.rest_vertices), 3), np.float32
        )
        torch = self._torch
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self._device.type, enabled=False),
        ):
            for begin in range(0, len(rotations), self._batch_size):
                end = min(begin + self._batch_size, len(rotations))
                pose = torch.as_tensor(rotations[begin:end], device=self._device)
                root = torch.as_tensor(roots[begin:end], device=self._device)
                dm, ds = self._network(pose)
                # Preserve upstream's separate residual masking and recombination.
                residual = (ds - dm) * self._mask
                dm = dm * self._mask
                ds = dm + residual
                transforms = _skinning_transforms(
                    torch, pose, root, self._rest, self.parents, self._rigid(pose)
                )
                skin[begin:end] = (
                    _pose(torch, self._skin + ds, self._weights, transforms)
                    .cpu()
                    .numpy()
                )
                for mesh, (rest, ids, blend, weights) in zip(
                    self.meshes, self._bindings
                ):
                    displacement = (blend[None, ..., None] * dm[:, ids]).sum(2)
                    posed = _pose(torch, rest + displacement, weights, transforms)
                    muscles[mesh.name][begin:end] = posed.cpu().numpy()
        if not np.isfinite(skin).all() or any(
            not np.isfinite(x).all() for x in muscles.values()
        ):
            raise ValueError("SOMA inference produced nonfinite geometry")
        return SOMAPrediction(muscles, skin)

    def pose_offsets(
        self, rotations: np.ndarray, offsets: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Transport canonical additive offsets using the released skinning.

        Args:
            rotations: Proper local rotations shaped ``(T,J,3,3)``.
            offsets: Canonical metre-valued vectors for every muscle and skin,
                keyed by mesh name and shaped ``(T,V,3)``.

        Returns:
            Posed vectors without joint/root translation. This is an optional
            visual correction; the released prediction is unchanged.
        """
        pose_array = np.asarray(rotations)
        if pose_array.dtype.kind not in "fiu":
            raise ValueError("offset rotations must contain real numbers")
        pose_array = np.array(pose_array, dtype=np.float32, order="C", copy=True)
        if (
            pose_array.ndim != 4
            or pose_array.shape[1:] != (len(self.joint_names), 3, 3)
            or not len(pose_array)
            or not np.isfinite(pose_array).all()
            or not np.allclose(
                pose_array @ pose_array.swapaxes(-1, -2), np.eye(3), atol=1e-4
            )
            or not np.allclose(np.linalg.det(pose_array), 1, atol=1e-4)
        ):
            raise ValueError("offset rotations must be proper (T,J,3,3) matrices")
        specs = (*self.meshes, self.skin_mesh)
        if set(offsets) != {spec.name for spec in specs}:
            raise ValueError("offsets must cover exactly every muscle and skin")
        arrays = {}
        for spec in specs:
            value = np.asarray(offsets[spec.name])
            if (
                value.dtype.kind not in "fiu"
                or value.shape != (len(pose_array), len(spec.rest_vertices), 3)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid canonical offsets for {spec.name}")
            arrays[spec.name] = np.array(value, dtype=np.float32, order="C", copy=True)
        result = {name: np.empty_like(value) for name, value in arrays.items()}
        weights = [binding[3] for binding in self._bindings] + [self._weights]
        torch = self._torch
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self._device.type, enabled=False),
        ):
            for begin in range(0, len(pose_array), self._batch_size):
                end = min(begin + self._batch_size, len(pose_array))
                pose = torch.as_tensor(pose_array[begin:end], device=self._device)
                roots = torch.zeros(
                    (len(pose), 3), device=self._device, dtype=torch.float32
                )
                transforms = _skinning_transforms(
                    torch, pose, roots, self._rest, self.parents, self._rigid(pose)
                )
                for spec, weight in zip(specs, weights):
                    vectors = torch.as_tensor(
                        arrays[spec.name][begin:end], device=self._device
                    )
                    result[spec.name][begin:end] = (
                        torch.einsum(
                            "vj,bjxy,bvy->bvx",
                            weight,
                            transforms[:, :, :3, :3],
                            vectors,
                        )
                        .cpu()
                        .numpy()
                    )
        return result


def _state_dict(torch: Any, path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"SOMA checkpoint missing: {path}; obtain the RELEASE_V2 assets separately"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, Mapping)
        or not state
        or any(
            not isinstance(key, str)
            or not torch.is_tensor(value)
            or not torch.isfinite(value).all()
            for key, value in state.items()
        )
    ):
        raise ValueError(f"Expected a finite plain SOMA state dictionary: {path}")
    return state


def _names(values: np.ndarray) -> tuple[str, ...]:
    if values.ndim != 1 or values.dtype.kind not in "US":
        raise ValueError("SKIM joint_names must be a string vector")
    return tuple(x.decode() if isinstance(x, bytes) else str(x) for x in values)


def load_soma_model(
    data_root: Path | str,
    subject: str,
    checkpoint: Path | str,
    *,
    device: str = "auto",
    individual_muscles: Sequence[str] | None = None,
    batch_size: int = 32,
) -> SOMAModel:
    """Load real-format RELEASE_V2 networks and independently downloaded SKIM v2.

    Args:
        data_root: Extracted SKIM dataset root containing ``metadata.json``.
        subject: Subject directory, S1 through S5.
        checkpoint: Released UNet ``*_best_data.pth``; its sibling
            ``*_best_data_rigid.pth`` is required too.
        device: Torch device, or ``auto`` for CUDA when available, otherwise CPU.
        individual_muscles: Exact output muscle names to load; omitted loads all.
        batch_size: Maximum number of frames evaluated together.

    Returns:
        A frozen visual predictor. No licensed data or weights are downloaded.

    Raises:
        ImportError: Torch is unavailable.
        FileNotFoundError: Required user-provided assets are absent.
        ValueError: Unsupported dataset, topology, rig, or checkpoint format.
    """
    if subject not in {"S1", "S2", "S3", "S4", "S5"}:
        raise ValueError("SOMA subject must be S1, S2, S3, S4, or S5")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    root, checkpoint = Path(data_root), Path(checkpoint)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"SKIM metadata missing: {metadata_path}; obtain and extract SKIM v2 separately"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("dataset_version") != 2:
        raise ValueError(
            "Only SKIM dataset_version 2 with RELEASE_V2 checkpoints is supported"
        )
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "SOMA inference requires PyTorch; install a build for your CPU or CUDA platform"
        ) from error
    selected_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    base = root / subject
    with np.load(base / "canonical.npz", allow_pickle=False) as canonical:
        parents = canonical["parents"]
        if parents.dtype.kind != "i":
            raise ValueError("SKIM parents must be signed integer indices")
        parents = parents.astype(np.int64)
        names = _names(canonical["joint_names"])
        canonical_rest = canonical["j_rest"].astype(np.float32)
    rig_path = base / "canonical_model" / f"{subject}_rig.npz"
    with np.load(rig_path, allow_pickle=False) as rig:
        rest = rig["j_rest"].astype(np.float32)
        weights = rig["weights"].astype(np.float32)
        if _names(rig["joint_names"]) != names:
            raise ValueError("SKIM canonical and rig joint ordering disagree")
    joints = len(names)
    if (
        joints < 2
        or len(set(names)) != joints
        or any(not name for name in names)
        or parents.shape != (joints,)
        or parents[0] != -1
        or np.any(parents[1:] < 0)
        or np.any(parents[1:] >= np.arange(1, joints))
        or rest.shape != (joints, 3)
        or not np.isfinite(rest).all()
        or canonical_rest.shape != rest.shape
        or not np.allclose(rest, canonical_rest, atol=1e-5)
    ):
        raise ValueError("Invalid or mismatched SKIM v2 rest skeleton")
    layers = base / "layers" / "tpose"
    layer = _read_obj(
        layers / f"musc_layer-{subject}-TPose.obj", "muscle_layer", "layer"
    )
    skin = _read_obj(layers / f"skin_layer-{subject}-TPose.obj", "Skin", "skin")
    if len(layer.rest_vertices) != len(skin.rest_vertices) or not np.array_equal(
        layer.faces, skin.faces
    ):
        raise ValueError(
            "SKIM muscle and skin layers must share vertex ordering and topology"
        )
    if (
        weights.shape != (len(layer.rest_vertices), joints)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.allclose(weights.sum(1), 1, atol=1e-4)
    ):
        raise ValueError("Invalid SKIM rig skinning weights")
    files = sorted((layers / "muscle_meshes_tpose").glob("*.obj"))
    available = {path.stem.replace("_t-pose", ""): path for path in files}
    if not available or len(available) != len(files):
        raise ValueError(
            "SKIM individual muscle meshes are missing or have duplicate names"
        )
    requested = (
        list(available) if individual_muscles is None else list(individual_muscles)
    )
    if (
        not requested
        or len(set(requested)) != len(requested)
        or set(requested) - available.keys()
    ):
        raise ValueError(
            f"Unknown, duplicate, or empty SOMA muscle selection: {requested}"
        )
    meshes = tuple(_read_obj(available[name], name, "muscle") for name in requested)
    state = _state_dict(torch, checkpoint)
    rigid_path = checkpoint.with_name(f"{checkpoint.stem}_rigid{checkpoint.suffix}")
    rigid_state = _state_dict(torch, rigid_path)
    try:
        widths = (state["m_enc1.0.weight"].shape[0], state["m_enc2.0.weight"].shape[0])
        network = _make_network(torch, len(layer.rest_vertices), joints, widths)
        rigid = _make_rigid(torch, joints, rigid_state["net.0.weight"].shape[0])
        network.load_state_dict(state, strict=True)
        rigid.load_state_dict(rigid_state, strict=True)
    except (KeyError, IndexError, RuntimeError) as error:
        raise ValueError(
            "Checkpoint architecture does not match the SKIM v2 UNet/rigid rig"
        ) from error
    for module in (network, rigid):
        module.to(selected_device).eval().requires_grad_(False)
    return SOMAModel(
        torch,
        selected_device,
        network,
        rigid,
        names,
        parents,
        rest,
        weights,
        layer,
        skin,
        meshes,
        batch_size,
        {
            "dataset_version": 2,
            "subject": subject,
            "data_root": str(root.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "rigid_checkpoint": str(rigid_path.resolve()),
            "rig": str(rig_path.resolve()),
            "model": "RELEASE_V2",
            "device": str(selected_device),
        },
    )
