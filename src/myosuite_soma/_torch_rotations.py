# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Extracted from the audited MyoSuite SOMA integration; imports adapted.
# Source attribution: THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Minimal quaternion helpers reused verbatim from MyoSuite.

Source: MyoHub/myosuite, myosuite/physics/quat_math_torch.py.
The released inference path preserves its float32 quaternion convention and
zero-norm behavior. This module is imported only when Torch inference is loaded.
"""

from __future__ import annotations

import torch

_EPS = torch.finfo(torch.float32).eps


def axis_angle2quat(axis: torch.Tensor, angle: torch.Tensor | float) -> torch.Tensor:
    """Quaternion of a rotation by *angle* about the unit vector *axis*."""
    half = torch.as_tensor(angle, dtype=axis.dtype, device=axis.device)[..., None] / 2
    xyz = torch.sin(half) * axis
    w = torch.cos(half).expand(xyz.shape[:-1] + (1,))
    return torch.cat([w, xyz], dim=-1)


def quat2mat(quat: torch.Tensor) -> torch.Tensor:
    """Unit quaternion → rotation matrix."""
    quat = quat.to(torch.float32)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = torch.sum(quat * quat, dim=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = torch.empty(quat.shape[:-1] + (3, 3), dtype=torch.float32, device=quat.device)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)

    eye = torch.eye(3, dtype=torch.float32, device=quat.device)
    return torch.where(
        (Nq > _EPS)[..., None, None].expand_as(mat), mat, eye.expand_as(mat)
    )
