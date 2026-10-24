"""Raw cache headers and allocation budgets are checked before NumPy loading."""

import io
from zipfile import ZipFile

import numpy as np
import pytest

from myosuite_soma import soma
from .test_soma_cache import _animation


def test_raw_cache_rejects_forged_array_shape_before_numpy_load(tmp_path, monkeypatch):
    path = tmp_path / "forged.npz"
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream, {"descr": "<f4", "fortran_order": False, "shape": (2**34, 3)}
    )
    with ZipFile(path, "w") as archive:
        archive.writestr("mesh_0000_points.npy", stream.getvalue())

    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy must not receive a forged allocation request")

    monkeypatch.setattr(soma.np, "load", forbidden)
    with pytest.raises(ValueError, match="shape or dtype"):
        soma.load_animation(path)


def test_raw_cache_budget_is_uncompressed_size_and_can_be_explicitly_raised(tmp_path):
    path = tmp_path / "valid.npz"
    original = _animation()
    soma.save_animation(path, original)
    with ZipFile(path) as archive:
        decoded_bytes = sum(info.file_size for info in archive.infolist())
    assert path.stat().st_size < decoded_bytes
    with pytest.raises(ValueError, match="Oversized"):
        soma.load_animation(path, max_bytes=decoded_bytes - 1)
    loaded = soma.load_animation(path, max_bytes=decoded_bytes)
    np.testing.assert_array_equal(loaded.meshes[0].points, original.meshes[0].points)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1000"])
def test_raw_cache_rejects_invalid_allocation_limit_before_opening(tmp_path, limit):
    with pytest.raises(ValueError, match="max_bytes"):
        soma.load_animation(tmp_path / "missing.npz", max_bytes=limit)
