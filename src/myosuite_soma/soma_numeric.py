# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under Apache-2.0.
# Extracted from the audited MyoSuite SOMA integration; imports adapted.
# Source attribution: THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Allocation and serialization guards for numeric SOMA archives."""

from __future__ import annotations

import math
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np


def inspect_numeric_archive(
    path: str | Path,
    *,
    max_bytes: int,
    allow_text: bool = False,
    max_members: int = 10000,
) -> None:
    """Validate ZIP/NPY headers before NumPy allocates or decompresses arrays.

    Args:
        path: NPZ archive containing only simple numeric arrays and optional text.
        max_bytes: Maximum total decompressed bytes including array headers.
        allow_text: Permit Unicode/byte arrays for explicit JSON metadata/names.
        max_members: Maximum number of arrays permitted in the archive.

    Raises:
        ValueError: Invalid members, object dtype, unsupported header, or shapes
            inconsistent with actual uncompressed bytes.
    """
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if (
                not entries
                or len(entries) > max_members
                or sum(entry.file_size for entry in entries) > max_bytes
                or len({entry.filename for entry in entries}) != len(entries)
            ):
                raise ValueError("Oversized or duplicated numeric archive arrays")
            for entry in entries:
                if (
                    not entry.filename.endswith(".npy")
                    or "/" in entry.filename
                    or "\\" in entry.filename
                ):
                    raise ValueError("Invalid numeric archive member")
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError("Unsupported NPY numeric archive version")
                    if dtype.hasobject:
                        raise ValueError(
                            "Invalid numeric array shape or dtype: "
                            "Object arrays are prohibited in numeric archives"
                        )
                    if (
                        dtype.kind not in ("fiuUS" if allow_text else "fiu")
                        or any(
                            not isinstance(size, int)
                            or isinstance(size, bool)
                            or size < 0
                            for size in shape
                        )
                        or math.prod(shape) * dtype.itemsize
                        != entry.file_size - stream.tell()
                    ):
                        raise ValueError("Invalid numeric array shape or dtype")
    except (OSError, EOFError, BadZipFile, RuntimeError, NotImplementedError) as error:
        raise ValueError(f"Invalid numeric archive: {path}") from error
