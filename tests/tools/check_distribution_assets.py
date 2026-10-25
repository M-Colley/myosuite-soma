"""Reject data/model/media members in new Python distribution archives."""

from pathlib import Path
import sys
import tarfile
import zipfile


def main():
    forbidden = {
        ".npz",
        ".npy",
        ".pth",
        ".pt",
        ".obj",
        ".glb",
        ".gltf",
        ".skn",
        ".stl",
        ".ply",
        ".mjb",
        ".mjcf",
        ".b3d",
        ".c3d",
        ".docx",
        ".xlsx",
        ".mat",
        ".nrrd",
        ".csv",
        ".jpg",
        ".jpeg",
        ".png",
        ".mp4",
        ".webm",
        ".blend",
    }
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check_distribution_assets.py DIST_DIRECTORY")
    artifacts = sorted(Path(sys.argv[1]).glob("*.whl")) + sorted(
        Path(sys.argv[1]).glob("*.tar.gz")
    )
    if len(artifacts) != 2:
        raise SystemExit("Expected exactly one wheel and one source distribution")
    for artifact in artifacts:
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                names = archive.namelist()
        else:
            with tarfile.open(artifact) as archive:
                names = archive.getnames()
        bad = [
            name
            for name in names
            if Path(name).suffix.lower() in forbidden
            or "local-assets/" in name
            or "physical-output/" in name
        ]
        if bad:
            raise SystemExit(f"Unexpected model/data/media in {artifact.name}: {bad}")
        print(f"{artifact.name}: {len(names)} members, no data/model/media files")


if __name__ == "__main__":
    main()
