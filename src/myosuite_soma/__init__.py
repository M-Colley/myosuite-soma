"""Pose-conditioned SOMA anatomy with optional MyoSuite integration."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("myosuite-soma")
except PackageNotFoundError:
    __version__ = "0.1.0a2"
