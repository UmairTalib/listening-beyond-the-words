"""Counterfactual paralinguistic gain (CPG) training for depression detection with audio LLMs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("listening-beyond-words")
except PackageNotFoundError:  # a source tree that is not installed
    __version__ = "0.0.0+unknown"
