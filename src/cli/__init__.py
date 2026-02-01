"""CLI modules for AutoQA."""

from .live import LiveTestRunner
from .mock import MockTestRunner

__all__ = ["LiveTestRunner", "MockTestRunner"]
