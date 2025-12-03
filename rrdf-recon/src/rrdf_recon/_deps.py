"""Helpers for optional, heavy dependencies."""

from importlib import import_module
from types import ModuleType


def require_dependency(module_name: str, *, extra: str) -> ModuleType:
    """Import a dependency or raise a helpful error pointing to an extra."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
        raise ImportError(
            f"`{module_name}` is required for this functionality. "
            f"Install `rrdf-recon[{extra}]` to enable it."
        ) from exc


def optional_dependency(module_name: str) -> ModuleType | None:
    """Import a dependency if available, otherwise return None."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        return None
