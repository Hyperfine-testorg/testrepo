"""Reconstruction utilities layered on top of rrdf-core."""

from importlib import import_module
from typing import Any

from rrdf_core.load import ArrayLabels

__all__ = [
    "ArrayLabels",
    "coil_combine",
    "numpy_save",
    "orient_images",
    "orientation_from_scanplane",
    "prewhiten",
    "reconstruct_image",
    "register_echo_fse",
    "save_png",
    "write_png",
    "xgrid",
    "xsdc",
]

_MODULE_MAP = {
    "coil_combine": "rrdf_recon.coil_combine",
    "numpy_save": "rrdf_recon.save_images",
    "orient_images": "rrdf_recon.orient_images",
    "orientation_from_scanplane": "rrdf_recon.orient_images",
    "prewhiten": "rrdf_recon.prewhiten",
    "reconstruct_image": "rrdf_recon.cli",
    "register_echo_fse": "rrdf_recon.register_echo_fse",
    "save_png": "rrdf_recon.save_images",
    "write_png": "rrdf_recon.save_images",
    "xgrid": "rrdf_recon.xgrid",
    "xsdc": "rrdf_recon.xsdc",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
