from __future__ import annotations

from pathlib import Path


def default_output_dir() -> Path:
    """Return the default output directory for recon outputs.

    Repo layout assumption:
      <repo_root>/rrdf-recon/src/rrdf_recon/output_dir.py
    So `parents[3]` is the repo root (rrdf).
    """

    here = Path(__file__).resolve()
    try:
        # .../rrdf/rrdf-recon/src/rrdf_recon/output_dir.py -> .../rrdf
        return here.parents[3] / "output"
    except IndexError:
        # Fallback for unexpected layouts.
        return here.parent / "output"

