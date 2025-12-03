# RRDF workspace

This repository now contains two packages with clear boundaries:

- **`rrdf-core`** – lightweight serialization, validation helpers, and shared data labels for RRDF files. Depends only on `h5py`, `numpy`, and `xarray`.
- **`rrdf-recon`** – optional reconstruction utilities and CLI built on top of `rrdf-core`. Pulls in heavier scientific dependencies (`scipy`, `numba`, `SimpleITK`, etc.).

## Install

```bash
# core-only (load/inspect)
pip install rrdf-core

# full reconstruction stack (installs rrdf-core automatically)
pip install rrdf-recon

# enable optional registration (SimpleITK)
pip install rrdf-recon[simpleitk]
```

## Quickstart

**Load RRDF metadata and arrays (core only)**
```python
from rrdf_core import read_rrdf

raw_data, coords2d, data_drift, meta, is_deepdish = read_rrdf("path/to/file.h5")
```

**Run reconstruction from the CLI (recon package)**
```bash
rrdf-recon data/10.1.0/fse_fast_t2_rrdf_20251105_162521.h5
```
Outputs default to `./output` as `.npy`/`.png` files.

**Programmatic reconstruction**
```python
from pathlib import Path
from rrdf_recon.cli import reconstruct_image

reconstruct_image(Path("data/11.0.0/fse_fast_t2_rrdf_20251105_163403.h5"), Path("output"))
```

## Compatibility

Use `rrdf_core` and `rrdf_recon` imports directly for all new code.

## Development

```bash
uv sync --group dev        # install shared dev tooling
uv pip install -e rrdf-core
uv pip install -e rrdf-recon

uv run pytest rrdf-core/tests                        # minimal core tests
uv run pytest rrdf-recon/tests -m "not recon_data"   # light recon tests (skip data-heavy)
uv run pytest rrdf-recon/tests -m recon_data         # full recon data exercise
```

## Versioning & releases

- `rrdf-core` and `rrdf-recon` are versioned independently (see their `pyproject.toml` files).
- Tags drive releases:
  - `core-vX.Y.Z` → builds/publishes `rrdf-core`
  - `recon-vX.Y.Z` → builds/publishes `rrdf-recon`
- Keep reconstruction imports limited to the public `rrdf_core` API to avoid cross-layer coupling.
