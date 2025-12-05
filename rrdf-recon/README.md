# rrdf-recon

Reconstruction utilities for RRDF (research raw data format) files. This package depends on `rrdf-core` for loading/metadata and layers in heavier scientific dependencies for NUFFT gridding, density compensation, coil combination, registration, and visualization helpers.

## Usage

Install with `pip install rrdf-recon` (which will pull in `rrdf-core`). Then:

```bash
rrdf-recon data/11.0.0/fse_fast_t2_rrdf_20251105_163403.h5
```

or programmatically:

```python
from rrdf_recon.cli import reconstruct_image

reconstruct_image("example.h5", "output-directory")
```

## Modules
- `xsdc`, `xgrid`: sampling density compensation and NUFFT gridding.
- `prewhiten`, `coil_combine`: noise whitening and coil combination utilities.
- `orient_images`, `register_echo_fse`: orientation and odd/even echo registration helpers.
- `save_images`: helpers for saving NPY/PNG outputs.

## Testing

- Lightweight CI-safe recon tests: `pytest rrdf-recon/tests -m "not recon_data"`
- Full data exercise (requires sample RRDF files): `pytest rrdf-recon/tests -m recon_data`
