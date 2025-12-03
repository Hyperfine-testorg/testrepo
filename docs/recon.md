# rrdf-recon

- Scope: reconstruction pipelines, transforms, saving helpers; depends on `rrdf-core`.
- Base dependencies: `numpy`, `xarray`, `imageio`, `numba`, `scipy`, `rrdf-core`.
- Optional extras:
  - `simpleitk` for echo registration (`pip install rrdf-recon[simpleitk]`).
  - `dev` for pytest.
- Public imports: `from rrdf_recon import xgrid, xsdc, prewhiten, coil_combine, reconstruct_image, orient_images, register_echo_fse`.
- CLI: `rrdf-recon path/to/file.h5` (expects base deps; registration features need `[simpleitk]`).
- Tests: light suite `pytest rrdf-recon/tests -m "not recon_data"`; full data exercise `pytest rrdf-recon/tests -m recon_data`.
