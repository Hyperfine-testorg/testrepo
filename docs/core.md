# rrdf-core

- Scope: serialization/deserialization of RRDF HDF5 files, schema helpers, and shared dimension labels (`ArrayLabels`).
- Dependencies: `h5py`, `numpy`, `xarray` only.
- API surface: `read_rrdf`, `load_dict_from_h5`, `ArrayLabels`, legacy helpers `get_dims_fromh5`, `get_nparray_dd`, `file_path`.
- Imports: `from rrdf_core import read_rrdf, ArrayLabels`.
- Tests: `pytest rrdf-core/tests`.
