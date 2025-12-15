# rrdf-core

Lightweight core for reading and inspecting RRDF (research raw data format) files. It keeps dependencies minimal and only exposes serialization helpers, metadata access, and shared enum labels used across the broader RRDF ecosystem.

## Usage

```python
from rrdf_core import read_rrdf

raw_data, coords2d, data_drift, meta, is_deepdish = read_rrdf("path/to/file.h5")
```

## Provided APIs
- `read_rrdf` / `load_dict_from_h5`: load RRDF files into Python structures.
- `ArrayLabels`: shared dimension labels for downstream reconstruction.
- `get_dims_fromh5` / `get_nparray_dd`: helpers for legacy RRDF variants.
