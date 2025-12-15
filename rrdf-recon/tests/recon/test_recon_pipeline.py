from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import shift

from rrdf_core import load
from rrdf_recon.xgrid import xgrid
from rrdf_recon.xsdc import xsdc


DATA_FILES = [
    Path("data/10.1.0/fse_fast_t2_rrdf_20251105_162521.h5"),
    Path("data/11.0.0/fse_fast_t2_rrdf_20251105_163403.h5"),
]


@pytest.mark.recon_data
@pytest.mark.parametrize("file_path", DATA_FILES)
def test_recon_from_rrdf(file_path: Path) -> None:
    if not file_path.exists():
        pytest.skip(f"Test data not present: {file_path}")

    raw_data_cleaned, coords2d, data_drift, rrdf_dict, is_deepdish = load.read_rrdf(
        str(file_path)
    )
    if data_drift is None:
        pytest.skip("Sample file does not include navigator data for comparison.")
    if load.ArrayLabels.CHAN not in raw_data_cleaned.dims:
        raw_data_cleaned = raw_data_cleaned.expand_dims(
            dim=load.ArrayLabels.CHAN, axis=0
        )  # add dummy channel dimension is missing for xgrid
    imsize = rrdf_dict["sequence"]["matrix_size"]  # [r p s]
    partial_fourier = rrdf_dict["sequence"]["partial_fourier"]  # [r p s]
    if is_deepdish:
        imsize = load.get_nparray_dd(imsize)
        partial_fourier = load.get_nparray_dd(partial_fourier)
    sdc_weights = xsdc(coords2d, imsize)

    img_recon = xgrid(
        kdata=raw_data_cleaned,
        coords=coords2d,
        weights=sdc_weights,
        res=imsize,
        partial_fourier=partial_fourier,
    )

    img_recon.data = shift(
        np.rot90(img_recon.data, k=2, axes=(2, 3)), (0, 0, 1, 1)
    )  # rotation and small shift between nufft and xgrid?
    epsilon = 1e-9
    ratio = np.abs(img_recon.data) / (np.abs(data_drift.data) + epsilon)
    ratio = ratio[np.isfinite(ratio)]
    scale_factor = np.median(ratio)

    recon_error = np.abs(img_recon / scale_factor) - data_drift.data
    median_error = np.median(recon_error)
    assert median_error < 0.005, f"Reconstruction error too high: {median_error:.6f}"
