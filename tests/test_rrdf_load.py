"""
Test suite for loading RRDF data from HDF5 files.
"""

import glob
import numpy as np
import pytest

from rrdf.xsdc import xsdc
from rrdf.xgrid import xgrid
from rrdf import load
from scipy.ndimage import shift


# --- Tests --- #
def get_test_file_path():
    """
    Returns the path to the HDF5 file containing RRDF data.
    """
    # Search recursively in current directory and subdirectories
    matches = glob.glob("**/*rrdf*.h5", recursive=True)
    file_path = matches[0] if matches else None
    assert file_path is not None, "No rrdf file found for testing."
    print(f"Found rrdf file: {file_path}")
    return file_path


@pytest.fixture
def file_path():
    """Provide rrdf .h5 path to tests, or skip when unavailable."""
    return get_test_file_path()


def test_load_dict_from_h5(file_path):
    """
    Test loading a dictionary from an HDF5 file. outside pyHFS context.
    """
    # Load the dictionary from the HDF5 file
    load_dict_from_h5 = load.load_dict_from_h5(file_path)

    # Check if the loaded dictionary is a dictionary
    assert isinstance(load_dict_from_h5, dict), "Loaded data is not a dictionary."

    assert "data" in load_dict_from_h5, "Key 'data' not found in loaded dictionary."

    # Check if the dictionary has expected keys (example keys, adjust as needed)
    expected_keys = [
        "data_drift",
        "data_matrix",
        "kspace_coordinate_pe",
        "data_drift_dims",
        "data_matrix_dims",
        "kspace_coordinate_pe_dims",
    ]
    for key in expected_keys:
        assert key in load_dict_from_h5["data"], (
            f"Key '{key}' not found in loaded dictionary."
        )


def test_recon_from_rrdf(file_path):
    """
    Test loading RRDF and reconstructing image outide pyHFS context.
    """
    raw_data_cleaned, coords2d, data_drift, rrdf = load.read_rrdf(file_path)
    raw_data_cleaned = raw_data_cleaned.expand_dims(
        dim="CHAN", axis=0
    )  # add dummy channel dimension is missing for xgrid
    imsize = rrdf["sequence"]["matrix_size"]  # [r p s]
    partial_fourier = rrdf["sequence"][
        "partial_fourier"
    ]  # [r p s], 0-1, where 1 = fully sampled dimension
    sdc_weights = xsdc(coords2d, rrdf["sequence"]["matrix_size"])
    print("\nReconstructing images")
    img_recon = xgrid(
        kdata=raw_data_cleaned,
        coords=coords2d,
        weights=sdc_weights,
        res=imsize,
        partial_fourier=partial_fourier,
    )

    img_recon.data = shift(
        np.rot90(img_recon.data, k=2, axes=(2, 3)), (0, 0, 1, 1)
    )  # there seems to be a rotation and small shift between nufft and xgrid?
    # There's an expected scaling factor due to the non uniform sampling density
    ratio = np.abs(img_recon.data) / np.abs(data_drift.data)
    ratio = ratio[np.isfinite(ratio)]
    scale_factor = np.median(ratio)
    print(f"Scale factor: {scale_factor:.3f}")

    recon_error = np.abs(img_recon / scale_factor) - data_drift.data
    median_error = np.median(recon_error)
    print(f"Median error: {median_error:.6f}")
    assert median_error < 0.005, f"Reconstruction error too high: {median_error:.6f}"
