"""
Test suite for loading RRDF data from HDF5 files.
"""

import glob

import pytest

from rrdf_core import load


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
