"""
Load RRDF (research raw data format) data from HDF5 files.
"""
import xarray as xr
import numpy as np
import glob
import h5py
from enum import Enum


def file_path():
    """
    Returns the path to the HDF5 file containing RRDF data.
    """
    # Search recursively in current directory and subdirectories
    matches = glob.glob('**/*rrdf*.h5', recursive=True)

    # Save the first match as a string (or handle if none found)
    file_path = matches[0] if matches else None
    assert file_path is not None, "No rrdf file found for testing."
    print(f"Found rrdf file: {file_path}")
    return file_path


def load_dict_from_h5(file_name):
    """
    Loads a dictionary from an HDF5 file.
    :param file_name: HDF5 file name to load
    :return: Dictionary with the loaded data
    """
    with h5py.File(file_name, 'r') as h5file:
        return _recursively_load_h5_group(h5file)


def _recursively_load_h5_group(group, path='/'):
    """
    Recursively loads a group from an HDF5 file into a dictionary and prints what's being loaded.
    :param group: h5py.Group or h5py.File
    :param path: Current path in the HDF5 file (used for printing)
    :return: Dictionary with the group contents
    """
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = _recursively_load_h5_group(
                item, f"{path.rstrip('/')}/{key}"
            )
        else:
            # if this dataset holds strings, decode via asstr()
            if h5py.check_string_dtype(item.dtype) is not None:
                value = item.asstr()[()]    # scalar or array of str
            else:
                value = item[()]            # numeric / other binary data
            result[key] = value
    return result


def read_rrdf(rrdf_path: str):
    """
    Read RRDF file and reconstruct raw_data_cleaned, coords2d, data_drift.
    Returns:
        raw_data_cleaned: xr.DataArray
        coords2d: xr.DataArray
        data_drift: xr.DataArray | None
        rrdf: dict
    """
    rrdf = load_dict_from_h5(rrdf_path)
    # --- Rebuild DataArrays ---
    data = rrdf["data"]
    # dims are stored as string repr of list of enum values, so eval is used
    data_matrix_dims = data["data_matrix_dims"]
    kspace_dims = data["kspace_coordinate_pe_dims"]
    raw_data_cleaned = xr.DataArray(np.array(data["data_matrix"]), dims=data_matrix_dims)
    coords2d = xr.DataArray(np.array(data["kspace_coordinate_pe"]), dims=kspace_dims)
    data_drift = None
    if data.get("data_drift", 0) is not None:
        drift_dims = data["data_drift_dims"]
        data_drift = xr.DataArray(np.array(data["data_drift"]), dims=drift_dims)

    return raw_data_cleaned, coords2d, data_drift, rrdf


class ArrayLabels(str, Enum):
    """Provides consistent labels for all recon methods."""

    CHAN = 'CHAN'  # receive coils (signal + noise)
    READOUT = 'READOUT'  # readout gradient encoded k-space lines
    PHASE = 'PHASE'  # phase encoded k-space lines
    LINE = 'LINE'  # before phases are organized into kdata they are just lines
    LOOP = 'LOOP'  # for ordering the input data or vectorizing arrays
    GROUP = 'group'  # the signpost channel contains 'group' info;
    SLICE = 'SLICE'  # slices in an image volume
    X = 'X'  # x dimension of an image volume
    Y = 'Y'  # y dimension of an image volume
    Z = 'Z'  # z dimension of an image volume
    R = 'R'  # R direction of an image volume (RSA/LIP)
    L = 'L'  # L direction of an image volume (RSA/LIP)
    S = 'S'  # S direction of an image volume (RSA/LIP)
    I = 'I'  # I direction of an image volume (RSA/LIP) # noqa: E741
    A = 'A'  # P direction of an image volume (RSA/LIP)
    P = 'P'  # P direction of an image volume (RSA/LIP)
    COORD = 'COORD'  # euclidean coordinates (e.g. kx, ky)
    FEATURE = 'FEATURE'  # a generic feature dimension (could represent anything)
    SAMPLES = 'SAMPLES'  # a generic dim for unstructured samples (e.g. noise)
    COMPONENT = 'COMPONENT'  # the dimension for data decompositions
    ECHO_NUMBER = 'ECHO_NUMBER'  # the number of the echo in an echo train

    # recon labels
    # according to `pyHFS.recon.node.index.collector.get_subset_order`
    AVG = 'AVG'  # averages
    AXIS = 'AXIS'  # axis of the gradient measurement
    DEFAULT = 'DEFAULT'  # unassigned loop dimension
    DELAY = 'DELAY'
    DW = 'DW'  # Diffusion weighting volume
    ECHO = 'ECHO'  # reconstructed volume #. may be comprised of multiple echoes (ie even/odd echo volumes in FSE)
    ENC = 'ENC'  # generic adc encoding step
    FLIP = 'FLIP'  # flip and loop
    FREQ = 'FREQ'  # SSFP frequency loop
    NOISE = 'NOISE'  # a noise loop within a sequence
    OFFON = 'OFFON'  # condition for gradient measurements on=measure, off=ref with 0 grad eg
    REPS = 'REPS'  # repetitions
    VOLT = 'VOLT'  # Voltage loop for coil tuning
    SWITCH = 'SWITCH'
    PARAM1 = 'PARAM1'  # Generic parameter
