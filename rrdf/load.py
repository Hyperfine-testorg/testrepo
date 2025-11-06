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
    matches = glob.glob("**/*rrdf*.h5", recursive=True)

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
    with h5py.File(file_name, "r") as h5file:
        return _recursively_load_h5_group(h5file)


def _recursively_load_h5_group_old(group, path="/"):
    """
    Recursively loads a group from an HDF5 file into a dictionary and prints what's being loaded.
    :param group: h5py.Group or h5py.File
    :param path: Current path in the HDF5 file (used for printing)
    :return: Dictionary with the group contents
    """
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            print(f"=====Group item: {item}")
            result[key] = _recursively_load_h5_group(item, f"{path.rstrip('/')}/{key}")
        else:
            print(f"=====item: {item}")
            # if this dataset holds strings, decode via asstr()
            if h5py.check_string_dtype(item.dtype) is not None:
                value = item.asstr()[()]  # scalar or array of str
            else:
                value = item[()]  # numeric / other binary data
            result[key] = value
    return result


def _recursively_load_h5_group(group, path="/"):
    """
    Recursively loads a group from an HDF5 file into a dictionary and prints what's being loaded.
    """
    result = {}
    # First, check if this group has attributes (deepdish stores scalar strings here)
    if len(group.items()) == 0 and len(group.attrs) > 0:
        # Try to extract attributes as Python values
        for k, v in group.attrs.items():
            if isinstance(v, bytes):
                v = v.decode()
            result[k] = v
        return result

    # Otherwise, walk through items
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = _recursively_load_h5_group(item, f"{path.rstrip('/')}/{key}")
        else:
            # Handle dataset reading (string or numeric)
            if h5py.check_string_dtype(item.dtype) is not None:
                value = item.asstr()[()]
            else:
                value = item[()]
            result[key] = value

    # Optionally capture group attributes too
    for attr_key, attr_val in group.attrs.items():
        if isinstance(attr_val, bytes):
            attr_val = attr_val.decode()
        result[attr_key] = attr_val

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
    try:
        sw_ver = rrdf["process_info"]["acquisition_software_version"]
    except KeyError:
        sw_ver = None
    try:
        deepdish_version = rrdf["process_info"]["deepdish_version"]
        print(f"deepdish version: {deepdish_version}, sw_version: {sw_ver}")
        is_deepdish = True
    except KeyError:
        deepdish_version = None
        is_deepdish = False
        pass

    data_matrix_dims = get_dims_fromh5(data, "data_matrix_dims", is_deepdish)
    kspace_dims = get_dims_fromh5(data, "kspace_coordinate_pe_dims", is_deepdish)
    # kspace_dims = eval(data["kspace_coordinate_pe_dims"])
    raw_data_cleaned = xr.DataArray(
        np.array(data["data_matrix"]), dims=data_matrix_dims
    )
    coords2d = xr.DataArray(np.array(data["kspace_coordinate_pe"]), dims=kspace_dims)
    data_drift = None
    if (
        data.get("data_drift", None) is not None
        and data.get("data_drift_dims", None) is not None
    ):
        drift_dims = get_dims_fromh5(data, "data_drift_dims", is_deepdish)
        data_drift = xr.DataArray(np.array(data["data_drift"]), dims=drift_dims)

    return raw_data_cleaned, coords2d, data_drift, rrdf, is_deepdish


def get_dims_fromh5(data, key, is_deepdish=False):
    """deepdish was used up through 10.1 SW. It saves lists differently from h5py."""
    dims_group = data[key]
    if is_deepdish and "CLASS" in dims_group:  # <= 10.1 beta SW
        print(f"dims_group = {dims_group}")
        dimlist = []
        for k in dims_group.keys():
            if k.startswith("i") and k[1].isdigit():
                dimlist.append(str(dims_group[k]))
    else:  # >= 11.0 SW
        dimlist = list(dims_group)
    return dimlist


def get_nparray_dd(arr):
    """
    deepdish stores arrays differently
    example: imsize {'CLASS': 'GROUP', 'TITLE': 'list:3', 'VERSION': '1.0', 'i0': np.int64(138), 'i1': np.int64(112), 'i2': np.int64(40)}
    """
    arr_out = []
    for k in arr.keys():
        if k.startswith("i") and k[1].isdigit():
            arr_out.append(arr[k])
    return np.asarray(arr_out)


class ArrayLabels(str, Enum):
    """Provides consistent labels for all recon methods."""

    CHAN = "CHAN"  # receive coils (signal + noise)
    READOUT = "READOUT"  # readout gradient encoded k-space lines
    PHASE = "PHASE"  # phase encoded k-space lines
    LINE = "LINE"  # before phases are organized into kdata they are just lines
    LOOP = "LOOP"  # for ordering the input data or vectorizing arrays
    GROUP = "group"  # the signpost channel contains 'group' info;
    SLICE = "SLICE"  # slices in an image volume
    X = "X"  # x dimension of an image volume
    Y = "Y"  # y dimension of an image volume
    Z = "Z"  # z dimension of an image volume
    R = "R"  # R direction of an image volume (RSA/LIP)
    L = "L"  # L direction of an image volume (RSA/LIP)
    S = "S"  # S direction of an image volume (RSA/LIP)
    I = "I"  # I direction of an image volume (RSA/LIP) # noqa: E741
    A = "A"  # P direction of an image volume (RSA/LIP)
    P = "P"  # P direction of an image volume (RSA/LIP)
    COORD = "COORD"  # euclidean coordinates (e.g. kx, ky)
    FEATURE = "FEATURE"  # a generic feature dimension (could represent anything)
    SAMPLES = "SAMPLES"  # a generic dim for unstructured samples (e.g. noise)
    COMPONENT = "COMPONENT"  # the dimension for data decompositions
    ECHO_NUMBER = "ECHO_NUMBER"  # the number of the echo in an echo train

    # recon labels
    # according to `pyHFS.recon.node.index.collector.get_subset_order`
    AVG = "AVG"  # averages
    AXIS = "AXIS"  # axis of the gradient measurement
    DEFAULT = "DEFAULT"  # unassigned loop dimension
    DELAY = "DELAY"
    DW = "DW"  # Diffusion weighting volume
    ECHO = "ECHO"  # reconstructed volume #. may be comprised of multiple echoes (ie even/odd echo volumes in FSE)
    ENC = "ENC"  # generic adc encoding step
    FLIP = "FLIP"  # flip and loop
    FREQ = "FREQ"  # SSFP frequency loop
    NOISE = "NOISE"  # a noise loop within a sequence
    OFFON = "OFFON"  # condition for gradient measurements on=measure, off=ref with 0 grad eg
    REPS = "REPS"  # repetitions
    VOLT = "VOLT"  # Voltage loop for coil tuning
    SWITCH = "SWITCH"
    PARAM1 = "PARAM1"  # Generic parameter
