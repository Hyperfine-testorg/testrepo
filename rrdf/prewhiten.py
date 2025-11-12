"""
Compute the noise covariance matrix and the prewhitening matrix from noise data - taken from k-space borders.
"""

import numpy as np
import xarray as xr

from rrdf.load import ArrayLabels as lbl


def prewhiten(data: xr.DataArray, coords: xr.DataArray) -> xr.DataArray:
    """Uses data from weighted outer kspace samples to get prewhitening matrix."""
    n_channels = data.sizes[lbl.CHAN]
    if lbl.REPS in data.dims:
        data_sum = data[{lbl.CHAN: 0}].sum(
            dim=(
                (lbl.READOUT, lbl.ENC)
                if lbl.ECHO not in data.dims
                else (lbl.READOUT, lbl.ENC, lbl.ECHO)
            )
        )
        max_rep = np.max(np.nonzero(data_sum.data))

        data = data[{lbl.REPS: max_rep}]
        coords = coords[{lbl.REPS: max_rep}]

    # make channel dimension first
    # get all volumes dimension names (eg: lbl.ECHO, lbl.AVG, ...)
    volume_dims = [_ for _ in data.dims if _ not in [lbl.CHAN, lbl.READOUT, lbl.ENC]]
    # flatten all volume dimensions with lbl.ENC
    data_volflat = data.stack({lbl.SAMPLES: volume_dims + [lbl.ENC]}).transpose(
        lbl.CHAN, lbl.READOUT, lbl.SAMPLES
    )
    crds_volflat = coords.stack({lbl.SAMPLES: volume_dims + [lbl.ENC]}).transpose(
        lbl.COORD, lbl.SAMPLES
    )

    # Use k-space weighting to derive a "noise" dataset
    kr_pe = np.sqrt((abs(crds_volflat) ** 2).sum(lbl.COORD)).data
    kr_ro = np.abs(np.linspace(-1, 1, data.sizes[lbl.READOUT]))
    kw = compute_kspace_weights(kr_pe, kr_ro, n_channels=n_channels)

    kw = kw.astype(np.float32)  # to ensure it's compatible with complex64
    kw = xr.DataArray(kw, dims=(lbl.CHAN, lbl.READOUT, lbl.SAMPLES))

    # get noise samples to compute covariance matrix
    noise_samples_da = kw * data_volflat
    noise_samples = np.reshape(noise_samples_da.data, (n_channels, -1))

    # compute covariance matrix
    covariance_matrix = np.dot(noise_samples, np.conj(noise_samples).T)

    # normalize by number of noise samples
    covariance_matrix = (1.0 / (noise_samples.shape[1] - 1)) * covariance_matrix

    # get prewhitening matrix from covariance matrix using Cholesky decomposition
    try:
        prewhitening_matrix = np.linalg.inv(np.linalg.cholesky(covariance_matrix))
    except np.linalg.LinAlgError:
        print("Singular covariance matrix, returning identity matrix")
        prewhitening_matrix = np.eye(n_channels)

    # apply prewhitening
    # Put channels dimension first
    dims_init = data.dims
    extra_dims = [d for d in data.dims if d != lbl.CHAN]
    dims_chan_first = [lbl.CHAN] + extra_dims
    data_pw = data.transpose(*dims_chan_first)

    # Stack all non-channel dimensions into 1D
    data_pw = data_pw.stack(samp=extra_dims)
    data_pw.data = np.dot(prewhitening_matrix, data_pw.data)

    # Put back in the original dimension order
    data_pw = data_pw.unstack("samp")

    # Remove the coords that were added by default by unstack
    data_pw = xr.DataArray(data_pw.data, dims=data_pw.dims)

    # Put back the dimension order
    data_pw = data_pw.transpose(*dims_init)
    return data_pw


def compute_kspace_weights(
    kr_pe=None,
    kr_readout=None,
    n_channels=None,
    threshold=None,
    var_readout=None,
    wgt_readout_factor=2.0,
    wgt_readout_power=0.5,
):
    """Compute signal weighing for a 3D Cartesian multichannel acquisision, given 2D k-space coordinates
    :param kr_pe: phase-encoding coords radius
    :param kr_readout: readout coords radius
    :param n_channels:
    :param threshold:
    :param var_readout: optional index for alternate-polarity readouts (eg: multi echo GRE or symmetric spiral navs)
    :param wgt_readout_factor:
    :param wgt_readout_power:
    :return: data weights as dimension (lines, channels, readout)
    """
    if kr_pe is None and kr_readout is None and n_channels is None:
        return 1.0
    if kr_pe is None:
        kr_pe = [0.0, 0.0]
    if kr_readout is None:
        kr_readout = 0.0
    if n_channels is None:
        n_channels = 1

    # Weight function in readout
    max_k = np.max(kr_readout)
    if max_k > 0:
        kr_readout /= max_k
    if isinstance(threshold, float):
        weight_readout = np.array(kr_readout > threshold, dtype=float)
    else:
        weight_readout = 1 - np.exp(-wgt_readout_factor * kr_readout**wgt_readout_power)

    weight_readout = weight_readout / np.max(weight_readout)

    # Weight function on line encoding
    max_k = np.max(kr_pe)
    if max_k > 0:
        kr_pe /= max_k
    if np.count_nonzero(kr_pe) == 0:
        # if no phase encoding, weight is set to 1 for all
        weight_pe = np.ones_like(kr_pe)
    elif isinstance(threshold, float):
        weight_pe = np.array(kr_pe > threshold, dtype=float)
    else:
        weight_pe = 1 - np.exp(-2.5 * kr_pe**2)

    # Combine all weights
    if weight_readout.ndim == 2:
        # Alternating polarity readouts
        assert var_readout is not None, "Missing index for readout polarity"
        assert weight_readout.shape[1] == 2
        weights_all = np.zeros((weight_readout.shape[0], weight_pe.shape[0]))
        for k in range(weight_readout.shape[1]):
            sel_var_readout = np.array(var_readout, dtype=int) % 2 == k
            weights_all[:, sel_var_readout] = np.dot(
                weight_readout[:, k][:, np.newaxis],
                weight_pe[sel_var_readout][:, np.newaxis].T,
            )
    else:
        # just one readout
        weights_all = np.dot(weight_readout[:, np.newaxis], weight_pe[:, np.newaxis].T)

    # Add multi-channel dimension to weights (identical weights on all channels)
    weights_all = np.tile(weights_all[:, :, np.newaxis], n_channels)
    # Set the weights as dimensions (channels, readout, lines)
    k_weights = np.transpose(weights_all, [2, 0, 1])

    return k_weights
