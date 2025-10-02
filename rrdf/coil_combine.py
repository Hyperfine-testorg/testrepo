"""Coil Combination"""

import itertools
from collections.abc import Sequence
from math import prod

import numpy as np
import xarray as xr
from numba import (
    c8,
    njit,
)
from scipy.interpolate import interpn

from rrdf.load import ArrayLabels as lbl


def coil_combine(
    data: xr.DataArray,
    encoding_resolution: tuple,
    dims_combine: Sequence[lbl] = (lbl.CHAN,),
) -> tuple[xr.DataArray, xr.DataArray]:
    """adaptive coil combination"""
    dims = data.dims
    dims_3d = (
        (lbl.X, lbl.Y, lbl.Z) if lbl.X in dims else (lbl.READOUT, lbl.PHASE, lbl.SLICE)
    )
    assert all(_ in dims for _ in dims_3d)

    # remove mean phase from all channels
    data = remove_mean_phase(data)
    data_image = data.copy()

    # input check and massage
    data = data.transpose(..., *dims_combine, *dims_3d)
    dims_out = data.dims  # must be after the transpose

    ndim = len(dims_3d)  # dimension of image
    ndimV = len(dims_combine) + ndim  # dimension of a volume

    shape = data.shape
    image = np.ascontiguousarray(data.data)
    ib1map = np.empty_like(image)  # "inverse" of b1map

    img_shape = image.shape[-ndim:]

    nV = prod(shape[:-ndimV])
    nC = prod(shape[-ndimV:-ndim])

    image_view = image.view()
    image_view.shape = (nV, nC, *img_shape)

    ib1map_view = ib1map.view()
    ib1map_view.shape = (nV, nC, *img_shape)

    NOMINAL_RES = [12, 12, 20]  # block size in [mm]

    # compute the block size
    psize = [
        round(r_nom / r)
        for (r_nom, r) in zip(NOMINAL_RES, encoding_resolution, strict=False)
    ]
    psize_min = [6, 6, 4]
    # size needs to be even
    psize = [
        max(round(p / 2) * 2, p_min)
        for (p, p_min) in zip(psize, psize_min, strict=False)
    ]

    def get_ib1map(img: np.ndarray) -> np.ndarray:
        """img (nimg_to_combine, *Nd)"""
        ib1map = compute_b1_adaptive(
            img,
            size_patch=psize,
            step_size=2,
        )
        return ib1map

    # "inverse" b1maps for every volume
    for iV in range(nV):
        ib1map_view[iV] = get_ib1map(image_view[iV])

    # outputs
    ib1maps_xr = xr.DataArray(ib1map, dims=dims_out)

    # all dim
    odin = set(ib1maps_xr.dims + data_image.dims) - set(dims_combine)

    # `squeeze`` cause xr.DataArray does NOT surport singleton broadcast.
    # apply the b1 maps
    img_combined: xr.DataArray = xr.dot(
        ib1maps_xr.squeeze(),
        data_image.squeeze(),
        dims=dims_combine,
    )
    # `expand_dims` undo the `squeeze` to add back singleton dimensions.
    img_combined = img_combined.expand_dims(tuple(odin - set(img_combined.dims)))

    return img_combined, ib1maps_xr


def remove_mean_phase(img: xr.DataArray) -> xr.DataArray:
    """Remove mean phase separately for each echo and coil.

    Args:
        img: DataArray with dims including lbl.CHAN and lbl.ECHO.

    Returns:
        DataArray with the same shape, mean phase removed per coil/echo.
    """
    img_out = xr.zeros_like(img)

    for echo_val in img[lbl.ECHO].values:
        for chan_val in img[lbl.CHAN].values:
            # Select the (echo, chan) slice
            slice_img = img.sel({lbl.ECHO: echo_val, lbl.CHAN: chan_val})

            # Smooth the phase reference for this echo/coil
            meanphs = blur(np.squeeze(slice_img.data), filter_fac=0.05)

            # Apply correction
            phs_corr = np.exp(-1j * np.angle(meanphs))
            img_out.loc[{lbl.ECHO: echo_val, lbl.CHAN: chan_val}] = slice_img * phs_corr

    return img_out


def compute_b1_adaptive(
    data_in: np.ndarray,
    cov_noise=None,
    size_patch=(8, 8, 4),
    step_size=2,
    # threads=5
):
    """Extract complex 3D coil sensitivity maps from data.

    Reference: Reconstruction of array data and computation of coil sensitivities based
    on: a) Adaptive Reconstruction of MRI array data, Walsh et al. Magn Reson Med. 2000; 43(5):682-90
    and b) Griswold et al. ISMRM 2002: 2410

    :param data_in: nd array of dimensions (nc, nx, ny, nz) with nc number of channels
    :param cov_noise: covariance matrix of noise-only data of dimensions (nc, nc)
    :param size_patch: 3d tuple
    :param step_size: int interpolation step size
    :param threads: Number of threads for parallel filter calculation
    :return: b1 maps of same size as data_in
    """
    data_coil = data_in.copy()
    while data_coil.ndim < 4:
        data_coil = data_coil[..., np.newaxis]
    # Compute patch dimensions
    size_im = data_coil.shape[1:]
    n_channel = data_coil.shape[0]
    size_patch = np.array(size_patch)
    size_patch[np.array(size_im) <= size_patch] = step_size

    if cov_noise is None:
        cov_noise = np.eye(n_channel, dtype=np.complex64)
        cov_noise_inv = np.eye(n_channel, dtype=np.complex64)
    else:
        cov_noise_inv = np.linalg.inv(cov_noise)

    # Pad input so patches are centered
    pad_width = (
        (0, 0),
        (size_patch[0] // 2, size_patch[0] // 2 - 1),
        (size_patch[1] // 2, size_patch[1] // 2 - 1),
        (size_patch[2] // 2, size_patch[2] // 2 - 1),
    )
    data_padded = np.pad(data_coil, pad_width, mode="constant")

    # Number of patches in each dimension
    nd_patch = tuple((size_im[d] + step_size - 1) // step_size for d in range(3))

    # Extract patches (manual replacement for view_as_windows)
    data_patches = np.zeros(
        nd_patch + (n_channel, np.prod(size_patch)), dtype=data_coil.dtype
    )
    for ix, iy, iz in itertools.product(
        range(nd_patch[0]), range(nd_patch[1]), range(nd_patch[2])
    ):
        x0, y0, z0 = ix * step_size, iy * step_size, iz * step_size
        patch = data_padded[
            :,
            x0 : x0 + size_patch[0],
            y0 : y0 + size_patch[1],
            z0 : z0 + size_patch[2],
        ]
        data_patches[ix, iy, iz] = patch.reshape(n_channel, -1)

    # Compute adaptive filters
    b1_patch = np.zeros((n_channel,) + nd_patch, dtype=np.complex64)
    for xyz_coord in itertools.product(
        range(nd_patch[0]), range(nd_patch[1]), range(nd_patch[2])
    ):
        result = _get_filter_for_patch(
            xyz_coord,
            data_patch=data_patches,
            cov_noise=cov_noise,
            cov_noise_inv=cov_noise_inv,
        )
        b1_patch[:, xyz_coord[0], xyz_coord[1], xyz_coord[2]] = result

    # Interpolate patch b1 to image size
    b1_map = grid_interpn(b1_patch, size_im, method="nearest")

    return np.conj(
        b1_map
    )  # conjugate to be consistent with CombineCoil node definition


@njit([c8[::](c8[:, ::1], c8[:, ::1], c8[:, ::1])], nogil=True, cache=True)
def get_patch_adaptive_filter(
    im_patch: np.ndarray, cov_noise: np.ndarray, cov_noise_inv: np.ndarray
):
    """Computes matched filter for im_patch given noise covariance.
    :param im_patch: 2D array of dimensions (nc, nx * ny * nz), c-contigous complex64
    :param cov_noise: 2D array of dimensions (nc, nc), c-contigous complex64
    :param cov_noise_inv: matrix inverse of cov_noise, c-contigous complex64
    :return: 1D complex64
    """
    cov_signal = np.dot(
        im_patch, np.conj(im_patch).T
    )  # this is a dot product of a c-contiguous and f-contiguous matrix, but it's faster to not pre-compute the hermitian
    eig_values, eig_basis = np.linalg.eig(np.dot(cov_noise_inv, cov_signal))
    m_opt = eig_basis[:, np.argmax(np.abs(eig_values))]
    m_opt /= np.dot(np.conj(m_opt).T, np.dot(cov_noise, m_opt))
    m_opt *= np.conj(m_opt[0]) / np.abs(m_opt[0])  # align phase against coil-0
    return m_opt


def grid_interpn(
    img: np.ndarray,
    Nd_out: tuple[int, ...],
    method: str = "linear",
) -> np.ndarray:
    """Grid interpolation along the last ``len(shape_out)`` dims"""
    ndim = len(Nd_out)
    Nd_xtra, Nd = img.shape[:-ndim], img.shape[-ndim:]

    img = img.reshape(-1, *Nd)
    nV = img.shape[0]

    coords_in = tuple(
        np.linspace(0, _o - 1, _i) for (_o, _i) in zip(Nd_out, Nd, strict=False)
    )
    grid_out = np.stack(
        np.meshgrid(*(np.arange(_) for _ in Nd_out), copy=False, indexing="ij"), axis=-1
    )

    img_out = np.empty(shape=Nd_xtra + Nd_out, dtype=img.dtype)
    img_out_view = img_out.view()
    img_out_view.shape = (nV, *Nd_out)

    for iV in range(nV):
        img_out_view[iV] = interpn(coords_in, img[iV], grid_out, method=method)

    return img_out


def _get_filter_for_patch(xyz_coord, data_patch, cov_noise, cov_noise_inv):
    dtype = cov_noise.dtype
    this_patch = data_patch[xyz_coord[0], xyz_coord[1], xyz_coord[2], ...]
    this_patch = this_patch.astype(dtype)
    cov_noise_inv = cov_noise_inv.astype(dtype)
    return get_patch_adaptive_filter(this_patch, cov_noise, cov_noise_inv)


def blur(image, filter_fac=0.1):
    """Blur the image with guassian kernel"""
    # dimensions of the image
    dims = np.array(image.shape)

    # coords
    xvec = np.linspace(-dims[1] / 2, dims[1] / 2, dims[1])
    yvec = np.linspace(-dims[0] / 2, dims[0] / 2, dims[0])
    zvec = np.linspace(-dims[2] / 2, dims[2] / 2, dims[2])

    y, x, z = np.meshgrid(xvec, yvec, zvec)
    r = np.sqrt(x**2 + y**2 + z**2)

    # define the k-space filter coarse
    sigma = filter_fac * np.mean(dims[0:2])  # width of blurring kernel
    kernel = np.exp(-(r**2) / (2 * sigma**2))
    kernel = kernel / np.max(kernel)

    image_blurred = np.fft.ifftn((np.fft.fftn(image)) * np.fft.fftshift(kernel))

    return image_blurred
