import time
from itertools import product
from typing import Tuple

# Third Party
import numpy as np
import numba as nb
import xarray as xr

from rrdf.load import ArrayLabels as lbl
from collections.abc import Sequence


def xgrid(
    kdata: xr.DataArray,
    coords: xr.DataArray,
    weights: xr.DataArray,
    res: np.ndarray,  # [r, p, s]
    partial_fourier: np.ndarray,  # [r, p, s]
) -> xr.DataArray:
    # gridding
    print("\nHELLO from XGrid : kdata", kdata.shape, kdata.dims)
    time0 = time.time()  # time the gridding operation

    # if the data has a repetition dimension just take coords from the first one
    if lbl.REPS in coords.dims:
        coords = coords.isel({lbl.REPS: 0})

    # Standardize dimensions
    kdata = kdata.transpose(..., lbl.CHAN, lbl.READOUT, lbl.ENC)
    coords = coords.transpose(..., lbl.ENC, lbl.COORD)
    weights = weights.transpose(..., lbl.ENC)

    volume_dims = kdata.dims[:-3]  # everything except (CHAN, READOUT, ENC)
    volume_sizes = kdata.shape[:-3]
    n_channels = kdata.sizes[lbl.CHAN]
    dtype = np.complex64

    res_rps = tuple(res)
    partial_fourier = tuple(partial_fourier)
    pocs_flag = bool(0 < partial_fourier.count(1) < 3)

    # sanity check on readout dimension
    assert res_rps[0] == kdata.sizes[lbl.READOUT]

    # Decide output shape + dims
    if res_rps[2] == 1 and volume_dims == (lbl.REPS,):
        shape = (n_channels, *res_rps)
        dims = (lbl.CHAN, lbl.READOUT, lbl.PHASE, lbl.REPS)
    else:
        shape = (n_channels, *res_rps, *volume_sizes)
        dims = (lbl.CHAN, lbl.READOUT, lbl.PHASE, lbl.SLICE) + volume_dims

    img_gridded = xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims)

    # Loop over all image "volumes"
    for vol in product(*(range(n) for n in volume_sizes)):
        vol_sel = dict(zip(volume_dims, vol))
        print(f"gridding {vol}")

        data = kdata[vol_sel].data
        crds = coords[vol_sel].data
        wgts = weights[vol_sel].data

        img_gridded[vol_sel] = gridFFTRolloff(
            crds, data, wgts, res_rps, res_rps, partial_fourier, pocs_flag
        )

    print("gridding took {:2.2f} seconds".format(time.time() - time0))
    return img_gridded


def gridFFTRolloff(
    crds: np.ndarray,
    kdata: np.ndarray,
    weights: np.ndarray,
    npts: tuple[int, int, int],
    imgRes: tuple[int, int, int],
    partial_fourier: tuple[float, float, float] = (1, 1, 1),
    pocs_flag: bool = False,
):
    """Perform grid-fft-rolloff functions in a single go."""
    if imgRes[2] == 1:
        crds = crds[:, 0]
    d = crds.shape[-1]

    if crds.ndim == 1:
        d = 1
    mtx_xy = int(max(npts[1], npts[2]))
    mtx_z = int(npts[0])  # readout

    # grid the data
    kspace_multichannel = grid(crds, kdata, weights, mtx_xy=mtx_xy, mtx_z=mtx_z)

    # pocs if using partial fourier, if not just fft along readout dimension
    fft_axes = (-1, -2, -3)
    if pocs_flag:
        img_multichannel = pocs(kspace_multichannel, partial_fourier, fft_axes=fft_axes)
    else:
        img_multichannel = shifted_fftn(
            kspace_multichannel, axes=fft_axes
        )  # do the readout dimension

    # apply rolloff
    # for 2D position scan, we do 1D gridding anf no rolloff
    if d == 1:
        img_rolloff = np.ones_like(img_multichannel)
    if d == 2:
        img_rolloff = rolloff((mtx_xy, mtx_xy))
    if d == 3:
        img_rolloff = rolloff((mtx_xy, mtx_xy, mtx_z))
    # Deapodization
    img_multichannel *= img_rolloff

    # when we have 2D image
    if len(img_multichannel.shape) == 3:
        img_multichannel = np.expand_dims(img_multichannel, axis=3)

    if d > 1:
        # should be (nChan, nRead, y, x)
        assert img_multichannel.ndim == 4

        # Crop the image volume data to the intended specs from the pulse seq.
        # design.
        # flip nPhase and nSlice
        img_multichannel = img_multichannel.transpose([0, 1, 3, 2])

        # Fix center crop so that it doesnt go out of bounds.
        # This fix should move into center_crop() as an option.
        res = tuple(min(d, r) for (d, r) in zip(img_multichannel.shape[-3:], imgRes))

        # print("res", res)
        img_multichannel = center_crop(img_multichannel, res)

    return img_multichannel


def grid(
    crds: np.ndarray,
    data: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    mtx_xy: int = 128,
    mtx_z: int = 128,
) -> np.ndarray:
    """Grid non-Cartesian samples onto a regular Cartesian grid.
    Args:
        crds: numpy array of coordinates, (N, d) where N is number of points
            and d is dimensionality ((N,) is also acceptable for 1D gridding).
        data: numpy array with shape matching the coordinates - that is
            data.shape[-1] == N
            (if not provided it will be initialized as all ones)
        weights: sample density weights shape (N,)
            (if not provided it will be initialized as all ones)
        mtx_xy, mtx_z: matrix size to grid onto (will do 2.5X overgridding)
            must be square in x-y so there's only one arg for that
            mtx_z is only used for 3D gridding, otherwise ignored
    """
    # TODO: input for "dims per set" or just make caller deal with it
    # TODO: accepting xr DataArray as well might be nice
    if crds.ndim == 1:
        N = crds.size
        d = 1
    elif crds.ndim == 2:
        N, d = crds.shape
    else:
        raise ValueError(f"Incompatible dimensionality of crds, ndim: {crds.ndim}")
    crds = np.require(crds, dtype=np.float64, requirements=["C", "W"])

    if data is None:
        data = np.ones((N,), dtype=np.complex128)

    if weights is None:
        weights = np.ones((N,), dtype=np.float64)
    else:
        weights = np.require(weights, dtype=np.float64, requirements=["C", "W"])

    if weights.shape[0] != N:
        raise ValueError(
            f"Size mismatch of crds ({N}) and weights ({weights.shape[0]})"
        )

    if data.shape[-1] != N:
        raise ValueError(f"Size mismatch of crds ({N}) and data ({data.shape[0]})")

    # collapse all but the last dim and loop over them
    if data.ndim > 1:
        data_shape = data.shape
        n_sets = int(np.prod(data.shape[:-1]))
    else:
        data_shape = None
        n_sets = 1

    data = np.reshape(data, (n_sets, N))

    # TODO: more validation of dims and sizes
    # Oversample only for 2D gridding
    if d > 1:
        # TODO: more validation of dims and sizes
        mtx_xy = int(6 * mtx_xy / 4)  # 1.5X OVERSAMPLE
        mtx_z = int(6 * mtx_z / 4)  # 1.5X OVERSAMPLE

    if d == 1:
        gridded_shape: Tuple[int, ...] = (mtx_xy,)
    elif d == 2:
        gridded_shape = (mtx_xy, mtx_xy)
    elif d == 3:
        gridded_shape = (mtx_xy, mtx_xy, mtx_z)

    # Assert correct data type.
    acceptable_dtypes = [np.complex64, np.complex128]
    if data.dtype not in acceptable_dtypes:
        raise TypeError

    # Require data to be c-contiguous and writeable
    data = np.require(data, requirements=["C", "W"])

    # Create output data array.
    gridded_data = np.zeros((n_sets, *gridded_shape), dtype=data.dtype)

    if d == 1:
        oned_grid(np.squeeze(crds), N, weights, data, gridded_data, mtx_xy)
    elif d == 2:
        twod_grid(crds, N, weights, data, gridded_data, mtx_xy)
    elif d == 3:
        threed_grid(crds, N, weights, data, gridded_data, mtx_xy, mtx_z)

    # HACK for 2.5d stuff, i guess?
    # I don't know why this was here, but it's included for compatibility with the old grid function
    if d == 3:
        gridded_data = np.transpose(gridded_data, (0, 3, 1, 2))
        gridded_shape = (mtx_z, mtx_xy, mtx_xy)

    if data_shape is not None:
        gridded_data = np.reshape(gridded_data, (*data_shape[:-1], *gridded_shape))
    else:
        gridded_data = np.reshape(gridded_data, gridded_shape)

    return gridded_data


def pocs(kspace_multichannel, partial_fourier, fft_axes):
    # Number of POCS iterations
    iterations_n = 5
    if partial_fourier[0] < 1:
        rotated_kspace = np.rot90(kspace_multichannel, -1, (1, 2))
        test_sum = abs(np.sum(np.sum(rotated_kspace[5, :, :, :], axis=0), axis=1))
        nonzeroind = np.amin(np.asarray(np.nonzero(test_sum)))

    elif partial_fourier[1] < 1:
        rotated_kspace = np.rot90(kspace_multichannel, -1, (2, 3))
        test_sum = abs(np.sum(np.sum(kspace_multichannel[5, :, :, :], axis=1), axis=0))
        nonzeroind = np.amin(np.asarray(np.nonzero(test_sum)))
        kspace_multichannel = rotated_kspace

    elif partial_fourier[2] < 1:
        test_sum = abs(np.sum(np.sum(kspace_multichannel[5, :, :, :], axis=0), axis=1))
        nonzeroind = np.amin(np.asarray(np.nonzero(test_sum)))

    kcenter = np.zeros_like(kspace_multichannel)

    if partial_fourier[0] < 1:
        kcenter[:, nonzeroind : kspace_multichannel.shape[1] - 1 - nonzeroind, :, :] = (
            kspace_multichannel[
                :, nonzeroind : kspace_multichannel.shape[1] - 1 - nonzeroind, :, :
            ]
        )
        kpartial = kspace_multichannel
        img_center = shifted_fftn(kcenter, axes=fft_axes)

        for pocs_count in range(iterations_n):
            img_partial = shifted_fftn(kpartial, axes=fft_axes)
            img_phased = np.exp(1j * np.angle(img_center)) * np.abs(
                img_partial
            )  # phase correct partial
            kspace_phased = shifted_ifftn(img_phased, axes=fft_axes)
            kpartial[
                :,
                kspace_multichannel.shape[1]
                - 1
                - nonzeroind : kspace_multichannel.shape[1],
                :,
                :,
            ] = kspace_phased[
                :,
                kspace_multichannel.shape[1]
                - 1
                - nonzeroind : kspace_multichannel.shape[1],
                :,
                :,
            ]
    else:
        kcenter[:, :, nonzeroind : kspace_multichannel.shape[3] - 1 - nonzeroind, :] = (
            kspace_multichannel[
                :, :, nonzeroind : kspace_multichannel.shape[3] - 1 - nonzeroind, :
            ]
        )
        kpartial = kspace_multichannel
        img_center = shifted_fftn(kcenter, axes=fft_axes)

        for pocs_count in range(iterations_n):
            img_partial = shifted_fftn(kpartial, axes=fft_axes)
            img_phased = np.exp(1j * np.angle(img_center)) * np.abs(
                img_partial
            )  # phase correct partial
            kspace_phased = shifted_ifftn(img_phased, axes=fft_axes)
            kpartial[:, :, 0 : nonzeroind + 2, :] = kspace_phased[
                :, :, 0 : nonzeroind + 2, :
            ]

    if partial_fourier[1] < 1:
        rotated_partial = np.rot90(kpartial, 1, (2, 3))
        kpartial = rotated_partial

    return shifted_fftn(kpartial, axes=fft_axes)


def rolloff(shape: Tuple[int, ...] = (118, 118, 36)) -> np.ndarray:
    """Generate deapodization for gridding correction.

    Typically used with the above "grid" method and matching matrix sizes.

    Args:
        shape: tuple of up to length 3 with (mtx_xy, [mtx_xy, [mtx_z]])
    """
    if len(shape) > 1 and shape[0] != shape[1]:
        raise ValueError

    # grid a single point at the center of a single grid matrix to esimate the
    # deapodization
    rodata = np.ones((1,), dtype=np.complex128)
    rowght = np.ones((1,), dtype=np.float64)
    rocrds = np.zeros((1, len(shape)), dtype=np.float64)
    gridded_point = grid(
        rocrds,
        rodata,
        rowght,
        mtx_xy=shape[0],
        mtx_z=shape[2] if len(shape) == 3 else 36,
    )

    # finish the rolloff kernel by transforming to the image domain
    rolloff = shifted_fftn(gridded_point, axes=None)  # do all dimensions

    # invert so that it can be applied to grids after fft
    rolloff_mag = np.abs(rolloff)
    rolloff = 1.0 / np.clip(
        np.abs(rolloff), a_min=0.2 * np.max(rolloff_mag), a_max=None
    )

    return rolloff


def center_crop(img, res):
    """Crop the image, centered, with the given dim lengths in 'res' starting
    from the end of the array.
    """
    # TODO: assert that res elements are integers

    # generate all slicing objects from the res array
    slices = []
    il = len(img.shape) - len(res)
    for i in range(len(res)):
        c = img.shape[il + i] // 2
        lo = c - res[i] // 2
        hi = c + res[i] // 2
        slices.append(slice(int(lo), int(hi)))

    # prepend some empty (pass all) slice objects to the beginning of the array
    for i in range(len(img.shape) - len(slices)):
        slices.insert(0, slice(None))

    # crop
    return np.ascontiguousarray(img[tuple(slices)])


@nb.njit(cache=True)
def grid_kernel():
    KERNSIZE = 250
    kernel = np.empty(KERNSIZE + 1, dtype=np.float64)
    for i in range(KERNSIZE):
        x = i / KERNSIZE
        kernel[i] = 0.5 * (1.0 + np.cos(np.pi * x)) * np.exp(-6.0 * x * x)
    kernel[KERNSIZE] = 0.0
    return kernel


@nb.njit(cache=True, parallel=True)
def oned_grid(crds, N, weights, data, data_out, mtx_xy):
    kernel = grid_kernel()
    fmtx_xy = mtx_xy & ~1  # round down to nearest even integer

    for n_slice in nb.prange(data.shape[0]):
        for i in nb.prange(N):
            # bounds check before gridding this coord
            if -0.5 <= crds[i] <= 0.5:
                val = data[n_slice, i] * weights[i]
                fj = (0.5 + crds[i]) * fmtx_xy
                minj = max(0, int(fj - 2.5) + 1)
                maxj = min(mtx_xy - 1, int(fj + 2.5))
                dj = int(np.floor((100.0 * (minj - fj)) + 0.5))
                for j in range(minj, maxj + 1):
                    data_out[n_slice, j] += kernel[abs(dj)] * val
                    dj += 100
    return 0


@nb.njit(cache=True, parallel=True)
def twod_grid(crds, N, weights, data, data_out, mtx_xy):
    kernel = grid_kernel()
    dmtx_xy = mtx_xy & ~1  # round down to nearest even integer

    for n_slice in range(data.shape[0]):
        for i in range(N):
            # bounds check before gridding this coord
            if -0.5 <= crds[i, 0] <= 0.5 and -0.5 <= crds[i, 1] <= 0.5:
                val0 = data[n_slice, i] * weights[i]

                fj = (0.5 + crds[i, 1]) * dmtx_xy
                fk = (0.5 + crds[i, 0]) * dmtx_xy

                minj = max(0, int(fj - 2.5) + 1)
                maxj = min(mtx_xy - 1, int(fj + 2.5))
                dj = int(np.floor((100.0 * (minj - fj)) + 0.5))

                mink = max(0, int(fk - 2.5) + 1)
                maxk = min(mtx_xy - 1, int(fk + 2.5))
                dk0 = int(np.floor((100.0 * (mink - fk)) + 0.5))

                for j in range(minj, maxj + 1):
                    val1 = kernel[abs(dj)] * val0
                    dk = dk0
                    for k in range(mink, maxk + 1):
                        data_out[n_slice, j, k] += kernel[abs(dk)] * val1
                        dk += 100
                    dj += 100
    return 0


@nb.njit(cache=True, parallel=True)
def threed_grid(crds, N, weights, data, data_out, mtx_xy, mtx_z):
    kernel = grid_kernel()
    dmtx_xy = mtx_xy & ~1  # round down to nearest even integer
    dmtx_z = mtx_z & ~1

    for n_slice in nb.prange(data.shape[0]):
        for i in nb.prange(N):
            # bounds check before gridding this coord
            if (
                -0.5 <= crds[i, 0] <= 0.5
                and -0.5 <= crds[i, 1] <= 0.5
                and -0.5 <= crds[i, 2] <= 0.5
            ):
                val0 = data[n_slice, i] * weights[i]

                fj = (0.5 + crds[i, 1]) * dmtx_xy
                fk = (0.5 + crds[i, 0]) * dmtx_xy
                fl = (0.5 + crds[i, 2]) * dmtx_z

                minj = max(0, int(fj - 2.5) + 1)
                maxj = min(mtx_xy - 1, int(fj + 2.5))
                dj = int(np.floor((100.0 * (minj - fj)) + 0.5))

                mink = max(0, int(fk - 2.5) + 1)
                maxk = min(mtx_xy - 1, int(fk + 2.5))
                dk0 = int(np.floor((100.0 * (mink - fk)) + 0.5))

                minl = max(0, int(fl - 2.5) + 1)
                maxl = min(mtx_z - 1, int(fl + 2.5))
                dl0 = int(np.floor((100.0 * (minl - fl)) + 0.5))

                for j in range(minj, maxj + 1):
                    val1 = kernel[abs(dj)] * val0
                    dk = dk0
                    for k in range(mink, maxk + 1):
                        val2 = kernel[abs(dk)] * val1
                        dl = dl0
                        for l in range(minl, maxl + 1):  # noqa: E741
                            data_out[n_slice, j, k, l] += kernel[abs(dl)] * val2
                            dl += 100
                        dk += 100
                    dj += 100
                    # We are adding 100 to dj, dk each step because the kernel size is 100 times the grid size
    return 0


def shifted_fftn(
    x: np.ndarray,
    axes: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> np.ndarray:
    _x = np.fft.ifftshift(x, axes=axes)
    _out = np.fft.fftn(_x, axes=axes, norm=norm)
    out = np.fft.fftshift(_out, axes=axes)

    return out


def shifted_ifftn(
    x: np.ndarray,
    axes: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> np.ndarray:
    _x = np.fft.ifftshift(x, axes=axes)
    _out = np.fft.ifftn(_x, axes=axes, norm=norm)
    out = np.fft.fftshift(_out, axes=axes)

    return out
