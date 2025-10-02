
from itertools import product

# Third Party
import xarray as xr
import numpy as np
import numba as nb  # this makes things faster, but commented out to keep this simple

from rrdf.load import ArrayLabels as lbl


def xsdc(
    coords: xr.DataArray,
    imsize: np.ndarray,  # [r p s]
) -> xr.DataArray:
    """Computes sampling-density-compensation (SDC) weights"""
    if lbl.REPS in coords.dims:
        coords = coords[{lbl.REPS: 0}]

    # initialize weights using self.coords input (just remove lbl.COORD dim)
    weights = xr.ones_like(coords[{lbl.COORD: 0}])

    # get volume dimensions and sizes
    volume_dims = [dim_name for dim_name in coords.dims if dim_name not in [lbl.COORD, lbl.ENC]]
    volume_sizes = [coords[dim_name].size for dim_name in volume_dims]
    n_volume_dims = len(volume_dims)

    # compute SDC weights over existing volumes
    for items in product(*[range(volume_sizes[k]) for k in range(n_volume_dims)]):
        # get volume indices
        vol_dim_sel = {volume_dims[k]: int(items[k])
                       for k in range(n_volume_dims)}
        # computer SDC weights
        weights[vol_dim_sel] = sdc(
            coords[vol_dim_sel].transpose(lbl.ENC, lbl.COORD).data,
            mtx_xy=max(imsize[1], imsize[2]),
            mtx_z=imsize[0],
        )

    return weights


def sdc(
    crds: np.ndarray,
    mtx_xy: int = 118,
    mtx_z: int = 36,
    numiter: int = 100,
    taper: float = 0.0,
    kradscale: float = 1.0,
) -> np.ndarray:
    """Calculate sample weights using an iterative method."""
    # compensate for "overgridding"
    mtx_xy = (2 * mtx_xy) + 6
    mtx_z = (2 * mtx_z) + 6

    crds = np.require(crds.copy(), dtype=np.float64, requirements="C")

    if crds.ndim == 1:
        N = crds.size
        d = 1
    elif crds.ndim == 2:
        N, d = crds.shape
    else:
        raise ValueError("Incompatible dimensionality of crds")

    weights = np.ones(N, dtype=np.float64)
    if d == 1:
        oned_sdc(np.squeeze(crds), N, mtx_xy, numiter, taper, weights)
    if d == 2:
        twod_sdc(crds, N, mtx_xy, numiter, taper, weights)
    if d == 3:
        threed_sdc(crds, N, mtx_xy, mtx_z, numiter, taper, kradscale, weights)

    return weights


@nb.njit(cache=True)
def kernel_2d(kernel_len):
    # the kernel, convolved with itself, should be equivalent to FT of a CIRCLE convolved with itself
    # See MRM paper by Zwart, Pipe
    # Doing this offline, can fit the result to a 5th order polynomial k(x), x from 0 to 1
    c0 = 1.0
    c1 = 0.03056504
    c2 = -3.01961845
    c3 = 0.6679865
    c4 = 2.77924058
    c5 = -1.45923643
    kernel = np.empty(kernel_len, dtype=np.float64)
    for i in range(kernel_len):
        x = i / kernel_len
        x2 = x ** 2
        x4 = x2 ** 2
        x3 = x * x2
        x5 = x * x4
        kernel[i] = c0 + (c1 * x) + (c2 * x2) + (c3 * x3) + (c4 * x4) + (c5 * x5)
    return kernel


@nb.njit(cache=True)
def kernel_3d(kernel_len):
    # the kernel, convolved with itself, should be equivalent to FT of a SPHERE convolved with itself
    # See MRM paper by Zwart, Pipe
    # Doing this offline, can fit the result to a 5th order polynomial k(x), x from 0 to 1
    c0 = 1.0
    c1 = 0.04522831
    c2 = -3.36020304
    c3 = 1.12417012
    c4 = 2.82448025
    c5 = -1.63447764
    kernel = np.empty(kernel_len, dtype=np.float64)
    for i in range(kernel_len):
        x = i / kernel_len
        x2 = x ** 2
        x4 = x2 ** 2
        x3 = x * x2
        x5 = x * x4
        kernel[i] = c0 + (c1 * x) + (c2 * x2) + (c3 * x3) + (c4 * x4) + (c5 * x5)
    return kernel


@nb.njit(cache=True)
def oned_sdc(crds, N, mtx_xy, numiter, taper0, sdc):
    # make the kernel
    kernel_len = 1000
    krad = 1.5 * 1.65  # = 2.475, kernel radius in cmtx coordinates
    knorm = kernel_len / krad
    kernel = kernel_2d(kernel_len)

    # taper Logic
    taper = np.empty(N, dtype=np.float64)
    maxcrd = 0
    for i in range(N):
        if (np.abs(crds[i]) > maxcrd):
            maxcrd = np.abs(crds[i])

    crdedge = (1 - taper0) * maxcrd
    if taper0 > 0:
        crdnorm = 1.0 / (maxcrd - crdedge)
    else:
        crdnorm = 0

    for i in range(N):
        if np.abs(crds[i]) > crdedge:
            crdmag = np.abs(crds[i])
            taper[i] = (maxcrd - crdmag) * crdnorm
        else:
            taper[i] = 1.0

        # scale coords, making sure to start within +/- 0.5
        # padding of 3 on each size means we can skip bounds checking later
        xcrd = min(max(-0.5, crds[i]), 0.5)
        crds[i] = 3.0 + ((xcrd + 0.5) * (mtx_xy - 6))

    # iterative algo convolves crds onto and off of cmtx
    cmtx = np.empty(mtx_xy, dtype=np.float64)
    for iter in range(numiter):
        cmtx.fill(0.0)

        # crds -> cmtx
        for i in range(N):
            ii0 = int(np.ceil(crds[i] - krad))
            ii1 = int(np.floor(crds[i] + krad))
            for ii in range(ii0, ii1 + 1):
                dx = np.abs(crds[i] - ii)
                ki = int(np.floor(knorm * dx))
                cmtx[ii] += sdc[i] * kernel[ki]

        # cmtx -> crds
        for i in range(N):
            sdcdenom = 0.0
            ii0 = int(np.ceil(crds[i] - krad))
            ii1 = int(np.floor(crds[i] + krad))
            for ii in range(ii0, ii1 + 1):
                dx = np.abs(crds[i] - ii)
                ki = int(np.floor(knorm * dx))
                sdcdenom += cmtx[ii] * kernel[ki]
            if (sdcdenom > 0):
                sdc[i] *= taper[i] / sdcdenom

    return 0


@nb.njit(cache=True)
def twod_sdc(crds, N, mtx_xy, numiter, taper0, sdc):
    # make the kernel
    kernel_len = 1000
    krad = 1.5 * 1.65  # = 2.475, kernel radius in cmtx coordinates
    krad_sq = krad ** 2
    knorm = kernel_len / krad
    kernel = kernel_2d(kernel_len)

    # taper Logic
    taper = np.zeros(N, dtype=np.float64)
    maxcrd = 0.0
    radius_sq = 0.0
    for i in range(N):
        radius_sq = crds[i, 0] * crds[i, 0] + crds[i, 1] * crds[i, 1]
        if (radius_sq > maxcrd):
            maxcrd = radius_sq
    maxcrd = np.sqrt(maxcrd)

    crdedge = (1 - taper0) * maxcrd
    crdnorm = 1.0 / (maxcrd - crdedge) if taper0 > 0.0 else 0.0

    crdedge_sq = crdedge * crdedge
    for i in range(N):
        radius_sq = crds[i, 0] ** 2 + crds[i, 1] ** 2
        if (radius_sq > crdedge_sq):
            crdmag = np.sqrt(radius_sq)
            taper[i] = (maxcrd - crdmag) * crdnorm
        else:
            taper[i] = 1.0

        # scale coords, making sure to start within +/- 0.5
        # padding of 3 on each size means we can skip bounds checking later
        xcrd = min(max(-0.5, crds[i, 0]), 0.5)
        ycrd = min(max(-0.5, crds[i, 1]), 0.5)
        crds[i, 0] = 3.0 + ((xcrd + 0.5) * (mtx_xy - 6))
        crds[i, 1] = 3.0 + ((ycrd + 0.5) * (mtx_xy - 6))

    # iterative algo convolves crds onto and off of cmtx
    cmtx = np.empty((mtx_xy, mtx_xy), dtype=np.float64)

    for iter in range(numiter):
        cmtx.fill(0.0)

        # crds -> cmtx
        for i in range(N):
            ii0 = int(np.ceil(crds[i, 0] - krad))
            ii1 = int(np.floor(crds[i, 0] + krad))
            jj0 = int(np.ceil(crds[i, 1] - krad))
            jj1 = int(np.floor(crds[i, 1] + krad))
            for ii in range(ii0, ii1 + 1):
                dx = np.abs(crds[i, 0] - ii)
                dx_sq = dx ** 2
                for jj in range(jj0, jj1 + 1):
                    dy = np.abs(crds[i, 1] - jj)
                    dy_sq = dy ** 2
                    rad_sq = dx_sq + dy_sq
                    if rad_sq < krad_sq:
                        ki = int(np.floor(knorm * np.sqrt(rad_sq)))
                        cmtx[ii, jj] += sdc[i] * kernel[ki]

        # cmtx -> crds
        for i in range(N):
            sdcdenom = 0.0
            ii0 = int(np.ceil(crds[i, 0] - krad))
            ii1 = int(np.floor(crds[i, 0] + krad))
            jj0 = int(np.ceil(crds[i, 1] - krad))
            jj1 = int(np.floor(crds[i, 1] + krad))
            for ii in range(ii0, ii1 + 1):
                dx = np.abs(crds[i, 0] - ii)
                dx_sq = dx ** 2
                for jj in range(jj0, jj1 + 1):
                    dy = np.abs(crds[i, 1] - jj)
                    dy_sq = dy ** 2
                    rad_sq = dx_sq + dy_sq
                    if rad_sq < krad_sq:
                        ki = int(np.floor(knorm * np.sqrt(rad_sq)))
                        sdcdenom += cmtx[ii, jj] * kernel[ki]
            if (sdcdenom > 0):
                sdc[i] *= taper[i] / sdcdenom

    return 0


@nb.njit(cache=True)
def threed_sdc(crds, N, mtx_xy, mtx_z, numiter, taper0, kradscale, sdc):
    # make the kernel
    kernel_len = 1000
    krad = kradscale * 1.93
    krad_sq = krad ** 2
    knorm = kernel_len / krad
    kernel = kernel_3d(kernel_len)

    # taper Logic
    taper = np.zeros(N, dtype=np.float64)
    maxcrd = 0.0
    radius_sq = 0.0
    for i in range(N):
        radius_sq = crds[i, 0] ** 2 + crds[i, 1] ** 2 + crds[i, 2] ** 2
        if (radius_sq > maxcrd):
            maxcrd = radius_sq
    maxcrd = np.sqrt(maxcrd)

    crdedge = (1 - taper0) * maxcrd
    crdnorm = 1.0 / (maxcrd - crdedge) if taper0 > 0.0 else 0.0

    crdedge_sq = crdedge ** 2
    for i in range(N):
        radius_sq = crds[i, 0] ** 2 + crds[i, 1] ** 2 + crds[i, 2] ** 2
        if (radius_sq > crdedge_sq):
            crdmag = np.sqrt(radius_sq)
            taper[i] = (maxcrd - crdmag) * crdnorm
        else:
            taper[i] = 1.0

        # scale coords, making sure to start within +/- 0.5
        # padding of 3 on each size means we can skip bounds checking later
        xcrd = min(max(-0.5, crds[i, 0]), 0.5)
        ycrd = min(max(-0.5, crds[i, 1]), 0.5)
        zcrd = min(max(-0.5, crds[i, 2]), 0.5)
        crds[i, 0] = 3.0 + ((xcrd + 0.5) * (mtx_xy - 6))
        crds[i, 1] = 3.0 + ((ycrd + 0.5) * (mtx_xy - 6))
        crds[i, 2] = 3.0 + ((zcrd + 0.5) * (mtx_z - 6))

    # iterative algo convolves crds onto and off of cmtx
    cmtx = np.empty((mtx_xy, mtx_xy, mtx_z), dtype=np.float64)
    dy_sq = np.empty(20, dtype=np.float64)
    dz_sq = np.empty(20, dtype=np.float64)

    for iter in range(numiter):
        cmtx.fill(0.0)

        # crds -> cmtx
        for i in range(N):
            ii0 = int(np.ceil(crds[i, 0] - krad))
            ii1 = int(np.floor(crds[i, 0] + krad))
            jj0 = int(np.ceil(crds[i, 1] - krad))
            jj1 = int(np.floor(crds[i, 1] + krad))
            kk0 = int(np.ceil(crds[i, 2] - krad))
            kk1 = int(np.floor(crds[i, 2] + krad))

            for jj in range(jj0, jj1 + 1):
                dy = crds[i, 1] - jj
                dy_sq[jj - jj0] = dy ** 2
            for kk in range(kk0, kk1 + 1):
                dz = crds[i, 2] - kk
                dz_sq[kk - kk0] = dz ** 2

            for ii in range(ii0, ii1 + 1):
                dx = np.abs(crds[i, 0] - ii)
                dx_sq = dx ** 2
                for jj in range(jj0, jj1 + 1):
                    if (dx_sq + dy_sq[jj - jj0] >= krad_sq):
                        continue
                    for kk in range(kk0, kk1 + 1):
                        rad_sq = dx_sq + dy_sq[jj - jj0] + dz_sq[kk - kk0]
                        if (rad_sq < krad_sq):
                            ki = int(np.floor(knorm * np.sqrt(rad_sq)))
                            cmtx[ii, jj, kk] += sdc[i] * kernel[ki]

        # cmtx -> crds
        for i in range(N):
            sdcdenom = 0.0
            ii0 = int(np.ceil(crds[i, 0] - krad))
            ii1 = int(np.floor(crds[i, 0] + krad))
            jj0 = int(np.ceil(crds[i, 1] - krad))
            jj1 = int(np.floor(crds[i, 1] + krad))
            kk0 = int(np.ceil(crds[i, 2] - krad))
            kk1 = int(np.floor(crds[i, 2] + krad))

            for jj in range(jj0, jj1 + 1):
                dy = crds[i, 1] - jj
                dy_sq[jj - jj0] = dy ** 2
            for kk in range(kk0, kk1 + 1):
                dz = crds[i, 2] - kk
                dz_sq[kk - kk0] = dz ** 2

            for ii in range(ii0, ii1 + 1):
                dx = np.abs(crds[i, 0] - ii)
                dx_sq = dx ** 2
                for jj in range(jj0, jj1 + 1):
                    if (dx_sq + dy_sq[jj - jj0] >= krad_sq):
                        continue
                    for kk in range(kk0, kk1 + 1):
                        rad_sq = dx_sq + dy_sq[jj - jj0] + dz_sq[kk - kk0]
                        if (rad_sq < krad_sq):
                            ki = int(np.floor(knorm * np.sqrt(rad_sq)))
                            sdcdenom += cmtx[ii, jj, kk] * kernel[ki]
            if (sdcdenom > 0):
                sdc[i] *= taper[i] / sdcdenom

    return 0
