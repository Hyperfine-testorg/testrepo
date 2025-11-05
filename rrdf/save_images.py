# Standard Library
import os

# Third Party
import imageio
import numpy as np
import xarray as xr


def numpy_save(
    image, data_dir, base_name, tag, complex_out=True, real_imag_separate=False
):
    """
    Save as .npy file.
    If complex_out is False, save with _r, _i tags as two files.
    """
    print("")
    if complex_out:
        if real_imag_separate:
            fn = data_dir / f"{base_name}_{tag}_r.npy"
            print(f"Saving real per-coil images to {fn}")
            np.save(fn, np.real(image.data))
            fn = data_dir / f"{base_name}_{tag}_i.npy"
            print(f"Saving imag per-coil images to {fn}")
            np.save(fn, np.imag(image.data))
        else:
            fn = data_dir / f"{base_name}_{tag}.npy"
            print(f"Saving complex per-coil images to {fn}")
            np.save(fn, image.data)
    else:
        fn_r = data_dir / f"{base_name}_{tag}_r.npy"
        print(f"Saving data to {fn_r}")
        np.save(fn_r, np.real(image.data))

        fn_i = data_dir / f"{base_name}_{tag}_i.npy"
        print(f"Saving data to {fn_i}")
        np.save(fn_i, np.imag(image.data))
    print("")


def flip_image(img: xr.DataArray, dims) -> xr.DataArray:
    """Flip an xarray image along given dims."""
    return img.isel({d: slice(None, None, -1) for d in dims})


def write_png(rimg, fn, maxp, rectify=False, in_memory=False):
    """
    Write a montage of a 3D or 4D image
    :param rimg: 2 to 4D image
    :param fn: filename to write to
    :param maxp: percentile of top end of dynamic range
    :param rectify: transpose to guess axial view
    :return: None
    """

    def iwr(f_, idata_, in_memory=False):
        assert len(idata_.shape) > 1
        if len(idata_.shape) == 3:
            M = montage(idata_, rectify, maxp)
        else:
            M = idata_
        Mi = M - M.min()
        Mi = Mi * ((2**8 - 1) / Mi.max())
        if in_memory:
            imageio.imwrite(f_, Mi.astype(np.uint8), format="PNG-PIL")
        else:
            imageio.imwrite(f_, Mi.astype(np.uint8))

    rimg = rimg - rimg.min()
    if not in_memory:
        pref, ext = os.path.splitext(fn)
        rimg = np.squeeze(rimg)
        if ext != ".png":
            fn = "%s.png" % pref
        if len(rimg.shape) == 4:
            for i_ in range(rimg.shape[0]):
                fn_ = "%s_%s.png" % (pref, i_ + 1)
                iwr(fn_, rimg[i_])
        else:
            iwr(fn, rimg)
    else:
        iwr(fn, rimg, in_memory=True)
    print(f"Wrote image data to {fn}")


def montage(xx, rectify=True, pmax=None, *, op=np.abs):
    """Montage function"""
    # Set operation to apply to image
    if callable(op):
        x = op(xx)
    else:
        x = xx
    # Prepare montage based on input of 3-D image
    assert len(x.shape) == 3
    # Make largest dimension on the y axis
    if rectify:
        x = x.transpose(np.argsort(x.shape)[::-1])
    m, n, count = x.shape
    mm = int(np.ceil(np.sqrt(count)))
    nn = int(np.ceil(count / mm))
    M = np.zeros((mm * m, nn * n))
    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceM : sliceM + m, sliceN : sliceN + n] = x[:, :, image_id]
            image_id += 1
    if pmax is not None:
        mx = np.percentile(M, pmax)
        M[M > mx] = mx
    return M
