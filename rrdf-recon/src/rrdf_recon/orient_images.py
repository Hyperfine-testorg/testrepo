# Third Party
import xarray as xr
from rrdf_core.load import ArrayLabels as lbl


def orient_images(image, scan_plane, image_orientation="axial"):
    """I orient images in LPI convention"""
    image = rps_to_xyz(image, scan_plane)
    # transpose to anatomical orientation
    axis_order = get_axis_order(image_orientation)
    spatial_dimensions = (lbl.X, lbl.Y, lbl.Z)
    if lbl.CHAN in image.dims:
        image = image.transpose(..., lbl.CHAN, *spatial_dimensions)
    else:
        image = image.transpose(..., *spatial_dimensions)
    image = image.transpose(..., *(tuple(axis_order)))
    image = flip_image(image, list(spatial_dimensions))  # flip along all dims
    return image


def flip_image(img: xr.DataArray, dims) -> xr.DataArray:
    """Flip an xarray image along given dims."""
    return img.isel({d: slice(None, None, -1) for d in dims})


def get_axis_order(
    image_orientation: str,
) -> str:
    """Axis order for writing png's"""
    match image_orientation:
        case "axial":
            axis_order = "ZYX"
        case "sagittal":
            axis_order = "XZY"
        case "coronal":
            axis_order = "XYZ"
        case _:
            raise ValueError(f"Unsupported `image_orientation`: {image_orientation}")
    return axis_order


def rps_to_xyz(
    image_rps: xr.DataArray,
    scan_plane: str,
) -> xr.DataArray:
    """Rename dim names from RPS to XYZ, according to scan_plane
    NOTE:
    This method does no transpose on the underlying data.
    It only renames the dims based on `scan_plane`.
    """
    image_xyz = image_rps.rename(
        dict(
            zip(
                (lbl.READOUT, lbl.PHASE, lbl.SLICE),
                tuple(scan_plane),
            )
        )  # type: ignore
    )
    return image_xyz


def xyz_to_rps(
    image_xyz: xr.DataArray,
    scan_plane: str,
) -> xr.DataArray:
    """Rename dim names from XYZ to RPS, according to scan_plane
    NOTE:
    This method does no transpose on the underlying data.
    It only renames the dims based on `scan_plane`.
    """
    image_rps = image_xyz.rename(
        dict(zip(tuple(scan_plane), (lbl.READOUT, lbl.PHASE, lbl.SLICE)))  # type: ignore
    )
    return image_rps


def orientation_from_scanplane(scan_plane):
    """Default orientation by `scan_plane. Not used but adding in case its useful"""
    match scan_plane:
        case "ZYX" | "YZX":  # end with 'X
            ornt = "axial"
        case "XZY" | "ZXY":  # end with 'Y
            ornt = "sagittal"
        case "XYZ" | "YXZ":  # end with 'Z
            ornt = "coronal"
        case _:
            raise ValueError(f"Unsupported scan_plane: {scan_plane}")
    return ornt
