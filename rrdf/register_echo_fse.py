"""
Registers distorted images from odd/even echo parity for fast spin echo sequences.
This assumes coils are not yet combined, and image is [Nc, Nr, Np, Ns, Necho_images]
"""

from pathlib import Path
import time
import numpy as np
import SimpleITK as sitk  # noqa: E402
import xarray as xr
from rrdf.load import ArrayLabels as lbl
from rrdf.coil_combine import coil_combine


def register_echo_fse(image, encoding_resolution):
    """
    I do 3D nonrigid registration using SITK on images with dimensions either [Ncoil, R, P, S, Necho=2]
    The odd echo parity image is registered to the even image
    """
    # if the echo dimension is already combined, do nothing
    if lbl.ECHO not in image.dims or image[lbl.ECHO].size == 1:
        return image

    if lbl.READOUT in image.dims:
        spatial_dim = (lbl.READOUT, lbl.PHASE, lbl.SLICE)
    elif lbl.X in image.dims:
        spatial_dim = (lbl.X, lbl.Y, lbl.Z)
    else:
        raise ValueError(f"Unexpected dimensions {image.dims}")
    image = image.transpose(..., *spatial_dim)

    # if has channel dim, do a simple coil combination before registration
    if lbl.CHAN in image.dims:
        image_coilcombined, _ = coil_combine(
            image, encoding_resolution=encoding_resolution, dims_combine=[lbl.CHAN]
        )
    else:
        image_coilcombined = image

    # sum even/odd echoes separately
    images_even = image_coilcombined[{lbl.ECHO: slice(0, None, 2)}].sum(dim=lbl.ECHO)
    images_odd = image_coilcombined[{lbl.ECHO: slice(1, None, 2)}].sum(dim=lbl.ECHO)

    # register odd to even echo image
    images_odd_registered, _, _ = Register3D(images_even, images_odd)

    debug_saveimages = False
    if debug_saveimages:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "output"
        fn = data_dir / "images_even.npy"
        np.save(fn, np.abs(images_even.data))
        fn = data_dir / "images_odd.npy"
        np.save(fn, np.abs(images_odd.data))
        fn = data_dir / "images_odd_reg.npy"
        np.save(fn, np.abs(images_odd_registered))

    images_odd_registered = xr.DataArray(
        images_odd_registered,
        dims=[d for d in image_coilcombined.dims if d != lbl.ECHO],
        coords={k: v for k, v in image_coilcombined.coords.items() if k != lbl.ECHO},
    )

    # replace all odd echoes with the registered version (broadcasted)
    image_coilcombined[{lbl.ECHO: slice(1, None, 2)}] = (
        images_odd_registered.expand_dims(lbl.ECHO).broadcast_like(
            image_coilcombined.isel({lbl.ECHO: slice(1, None, 2)})
        )
    )
    return image_coilcombined


def Register3D(img1, img2, registertype="nonrigid", mask=None):
    """Performs 3D registration
    input:
        img1 = fixed image [x, y, z]
        img2 = moving_image: [x, y, z]
        phantom = flag:  1= you're scanning a phantom, 0= you're scanning a head
        registertype = 'nonrigid', 'affine'
        mask is optional, if there it's a mask on img1.
    return:
        img2, resampled
        transform: sitk Transform, affine + bspline, can be further applied to images using sitk.Resample
        disp_field: [x, y, z, 3] vector field, affine + bspline transforms

    I run two stages of registration: Affine then BSpline
    Example: img2, transform, disp_field = Register3D(img1, img2, registertype='affine')
    """
    print("\nStarting Register3D to register odd/even echo parity images.")
    start_time = time.time()
    fixed_image = sitk.GetImageFromArray(abs(img1))
    moving_image = sitk.GetImageFromArray(abs(img2))

    initial_transform = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_image, moving_image.GetPixelID()),
        moving_image,
        sitk.AffineTransform(fixed_image.GetDimension()),
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()

    if mask is not None:
        fixed_mask = sitk.GetImageFromArray(mask)
        registration_method.SetMetricFixedMask(fixed_mask)

    print("Running sitk 3D affine registration")
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01, seed=0)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100
    )  # 1. Increase to 1000
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    affine_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )
    # apply the transform to moving image, to pass into next step
    moving_resampled = sitk.Resample(
        moving_image,
        fixed_image,
        affine_transform,
        sitk.sitkBSpline,
        0.0,
        moving_image.GetPixelID(),
    )

    if registertype == "affine":
        # apply the affine transform to moving image and exit function
        moving_image = sitk.Resample(
            moving_image,
            fixed_image,
            affine_transform,
            sitk.sitkBSpline,
            0.0,
            moving_image.GetPixelID(),
        )
        disp_field_affine = sitk.TransformToDisplacementField(
            affine_transform, sitk.sitkVectorFloat32, fixed_image.GetSize()
        )
        disp_field_affine = np.array(sitk.GetArrayFromImage(disp_field_affine))

        moving_image = sitk.GetArrayFromImage(moving_image)
        return moving_image, affine_transform, disp_field_affine

    print("Running sitk 3D bspline registration")
    transformDomainMeshSize = [8] * moving_resampled.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)
    R = sitk.ImageRegistrationMethod()
    gradientConvergenceTolerance = 1e-4
    numberOfIterations = 10
    maximumNumberOfCorrections = 5
    maximumNumberOfFunctionEvaluations = 50
    costFunctionConvergenceFactor = 1e7
    R.SetMetricAsCorrelation()

    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance,
        numberOfIterations,
        maximumNumberOfCorrections,
        maximumNumberOfFunctionEvaluations,
        costFunctionConvergenceFactor,
    )

    if mask is not None:
        fixed_mask = sitk.GetImageFromArray(mask)
        registration_method.SetMetricFixedMask(fixed_mask)

    R.SetInitialTransform(tx, True)
    R.SetInterpolator(sitk.sitkLinear)

    def print_metric():
        if not (R.GetOptimizerIteration(), 10):
            print(f"{R.GetOptimizerIteration():3} = {R.GetMetricValue():10.5f}")
        return

    R.AddCommand(sitk.sitkIterationEvent, print_metric())
    bsplinetransform = R.Execute(fixed_image, moving_resampled)

    composite_transform = sitk.CompositeTransform([affine_transform, bsplinetransform])

    # make displacement vector field, for imaging test
    disp_field = sitk.TransformToDisplacementField(
        composite_transform, sitk.sitkVectorFloat32, fixed_image.GetSize()
    )
    disp_field = np.array(sitk.GetArrayFromImage(disp_field))

    # Just resample in one go....
    moving_resampled = sitk.Resample(
        moving_image,
        fixed_image,
        composite_transform,
        sitk.sitkBSpline,
        0.0,
        moving_image.GetPixelID(),
    )

    img_resampled = sitk.GetArrayFromImage(moving_resampled)
    elapsed_time = time.time() - start_time
    print("registration time = {}".format(elapsed_time))

    return img_resampled, composite_transform, disp_field
