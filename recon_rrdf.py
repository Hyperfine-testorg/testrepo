"""
This reconstructs Hyperfine "RRDF (research raw data format) .h5 files to images.
See readme.txt for install instructions.
Call as: python3 recon_rrdf.py filename_rrdf_.h5. Outputs to /data/
Example: python3 recon_rrdf.py data/flair_axi_rrdf_20250822_153046.h5

- RRDF files have .h5 format, and have "data" of size [Nc, Nr, odd/even echo parity, Nl].
- Nc = 8 channels, Nr = readout, Nl = # acquired lines.
- Note: these functions all use xarrays. dimension labels are defined in ArrayLabels in rrdf_load.py. That makes array dimension ordering not matter.

- All sequences i've tested this with are spin echo right now.
- Odd and even spin echo lines are reconstructed into two separate images because eddy currents and concomitant gradient terms make different phase and image warping between the two images.
- coords2D is scaled between +/- 0.5 (actually may end up being a bit beyond those boundaries- not sure why but our recon seems to tolerate this)
- We also store 'data_drift' in that rrdf file. This is a short fid navigator signal that is acquired once per TR immediately following excitation.
    It's used for live frequency drift correction during acquisition, but isnt used here right now.

This recon does:
- sample density compensation (sdc)
- prewhitening
- nufft gridding  + POCS (if acquisition uses partial fourier for any dimension)
- coil combination (3D adaptive filter)
- outputs images as .npy files, and puts out png's for coil combined odd/even echo parity images.

Steps it doesnt do:
- apodization
- unwarping, shading corrections
- combine the odd/even echo images (requires nonrigid registration to cleanly combine them)

The output from this script is going to look worse than our product image reconstruction
- This doesnt have odd/even echo image combination (needs nonrigid registration + complex combination to optimally combine)
- this only uses a single step nufft stage (we use an iterative CG-like recon in the product)
- Coil combination  method's not great- just used adaptive filtering bc its simple.
- We also use a DL model inside the iterative recon loop to improve estimates of images and coil maps
- no apodization is implemented here

"""

import argparse
from pathlib import Path

import numpy as np

import rrdf.load as rd
from rrdf.coil_combine import coil_combine
from rrdf.load import ArrayLabels as lbl
from rrdf.prewhiten import prewhiten
from rrdf.xgrid import xgrid
from rrdf.xsdc import xsdc


def main():
    """python3 recon_rrdf.py filename_rrdf_.h5"""
    parser = argparse.ArgumentParser(description="Reconstruct RRDF data")
    parser.add_argument(
        "filename", type=str, help="Input HDF5 filename (inside ./data directory)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)  # create if missing

    # ---------------------
    # Input file
    fn = Path(args.filename)
    if not fn.exists():
        raise FileNotFoundError(f"File not found: {fn}")
    print(f"Loading data from {fn}")

    # ---------------------
    # load data and coordinates from h5 file
    raw_data, coords2d, data_drift, rrdf = rd.read_rrdf(
        fn
    )  # raw_data is an xarray. can convert to numpy array by doing raw_data.data
    print(f"Raw data shape: {raw_data.shape} = [coils, read, odd/even echoes, lines]")
    print(
        f"Coords2D shape: {coords2d.shape} = [dims, lines, odd/even echos], "
        f"scaled between {np.min(coords2d.data)} and {np.max(coords2d.data)}"
    )

    # Imaging parameters
    imsize = rrdf["sequence"]["matrix_size"]  # [r p s]
    partial_fourier = rrdf["sequence"][
        "partial_fourier"
    ]  # [r p s], 0-1, where 1 = fully sampled dimension

    # ---------------------
    # Sample density compensation
    print("\nComputing Sample density compensation weights")
    sdc_weights = xsdc(coords2d, imsize)
    print(
        f"Sample density compensation weights are [lines, odd/even echoes] {sdc_weights.shape}"
    )

    # ---------------------
    # Prewhitening
    print("\nComputing and applying prewhitening matrix")
    raw_data = prewhiten(raw_data, coords2d)

    # ---------------------
    # NUFFT reconstruction
    print(f"\nReconstructing images with NUFFT, matrix size [r, p, s] = {imsize}")
    image = xgrid(
        kdata=raw_data,
        coords=coords2d,
        weights=sdc_weights,
        res=imsize,
        partial_fourier=partial_fourier,
    )
    print(
        f"Image data pre-coil combine has size {image.shape}, {image.dims}"
    )  # [coils read phase slice odd/even echoes]

    # Odd/even image registration (not implemented)
    # Do an initial coil combination for each of the odd/even echo parity imaages,
    # then compute and apply a nonrigid 3D registration between odd/even echoes (example)

    # ---------------------
    # Receive coil combination (not best coil combination, just using this method here bc its simple)
    print("\nCombining coils using 3D adaptive filtering")
    encoding_resolution = rrdf["sequence"]["resolution"]
    image, b1maps = coil_combine(
        image, encoding_resolution=encoding_resolution, dims_combine=[lbl.CHAN]
    )
    image = image.transpose(lbl.READOUT, lbl.PHASE, lbl.SLICE, lbl.ECHO)

    print(
        f"Images following coil combination: {image.shape}, {image.dims}"
    )  # [coils read phase slice odd/even echoes]
    print(
        f"Receive b1 maps following coil combination: {b1maps.shape}, {b1maps.dims}"
    )  # [coils read phase slice odd/even echoes]


if __name__ == "__main__":
    main()
