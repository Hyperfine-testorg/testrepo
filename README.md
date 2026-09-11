# RRDF - Research Raw Data Format Reconstruction

A Python library for reconstructing MRI images from Hyperfine's Research Raw Data Format (RRDF) files.

The repository contains two packages:

- `rrdf-core`: reading and inspecting RRDF HDF5 files (light dependencies).
- `rrdf-recon`: the reconstruction pipeline (density compensation, prewhitening, NUFFT gridding, coil combination, odd/even echo registration, orientation, NPY/PNG output).

## Installation

### Prerequisites
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- [Git LFS](https://git-lfs.com/). The sample RRDF files in `data/`, the coil maps in `system/coil_maps/`, and the gradient maps in `system/gradient_maps/` are stored with Git LFS. The reconstruction needs the `system/` files, so LFS is required even if you bring your own data.

### Setup

1. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install Git LFS and fetch the large files (skip if you cloned with LFS already installed):
   ```bash
   git lfs install
   git lfs pull
   ```

3. Install dependencies (including development extras) from the repository root:
   ```bash
   uv sync --group dev
   ```

4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Usage

### Basic Reconstruction

Reconstruct a single RRDF file, or every file in a folder, with:

```bash
rrdf-recon data/10.1.0/fse_fast_t2_rrdf_20251105_162521.h5
rrdf-recon path/to/folder_of_rrdf_files/
```

`python recon_rrdf.py <file-or-folder>` is an equivalent wrapper.

Outputs are written to `output/` at the repository root:

- `<name>_image.npy`: complex reconstructed image (or `<name>_image_r.npy` / `<name>_image_i.npy` if saved as separate real/imaginary parts)
- `<name>.png`: montage of the coil-combined image

You can read `.npy` files into MATLAB using [readNPY](https://github.com/kwikteam/npy-matlab). If the output was saved as separate real and imaginary parts:

```matlab
img = readNPY('filename_r.npy') + 1i*readNPY('filename_i.npy');
```
(readNPY doesn't support complex input, which is why there is an option to save outputs as separate real and imaginary parts).

### Notes on scope

This is a simple reference reconstruction; see the docstring at the top of `rrdf-recon/src/rrdf_recon/cli.py` for what it does and does not do compared to the product reconstruction. It has been tested with spin-echo (FSE) acquisitions.

## Testing

```bash
uv run pytest rrdf-core/tests
uv run pytest rrdf-recon/tests -m "not recon_data"   # lightweight
uv run pytest rrdf-recon/tests -m recon_data         # full reconstruction on bundled sample data
```

## License

MIT, see [LICENSE](LICENSE).
