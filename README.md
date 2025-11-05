# RRDF - Research Raw Data Format Reconstruction

A Python library for reconstructing MRI images from Hyperfine's Research Raw Data Format (RRDF) files.

## Installation

### Prerequisites
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the project:
   ```bash
   git clone https://github.com/Hyperfine/rrdf.git
   ```
3. Install dependencies:
   ```bash
   cd rrdf && uv sync
   ```

4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Usage

### Basic Reconstruction

Reconstruct an RRDF file with:

```bash
python recon_rrdf.py data/fse_fast_t2_rrdf_20251009_144749.h5
```

Outputs will be placed in `/data` as `.npy` files. You can read the `.npy` files into MATLAB using [readNPY](https://github.com/kwikteam/npy-matlab) like this:

```matlab
img = readNPY('filename_r.npy') + 1i*readNPY('filename_i.npy');
```
(readNPY doesn't support complex input, which is why we save outputs as separate real and imaginary parts).
