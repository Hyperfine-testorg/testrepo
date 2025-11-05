Basic image recon demo for Hyperfine "Research Raw Data Format" (RRDF) files
by Laura, Rui, Dave 2025 Hyperfine

Install dependencies:
    1. Install UV by following instructions on https://docs.astral.sh/uv/getting-started/installation/
    2. uv sync

Activate the environment:
    source .venv/bin/activate

Reconstruct a sample dataset:
    python recon_rrdf.py data/fse_fast_t2_rrdf_20251009_144749.h5



Outputs will be placed in /data/.
You can read the .npy files into matlab as img = readNPY('filename_r.npy') + 1i*readNPY('filename_i.npy');  
readNPY is available from https://github.com/kwikteam/npy-matlab. it doesnt read complex- thats why i output _r, _i
