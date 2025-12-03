# rrdf monorepo reorganization

Goals: split the lightweight core save/load layer from heavier reconstruction features; keep public APIs predictable; isolate dependencies; prepare CI/tests for two packages: `rrdf-core` and `rrdf-recon`.

## Target layout
repo-root/
│
├── rrdf-core/
│   ├── pyproject.toml
│   ├── README.md
│   ├── LICENSE
│   ├── src/
│   │   └── rrdf_core/
│   │       ├── __init__.py
│   │       └── ... core package code ...
│   └── tests/
│       └── ... unit tests for core ...
│
├── rrdf-recon/
│   ├── pyproject.toml
│   ├── README.md
│   ├── LICENSE
│   ├── src/
│   │   └── rrdf_recon/
│   │       ├── __init__.py
│   │       └── ... recon functions ...
│   └── tests/
│       └── ... unit tests for recon ...
│
├── shared/  (optional)
│   └── ... scripts, docs, CI workflows, dev tooling, etc ...
│
├── .github/workflows/
│   ├── rrdf-core-ci.yml
│   ├── rrdf-recon-ci.yml
│   └── monorepo-global.yml
│
└── README.md (top-level)

## Responsibilities and boundaries
- Core (`rrdf-core`)
  - Serialization/deserialization (save/load), schema validation, data models, lightweight CLI
  - Only minimal deps required for the file format and validation
  - No reconstruction logic, no heavy numeric/image deps
- Recon (`rrdf-recon`)
  - Reconstruction pipelines, transforms, utilities, visualization helpers
  - Depends on `rrdf-core`; must not be imported by core
  - Heavy/back-end deps isolated via extras (e.g., `torch`, `jax`, `skimage`, `vis`)

## Public API and imports
- Core users: `from rrdf_core import save, load, validate, Model, RRDFError`
- Recon users: `from rrdf_recon import reconstruct, Pipeline, transforms`

## Dependency and packaging guidance
- `rrdf-core` pins only essentials (e.g., schema/validation and serialization libs)
- `rrdf-recon` declares optional extras per backend: `[torch]`, `[jax]`, `[skimage]`, `[cuda]`, `[vis]`
- Lock separately (per-package `requirements-dev.txt` or `uv.lock`/`pip-tools`) to avoid leaking heavy deps into core

## Versioning and releases
- Prefer independent versions:
  - `rrdf-core`: semantic versioning for serialization changes
  - `rrdf-recon`: semantic versioning with dependency bounds on core (e.g., `rrdf-core >=x.y,<x+1`)
- Tagging suggestion: `core-vX.Y.Z` and `recon-vA.B.C`; CI builds/publishes wheels for the tagged package
- Compatibility table in docs noting which recon versions support which core versions

## Testing strategy
- `rrdf-core/tests`: fast unit/property tests for save/load/validate and data models
- `rrdf-recon/tests`: integration tests for pipelines; mark heavy backends (`@pytest.mark.torch`, `@pytest.mark.jax`, etc.)
- Default CI runs core suite and recon light tests; optional matrix jobs run heavy markers
- Contract test: recon CI installs the built core wheel artifact to ensure published compatibility

## Migration steps
1) Create `rrdf-core` with `src/rrdf_core` and move serialization/data-model modules there; update imports in tests/examples to `rrdf_core`.
2) Create `rrdf-recon` with `src/rrdf_recon` and move reconstruction/transforms/pipeline modules there; adjust imports to consume `rrdf_core`.
3) Update `pyproject.toml` files for each package, including extras for recon and minimal deps for core.
4) Update docs (`README.md`, `docs/core.md`, `docs/recon.md`) and examples to show new install commands: `pip install rrdf-core` and `pip install rrdf-recon[torch]` (or other extras).
5) Split tests into `rrdf-core/tests` and `rrdf-recon/tests`, adding markers for heavy backends and updating CI to run matrix jobs per package.
6) Adjust CI release flow: tag-driven builds for each package, build/upload wheels to Releases/PyPI, and ensure recon job consumes the tagged/published core version.

## Developer workflow
- Core changes: work in `rrdf-core`, run `pytest rrdf-core/tests`
- Recon changes: work in `rrdf-recon`, run `pytest rrdf-recon/tests -m "not torch and not jax"`; run heavier markers in opt-in jobs
- Releases: bump/tag per package; recon version bounds updated when core has breaking changes
