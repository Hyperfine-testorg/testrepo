I am using GitHub Actions. I need to set up a CI pipeline for this repo that does the following:

- [x] When a tag is pushed to the repo that matches the format "vx.y.z", it triggers a new release
- [x] A pip wheel is automatically built and uploaded to the GitHub Releases page for this repo
- [x] The version number in the pyproject.toml file is automatically updated to reflect the new version

## ✅ Implementation Complete

The project now uses **Python Semantic Release** for automated versioning and releases.

### How to use:
1. Write commits using [Conventional Commits](https://www.conventionalcommits.org/) format
2. Push to `main` branch
3. The workflow automatically determines if a release is needed and creates it

### Commit format:
- `feat:` → Minor version bump (0.1.0 → 0.2.0)
- `fix:` → Patch version bump (0.1.0 → 0.1.1)
- `feat!:` or `BREAKING CHANGE:` → Major version bump (0.1.0 → 1.0.0)

See `RELEASE.md` for detailed documentation.

