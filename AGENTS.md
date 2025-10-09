I am using GitHub Actions. I need to set up a CI pipeline for this repo that does the following:

- [] When a tag is pushed to the repo that matches the format "vx.y.z", it triggers a new release
- [] A pip wheel is automatically built and uploaded to the GitHub Releases page for this repo
- [] The version number in the pyproject.toml file is automatically updated to reflect the new version

Please help with the above tasks.
