# Contributing

This repository follows the RAWS Labs contributing guide for branches, pull requests,
commit subjects and the code of conduct:
https://github.com/raws-labs/.github/blob/main/CONTRIBUTING.md. Below is only what is
specific to this repository.

## Build and test

`README.md` describes the CMake build, `ctest`, and the local gate checks. Run
`scripts/local_ci.sh --strict` before opening a pull request; it covers the gates that
CI enforces beyond `ctest`.

## Releasing

- A release is a `vX.Y.Z` tag on `main`. The `version` field in `idf_component.yml`
  must equal the tag without the `v`.
- Pushing the tag runs `.github/workflows/publish-component.yml`, which uploads this
  repository as the `raws-labs/tigris-runtime` component to the ESP Component
  Registry. The `files` section of `idf_component.yml` decides what ships; the
  getting-started example is included on purpose.
- The workflow authenticates with the `IDF_COMPONENT_API_TOKEN` repository secret.
  The registry refuses a version that already exists, so a failed upload is fixed
  with a new patch version, not by re-tagging.
- Check the result at https://components.espressif.com/components/raws-labs/tigris-runtime.
  Consumers install it with `idf.py add-dependency "raws-labs/tigris-runtime"`.
