# Contributing

This repository follows the RAWS Labs contributing guide for branches, pull requests,
commit subjects and the code of conduct:
https://github.com/raws-labs/.github/blob/main/CONTRIBUTING.md. Below is only what is
specific to this repository.

`develop` is the default branch. Pull requests target `develop` and use
rebase-merge only; linear history is required.

## Build and test

`README.md` describes the CMake build, `ctest`, and the local gate checks. Run
`scripts/local_ci.sh --strict` before opening a pull request; it covers the gates that
CI enforces beyond `ctest`.

## Releasing

Compiler and runtime release as a pair with the same version. The release
procedure for the pair is in the compiler's
[CONTRIBUTING](https://github.com/raws-labs/tigris/blob/main/CONTRIBUTING.md#releasing).
Below is what is specific to the runtime.

`scripts/check_version_sources.py --expect X.Y.Z` verifies the runtime's CMake,
ESP component, example, and consumer version sources.

The capability contract uses the matching compiler branch or tag when available,
otherwise `develop`. A runtime release can precede its compiler tag; the compiler
release then runs the complete pair checks against the matching runtime tag.

The host-library workflow builds and tests native archives for the listed OS and
architecture combinations. On a published release it attaches those tested
archives and their SHA-256 files without rebuilding or replacing existing assets.
`TIGRIS_HOST_PLATFORM` records the distribution platform tag; Linux builds use
the manylinux build images, and macOS builds set an explicit deployment target.
Consumers pin the release and archive hashes separately.

- The `version` field in `idf_component.yml` must equal the release tag without
  the `v`.
- Pushing the tag runs `.github/workflows/publish-component.yml`, which uploads this
  repository as the `raws-labs/tigris-runtime` component to the ESP Component
  Registry. The `files` section of `idf_component.yml` decides what ships; the
  getting-started example is included on purpose.
- The workflow authenticates with the `IDF_COMPONENT_API_TOKEN` repository secret.
  The registry refuses a version that already exists, so a failed upload is fixed
  with a new patch version, not by re-tagging.
- Check the result at https://components.espressif.com/components/raws-labs/tigris-runtime.
  Consumers install it with `idf.py add-dependency "raws-labs/tigris-runtime"`.
