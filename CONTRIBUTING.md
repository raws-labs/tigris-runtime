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

Compiler and runtime release as a pair with the same version, starting with
`v0.11.0`. The number identifies the tested combination. The plan schema,
accepted schema range, host ABI, `compatibility.json`, and zoo runtime ranges
retain their separate compatibility meanings.

1. Prepare and test both components with the same release number. Run
   `scripts/check_version_sources.py --expect X.Y.Z` to verify the runtime's
   CMake, ESP component, example, and consumer version sources.
2. Publish runtime `vX.Y.Z` first and wait for `host.yml` to attach all native
   archives and their SHA-256 files. The first pair is `v0.11.0`;
   `v0.10.2` has no host archives.
3. Pin the compiler's `runtime-host.json` to that tag and the per-platform
   archive hashes, then run its wheel tests.
4. Tag and publish compiler `vX.Y.Z`. Its tag, setuptools-scm version, and
   runtime pin must agree; the release-wheel build checks this equality.

A compiler-only change still releases the runtime and ESP component with the
same new version. The runtime release note says: "No runtime implementation
changes; released with the matching compiler version."

The capability contract uses the matching compiler branch or tag when available,
otherwise `develop`. A runtime release can precede its compiler tag; the compiler
release then runs the complete pair checks against the matching runtime tag.

The host-library workflow builds and tests native archives for the listed OS and
architecture combinations. On a published release it attaches those tested
archives and their SHA-256 files without rebuilding or replacing existing assets.
`TIGRIS_HOST_PLATFORM` records the distribution platform tag; Linux builds use
the manylinux build images, and macOS builds set an explicit deployment target.
Consumers pin the release and archive hashes separately.

- `main` only fast-forwards to a tested `develop` commit, through the "Promote to
  main" workflow; there are no release or back-merge pull requests.
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
