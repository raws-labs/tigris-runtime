# Compiler and Plan Compatibility

The canonical machine-readable release matrix is maintained by the compiler in
[`compatibility.json`](https://github.com/raws-labs/tigris/blob/main/compatibility.json).
Use that record instead of assuming that similarly numbered compiler and
runtime releases are compatible.

| Runtime | Compiler | Accepted plan schemas | Status |
| --- | --- | --- | --- |
| `v0.4.0` (`73bd717`) | `v0.4.0` (`d61c2bc`) | 2, 3, and 4 | Supported |

The compiler owns the canonical wire definitions and publishes a generated
schema artifact with each coordinated release. This runtime vendors the C wire
representation so it keeps no build-time or run-time dependency on the Python
compiler. The compiler's cross-repository CI builds this runtime, executes
generated plans, and verifies that its accepted schema set matches the
integration manifest.

Development integration uses `develop` in both repositories. A branch or
commit pair is not a supported release until the compiler manifest records its
exact tagged commits and the differential contract gate passes.
