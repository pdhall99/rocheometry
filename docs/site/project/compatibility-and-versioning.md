# Compatibility and versioning

## Package versioning

This project follows [Semantic Versioning](https://semver.org/).
Releases have version numbers of the form `MAJOR.MINOR.PATCH`:

- **MAJOR** releases may contain backwards-incompatible changes to the public API
- **MINOR** releases may add functionality and deprecate public APIs while remaining backwards compatible
- **PATCH** releases contain backwards-compatible fixes

APIs explicitly documented as experimental are not covered by the same backwards-compatibility guarantees.

For releases before `1.0.0`, the public API should be considered under development and may change between minor releases, as permitted by Semantic Versioning.

## Python-version compatibility

This project supports Python feature releases from their official final release until their [official end-of-life](https://devguide.python.org/versions/).

Support for a new Python feature release is generally introduced in the first minor release of this project following the upstream Python release.
Python feature releases may be dropped once they reach the end of their upstream support cycle.
The currently supported Python versions are declared in the package metadata.

Dropping an end-of-life Python version is considered a change to the supported runtime environment rather than a backwards-incompatible change to this project's public API, and therefore does not by itself require a new major release.
