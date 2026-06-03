# Compatibility and support promise

This is the launch-readiness support promise for TensorGuard. It is
generated from existing policy and CI surfaces rather than asserted as
marketing copy.

## Versioning

- Current package version: `0.1.0`.
- The launch track is `1.0-readiness`; it does not claim the package has already shipped as `1.0.0`.
- Public API stability follows `DEPRECATION_POLICY.md`.

## Compatibility

- Python requirement is `>=3.9` from `pyproject.toml`.
- The compatibility matrix currently enumerates 8 stable OS/Python/torch jobs and a nightly early-warning path in `.github/workflows/matrix.yml`.
- Release-readiness gates cover `conda`, `docker`, `pypi` in `reproducibility/release_readiness.md`.

## Security and maintenance

- Security reports and the static-only untrusted-source boundary are governed by `SECURITY.md`.
- User-visible compatibility changes are recorded in `CHANGELOG.md` and deprecated through the public policy before removal.
- UNKNOWN is a supported outcome, not a failed launch: out-of-fragment models abstain rather than silently passing.
