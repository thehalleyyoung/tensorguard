# Release-readiness checklist (Step 286)

A TensorGuard release is shippable only when PyPI, conda, and Docker all pass the same benchmark, audit, security, and artifact-freshness gates plus their channel-specific metadata checks.

- version: `0.1.0`
- all channels release-ready: **True**
- check command: `python reproducibility/release_readiness.py --check`

## pypi

- publish command: `python -m build --sdist --wheel && twine upload dist/tensorguard-*`
- passed items: **6/6**
- release-ready: **True**

| Area | Gate | Status | Evidence | Detail |
| --- | --- | --- | --- | --- |
| benchmark dashboards | `benchmark-dashboard` | pass | `evaluation/dashboard.py`, `evaluation/dashboard.md`, `evaluation/dashboard_baseline.json` | 20 metrics, regressions=0, orphans=0, unregistered=0, markdown_fresh=True |
| benchmark dashboards | `deployment-dashboard` | pass | `evaluation/deployment_dashboard.py`, `evaluation/deployment_dashboard.md`, `evaluation/deployment_dashboard_baseline.json` | rows=4, regressions=0, missing=0, unregistered_supported=0, markdown_fresh=True |
| audit status | `numeric-claim-audit` | pass | `reproducibility/audit_numeric_claims.py`, `reproducibility/numeric_claims_audit.json` | statuses=QUALIFIED_ENV:1, QUALIFIED_REGIME:1, VERIFIED:15 |
| security review | `security-review` | pass | `SECURITY.md`, `tests/test_security.py`, `.github/workflows/matrix.yml` | static-only threat model, regression test, and CI coverage are present |
| artifact freshness | `source-artifact-package` | pass | `reproducibility/artifact_package.py`, `reproducibility/artifact_package.json` | checks=6, failed=none |
| versioning and supply chain | `pypi-metadata` | pass | `pyproject.toml`, `MANIFEST.in`, `CHANGELOG.md`, `CITATION.cff` | version=0.1.0, roadmap_excluded=True |

## conda

- publish command: `conda build conda-recipe/ && anaconda upload <built-package>`
- passed items: **6/6**
- release-ready: **True**

| Area | Gate | Status | Evidence | Detail |
| --- | --- | --- | --- | --- |
| benchmark dashboards | `benchmark-dashboard` | pass | `evaluation/dashboard.py`, `evaluation/dashboard.md`, `evaluation/dashboard_baseline.json` | 20 metrics, regressions=0, orphans=0, unregistered=0, markdown_fresh=True |
| benchmark dashboards | `deployment-dashboard` | pass | `evaluation/deployment_dashboard.py`, `evaluation/deployment_dashboard.md`, `evaluation/deployment_dashboard_baseline.json` | rows=4, regressions=0, missing=0, unregistered_supported=0, markdown_fresh=True |
| audit status | `numeric-claim-audit` | pass | `reproducibility/audit_numeric_claims.py`, `reproducibility/numeric_claims_audit.json` | statuses=QUALIFIED_ENV:1, QUALIFIED_REGIME:1, VERIFIED:15 |
| security review | `security-review` | pass | `SECURITY.md`, `tests/test_security.py`, `.github/workflows/matrix.yml` | static-only threat model, regression test, and CI coverage are present |
| artifact freshness | `conda-artifact-package` | pass | `reproducibility/artifact_package.py`, `reproducibility/artifact_package.json` | checks=6, failed=none |
| versioning and supply chain | `conda-metadata` | pass | `conda-recipe/meta.yaml`, `pyproject.toml`, `LICENSE` | pyproject_version=0.1.0, conda_version=0.1.0 |

## docker

- publish command: `docker build -t ghcr.io/thehalleyyoung/tensorguard:<version> . && docker push ghcr.io/thehalleyyoung/tensorguard:<version>`
- passed items: **6/6**
- release-ready: **True**

| Area | Gate | Status | Evidence | Detail |
| --- | --- | --- | --- | --- |
| benchmark dashboards | `benchmark-dashboard` | pass | `evaluation/dashboard.py`, `evaluation/dashboard.md`, `evaluation/dashboard_baseline.json` | 20 metrics, regressions=0, orphans=0, unregistered=0, markdown_fresh=True |
| benchmark dashboards | `deployment-dashboard` | pass | `evaluation/deployment_dashboard.py`, `evaluation/deployment_dashboard.md`, `evaluation/deployment_dashboard_baseline.json` | rows=4, regressions=0, missing=0, unregistered_supported=0, markdown_fresh=True |
| audit status | `numeric-claim-audit` | pass | `reproducibility/audit_numeric_claims.py`, `reproducibility/numeric_claims_audit.json` | statuses=QUALIFIED_ENV:1, QUALIFIED_REGIME:1, VERIFIED:15 |
| security review | `security-review` | pass | `SECURITY.md`, `tests/test_security.py`, `.github/workflows/matrix.yml` | static-only threat model, regression test, and CI coverage are present |
| artifact freshness | `docker-artifact-package` | pass | `reproducibility/artifact_package.py`, `reproducibility/artifact_package.json` | checks=6, failed=none |
| versioning and supply chain | `docker-metadata` | pass | `Dockerfile`, `.dockerignore` | multi-stage wheel-only, non-root image with local roadmap excluded |
