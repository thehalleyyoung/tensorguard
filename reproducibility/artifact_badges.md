# Artifact-evaluation appendix (Step 123)

Repository: https://github.com/thehalleyyoung/tensorguard

This appendix maps each reproducibility badge to concrete, in-tree evidence and verifies every referenced artifact exists. Badge systems: ACM SIGPLAN/SIGSOFT, USENIX.

**4 of 4 badges have complete in-tree evidence.**

## Artifacts Available

*USENIX equivalent:* Available  
*Criteria:* Publicly and permanently retrievable with a license and citation metadata.

Evidence complete: **True** (4/4 present)

| evidence | path | present |
| --- | --- | --- |
| OSI-approved license | `LICENSE` | True |
| machine-readable citation metadata | `CITATION.cff` | True |
| packaging metadata (public, versioned) | `pyproject.toml` | True |
| project overview | `README.md` | True |

## Artifacts Evaluated -- Functional

*USENIX equivalent:* Functional  
*Criteria:* Documented, consistent, complete and exercisable: installs, runs, and is covered by an automated test suite.

Evidence complete: **True** (6/6 present)

| evidence | path | present |
| --- | --- | --- |
| installable package + console entry-points | `pyproject.toml` | True |
| command-line interface | `src/cli/main.py` | True |
| container image for the tool | `Dockerfile` | True |
| automated test suite | `tests` | True |
| worked examples | `examples` | True |
| artifact-evaluation install guide | `docs/artifact/INSTALL.md` | True |

## Artifacts Evaluated -- Reusable

*USENIX equivalent:* (no direct USENIX equivalent)  
*Criteria:* Exceeds Functional: carefully documented, with pinned dependencies and structure that facilitate reuse and repurposing.

Evidence complete: **True** (6/6 present)

| evidence | path | present |
| --- | --- | --- |
| documentation tree | `docs` | True |
| pinned reproducibility lock | `capsule/requirements.lock.txt` | True |
| reproducibility capsule image | `capsule/Dockerfile.reproduce` | True |
| public, typed API surface | `src/api.py` | True |
| pre-commit / pytest integrations | `src/precommit.py` | True |
| artifact-evaluation requirements doc | `docs/artifact/REQUIREMENTS.md` | True |

## Results Reproduced

*USENIX equivalent:* Reproduced  
*Criteria:* The paper's quantitative results are regenerated from source by a third party with one command.

Evidence complete: **True** (5/5 present)

| evidence | path | present |
| --- | --- | --- |
| one-command capsule entrypoint | `capsule/reproduce.sh` | True |
| from-scratch reproduction + determinism check | `reproducibility/reproduce_all.py` | True |
| capsule manifest + env gate | `reproducibility/capsule_manifest.py` | True |
| numeric-claim audit (validates README numbers) | `reproducibility/audit_numeric_claims.py` | True |
| artifact-evaluation status report | `docs/artifact/STATUS.md` | True |

## How to evaluate

```bash
# Available: clone the public repository (archival DOI at camera-ready).
git clone https://github.com/thehalleyyoung/tensorguard

# Functional: install and run the tool + its test suite.
pip install -e .[dev] && pytest -q

# Reusable: build the pinned reproducibility capsule.
docker build -f capsule/Dockerfile.reproduce -t tensorguard-capsule .

# Reproduced: one command regenerates + byte-verifies every result.
docker run --rm tensorguard-capsule   # or: bash capsule/reproduce.sh
```

> An archival DOI (e.g. Zenodo) must be minted at camera-ready time to upgrade Available from 'public repository' to 'permanently archived'; every other badge's evidence is in-tree and verified here.
