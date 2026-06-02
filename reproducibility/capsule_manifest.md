# Reproducibility capsule manifest (Step 122)

One command — `docker run --rm tensorguard-capsule` — regenerates **78** deterministic artifacts from source and verifies each is byte-identical to the committed tree, then re-audits every README numeric claim.

- base image: `python:3.12-slim`
- entrypoint: `bash capsule/reproduce.sh`
- determinism gate: `reproducibility/reproduce_all.py --check`
- numeric audit: `reproducibility/audit_numeric_claims.py`

## Pinned wheels

| package | version |
| --- | --- |
| hypothesis | 6.148.7 |
| numpy | 2.4.3 |
| pytest | 8.4.2 |
| torch | 2.9.1 |
| z3-solver | 4.15.4 |

## Capsule file hashes (sha256)

| file | sha256 |
| --- | --- |
| `capsule/requirements.lock.txt` | `7d5d51b4a51e1134d7387b1437130bdbf46d71488a4cb1c76e80b5acc756b9cc` |
| `capsule/Dockerfile.reproduce` | `3729d1530761b8c6ac1d06cba4e2a1604ba271fdc006fe9b3a0c6ad72a3949a5` |
| `capsule/reproduce.sh` | `5fb9d06bd7c0269230f02b15ae707cfe43006c6baad38d726279f2312c4aefbf` |
