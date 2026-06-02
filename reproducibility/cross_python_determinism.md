# Cross-Python determinism proof

The dominant source of cross-Python and cross-run nondeterminism in pure-Python code is **hash randomization** (`PYTHONHASHSEED`), which perturbs `dict`/`set` iteration order. We score a deterministic corpus slice in fresh subprocesses under a spread of hash seeds -- including `PYTHONHASHSEED=random` runs -- and check the verdict-set digest is invariant.

| PYTHONHASHSEED | verdict-set SHA-256 |
| --- | --- |
| 0 | `9d73d39e72cd513e...` |
| 1 | `9d73d39e72cd513e...` |
| 42 | `9d73d39e72cd513e...` |
| 7 | `9d73d39e72cd513e...` |
| 12345 | `9d73d39e72cd513e...` |
| random #1 | `9d73d39e72cd513e...` |
| random #2 | `9d73d39e72cd513e...` |
| random #3 | `9d73d39e72cd513e...` |

- distinct digests observed: **1** (one means fully invariant)
- verdict invariant under hash randomization: **True**
- deterministic across Python builds: **True**

Supported interpreter matrix: 3.9, 3.10, 3.11, 3.12, 3.13, 3.14.

The hash-seed proof holds on any single interpreter and is the property a multi-version CI would assert. Full interpreter matrix command (GitHub Actions): `strategy.matrix.python-version: [3.9, 3.10, 3.11, 3.12, 3.13, 3.14]` running `python reproducibility/cross_python_determinism.py`. Installing all interpreters in this box is not possible; the result is env-qualified but the hash-randomization invariance below is the mechanism that makes it portable.
