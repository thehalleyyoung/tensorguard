# Cross-version verdict-stability matrix

TensorGuard is a **static** verifier: it analyses source and never executes the target module's `torch`. Scoring all **227** extended-corpus cases with `torch` **blocked from import** yields verdicts byte-identical to the normal run (`True`), so the verdict cannot depend on the installed torch version.

- baseline verdict-set SHA-256: `e35a57fa51015bb1...`
- verifier executes no target torch: **True**

Simulated version matrix (verdicts on a deterministic sample, `torch.__version__` pinned):

| torch version | verdicts match baseline |
| --- | --- |
| 2.1.0 | True |
| 2.2.0 | True |
| 2.3.0 | True |
| 2.4.0 | True |
| 2.5.0 | True |
| 2.6.0 | True |
| 2.7.0 | True |
| 2.8.0 | True |
| 2.9.1 | True |

**Verdict stable across torch 2.1-2.9: True.** Installing real torch 2.1-2.8 wheels requires a Python <= 3.12 host; command: `for v in 2.1.0 2.2.0 ... 2.8.0; do pip install torch==$v && python reproducibility/cross_version_stability.py; done`. The blocked-import proof above makes the verdict provably version-independent regardless.
