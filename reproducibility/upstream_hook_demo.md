# Proposed upstream verification hook vs real PyTorch

Reference implementation (`src/upstream_hook.py`) for the upstream proposal in [`docs/upstream/pytorch_proposal.md`](../docs/upstream/pytorch_proposal.md). It shows how PyTorch could let any `nn.Module` be statically verified with zero changes to model code.

| property | value |
| --- | --- |
| buggy module static verdict | `UNSAFE` |
| clean module static verdict | `SAFE` |
| buggy real `forward` raises at runtime | True |
| attached hook rejects buggy *before* forward | True |
| hook transparent on clean (forward ran) | True |
| clean forward output shape | [2, 2] |
| @verifiable accepts clean | True |
| @verifiable rejects buggy | True |
| static rejection iff runtime failure | True |

**All consistent: True.** The hook turns a deep `aten`-level runtime stack trace into a precise diagnostic raised at the module boundary, while remaining completely transparent (and non-breaking) for modules it proves safe.
