<!-- Thanks for contributing to TensorGuard! Please fill in the sections below. -->

### What does this change do?

(brief description + linked issue, e.g. "Fixes #123")

### Type of change

- [ ] Bug fix (false SAFE / false UNSAFE)
- [ ] New analysis capability (new operator / theory / transfer function)
- [ ] Integration or tooling (CI, packaging, reporters, hooks)
- [ ] Documentation only

### Soundness

- [ ] This change cannot cause the verifier to report **SAFE** for an unsound
      program, **or** the new behavior is opt-in and documented with a test that
      demonstrates the boundary.
- [ ] If a new transfer function/abstraction was added, the abstain cases are
      handled (we never silently assume).

### Testing (proven against real code)

- [ ] Added/updated a regression test under `tests/` that fails before and
      passes after this change.
- [ ] Demonstrated the behavior on a real `torch.nn.Module` (differentially vs.
      eager PyTorch where applicable).
- [ ] The supported-surface suite passes locally.
- [ ] If README/claims changed: `python reproducibility/audit_numeric_claims.py`
      prints `RESULT: PASS`.

### Compatibility

- [ ] No breaking change to the public API, **or** it goes through the
      `DEPRECATION_POLICY.md` process (and `tests/test_api_stability.py` is
      updated accordingly).

### Notes for reviewers
