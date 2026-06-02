# Contributing to TensorGuard

Thank you for your interest in improving TensorGuard — a sound static
shape/device/phase/dtype/gradient verifier for PyTorch `nn.Module`s. This guide
explains how to get set up, what we expect from a change, and how the project is
governed.

## Code of conduct

Participation in this project is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
By participating you agree to uphold it.

## Getting set up

```bash
git clone https://github.com/thehalleyyoung/tensorguard
cd tensorguard
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]" z3-solver torch
```

Run the supported-surface test suite (the same set the CI matrix runs):

```bash
python -m pytest tests/test_security.py tests/test_reporters.py \
  tests/test_torch_integration.py tests/test_framework_hooks.py -q
```

## What we expect from a contribution

TensorGuard's value rests on **soundness** — when it reports SAFE, that must be
trustworthy. Contributions are held to a high bar:

1. **Prove it against real code.** New analysis behavior must be demonstrated on
   a real `torch.nn.Module`, ideally differentially against eager PyTorch.
2. **Add a regression test.** Every change ships with a test that fails before
   and passes after. Put it under `tests/`.
3. **Keep the supported surface green.** The public API and the Phase 7/8
   integration modules carry a coverage gate (see
   `reproducibility/coverage_gate.py`); it must stay at or above the threshold.
4. **Respect the security boundary.** Analysis must never execute untrusted
   model source. See [`SECURITY.md`](SECURITY.md); `tests/test_security.py`
   enforces this.
5. **Don't break the public API silently.** Stability-guaranteed symbols are
   pinned by `tests/test_api_stability.py`; removals go through the deprecation
   process in [`DEPRECATION_POLICY.md`](DEPRECATION_POLICY.md).

## Submitting a change

1. Fork and create a topic branch.
2. Make your change with a focused, well-described commit.
3. Run the relevant tests and, for README/claims edits, the numeric-claims
   audit: `python reproducibility/audit_numeric_claims.py` (must print
   `RESULT: PASS`).
4. Open a pull request using the template; fill in the soundness/testing
   sections.
5. A maintainer will review. Expect questions about soundness and edge cases.

## Reporting bugs and proposing features

Use the issue templates under `.github/ISSUE_TEMPLATE/`. For a **false SAFE**
(an unsound result) or a **false UNSAFE** (a false positive), include a minimal
`nn.Module` reproducer and the exact `input_shapes`.

## Security issues

Do **not** open a public issue for a vulnerability. Follow the private
disclosure process in [`SECURITY.md`](SECURITY.md).

## Governance & maintainers

Project governance, decision-making, and the maintainer rotation are documented
in [`GOVERNANCE.md`](GOVERNANCE.md) and [`MAINTAINERS.md`](MAINTAINERS.md).
