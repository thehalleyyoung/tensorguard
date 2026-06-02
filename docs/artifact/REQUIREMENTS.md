# TensorGuard — Artifact Evaluation: REQUIREMENTS

## Hardware

- **No special hardware is required** for the core functional evaluation:
  TensorGuard is a *static* analyser, so reproducing the headline results needs
  no GPU, no concrete inputs, and no model execution. Any x86-64 or Apple
  Silicon machine with ~2 GB RAM and ~2 GB disk is sufficient.
- A CUDA-capable GPU is required **only** to *regenerate* (as opposed to
  validate) the device-domain and Dynamo end-to-end artifacts. These are
  clearly labeled `QUALIFIED_ENV` by the numeric audit and are not needed for
  the Functional badge.

## Software

The minimal, CI-reproducible path needs only:

- Python **3.9–3.13** (the repository's own CI matrix; the maintainers'
  dev box additionally runs 3.14).
- The pinned solver dependency **z3-solver** and the test/eval extras, installed
  via `pip install -e ".[dev]"` (see `docs/artifact/INSTALL.md`).
- PyTorch (CPU build is fine) for the runtime baselines in the precision/recall
  comparison.

Optional toolchains, needed only to regenerate the env-qualified artifacts:

- **Lean 4** (`leanprover/lean4:v4.14.0`, via `elan`/`lake`) to rebuild the
  machine-checked soundness proofs.
- **transformers** + network access to rebuild the HuggingFace cross-family
  bug study.
- A **CUDA** host to rebuild the device/Dynamo artifacts.

A fully pinned, hardware-independent environment is provided by the
`Dockerfile`; using the container removes all of the above version concerns for
the functional path.

## Estimated time and disk

- `pip install -e ".[dev]"`: a few minutes (dominated by the PyTorch wheel).
- `python reproducibility/reproduce_all.py --check`: well under a minute on a
  laptop (the static analysis and the significance tests are cheap).
- Disk: the source tree plus a CPU PyTorch install is roughly 2 GB.

## Security / claims of safety

Analysing an untrusted model file never executes its code (read-as-text +
AST-only static path). See `SECURITY.md` and `tests/test_security.py`.
