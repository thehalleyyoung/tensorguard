# TensorGuard 1000-star growth playbook

A concrete, sequenced plan to take TensorGuard from a strong research artifact
to a widely-adopted, 1000★ default in the PyTorch ecosystem. Each item names the
asset it depends on (most already exist in this repo) so the plan is executable,
not aspirational.

## 0. The one-sentence pitch

> **TensorGuard catches shape/device/dtype/phase/gradient bugs in your PyTorch
> model *statically*, before you run it — sound mode has zero false alarms.**

Every asset below leads with that sentence and a 10-second copy-pasteable demo
([`examples/quickstart.py`](../examples/quickstart.py),
[`examples/tutorials/01_quickstart.ipynb`](../examples/tutorials/01_quickstart.ipynb)).

## 1. First-run experience (week 1)

- **GitHub-source install works and the README's first code block runs as-is.**
  Backed by `tests/test_api_stability.py` + the public import surface.
- **Five Colab notebooks**, one per integration
  ([`examples/tutorials/`](../examples/tutorials/)) — each has an "Open in Colab"
  badge and runs in seconds. CI executes them (`tutorials.yml`) so they never rot.
- **GETTING_STARTED.md** reaches a green check in <5 minutes.

## 2. Distribution channels (weeks 1–4)

| Channel | Asset to lead with | CTA |
| --- | --- | --- |
| Hacker News / Lobsters "Show" | quickstart notebook + the zero-false-positive claim | star + try on your repo |
| r/MachineLearning, r/pytorch | the 60-bug head-to-head vs. PyTea/runtime baselines | leaderboard PR |
| X/Bluesky thread | 4 GIFs: a typo → red squiggle (VS Code), CI fail, ONNX gate, compile gate | repo link |
| PyTorch forums / dev-discuss | the upstream `nn.utils.verify_module` RFC (`docs/RFC_pytorch_companion.md`) | RFC feedback |
| Weekly newsletters (PyTorch, MLOps, Deep Learning Weekly) | one-paragraph + Colab | — |

## 3. Docs site + examples gallery (weeks 2–6)

- Publish the docs site (`docs/site/`, GitHub Pages via `pages.yml`) with: pitch,
  install, the five tutorials, the API reference (`API.md`), the soundness
  contract (`SOUNDNESS_CONTRACT.md`), and the verifiable fragment
  (`VERIFIABLE_FRAGMENT.md`).
- **Examples gallery**: expand [`examples/`](../examples/) into a browsable
  "bug zoo" (one file per bug class from `bugclasses.jsonl`), each with the red
  TensorGuard diagnostic and the one-line fix.

## 4. Editor + CI surface (weeks 2–8)

- Ship the **VS Code extension** (`editors/vscode/`) to the Marketplace; the LSP
  server (`src/lsp_server.py`) powers inline diagnostics.
- Promote the **GitHub Action** (`action.yml`) and **pre-commit hook**
  (`.pre-commit-hooks.yml`) — "add 3 lines to your CI, never ship a shape bug".
- A **public leaderboard** (`benchmarks/leaderboard_entries/`, `leaderboard.yml`)
  invites other tools to submit; being measured against TensorGuard drives links.

## 5. Content cadence (ongoing)

Five blog posts, each anchored on an existing artifact:

1. *"Why your `nn.Linear` typo costs 40 minutes"* — the case for static checks.
2. *"Sound by construction: zero false positives, and how we prove it"* —
   `SOUNDNESS_CONTRACT.md` + the Lean audit (`lean/`).
3. *"Gating the whole PyTorch lifecycle"* — compile/ONNX/AOT/accelerate/Lightning.
4. *"Teaching the verifier a third-party layer in 20 lines"* —
   the community stub registry (`community_stubs/`).
5. *"Beyond PyTorch: the same verifier on Flax"* — the second-framework frontend
   (`src/flax_extractor.py`).

## 6. Community + governance (ongoing)

- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `GOVERNANCE.md`, and `MAINTAINERS.md`
  are in place; **good-first-issue** labels target new operator stubs and
  community shape-stub manifests (declarative, CI-validated — see
  `community_stubs/README.md`).
- Triage SLA: respond to issues within 48h; tag every merged stub with provenance.
- Monthly "new operators / new corpora" release notes (`CHANGELOG.md`).

## 7. Credibility flywheel (ongoing)

- The **artifact capsule** (`bash capsule/reproduce.sh`) makes every number in
  the README and papers one-command reproducible — reviewers and skeptics become
  advocates.
- Land the **workshop/conference paper** (wired to the capsule, Step 177) and the
  **upstream RFC**; each citation and cross-link compounds discovery.

## 8. Metrics to watch

| Metric | Target (6 months) |
| --- | --- |
| GitHub stars | 1000 |
| GitHub-source installs / clones | 5k |
| External leaderboard submissions | ≥3 tools |
| Community stub manifests merged | ≥25 |
| Notebooks run / Colab opens | tracked via badge referrer |

The north star is **time-to-first-caught-bug**: minimize the seconds between
`pip install` and a user seeing TensorGuard flag a real bug in *their* model.
