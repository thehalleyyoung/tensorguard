# Merging PromptABI into TensorGuard: the interface layer

This note records the merge of **PromptABI** (a static verifier for the discrete
tokenizer / chat-template / stop-sequence / tool-calling boundary of LLM apps)
into **TensorGuard** (a static verifier for the PyTorch tensor plane). It states
exactly **what was merged, what was redundant, and what is synergistic**, so the
union is coherent rather than a pile of tools.

## The two planes (why the merge is coherent, not redundant)

TensorGuard verifies the **tensor plane**: a program's shapes, broadcasting,
devices, dtypes, training-loop numerics, and semantic axis roles. PromptABI
verifies the **interface plane**: the *text/token* boundary where data crosses
into and out of an LLM. They are the two halves of "is the data crossing the model
boundary well-formed?" Both follow the same discipline — **sound-or-abstain
decisions over finite abstractions (automata, SMT), CPU-only, no weights, with
replayable witnesses** — so they compose into one tool with one philosophy.

Merged code lives in `src/interface_layer/` (a closed 15-module dependency
closure copied verbatim; all imports are internal). Four capabilities are wired
into the `tensorguard` CLI; the rest are available as a library.

## What was MERGED (new + synergistic)

### 1. The discrete LLM text/token interface plane — entirely new to TensorGuard

TensorGuard had **nothing** at the tokenizer/template/stop/tool boundary. These
analyzers add a whole new, non-overlapping capability area:

| module | decides | CLI |
| --- | --- | --- |
| `constrained_decoding_feasibility` | does a guided-decoding grammar admit a tokenization? (decoder-stall via greatest-fixpoint; expressivity-gap via bounded SMT) | `prove-decoding-feasibility` |
| `surface_ban_soundness` | does an id-level `bad_words`/`suppress_tokens` ban actually prevent a forbidden *surface*? (KMP×Aho–Corasick product automaton, unbounded) | `prove-surface-ban` |
| `streaming_stop_soundness` | can a server truncate output exactly at a stop string? (overshoot / split-stop, unbounded) | `prove-streaming-stop` |
| `smt_tokenizer_forgery` | can attacker text forge a control-token boundary through the tokenizer? (SMT forge-or-prove) | library |
| `role_separator_forgery`, `control_token_injection`, `normalization_confusables`, `chat_templates`, `tokenizers`, `normalizers` | role/control-token forgery and Unicode-normalization confusables at the template boundary | library |
| `formal` | the shared automata kernel (DFA, products) these provers stand on | — |

These are synergistic with TensorGuard's mission ("catch silent ML bugs before
deployment") but cover a plane TensorGuard could not previously see.

### 2. `torch_data_misuse` — a data-pipeline complement to `training_loop_checks`

This one lands squarely in TensorGuard's own PyTorch domain and is **additive, not
duplicative**. `src/training_loop_checks.py` already covers gradient-flow breaks,
missing `zero_grad`, and autocast/`GradScaler`. `torch_data_misuse` adds three
silent `DataLoader`/split bugs TensorGuard did **not** check:

* **worker-rng-duplication** — `numpy`/`random` augmentation in `Dataset.__getitem__`
  with `DataLoader(num_workers>0)` and no `worker_init_fn` → identical
  augmentations across workers.
* **drop-last-on-eval** — `drop_last=True` on a val/test loader → silently dropped
  eval samples.
* **fit-before-split-leakage** — a `fit`/`fit_transform` over the full data before
  `train_test_split`/`random_split` → leaked test statistics.

Wired as the `scan-torch-data` CLI command.

## What was REDUNDANT (deliberately NOT merged)

PromptABI's data-plane ambitions (see its `NORTH_STAR.md`) overlap with mature
TensorGuard subsystems. We did **not** copy anything in these areas; TensorGuard's
implementations are the canonical ones:

* **Tensor shape / broadcast / device / dtype verification** — TensorGuard
  `tensor_shapes.py`, `domains/broadcast_analysis.py`, `domains/*`.
* **Loss numeric/shape contracts** (`cross_entropy`/`nll_loss` operand alignment,
  reduction dims) — TensorGuard `loss_verify.py`.
* **Training-loop hazards** (detached loss, missing `zero_grad`, autocast scaler)
  — TensorGuard `training_loop_checks.py`.
* **Semantic axis/role + grad-flow typing** (the "axis/role type system" PromptABI's
  North Star proposed building) — TensorGuard **already has it** in
  `intent_bugs.py` (DimRoleSort / DeviceSort / DTypeSort / GradFlowSort / PhaseSort).
* **Runtime/object-level silent-bug gates** — TensorGuard `silent_bug_checks.py`.

We also dropped PromptABI's large **ecosystem/marketing/governance sprawl**
(adoption playbooks, conference demos, benchmark leaderboards, roadmap/award
modules, etc.) — it carries no verification capability and TensorGuard already has
its own packaging/CI/governance surface.

## Net result

TensorGuard now verifies **both** sides of the model boundary:

* the **tensor plane** (its existing core), and
* the **interface plane** (the merged `src/interface_layer/`), with four new CLI
  provers (`scan-torch-data`, `prove-surface-ban`, `prove-streaming-stop`,
  `prove-decoding-feasibility`) and a library surface for the template/forgery
  analyzers.

Tests: `tests/interface_layer/` (39, all passing). The merge added no new
dependency — `z3-solver` is already a TensorGuard core dependency.
