# Symbolic-execution corpus

A curated, data-driven corpus exercised by `tests/test_symexec_corpus.py`.

* **`wild/`** — minimal trimmed reproductions of real-world (and representative)
  defects. Each file is expected to fire **exactly one** named bug kind at a
  pinned line (Step 71, the regression corpus). The files include the original
  motivating issues — `titans-pytorch#60` (return-arity contract),
  `OpenStrawberry#113` (rank-dependent indexing), `vector-quantize-pytorch#248`
  (file does not parse) — plus one representative repro per shape detector
  (broadcast / matmul / reshape / `nn.Linear` / `cat` / `einsum`).

* **`correct/`** — correct models that must produce **zero reports**: the
  no-false-positive half of the soundness suite (Step 72). They cover the same
  operators as `wild/` in their *correct* form, plus cases the engine must stay
  silent on by **abstaining** (unknown-rank receivers).

Every file is a pure, faithful repro: no headers or docstrings are added (so the
bug line numbers stay stable). All expectations live in `manifest.json`, keyed by
filename, so the corpus is purely additive — drop in a new `.py` and add a
manifest entry and the test picks it up automatically.
