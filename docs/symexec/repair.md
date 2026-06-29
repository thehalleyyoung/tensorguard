# Verified & SMT-Synthesized Auto-Repair (`tensorguard fix`)

TensorGuard does not only *find* shape, autograd, and module-registration bugs —
it **repairs** them. This page explains how the repair engine works, why every
fix it surfaces is sound, and how the SMT solver is used to *synthesize* fixes
that have no obvious syntactic patch.

The implementation lives in `src/symexec/autofix.py` (the verified-repair loop and
the pattern strategies) and `src/symexec/fix_synth.py` (the SMT-synthesized
proposals). The CLI is `tensorguard fix`.

---

## 1. The guarantee: re-verified, not guessed

Most auto-fix tools emit a textual suggestion and hope. TensorGuard can do
something strictly stronger, because the analyzer is **sound and fast**: it
*re-runs itself on the patched program* and only surfaces a fix when

1. the **targeted bug is gone**, and
2. **no new bug *kind*** appears anywhere in the patched program.

```
              ┌──────────────── propose ────────────────┐
   bug ──►  pattern strategy  ∪  SMT synthesizer   ──►  FixCandidate
                                                          │
                                                          ▼
                                                   verify_fix(): re-run
                                                   the engine on the patch
                                                          │
                                   targeted bug gone  ∧  no new bug kind
                                                          │
                                                          ▼
                                                   surfaced as a VerifiedFix
```

This is the central design choice (`autofix.verify_fix`): the **proposal** step
is allowed to be heuristic or solver-driven or speculative, because the
**acceptance** step is a sound re-verification. A fix is only ever shown — or
written to disk with `--write` — once the engine has *proved* the program no
longer triggers the bug. Comparing **bug *kinds*** (not exact line/coordinates)
on the "no new bug" check makes the gate robust to line shifts introduced by a
fix that inserts or deletes lines.

Because repair only ever *adds a second consumer* of the analyzer, it cannot
change which bugs report on the original program, and it never perturbs the
deterministic analysis fingerprint.

---

## 2. Two proposal channels

`propose_fix(bug, source)` dispatches by `bug.kind` to a registered strategy.
There are two kinds of strategy, both with the same signature
`(_lines, bug) -> (patched_source, name, description) | None`:

### 2a. Pattern strategies (`autofix.py`)

Direct, canonical rewrites for bugs whose repair is syntactically unambiguous:

| Bug kind | Repair |
|---|---|
| `missing_super_init` | insert `super().__init__()` as the first statement of `__init__` |
| `direct_forward_call` | `module.forward(x)` → `module(x)` (so hooks/`__call__` run) |
| `tensor_data_access` | `.data` → `.detach()` (autograd-safe) |
| `negative_dimension` | drop the sign on a negative dimension literal |
| `layer_dim_mismatch` | set the unique layer's first ctor arg (`nn.Linear`/`nn.Conv{1,2,3}d`/`nn.BatchNorm{1,2,3}d`) to the size flowing in |

Each abstains (returns `None`) the instant the edit would be ambiguous — e.g.
`layer_dim_mismatch` only fires when *exactly one* layer definition matches the
declared size (`nn.Linear(<declared>, …)`, `nn.Conv2d(<declared>, …)`, …), never
when two layers share it.

### 2b. SMT-synthesized strategies (`fix_synth.py`)

For bugs where the *correct value* is a function of shapes elsewhere in the
graph, we pose the repair as a constraint-satisfaction query and read the fix off
a **satisfying model** from the same Z3 bridge the prover uses
(`src/symexec/smt_bridge.py`). The solver does the thinking; the re-verification
gate keeps it honest.

---

## 3. The synthesis primitive

`smt_bridge.model(constraints)` returns a concrete satisfying assignment
(variable → integer) for a conjunction of affine `SymDim` constraints, or `None`
when unsatisfiable/unknown. That is the whole trick: **encode "what value makes
the precondition hold?" as constraints, then read the value out of the model.**

`fix_synth.solve_inferred_factor(prod_others, numel)` is the canonical example.
It solves

> `prod_others · f == numel ∧ f ≥ 1`

for a positive integer `f`, and additionally checks **uniqueness** by asserting
`f ≠ cand` and requiring that to be `unsat`. Routing this through the solver
(rather than a bare `numel % prod_others`) means the same primitive keeps working
when `numel` becomes a *symbolic* dimension product — `f` is still read off the
model. When Z3 is unavailable it falls back to an exact concrete division, so it
is never unsound, only less general.

---

## 4. The synthesized repairs

### 4a. Reshape — *preserve the user's intent* (`R1`)

A `reshape_size_mismatch` fires when a `.reshape(t₀,…,tₘ)` / `.view(...)` target
has `∏tⱼ ≠ numel`. The blunt fix is `.reshape(-1)` (flatten), which always works
but **discards every dimension the user intended**.

The synthesized fix is smarter. For each position `j`, let
`prod_others = ∏_{i≠j} tᵢ` and ask the solver for the inferred factor
`f = solve_inferred_factor(prod_others, numel)`. If **exactly one** position
`j` admits a positive integer `f ≠ tⱼ`, we rewrite only that position to `-1`:

```
x.reshape(6, 5)   # numel = 24
        ▲
        └── 6·f = 24 ⇒ f = 4, unique ⇒ keep the 6, infer the other factor
x.reshape(6, -1)  # ✓ verified
```

If no single-factor repair exists (e.g. `(5, 5)` for `numel = 24`: neither kept
product `5` divides `24`), the synthesizer abstains and `autofix` falls back to
the flatten strategy. This is the "ingenious" part: the solver lets us make the
*minimal* edit that respects the programmer's stated shape, instead of collapsing
everything.

### 4b. Matmul — *detect a transposed operand* (`R2`)

A `matmul_dim_mismatch` on `a @ b` means `a[-1] ≠ b[-2]`. A very common cause is
that the right operand is **stored transposed** (weights saved as `(out, in)`,
data laid out column-major, etc.). Transposing `b`'s last two axes changes the
contraction requirement to `a[-1] == b[-1]`, so the transpose is the right fix
**iff** `a[-1] == b[-1]` (and the original `a[-1] ≠ b[-2]` really is a mismatch).

`_transpose_fixes_contraction(a, b)` checks exactly this, routing the
mismatch/fixed conditions through the Z3 bridge so the reasoning extends to
symbolic dims:

```
a:(2,3) @ b:(5,3)      # 3 ≠ 5  → mismatch;  a[-1]=3 == b[-1]=3  → transpose fixes it
c = a @ b.transpose(-1, -2)   # ✓ verified
```

When the transpose does **not** restore compatibility (`a:(2,3) @ b:(4,5)`), the
synthesizer abstains rather than emit a wrong rewrite. It also requires the line
to contain a single, simply-named `@` so the operand is locatable unambiguously.

### 4c. Repeat — *left-pad the size list to the rank* (`R5a`)

`tensor.repeat(*sizes)` requires **at least one size per tensor dimension**;
`repeat_dims_too_few` fires when fewer are given. Torch aligns the provided sizes
to the **trailing** axes (broadcasting-style), so the unique, intent-preserving
fix is to **prepend `1`s** — no-op repeats on the new leading axes — until the
list has one entry per dimension:

```
x:(2,3,4)                 # rank 3
x.repeat(2)               # only 1 size  → RuntimeError
x.repeat(1, 1, 2)         # ✓ verified — user's 2 stays on the last axis
```

Both call forms are handled (`x.repeat(2)` and `x.repeat((2,))`); the
synthesizer abstains when the receiver is not uniquely locatable (e.g. two
`.repeat(` calls on one line) or the arguments are not a simple literal list.

---

## 5. Applying fixes (`tensorguard fix`)

```bash
tensorguard fix model.py            # show verified unified diffs
tensorguard fix model.py --write    # apply them in place
tensorguard fix src/ --format json  # machine-readable (for CI bots)
tensorguard fix model.py --unverified   # also show rejected candidates + reason
```

`--write` applies fixes **iteratively**: after each edit it re-runs `repair()` on
the evolving buffer, so multiple fixes compose deterministically and line numbers
stay consistent. A second `fix` run on an already-repaired file reports nothing
and leaves the file byte-identical (idempotence is covered by the test suite).

---

## 6. Soundness summary

- **Synthesized-and-re-verified** fixes inherit the engine's soundness: the
  patched program is *proved* to no longer trigger the bug, with no new bug kind.
  This is strictly stronger than any LLM/codemod fixer — "re-checked", not "looks
  right".
- **Abstention over risk.** Every proposal channel returns `None` on ambiguity
  (≥2 minimal fixes, an un-locatable token) or when the solver answers
  `unknown`/`unsat`. The same discipline that gives the *detector* zero false
  positives gives the *fixer* zero unsafe edits.
- **No effect on analysis.** Repair is a fix-only code path. It never changes
  which bugs report on the original program, and never perturbs the analysis
  fingerprint.

---

## 7. Roadmap

The synthesis approach generalizes to many more bug kinds (broadcast-aligning
`unsqueeze`, `expand`/`repeat` argument vectors, in-range axis selection for
`cat`/reductions, einsum subscript repair, and a CEGAR-style quantified
"∃ edit. ∀ inputs. precondition holds" driver). Each new synthesizer plugs into
the same `_STRATEGIES` table and is gated by the same re-verification loop, so the
soundness story above holds unchanged as coverage grows.
