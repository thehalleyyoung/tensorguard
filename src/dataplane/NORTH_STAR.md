# DataRefine — North Star

> **One sentence.** DataRefine is a *refinement-type, effect, and
> information-flow system for the data plane of deep learning*: every value that
> moves through a PyTorch / pandas pipeline carries a typed contract, every
> operator is a typed transfer function over those contracts, and **every "wrong
> way to use data" is a typing or non-interference violation** — discovered by
> abstract interpretation over a small *decidable* intermediate representation and
> discharged by an SMT solver, **without ever modelling the full semantics of
> Python or of tensors.**

This document is the load-bearing vision. Everything in the repo should either
*be* a piece of this system or be deleted. The seven scanners that exist today
(pipeline leakage, split contracts, temporal causality, group leakage, sampling
determinism, join cardinality, value domain) are **not** the product — they are
seven early *instances* of one calculus that was never given a spine. The work
is to build that spine and let the instances fall out of it.

---

## 1. The thesis: a deep-learning program has two layers, and the data layer is untyped

A PyTorch program is two interleaved computations:

1. a **model plane** — the differentiable tensor algebra (`nn.Module.forward`,
   autograd, optimizers). This is what shape/dtype verifiers like
   [`thehalleyyoung/tensorguard`](https://github.com/thehalleyyoung/tensorguard)
   already type. **DataRefine does not touch this plane.**
2. a **data plane** — everything that decides *which numbers reach the model and
   what they mean*: dataset construction, splitting, normalisation, augmentation,
   joins/merges, resampling, batching, label construction, loss-input
   preparation, metric computation.

The model plane is heavily typed (shapes, dtypes, devices, even gradients). The
**data plane is almost entirely untyped**, and that is exactly where the
expensive, silent, irreproducible bugs live — the ones that produce a model that
*runs*, *trains*, and *reports a great number* while being quietly wrong. A leaked
test row, a normaliser fit on the full dataset, a label shifted the wrong way, a
loss fed un-normalised logits: none of these throw. They corrupt the *meaning* of
the result.

**The north star is to give the data plane a type system as rigorous as the one
the model plane already enjoys.**

---

## 2. The unifying formalism

Three composable judgements over the data plane, each decided by the *cheapest
sound mechanism*, never by symbolic execution of Python:

### 2.1 Refinement types (what a value *is*)
Every data value `v` (tensor, column, frame, dataset, batch, scalar) is assigned
a **refinement** drawn from a product lattice of independent sub-domains:

| sub-domain | example points | decided by |
|---|---|---|
| **value-domain** | `prob∈[0,1]`, `log_prob≤0`, `logit∈ℝ`, `unit`, `nonneg`, `integer-id` | SMT (LRA/LIA) |
| **distributional** | `standardised(μ≈0,σ≈1)`, `min-max`, `raw`, `clipped` | SMT + abstract domain |
| **schema/role** | `feature`, `target`, `group-key`, `time-index`, `weight`, `id` | lattice |
| **shape** *(delegated)* | `(B,C)` — **owned by TensorGuard, never re-derived** | external |
| **missingness** | `complete`, `may-be-nan`, `imputed(train-stats)` | lattice + SMT |
| **units / encoding** | `radians`, `log-space`, `one-hot`, `token-ids` | lattice |
| **cardinality** | `rows=n`, `key-unique`, `multiplicity≥2` | SMT (LIA) |

A refinement type **`{v : τ | φ(v)}`** is a base sort plus an SMT-checkable
predicate. Operators have **dependent function types**: `torch.sigmoid :
{x:Tensor|⊤} → {y:Tensor| 0≤y≤1}`; `BCELoss.__call__ : {p | 0≤p≤1} × {y} → loss`.
Calling `BCELoss` on a value whose inferred refinement does **not** entail
`0≤p≤1` is a **subtyping failure**, decided by `entail(φ_actual ⇒ φ_required)` in
z3. This already subsumes the `value_domain` scanner — V1/V2 are just two
operators in the signature table.

### 2.2 Effects (what an operator *does to obligations*)
Borrowing the existing `effects.TransformEffect`, every operator carries an
effect signature over a set of **standing obligations**: `required`,
`preserved`, `strengthened`, `weakened`, `discharged`, `invalidated`. A
pipeline is a *composition* of effects (`effects.propagate_obligations` /
`ObligationGraph`), and a bug is an obligation that is **`required` at a sink but
not `discharged` upstream** (or one `invalidated` and silently relied on). This is
the algebra that makes "fit the scaler, *then* split" different from "split,
*then* fit the scaler": same operators, different effect ordering, one discharges
the isolation obligation and one violates it.

### 2.3 Information flow / non-interference (what a value is *allowed to influence*)
The deepest class — **leakage** — is an *information-flow* property, not a value
property. Label a partition of provenance:

```
HIGH  = {test/held-out rows, the target column, the future w.r.t. a cut, the
         test-time distribution, any statistic fitted over HIGH data}
LOW   = {train rows, features, the past, anything the model may legitimately see}
```

The non-interference theorem we want: **no HIGH value may influence a LOW
computation that the model or its selection consumes.** Every leakage bug is a
violation of this single theorem:

- fit/normalise **before** split → a statistic fitted on `{all rows}` (HIGH-tainted)
  flows into the train transform → interference.
- group split that puts a patient in both folds → the test partition's identity
  influences a train row → interference.
- temporal lookahead → a *future* (HIGH) value flows into a *past* feature.
- join fan-out → test rows duplicated into train via a many-to-many merge.
- target leakage → the label flows into a feature.

Non-interference is decided by **lattice monotonicity over the provenance
labelling** (cheap, sound) — *not* SMT — with SMT only for the arithmetic side
conditions (e.g. "do these two index intervals overlap?"). The `information_flow`
module is the seed of this; it has never been driven by a real program.

> **The punchline:** refinement subtyping (§2.1) + effect composition (§2.2) +
> non-interference (§2.3) is, conjecturally, *enough to express every data-plane
> correctness property we care about.* The three judgements share one
> obligation IR and one certifier. That shared IR is the spine that is missing
> today.

---

## 3. The engine: abstract interpretation over a decidable data-plane IR

The genius constraint — repeated throughout this project — is **never reason
about full Python/tensor semantics.** We honour it by analysing a *radically
abstracted* IR:

1. **Lower** source (`.py`, notebook cells) to a **data-plane dataflow graph**:
   nodes are operator applications drawn from a known vocabulary (torch, pandas,
   sklearn, numpy data ops); everything unrecognised is an **opaque** node that
   produces `⊤` (unknown refinement) and is **never blamed** (precision-first).
2. **Abstractly interpret** the graph: maintain an environment `name → Refinement`
   and a standing `ObligationGraph`. Each operator resolves to (a) a refinement
   transfer function and (b) an effect signature. Control-flow merges take the
   **lattice join** `⊔`. This is a *finite-height* abstract interpretation — it
   terminates, it never executes user code, it needs no loop unrolling beyond
   widening on the (tiny) refinement lattice.
3. **Emit obligations at sinks** (loss calls, `.fit`/`.transform`, splits,
   merges, samplers, metric calls). Discharge them with the existing
   `StructuralCertifier`: SMT for refinement/cardinality, lattice for flow.
4. **Witness or admit.** A violated obligation comes back with a *concrete
   counterexample* (an out-of-domain value, an overlapping index, a fan-out
   factor, a HIGH→LOW path), re-checkable by `smt_backend.recheck`. Admissions
   carry the discharging proof.

This is one front end + one lattice + one obligation IR + one certifier. **Adding
a new bug class becomes: add operator signatures and a sink — not a new
scanner.** The seven existing scanners collapse into seeded entries of the
operator/effect tables.

```
source ──lower──▶ data-plane IR ──abstract interp.──▶ obligations ──certify──▶ {witness | proof}
                       │                  │                                  ▲
              operator vocabulary   Refinement lattice  ◀── effect signatures ┘
              (torch/pandas/sklearn)  (§2.1 product)        (§2.2 algebra)
```

---

## 4. Speculative frontier — bugs we want to reach (ideate widely)

The point of the spine is *reach*. Once values are refinement-typed and flows are
labelled, an enormous catalogue of currently-invisible data-plane bugs becomes a
typing query. A deliberately ambitious, partly-speculative wishlist:

**Leakage / non-interference (the crown jewels)**
- Normalisation, imputation, feature selection, target encoding, PCA, vocabulary
  building, tokenizer fitting, or class-weight computation performed over
  test-or-full data (statistic-fit leakage — *the* most common real bug).
- Group/entity leakage across *any* equivalence key (patient, user, session,
  molecule, document, image-series) — including transitive keys via joins.
- Temporal lookahead in *any* form: negative shifts, rolling windows that include
  the current row, resampling that averages future into past, target built from
  `t+k`, train/test split not respecting time order, embargo/purge violations in
  cross-validation.
- Duplicate / near-duplicate rows straddling the split (exact, hash, or join-fanout).
- Test-set tuning: hyperparameters / early-stopping / thresholds selected on the
  test split; "peeking" loops.
- Distribution leakage: BatchNorm running stats updated in eval; normalising the
  *test* batch by its *own* statistics.

**Refinement / value-domain**
- Loss/activation domain mismatches (BCE-on-logits, NLL-on-probs, `log` of a
  possibly-zero/negative, `sqrt`/`log` of an un-clamped value, softmax over the
  wrong axis *as a value property*).
- Label encoding faults: off-by-one class indices, `ignore_index` mismatch,
  one-hot vs index confusion, regression target on a classification loss.
- Units / scale faults: mixing standardised and raw features, angles in
  degrees vs radians, log-space vs linear, probabilities summed across the wrong
  axis, metrics computed in the wrong space.
- Masking/padding faults: padded positions contributing to the loss; attention
  mask polarity inverted; `nan`/`inf` propagating through an unguarded reduction.

**Distribution & sampling**
- Class-imbalance handling that double-corrects (weighted sampler *and* weighted
  loss), or that biases the eval metric.
- Non-deterministic eval; uncontrolled RNG across workers (the documented NumPy
  fork bug) — generalised to *any* impurity that breaks per-sample independence.
- Train/serve skew: a transform present in training but absent at inference (or
  vice-versa) — an effect-signature mismatch between two pipelines.
- Data-augmentation applied to the *label* when it should only touch the input
  (or to eval data).

**Structural / relational**
- Many-to-many merges, accidental Cartesian products, index misalignment in
  concat/merge, row-order dependence after a shuffle, `reset_index` faults.
- Silent dtype coercions that lose information (float64→float32 of an id,
  int overflow in a count, categorical→object).

**Truly speculative (north-of-north-star)**
- *Differential pipeline equivalence*: prove that the training and serving
  pipelines induce the **same** transform on a feature (refutational train/serve
  skew detection) by comparing their effect signatures.
- *Statistical-power / metric-validity obligations*: flag a reported metric whose
  obligation graph shows its evaluation set was contaminated, so the number is
  **not entitled to be believed** — a "this benchmark is inadmissible" verdict.
- *Provenance-carrying tensors at runtime*: an opt-in `RefinedTensor` that
  carries its refinement and split-label through eager execution, turning the
  static obligations into runtime contracts that *fail loudly* the first time a
  HIGH value reaches a LOW sink.
- *Cross-framework*: the same IR fed by JAX/TF/polars front ends — the data plane
  bugs are framework-independent; only the operator vocabulary changes.
- *Proof-carrying datasets*: a dataset ships with a DataRefine certificate that
  downstream consumers re-check, so "this split is leakage-free" composes across
  teams the way TensorGuard's shape proofs compose across modules.

---

## 5. Why this is the *right* north star

- **It is one idea, not seven.** Coherence is the whole point of this rewrite.
- **It composes with TensorGuard, not against it.** Shapes are theirs; meaning,
  provenance, and flow are ours. Together they fully type a PyTorch program.
- **It honours the decidability constraint.** Abstract interpretation over a
  finite refinement lattice + SMT side conditions is sound and terminating; we
  never chase Python's semantics.
- **It is honest about recall.** Opaque producers are `⊤` and never blamed; every
  blame ships a re-checkable counterexample. The system is *sound on what it
  claims*, explicit about what it cannot see.
- **It has unbounded headroom.** §4 shows years of bug classes reachable by
  adding signatures to one engine.

---

## 6. Build order (the spine first, then let the instances fall out)

1. **The engine** (`datarefine/dataplane.py`): the data-plane IR, the `Refinement`
   product lattice (with `⊔` and `⊑`), the operator/effect registry, the abstract
   interpreter, sink obligation emission, and discharge via the existing
   certifier. ← *start here.* **✅ done** (the `DataPlaneInterpreter`).
2. **Re-express `value_domain`** as operator signatures over the engine (proof
   that the calculus reproduces a shipped scanner exactly). **✅ done** — engine
   matches `value_domain` on all 12 validator fixtures.
3. **Re-express the leakage family** as non-interference over the provenance
   labelling (proof that the *same* engine spans a second, very different axis).
   **✅ done** — fit-before-split sink over `fit_transform_isolation`.
4. Migrate temporal, group, sampling, join — each becomes operator signatures +
   a sink. **✅ done** — temporal is a native sink (`_emit_temporal`); group,
   join, sampling, and split-contracts are unified through the `analyze_all`
   front door via `_emit_structural_families`, each re-born as an engine
   obligation and (where a faithful structural reconstruction exists)
   re-discharged by the engine's own certifier, with per-family parity to every
   shipped scanner. **One `analyze_all` run now spans all seven bug axes.**
5. Retire `scanners.py`'s reconstruction bridge: once findings are *born* as
   obligations from the engine, nothing needs to lift them back. ← *next:* the
   delegated families currently still *recognise* via their scanner modules and
   lift via `finding_to_obligation`; the remaining work is to move their
   recognition into native operator signatures so the lift becomes unnecessary.
6. Expand the operator vocabulary toward §4; each addition is a few table rows.

**Definition of done for step 1 (achieved):** a single interpreter that, from real
source, infers refinements through a chain of ops, emits a value-domain obligation
at a loss sink *and* a non-interference obligation at a fit-before-split sink, and
discharges both — reproducing the existing value-domain verdicts while proving the
engine already spans two independent bug axes.

**Unification milestone (achieved):** `analyze_all` is the single DataRefine front
door — one call emits obligations across **all seven** structural bug axes
(refinement, non-interference, temporal, split, group, join, sampling), each
discharged by one certifier, each at per-family parity with its shipped scanner.
The remaining north-star work is depth (steps 5–6): migrate the delegated
families' *recognition* into native operator signatures and widen the vocabulary
toward §4.

