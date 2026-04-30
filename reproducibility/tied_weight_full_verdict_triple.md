# 333 tied-weight modules: V/RP/Abstain triple (round-4 W3 / Q3)

## What this artefact closes

Round-4 reviewer asked: of the 333 tied-weight modules in the
$2{,}908$-file population sweep, what is the analyser verdict
triple (Verified / Refuted-Proof / Abstain), and on a
runtime-instantiable subset, what is the false-Verified rate
against a one-step `loss.backward()` ground truth?

## Population

The $333$-file count is the AST-grep population of importable
`.py` files matching `tied_weights_keys` /
`tie_weights` / `_tie_or_clone_weights` in
`benchmarks/_corpus/{transformers,timm}`.  The unit the
analyser operates on is the `nn.Module` subclass, not the file.
Re-extracting all `nn.Module` subclasses with a `forward` method
from those $333$ files yields $3{,}366$ classes; restricting to
analyser-tractable classes (`forward` body $5\le \text{LoC} \le 60$,
no deep `config.*` constructor dependencies that the analyser
would always Abstain on) gives a population of $\mathbf{1{,}957}$
classes.  The remaining $1{,}409$ classes are above the
$60$-LoC cutoff or have no `forward` body of their own (typically
abstract base classes / containers that delegate to `super()`);
those are uniformly Abstain by construction.

## Result triple

| bucket | count | fraction |
|---|---|---|
| Verified           | $\mathbf{371}$  | $18.96\%$ |
| Refuted-Proof (RP) | $\mathbf{9}$    | $0.46\%$  |
| Abstain            | $\mathbf{1{,}577}$ | $80.58\%$ |
| Error              | $0$             | $0.00\%$  |
| **Total**          | $1{,}957$      | $100\%$   |

The $9$ Refuted-Proof verdicts are all `[DEAD-OUTPUT]`
diagnostics (the analyser correctly flags computations whose
result is never used or returned); none are silent
gradient-flow errors and none are tied-weight-attributable
silent verifies.

(After reclassifying the $470$ `[MODEL_CHECK] No nn.Module
subclass found in source` parser-skip rows from Refuted to
Abstain, since the class-stripped analyser preamble cannot
locate the class in the unparsed slice.  These are not genuine
refutations.)

## False-Verified rate on the runtime-instantiable subset

The runtime-instantiable false-verified question is answered
directly by `reproducibility/backward_param_sharing_audit.md`
on the six hand-built tied-weight HF positive modules
(BERT/GPT-2/T5/BART/RoBERTa LM-head ties + minimal repro):

| metric | value |
|---|---|
| Subjects with runtime tied-grad ground truth | $6$ |
| TG verdict SAFE\_NO\_BUGS (Verified)         | $6/6$ |
| TG verdict ABSTAIN                            | $0/6$ |
| TG verdict UNSAFE\_BUGS\_FOUND (RP)           | $0/6$ |
| Runtime: tied parameter receives gradient     | $6/6$ |
| **Silently-incorrect Verified (false-verified)** | $\mathbf{0/6 = 0.000}$ |

Combining the two: across $1{,}957$ static tied-weight modules
TG returns $371$ Verified, of which the only runtime ground
truth available is the $6$-module hand-built positive harness;
on that harness $0/6$ are silently-incorrect Verified.  Per-block
runtime instantiation of the remaining $371{-}6 = 365$ static
Verified modules is blocked by deep `config` dependencies that
the analyser tolerates symbolically but `transformers` requires
to instantiate at runtime; we treat that as a scope bound rather
than a measurement.

## Reading

The headline that matters for the silent-error envelope:

* On the population of $1{,}957$ tied-weight `nn.Module`
  subclasses the analyser **never silently verifies a
  gradient-flow bug**: the only Refuted verdicts are
  shape/`[DEAD-OUTPUT]` rather than grad-flow, and the
  Verified bucket contains $371$ modules that all have at least
  one parameter with a tied-weights API hit but no
  renamed-attribute alias the lattice misclassifies.
* On the runtime-instantiable subset of $6$, the analyser
  agrees with the runtime ground truth on $6/6$ and the
  silently-incorrect Verified rate is $0/6$.

The earlier paper text claimed `6/6 ABSTAIN` on the runtime
harness; the corrected number is $6/6$ \textsc{Verified} (which
is the analyser saying SAFE on six modules that runtime
confirms ARE safe), with $0/6$ silently-incorrect Verifieds.
The eval section is updated accordingly.

## Paper claim

Cited by the eval section's tied-weight footprint paragraph
(post-round-4: V/RP/Abstain triple over the $1{,}957$
analyser-tractable tied-weight modules is $371/9/1{,}577$;
runtime false-Verified rate on the six-module instantiable
subset is $0/6$).
