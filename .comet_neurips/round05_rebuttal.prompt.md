# Role: paper authors writing a brief rebuttal

A NeurIPS reviewer just posted the review below on your paper. Before
you start any code or paper changes this round, you have ONE chance to
push back on weaknesses you believe are misweighted, factually wrong,
or already-resolved-in-the-current-repo. The next round's reviewer
WILL read this rebuttal and must either accept it (drop the weakness)
or sharpen it (restate with a concrete counter-example).

## The review you are rebutting
## Summary
This paper presents TensorGuard, a static refinement-type verifier for PyTorch `nn.Module` forward methods that reasons about tensor shapes and a coarse gradient-flow property directly from class source, without instantiation or tracing. The technical core is a refinement calculus with assume/guarantee composition at module boundaries, plus a Lean-audited operator-rule table and a mechanized 17-operator composition fragment. Empirically, the paper claims `53/60` high-confidence refuted bugs on a historical bug corpus, a fragment-fair `32/34` vs. `25/34` win over Pytea, and `9/9` catches on naturally occurring HuggingFace-family bugs. On real-source library code, the canonical high-confidence regime reports `57` Verified, `128` Contract-Violation, `78` Library-Warn, and `225` Abstain on `488` blocks, with `0` unconditional RP on the unrestricted corpus but `26/356` on the empty-assume subset and `5` of those inside the audited footprint. The paper also calibrates its soundness claims by separating Lean-audited, pen-and-paper, tested-only, and out-of-scope handlers, and by quantifying a known backward-verifier limitation on tied / renamed-attribute parameter sharing.

## Prior weakness disposition
- [RESOLVED] The headline numerical claim in the abstract is in tension with Table `tab:headline`. The abstract advertises "26 unconditional... -- rebuttal accepted: Table `tab:headline` and the surrounding §4.1 text now explicitly distinguish unrestricted RP=`0` from the empty-`assume_M` subset count `26/356`, and the released artifacts support that reconciliation.
- [PARTIAL] C1's "joint shape-plus-grad" novelty rests on a grad lattice that is admitted to be silently incorrect on a 25% slice... -- the paper now quantifies and foregrounds the limitation, but the backward verifier still silently misverifies `2/8` positives on the worst-case tied/renamed-sharing family, so the novelty claim remains materially qualified.
- [RESOLVED] The "fragment-fair head-to-head" 32/34 vs 25/34 against Pytea is the only result with a frequentist significance... -- rebuttal accepted: the appendix now gives the full 34-row matched-pair table, the 60→34 filter rule is stated deterministically, and the released JSON reproduces the McNemar statistic.
- [RESOLVED] The "Bookkeeping note on the headline triple" ... reports four different `{V, R, A}` triples for the same 488-block corpus... -- rebuttal accepted: the current draft cleanly identifies the two axes (high-confidence vs default, original capture vs rerun), names the canonical snapshot, and the current released logs fit that bookkeeping story.
- [PARTIAL] C2 (assume/guarantee at the `nn.Module` boundary with contravariant/covariant subclassing) is, novelty-wise... -- the draft now cites Jones/Meyer/Findler and states the mechanized fragment more honestly, but the conceptual step still reads primarily as a framework-specific instantiation of standard contract subtyping.
- [PARTIAL] The "stub-mocked runtime sample on the 371-Verified subset" ... reports `0/25` silently-incorrect Verified with Wilson 95% CI... -- the authors added stronger complementary audits, but this specific sample is still only `25` rows with a `[0,13.32%]` interval and a shortest-LoC-first selection rule that overweights simple modules.
- [PARTIAL] The paper's distinctive empirical novelty — verdicts on un-instantiated class source — is most cleanly demonstrated by the inapplicability gap... -- the draft now adds `15/488`, `26/356`, and `5` audited-footprint unconditional catches, but the canonical unrestricted real-source headline still has `0/488` unconditional RP.

## Strengths
- The paper is unusually explicit about what is and is not covered by the theorem: RP/CV only, `Cat_sound` only, with tested-only and out-of-scope handlers separated rather than quietly absorbed.
- The mechanized artifact is substantive: the Lean development builds, the advertised 17-operator assume/guarantee fragment is concrete, and the operator-table audit is more serious than a typical ML-systems proof appendix.
- The empirical comparison to Pytea is now genuinely auditable; the filter, contingency table, and significance calculation are no longer black-box prose.
- The bug-finding results on historical bugs and naturally occurring HuggingFace-family bugs are strong and relevant to practice.

## Weaknesses
- The main soundness limitation remains substantial on real source: only `62/185` of the paper’s real-source Verified+CV verdicts lie wholly inside the Lean-or-pen-and-paper footprint, while `66/185` touch tested-only handlers and `57/185` touch only out-of-scope operators (`§4`, Table `tab:soundness-footprint-185`).
- The gradient-flow story is still materially weakened by the tied / renamed-attribute parameter-sharing failure mode: the runtime harness reports a `2/8 = 25%` false-Verified rate on that worst-case construct family (`§6`, `§4` runtime trainer audit).
- The stub-mocked validation on the `371` Verified tied-weight rows is not very convincing as population evidence: it samples shortest-LoC-first, succeeds on only `25` rows, and those rows are dominated by simple RMSNorm-like modules, so the reported Wilson interval `[0.00%, 13.32%]` is not tight and is selection-biased.
- The conceptual contribution around C2 still feels overstated. The theorem mechanizes composition for this DSL, but the core contravariant/covariant contract rule is standard, so the novelty seems to lie more in the PyTorch adaptation and audit packaging than in a new contract-theoretic idea.
- The paper’s most distinctive real-source claim is still weaker than the abstract framing suggests: the unrestricted `488`-block corpus yields `0` unconditional RP in the canonical regime, so the positive real-source story depends on the empty-assume subset, a rule-extension rerun, or the very small `5`-catch audited-footprint slice.
- The released artifact is not completely stable: the current test suite fails on a known bug-detection regression (`missing unsqueeze before broadcast`), which is uncomfortable for a paper whose empirical case leans heavily on a bug-catching benchmark and on implementation calibration.

## Questions
- What is the strongest real-source result that holds **strictly inside** the theorem-backed footprint, with no tested-only or out-of-scope handler anywhere on the relevant path?
- Why should C2 be read as a conceptual contribution beyond a framework-specific instantiation of standard contract/subtyping principles? What theorem obligation here is genuinely new?
- For the `371`-Verified tied-weight population, why use shortest-LoC-first rather than a stratified or random sample across handler families, and how sensitive is the `0/25` result to that selection rule?
- How should readers reconcile the current artifact regression on a broadcast/unsqueeze bug pattern with the paper’s broader bug-detection claims?

## Scores
Soundness: 2
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
I would move this to a 6 if the paper delivered a stronger **theorem-backed real-source result**, i.e. if a much larger fraction of the `488`-block Verified/CV/RP story were brought inside `Cat_sound` rather than resting on tested-only or out-of-scope handlers.

Round: 5


Changes   +0 -0
Requests  1 Premium (5m 10s)
Tokens    ↑ 952.7k • ↓ 9.4k • 826.9k (cached) • 4.3k (reasoning)

## Output requirements

Pick **at most 3** of the listed weaknesses. For each, write a
paragraph of strict format:

  ### Rebuttal of weakness: <verbatim wording, truncated to ~100 chars>
  Concise argument (4-8 sentences) for why this weakness is
  overweighted, factually wrong, or already addressed. Cite specific
  artifacts in the repo (concept names, theorem names, table numbers
  — NOT file paths) that prove your point. Do NOT add caveats. Do
  NOT use the word "honest" or any rebuttal-style narration that
  mentions the reviewer.

If you have nothing strong enough to rebut, write only the line:
`(no rebuttal this round — addressing all weaknesses in the improver pass)`

Do not preface with anything; the first non-blank line of your output
must be either the first `### Rebuttal of weakness:` header or the
`(no rebuttal this round...)` sentinel. Do not write to a file.

Round: 5
