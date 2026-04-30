# Role: speculative-extension brainstormer

You are a senior researcher brainstorming bold but tractable
extensions to the work in this repo, to be attempted by Sonnet
subagents under a 10-minute wall-clock budget each. Each candidate
will be tried in isolation under a git snapshot; if it fails, the
harness silently reverts and the reviewer never sees the failed
attempt. So bias HARD toward ambitious bets.

## Context

The current reviewer report and the active obligations are below.

### Latest reviewer report
## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that operates on class source without instantiation, computing a refinement-typed signature `Tensor{s,g | φ}` for symbolic shape `s`, static gradient flag `g`, and a Z3-decidable refinement `φ`. Headline empirics on the curated 60-bug historical corpus are 53/60 Refuted-Proof and a 32/34 vs 25/34 win over Pytea (McNemar p=0.0156) on a fragment-fair subset; on 488 real `nn.Module` blocks, 0 unconditional RP under the high-confidence Z3 regime, but 15/488 (and 26/356 on the contract-empty subset) under a derived "unconditional" classifier. A Lean 4 mechanisation closes 17/17 per-operator soundness lemmas on a DSL backing the assume/guarantee composition theorem; 28 of 79 handlers are Lean-audited, with the analyser, AST extractor, and backward verifier in the TCB. The backward verifier reports 8/8 / 0/50 on canonical synthetic bugs and a worst-case 2/8 false-Verified rate on tied/renamed-attribute parameter sharing, bounded to ≤3.0% deployment-weighted via a regex-screened HF prevalence sweep.

## Prior weakness disposition
- [RESOLVED] The CEGAR contribution (C5) is effectively unimplemented on the real corpora -- C5 has been explicitly rewritten in `intro_v6.tex` so that "the unused CEGAR loop and the always-satisfiable phase encoder ship with the analyser but are not claimed as contributions"; the contribution is now restricted to the three knobs (device, grad, low-confidence gating) that actually move verdicts.
- [RESOLVED] The pre-registered unfiltered corpus (Table 3, 5/15) provides no statistical separation -- the conclusion in `limconc_v6.tex` now reports the 5/15 vs 2/15/3/15 line as "a directional trend, not a significance claim" under BH correction at α=0.05, and the abstract no longer claims separation from this corpus.
- [RESOLVED] test_config_qkv_upgrade.py is a known-failing test -- rebuttal accepted: the test is the regression anchor for the disclosed qkv silent-Verified false positive that the paper itself catalogues as a known limitation, and `verify_neurips_revision.py` runs to completion under the documented xfail policy; this is a documented soundness boundary, not a hidden failure.
- [PARTIAL] C3 "8/8 canonical bugs caught, 0/50 false positives" eval corpus is entirely synthetic -- rebuttal partially accepted: the upstream `transformers` sweep, the 6/8 GRADIENT-OUT-OF-FRAGMENT firings on real checkpoint patterns, and the 1/42 held-out PyTorch-examples rate are genuine non-author-authored validation. But the paper still leads C3 in the contribution list (`intro_v6.tex` ll.84-85) with the 8/8/0/50 number drawn from author-authored fixtures, and the same paragraph admits a worst-case 2/8 = 25% false-Verified rate on the tied/renamed-attribute family — the natural-source counterpart of "8/8" (i.e., a count of real upstream silent-zero-grad bugs caught) is still not given as a single headline number parallel to the 9/9 cross-family shape result.
- [RESOLVED] The 57/185 verdicts touching only handlers outside any soundness scope are not discussed defensively -- `eval_v6.tex` now contains Table `tab:soundness-footprint-185` decomposing the 185 in-soundness verdicts into a 4-way partition (only-Lean / Lean+pen-and-paper / tested-only-touch / only-out-of-scope), explicitly enumerates the 57 outside-scope cell as a TCB obligation, and further decomposes it as 15 no-handler-detected + 42 out-of-catalogue.

## Strengths
- The novelty fingerprint that survives skepticism — joint refinement of static `requires_grad` *and* symbolic shape on un-instantiated class source — is a genuine extension of Pytea's no-execution stance: Pytea is shape-only, and `FakeTensorMode`/`torch.export` require an instantiable, traceable model. This is a plausible contribution beyond a "we apply X to Y" framing, even though the calculus itself is acknowledged as a reorganisation of constraint-based shape typing.
- The calibration package is unusually disciplined for an empirical PL paper: every verdict is partitioned by a Lean-audited / pen-and-paper / tested-only / out-of-scope handler footprint (Table `tab:soundness-footprint-185`), every CV verdict is checked for caller-rely satisfiability (118/128 witnessed), and the abstract's column triple is exactly the column total of that table. This kind of TCB accounting is rare and is the right way to report a partially-mechanised verifier.
- The Lean 4 mechanisation actually delivers what it advertises: 17/17 per-operator `applyOp_sound_*` lemmas closed sorry-free, an operator-agnostic composition theorem, and JSON export of the operator registry that prevents the Python analyser from referring to a Lean-undeclared operator. The four operators that fire on the post-freeze real-PR catches (view/reshape, conv2d, einsum, unbind) are inside the mechanised fragment.
- The TCB fault-injection footprint (F1–F4) with both an exposure ceiling and a measured RP→Verified flip count of 0/60 under each injected fault is a concrete soundness test that goes beyond proof-vs-implementation hand-waving.

## Weaknesses
- The headline numerical claim in the abstract is in tension with Table `tab:headline`. The abstract advertises "26 unconditional Refuted-Proof verdicts on the 356-block subset" and "the unconditional count is 15/488", yet Table `tab:headline` reports `TG: 57 V / 0 RP / 128 CV / 78 LW / 225 A` on the same 488-block corpus. The paper later (Section 4.1, "Calibration first") concedes that on real library source the unconditional-RP claim is *not* carried by the block corpus and is "carried by the bug corpora, not by the block corpus" — yet the abstract still leads with 26/356 and 15/488 as if they were unconditional refutations under the same regime as the headline table. Either these are derived from a different (post-hoc, contract-empty) subset and the abstract should say so plainly, or they need to be reconciled with the 0-RP row in `tab:headline`.
- C1's "joint shape-plus-grad" novelty rests on a grad lattice that is admitted to be silently incorrect on a 25% slice of the worst-case construct family (tied/renamed-attribute parameter sharing; `limconc_v6.tex` ll.124-131, `eval_v6.tex` ll.1700-1707). The paper rescues this with a deployment-side prevalence-weighted product `≤0.12·0.25 = 3.0%`, but that bound is the *product of two upper bounds* on disjoint populations (regex-screened prevalence ceiling × construct-family conditional rate from a 2/8 worst-case probe). The product is not an upper bound on the deployment-side rate unless the two estimates are independent and the population matches; the paper does not justify either. The novelty premise of C1/C3 is exactly the gradient layer, so a 25% conditional false-Verified rate on the construct family that the contribution most directly targets is a substantive Soundness deduction.
- The "fragment-fair head-to-head" 32/34 vs 25/34 against Pytea is the only result with a frequentist significance test (McNemar p=0.0156) and is leaned on heavily in the abstract, but the paper does not make it possible to audit how 60 bugs were filtered to 34. The reader needs (a) a deterministic filter rule, (b) the 26 excluded bugs and the rule that excluded each, and (c) the per-bug verdict for both tools. Without this, the comparison is open to a selection-bias critique against a system whose comparator is publicly known to abstain on most modern transformer code.
- The "Bookkeeping note on the headline triple" (`eval_v6.tex` ll.83-97) reports four different `{V, R, A}` triples for the same 488-block corpus across regimes and re-runs (`{57,206,225}`, `{50,213,225}`, `{62,201,225}`, `{55,208,225}`). Each shift is "bookkeeping-clean", but a reader cannot tell from the paper which numbers were produced by which commit, nor whether the abstract's `15/488` and `26/356` refer to the original or the re-executed regime. For a paper whose central calibration claim is exact partitioning of every verdict, this much numerical drift in the headline corpus is itself a presentation/soundness concern.
- C2 (assume/guarantee at the `nn.Module` boundary with contravariant/covariant subclassing) is, novelty-wise, the application of Jones-Meyer-Findler to the class boundary of a particular framework. The paper acknowledges this by routing C2 through `ag_composition_ext` — but the composition theorem is mechanised on a 17-operator DSL, while the analyser implements 79 handlers; the gap is named (62 outside the mechanised composition fragment), but the contribution claim "an assume/guarantee discipline at the `nn.Module` class boundary" is not substantiated by a result that genuinely could not be obtained by composing existing rely/guarantee work with a hand-written PyTorch handler table.
- The "stub-mocked runtime sample on the 371-Verified subset" (`eval_v6.tex` ll.1717-1745) reports `0/25` silently-incorrect Verified with Wilson 95% CI `[0%, 13.32%]`. A 13.3% upper bound is wide enough that this sample cannot rule out a deployment false-Verified rate roughly comparable to the 25% worst-case figure quoted earlier; the paper presents it as a substantial improvement over an "abstention-bounded silent-error envelope" without acknowledging that a one-shot `loss.backward()` on a stubbed config does not exercise checkpointing, multi-step optimiser interaction, or tied-weight backward — i.e., the very constructs the gradient layer is silently incorrect on.
- The paper's distinctive empirical novelty — verdicts on un-instantiated class source — is most cleanly demonstrated by the inapplicability gap (`481/488` for execution-based baselines in Table `tab:headline`). But this is the architectural premise, not an experimental result: any analyser that does not require instantiation will exhibit the same gap by construction. The paper would be stronger if the inapplicability gap were paired with a result on the 481-block subset that *only* an un-instantiated analyser could produce, e.g. a counted set of unconditional RP verdicts on blocks for which no ShapeProp or `FakeTensor` invocation is even definable.

## Questions
- Please reconcile the abstract's "26 unconditional RP / 356" and "15/488" with Table 1's `0 RP / 488` row. Are these the same RP definition? If "unconditional" in the abstract is a different post-hoc classifier (e.g., RP-only-where-`assume_M`-is-empty), please name it in the abstract and in the table caption.
- Provide the deterministic filter that maps the 60-bug corpus to the 34-bug fragment-fair head-to-head and, in the appendix, a per-bug `(TG verdict, Pytea verdict)` row over all 34 + the 26 excluded bugs with the per-row exclusion reason. Without this, the McNemar p=0.0156 is not auditable.
- The 3.0% deployment-side false-Verified upper bound is a product of two conditional estimates from disjoint populations (regex prevalence × construct-family worst-case rate). Please state explicitly which independence or population-overlap assumption justifies treating the product as an upper bound, and ideally a single end-to-end measurement on a prevalence-weighted sample.
- Across the four headline triples in the bookkeeping note, which one is the canonical version cited in the abstract and Table 1, and at which commit SHA? A single per-id audit table with that SHA would close this.
- For C3, can you supply a count of *real upstream* silent-zero-grad bugs caught, parallel to the 9/9 real upstream shape bugs from HF issues? The 6/8 GRADIENT-OUT-OF-FRAGMENT firings are on author-constructed positives; a single real-issue-mined number would directly satisfy the prior weakness.
- The C2 assume/guarantee composition theorem is mechanised over a 17-operator DSL but the analyser uses 79 handlers. Of the 128 CV verdicts on the 488-block corpus, how many are produced entirely under handlers that have a Lean composition witness (i.e., what fraction of the *operationally important* CV traffic is in the mechanised fragment)?

## Scores
Soundness: 3
Presentation: 2
Contribution: 2
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would push this up by one point is reconciling the abstract's headline numbers (26/356 and 15/488 unconditional RP) with Table 1's `0 RP / 488` row, and committing to one canonical headline triple at a named commit — together with a deterministic, auditable derivation of the 60→34 fragment-fair filter. That alone would convert a borderline-reject calibration story into a credible-positioning one and would let the reader take the 32/34-vs-25/34 result at face value.


Changes   +0 -0
Requests  7.5 Premium (3m 2s)
Tokens    ↑ 741.2k • ↓ 9.0k • 667.0k (cached)

### Active obligations
- [reviewer, w=1.00, added round 5, streak=2] **[ESCALATED — reviewer rejected your last 2 attempts; you must EITHER ship the missing artifact this round OR remove the disputed claim from the abstract+contributions list, no further paraphrase is allowed]**
  Last round you only edited paper-side files (.tex/.bib/.md) and shipped no code, no new data artifact, no Lean theorem, no new figure regenerated from a script. The reviewer's weaknesses require substantive evidence, not paraphrase. This round, EITHER ship the requested artifact via spawn_sonnet_subagent.sh (preferred), OR remove the disputed claim from the abstract+contributions list.
- [reviewer, w=1.00, added round 4, streak=0] The headline numerical claim in the abstract is in tension with Table `tab:headline`. The abstract advertises "26 unconditional Refuted-Proof verdicts on the 356-block subset" and "the unconditional count is 15/488", yet Table `tab:headline` reports `TG: 57 V / 0 RP / 128 CV / 78 LW / 225 A` on the same 488-block corpus. The paper later (Section 4.1, "Calibration first") concedes that on real library source the unconditional-RP claim is *not* carried by the block corpus and is "carried by the bug corpora, not by the block corpus" — yet the abstract still leads with 26/356 and 15/488 as if they were unconditional refutations under the same regime as the headline table. Either these are derived from a different (post-hoc, contract-empty) subset and the abstract should say so plainly, or they need to be reconciled with the 0-RP row in `tab:headline`.
- [reviewer, w=1.00, added round 4, streak=0] C1's "joint shape-plus-grad" novelty rests on a grad lattice that is admitted to be silently incorrect on a 25% slice of the worst-case construct family (tied/renamed-attribute parameter sharing; `limconc_v6.tex` ll.124-131, `eval_v6.tex` ll.1700-1707). The paper rescues this with a deployment-side prevalence-weighted product `≤0.12·0.25 = 3.0%`, but that bound is the *product of two upper bounds* on disjoint populations (regex-screened prevalence ceiling × construct-family conditional rate from a 2/8 worst-case probe). The product is not an upper bound on the deployment-side rate unless the two estimates are independent and the population matches; the paper does not justify either. The novelty premise of C1/C3 is exactly the gradient layer, so a 25% conditional false-Verified rate on the construct family that the contribution most directly targets is a substantive Soundness deduction.
- [reviewer, w=1.00, added round 4, streak=0] The "fragment-fair head-to-head" 32/34 vs 25/34 against Pytea is the only result with a frequentist significance test (McNemar p=0.0156) and is leaned on heavily in the abstract, but the paper does not make it possible to audit how 60 bugs were filtered to 34. The reader needs (a) a deterministic filter rule, (b) the 26 excluded bugs and the rule that excluded each, and (c) the per-bug verdict for both tools. Without this, the comparison is open to a selection-bias critique against a system whose comparator is publicly known to abstain on most modern transformer code.
- [reviewer, w=1.00, added round 4, streak=0] The "Bookkeeping note on the headline triple" (`eval_v6.tex` ll.83-97) reports four different `{V, R, A}` triples for the same 488-block corpus across regimes and re-runs (`{57,206,225}`, `{50,213,225}`, `{62,201,225}`, `{55,208,225}`). Each shift is "bookkeeping-clean", but a reader cannot tell from the paper which numbers were produced by which commit, nor whether the abstract's `15/488` and `26/356` refer to the original or the re-executed regime. For a paper whose central calibration claim is exact partitioning of every verdict, this much numerical drift in the headline corpus is itself a presentation/soundness concern.
- [reviewer, w=1.00, added round 4, streak=0] C2 (assume/guarantee at the `nn.Module` boundary with contravariant/covariant subclassing) is, novelty-wise, the application of Jones-Meyer-Findler to the class boundary of a particular framework. The paper acknowledges this by routing C2 through `ag_composition_ext` — but the composition theorem is mechanised on a 17-operator DSL, while the analyser implements 79 handlers; the gap is named (62 outside the mechanised composition fragment), but the contribution claim "an assume/guarantee discipline at the `nn.Module` class boundary" is not substantiated by a result that genuinely could not be obtained by composing existing rely/guarantee work with a hand-written PyTorch handler table.
- [reviewer, w=1.00, added round 4, streak=0] The "stub-mocked runtime sample on the 371-Verified subset" (`eval_v6.tex` ll.1717-1745) reports `0/25` silently-incorrect Verified with Wilson 95% CI `[0%, 13.32%]`. A 13.3% upper bound is wide enough that this sample cannot rule out a deployment false-Verified rate roughly comparable to the 25% worst-case figure quoted earlier; the paper presents it as a substantial improvement over an "abstention-bounded silent-error envelope" without acknowledging that a one-shot `loss.backward()` on a stubbed config does not exercise checkpointing, multi-step optimiser interaction, or tied-weight backward — i.e., the very constructs the gradient layer is silently incorrect on.
- [reviewer, w=1.00, added round 4, streak=0] The paper's distinctive empirical novelty — verdicts on un-instantiated class source — is most cleanly demonstrated by the inapplicability gap (`481/488` for execution-based baselines in Table `tab:headline`). But this is the architectural premise, not an experimental result: any analyser that does not require instantiation will exhibit the same gap by construction. The paper would be stronger if the inapplicability gap were paired with a result on the 481-block subset that *only* an un-instantiated analyser could produce, e.g. a counted set of unconditional RP verdicts on blocks for which no ShapeProp or `FakeTensor` invocation is even definable.
- [reviewer, w=1.00, added round 4, streak=0] Please reconcile the abstract's "26 unconditional RP / 356" and "15/488" with Table 1's `0 RP / 488` row. Are these the same RP definition? If "unconditional" in the abstract is a different post-hoc classifier (e.g., RP-only-where-`assume_M`-is-empty), please name it in the abstract and in the table caption.
- [reviewer, w=1.00, added round 4, streak=0] Provide the deterministic filter that maps the 60-bug corpus to the 34-bug fragment-fair head-to-head and, in the appendix, a per-bug `(TG verdict, Pytea verdict)` row over all 34 + the 26 excluded bugs with the per-row exclusion reason. Without this, the McNemar p=0.0156 is not auditable.

## Your output

Propose EXACTLY 2 candidate bold extensions, each one a separate
attempt. Format as a numbered list, one block per candidate, in this
exact shape:

```
### Candidate 1: <one-line title (no filenames)>
goal: <2-3 sentence description of the extension as a research
       deliverable. State which sub-score (Soundness / Presentation /
       Contribution) it would lift, and by how much you expect.>
plan: <2-5 imperative bullets the subagent should follow.>
success_criterion: <a single verifiable test the subagent runs at the
       end. Must be objectively pass/fail (e.g. "pytest tests/new_X.py
       exits 0 AND the new benchmark CSV has >=N rows", "lake build
       succeeds AND theorem X is checked", "python -m repo.eval
       --model M produces a numeric accuracy value"). NEVER use vague
       criteria like "the result looks reasonable".>
fallback_message: <one sentence: if the candidate is fundamentally
       infeasible in 10 minutes, what should the subagent emit
       instead so the harness can revert cleanly?>
```

Constraints on candidates:
  * Each candidate must be SUBSTANTIAL — adding a whole new feature,
    benchmark suite, model family, theorem, or dataset. Not "fix a
    typo", not "rephrase the abstract".
  * Each candidate must be EXECUTABLE end-to-end by a Sonnet subagent
    in ~10 minutes wall-clock with no human review.
  * Each candidate must have a HARD success criterion the harness can
    parse from a single command's exit code or stdout.
  * Candidates may be entirely independent of one another (they are
    attempted on separate git branches).
  * Do NOT propose candidates that only edit `.tex` / `.bib` / `.md`
    files; those are paper polish, not exploration.

Emit only the 2 candidate blocks — no preamble, no closing remarks.
