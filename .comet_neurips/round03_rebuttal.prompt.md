# Role: paper authors writing a brief rebuttal

A NeurIPS reviewer just posted the review below on your paper. Before
you start any code or paper changes this round, you have ONE chance to
push back on weaknesses you believe are misweighted, factually wrong,
or already-resolved-in-the-current-repo. The next round's reviewer
WILL read this rebuttal and must either accept it (drop the weakness)
or sharpen it (restate with a concrete counter-example).

## The review you are rebutting
## Summary

TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that verifies tensor shapes and gradient flow statically from class source, without instantiating or tracing the module. The paper formalises a refinement-type calculus Tensor{s,g|φ} unifying shape and gradient-flag analysis under Z3, proves a soundness theorem over a 44-handler sub-catalogue Catsound (17 operators mechanised in Lean 4, 17/17 sorry-free lemmas, 16 pen-and-paper), and reports 53/60 REFUTED-PROOF on a curated 60-bug corpus, 32/34 vs. Pytea (McNemar p=0.0156) on a fragment-fair subset, 9/9 naturally-occurring cross-family HuggingFace bugs, and 15/488 (3.07%) unconditional REFUTED-PROOF on a 488-block real-source corpus. The paper is explicit that the analyser implementation, AST extractor, and backward verifier sit in a trusted computing base outside the Lean audit. The primary claimed contribution over execution-based tools is applicability to constructor-argument-dependent class source that cannot be instantiated without a full configuration.

## Prior weakness disposition

- [RESOLVED] On the fairest directly comparable bug subset, the strongest maintained baseline is actually `torch.compile`, which catches 34/34 while TG catches 32/34 -- Rebuttal accepted: the paper clearly delineates the two regimes (execution vs. no-execution); `torch.compile` requires concrete inputs and a traceable module, TG operates on un-instantiated class source; the 32/34 vs. 25/34 head-to-head explicitly excludes `torch.compile` because it belongs to the different regime; the regime asymmetry is now documented in the abstract.
- [PARTIAL] The user-visible real-source result remains weak: on the 488-block corpus the free-symbolic regime yields 0 unconditional RP -- The paper now reports 26/356 unconditional RP on the zero-contract-obligation subset and 15/488 overall, which is non-zero, but 3.07% RP rate with 418/488 abstains still means the tool provides almost no signal on the bulk of real-source code; the improvement from 0 to 15–26 is real but the practical yield remains very low.
- [UNRESOLVED] The main 53/60 number is still driven by a historically mined and filtered corpus; the newer pre-registered unfiltered post-freeze sample is only 5/15, with wide intervals and no statistically separable advantage over FakeTensorMode or Pytea -- The 5/15 result (Wilson 95% CI [15.2%, 58.3%]) is still reported in the paper with explicit acknowledgment that the CIs overlap with baselines (FakeTensorMode/torch.compile at 2/15); the paper states "not a separation"; no additional real-PR corpus evidence is offered.
- [PARTIAL] The soundness footprint on real-source verdicts is still limited: only 62/185 in-soundness verdicts touch handlers entirely inside the Lean-or-pen-paper audited footprint -- Rebuttal partially accepted: the floor/ceiling argument is methodologically sound (any verdict touching one tested-only handler drops out entirely), and the rebuttal notes Catsound concentrates on high-frequency operators; however, the 62/185 number has not moved and the claimed "propagation of audited-handler coverage across composition" is not demonstrated empirically on the 123 remaining verdicts; the gap between 44/79 audited handlers and the 57/185 verdicts touching only outside-any-scope handlers remains a concrete concern.
- [PARTIAL] The public artifact surface still looks immature relative to the paper's architectural narrative: the README states that `check_devices`, `check_phases`, and `check_gradients` are currently not forwarded by the public API/CLI -- The README now explicitly documents this limitation under "Known limitations." However, the feature_ablation.json metadata notes "check_devices, check_phases, check_gradients are accepted by the API but NOT forwarded to verify_model in the current implementation; L2/L3/L4 rows therefore replicate L1 verdict counts," meaning the paper's advertised "5-theory product domain" (Shape × Device × Phase × Stride × Permutation) produces no device, phase, or gradient-specific verdicts on either real corpus; the per-feature ablation confirms this is not just a reporting choice but an implementation gap.

## Strengths

- **Genuine regime contribution**: The no-execution, un-instantiated class-source regime is a real gap in the ecosystem. The ≥ 435/488 mechanical N/A of all execution-based baselines on the 488-block corpus is a structural fact that strongly motivates the approach.
- **Carefully scoped theoretical claims**: The paper gives explicit TCB declarations, partitions verdicts by handler-audit tier (Lean/pen-and-paper/tested-only), restricts the soundness theorem to Catsound, and reports ABSTAIN as a first-class outcome. This level of epistemic hygiene is uncommon and commendable.
- **Compelling naturally-occurring bug evidence**: 9/9 REFUTED-PROOF on real HuggingFace PR/issue repros across five decoder families (Llama, Qwen2, Mistral, Phi-3, Gemma 2), with per-PR citations, is the most externally-valid result in the paper.
- **Reproducible artifacts**: `verify_neurips_revision.py` runs to completion and corroborates the revision headlines; benchmark JSON artifacts are committed; the Lean development builds sorry-free under `lake build`.
- **Honest calibration**: The paper reports 418/488 abstains, explicit CI intervals on every headline, the qkv known false-positive, and the silent-verified gap on 2/10 upstream-faithful bugs. These are not buried in appendices.

## Weaknesses

- **The CEGAR contribution (C5) is effectively unimplemented on the real corpora.** `feature_ablation.json` explicitly documents: "CEGAR predicates are stored as metadata only (not fed back as Bug objects). check_devices, check_phases, check_gradients are accepted by the API but NOT forwarded to verify_model in the current implementation; L2/L3/L4 rows therefore replicate L1 verdict counts." The 25-case stress benchmark activates these knobs, but the paper's own ablation on the real corpora produces a flat line. C5 as stated ("CEGAR predicate discovery … discovers shape predicates automatically") and the claimed "5-theory product domain" are misleading characterisations of what the shipped tool actually does on any real-world input, and the paper's disclosure of this (README Known Limitations section) is too understated relative to the abstract's claims.
- **The pre-registered unfiltered corpus (Table 3, 5/15) provides no statistical separation from baselines.** The paper states this explicitly ("not a separation") and provides a power calculation, but then the abstract and Section 4.1 headline the 53/60 curated figure without a comparable disclaimer. The curated corpus was constructed by historical triage of known bugs; the 5/15 result on the one corpus collected without that foreknowledge is the cleanest unbiased estimate of real-world utility, and its Wilson 95% CI [15.2%, 58.3%] overlaps completely with the execution-based baselines' 2/15 (CI [3.8%, 40.7%]).
- **test_config_qkv_upgrade.py is a known-failing test that must be explicitly ignored.** The prior round's experiment log shows this test was skipped with `--ignore=tests/test_config_qkv_upgrade.py` to get a passing suite. A reproducible artifact should not require ignoring a test that presumably validates a real analysis behaviour. Neither the paper nor the README explains what this test exercises or why it fails.
- **The backward verifier's gradient-flow analysis (C3) is claimed "8/8 canonical bugs caught, 0/50 false positives" but the eval corpus is entirely synthetic.** The 8 canonical bug classes and 50 clean scripts are author-authored; there is no third-party or natural-occurrence validation analogous to the 9/9 cross-family HF result for shape bugs. The ≤ 3.0% false-Verified bound on "regex-screened training-script population" depends on a regex screen that is not defined or validated in the paper's main body.
- **The 57/185 verdicts touching only handlers outside any soundness scope (Table 8 bottom row) are not discussed defensively.** These are not "tested-only" — they are outside all three tiers (Lean, pen-and-paper, tested-only). This is acknowledged with "TCB obligation explicitly tracked" but the paper does not report whether these 57 verdicts are concentrated in particular operator families, which would let the reader gauge whether the theoretical apparatus covers the majority of practical fires.

## Questions

- The feature ablation JSON metadata states that `check_devices`, `check_phases`, and `check_gradients` "are accepted by the API but NOT forwarded to verify_model." Does the paper intend to claim device, phase, and gradient analysis as live, callable contributions, or are these planned features? If the former, which specific commits implement the forwarding, and which benchmark script exercises them end-to-end?
- What does `test_config_qkv_upgrade.py` test, why is it currently failing, and can the authors either fix it or remove it and explain what real behaviour it was meant to validate?
- On the 5/15 pre-registered corpus (Table 3), the paper cites a power calculation "conditioning on 5/15 for TG, 2/15 for baselines." What sample size N would be required to achieve 80% power to reject H₀ under these observed proportions, and is there a concrete plan to collect it?
- The abstract claims "26 unconditional REFUTED-PROOF verdicts on the 356-block subset whose contract obligation is empty." Is there a single command (without cached benchmarks) that regenerates this 26/356 number from the committed corpus, analogous to `python3 verify_neurips_revision.py`?

## Scores

Soundness: 2
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons

Adding a live end-to-end demonstration that `check_devices`, `check_phases`, and `check_gradients` actually flip verdicts on at least one committed real-source example — backed by a JSON artifact — would convert C5's "5-theory product domain" from a documented no-op on real corpora to a demonstrated contribution; this single change would push the overall score to 6 by substantiating the tool's architectural narrative with evidence rather than stress-benchmark proxies.


Changes   +0 -0
Requests  1 Premium (3m 27s)
Tokens    ↑ 322.6k • ↓ 9.1k • 273.8k (cached)

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

Round: 3
