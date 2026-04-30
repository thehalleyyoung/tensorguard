● Read round10_reviewer.md
  │ .comet_neurips/round10_reviewer.md
  └ 119 lines read

● List paper files (shell)
  │ ls -la *.tex *.pdf 2>/dev/null | head -20 && echo "---" && wc -l neurips.tex
  │ neurips.txt 2>/dev/null
  └ 7 lines...

● Read neurips.tex
  └ 89 lines read

● List section files (shell)
  │ ls docs/paper/sections_v5/ && wc -l docs/paper/sections_v5/*.tex
  └ 42 lines...

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ 135 lines read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:200 (200 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L200:500 (301 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L500:900 (396 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L896:1421 (368 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1264:1421 (158 lines read)

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ 186 lines read

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ 171 lines read

● Read impl_v6.tex
  │ docs/paper/sections_v5/impl_v6.tex
  └ 86 lines read

● Check lean directory (shell)
  │ ls lean/ 2>/dev/null && echo "---" && grep -l "applyOpExt_sound" lean/ -r
  │ 2>/dev/null | head -5
  └ 10 lines...

● Check for sorries (shell)
  │ grep -c "sorry" lean/TensorGuard/*.lean 2>/dev/null | grep -v ":0$" | head
  │ -10
  └ 5 lines...

● Show sorry locations (shell)
  │ grep -n "sorry" lean/TensorGuard/AssumeGuaranteeExtended.lean
  │ lean/TensorGuard/V5OperatorRules.lean lean/TensorGuard/Extended.lean
  │ lean/TensorGuard/Parity.lean 2>/dev/null
  └ 9 lines...

## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that statically verifies tensor shapes and gradient-flow flags from class source via a Z3-decidable refinement calculus and an assume/guarantee discipline at the class boundary. The shape-transfer rule table is partially mechanised in Lean 4 (28/79 handlers, sorry-free), and the assume/guarantee composition theorem is mechanised on a 17-operator DSL. Empirically, the system reports 53/60 Refuted-Proof on a curated historical bug corpus and a 32/34 vs. 25/34 fragment-fair comparison against Pytea (McNemar p=0.0156), while honestly disclosing 0 unconditional Refuted-Proof on the 488-block real-source corpus under the user-visible free-symbolic-config regime. Auxiliary results include a backward verifier (500/500 randomized agreement, 8/8 canonical bugs, 25% worst-case false-verified on tied/renamed-attribute parameter sharing), an exploratory Dynamo-guard inclusion lemma audited on 14 modules, and a hand-built 25-block hybrid stress benchmark.

## Prior weakness disposition
- [PARTIAL] The aggregate mutation kill rate is 7/50 = 14% at union across three corpora. -- A targeted enumeration on conv2d/einsum lifts those handlers to ~60% comparison-flip kill, but the analyser-wide multi-corpus union rate is still 7/50 with 43 survivors and the structural classification of survivors doesn't itself kill mutants.
- [PARTIAL] Zero unconditional Refuted-Proof on real library source without a user-supplied contract. -- The user-visible free-symbolic-config headline remains 0/488 unconditional RP; an input-shape-contract rerun yields 15/488 (3.07%) and 3/12 LW→RP candidates were measured-flipped, but the headline gap on the natural distribution persists.
- [PARTIAL] Theorem 5 grounded primarily on CNN-type modules; transformers via forward-signature surrogate. -- Three additional T5/BERT sublayer modules were added end-to-end (all 3/3 Safe with 1 warm-up recompile), but the non-surrogate transformer base remains tiny (4 end-to-end transformer/sublayer subjects against 9 CNN), so the transformer instantiation is still thin.
- [PARTIAL] Backward verifier false-verified rate is 2/8 = 25% on tied/renamed-attribute parameter sharing. -- A held-out 42-script HF examples sweep (1/42=2.4% silent-error-positive) and a stub-mocked runtime sample on 25 of the 371 Verifieds (0/25 silently incorrect) tighten prevalence and held-out behaviour, but the 25% worst-case-construct-family rate itself is unchanged and the regex prevalence bound (≤12%) is still acknowledged as not semantic.
- [RESOLVED] The hybrid-mode complementarity result is demonstration-only. -- The paper now explicitly labels Table 4 as an existence demonstration in caption and body, reports zero hybrid gain on the 488-block natural distribution, and does not claim distributional complementarity.
- [RESOLVED] CEGAR loop ships in the implementation but never fires. -- CEGAR is removed from the claimed contributions, marked as zero-delta no-op in Table 5 (L1), and a source-level deletion audit confirms no verdict-touching call sites; the same treatment is applied to the always-satisfiable phase encoder.

## Strengths
- Refreshingly calibrated reporting with a five-way verdict taxonomy (V/RP/CV/LW/A) and a soundness theorem that explicitly covers only RP+CV, plus per-block scope bookkeeping (36/185 in-soundness-only verdicts) that lets readers audit which results sit inside the mechanised footprint.
- Genuine, sorry-free Lean mechanisation of a non-trivial fragment: 28 shape-transfer rules, 11/11 previously-axiomatic soundness lemmas closed, plus a 17-operator AG composition theorem with 15 per-operator `applyOpExt_sound_*` witnesses; the rule table additionally cross-checks 28,000/28,000 in-envelope samples against torch 2.9.1.
- Inapplicability gap on 481/488 real `nn.Module` blocks for execution-based baselines is a genuinely substantive empirical observation about why class-source static analysis is needed for modern HuggingFace/timm code.
- Honest disclosure of negative or null findings (zero hybrid gain on natural distribution, 0 unconditional RP on free-symbolic-config 488-block corpus, dead CEGAR loop, 25% false-verified on tied/renamed-attribute lattice gap) avoids overclaim and is unusual in this niche.
- Strict pre-registered post-freeze evaluation (catalogue freeze 2026-04-07, GitHub query pre-registered 2026-04-08) on N=15 unfiltered real PRs is a real attempt at avoiding retrofit; even though the headline 5/15 vs. 2/15 / 3/15 is not statistically separable on N=15, the protocol is the right one.

## Weaknesses
- The headline empirical claim of the paper is carried almost entirely by the 60-bug + 10-real-bug + 6+15 post-freeze corpora — a combined N≈91 — with the 488-block real-source corpus contributing essentially zero unconditional RP under the user-visible regime. For a verification paper aimed at "real library source," the load-bearing evidence base is small and bug-corpus-skewed; the 53/60 figure in the abstract should be read in this light, not as evidence of broad real-source utility.
- The fragment-fair Pytea head-to-head (32/34 vs. 25/34, McNemar p=0.0156) compares against a tool whose upstream has had zero commits since April 2022. The paper acknowledges this structurally but the comparison is then asked to do a lot of work (significance, 95% CI on the difference, BH correction). A contemporary baseline (e.g., a simple jaxtyping/beartype harness with handwritten shape annotations on the same 34 bugs, or even a recent LLM-based linter on the same minimal repros) would meaningfully strengthen the head-to-head; right now the only competitor that survives the fragment intersection is a 4-year-stale tool.
- The Lean audit covers 28 of 79 handlers and the AG composition theorem covers 17 of 79 operators, with the parser, AST extractor, backward verifier, and Z3 dispatch in the TCB. The TCB fault-injection footprint (4 hand-picked faults, all measuring 0/60 RP→V flip on the bug corpus) is a small bound on a small set of pre-selected fault sites in a much larger TCB, and the mutation sweep on the analyser as a whole still kills only 7/50 mutants at multi-corpus union. Together these do not yet approach the assurance level the formal apparatus implies.
- Theorem 1 (Soundness) covers RP and CV verdicts, but in the user-visible regime there are 0 RP and 128 CV verdicts on the 488-block corpus, of which 10/128 are "single-default-omitted." The soundness story for the headline dataset therefore reduces almost entirely to soundness of CV under a mechanically-synthesised `assume_M`. The AST-extractor cross-validation against an oracle reaches 140/140 subset agreement on `symbolic_config_attrs`, which is reassuring, but the oracle itself is a hand-built Python AST sweep that has not been independently audited; the soundness of CV verdicts ultimately rests on the AST extractor + oracle pair, both in the TCB.
- The cross-family Llama 2/3 sanity result (4/6 V, 2/6 RP including a buggy variant) is worth keeping but is six modules with one deliberately-buggy fixture; the LlamaAttention "RP" is a fired division guard rather than a caught bug, which the paper notes but the reader should not over-interpret.
- The stub-mocked runtime sample on the 371-Verified subset (0/25 silently incorrect) is a Wilson upper bound of 13.32%; this is an audit, not a guarantee, on a self-selecting "instantiation completed" subsample of the easiest-to-instantiate rows by LoC. The interaction with the 2/8 = 25% worst-case-construct-family rate is not fully resolved by these two complementary samples.
- The worked GPT-NeoX symbolic-calculus example is helpful as an illustration but a single end-to-end SMT trace is the only fully-worked symbolic example in the body; for a refinement-type calculus paper the calculus section would benefit from a second worked case on a non-`view` operator (e.g., einsum with a batch broadcast or conv2d with a groups divisibility witness) to demonstrate the calculus's expressiveness beyond divisibility-on-view.

## Questions
- What is the unconditional Refuted-Proof rate on the 488-block corpus when the input-shape contract is supplied per block but the symbolic-config envelope remains free? You report 15/488 (3.07%) on the input-shape-contract rerun in the body but the abstract still cites the 0-RP free-symbolic-config number; can you give the same point estimate plus 95% CI for the input-shape-contract-only intermediate regime that isolates the contribution of input-shape contracts from config envelopes?
- The mutation sweep classifies 17/43 surviving mutants as residing in "unannotated helper functions" without a verdict-flipping consequence. Of those 17, how many are syntactically reachable from a `_propagate_*` operator handler under any forward path in the three corpora, and how many sit on dead branches of the analyser (i.e., is the survivor rate a coverage problem or a true-equivalence problem)?
- For the 2/8 worst-case false-verified rate on tied/renamed-attribute parameter sharing, what is the smallest extension of the grad-flag lattice that would catch both? A flat alias-class lattice would seem to suffice; is the omission a deliberate scoping choice or a representation limitation that interacts badly with the AG composition rule?
- The 32/34 vs. 25/34 Pytea comparison is on 4-year-stale software. Can you supply a contemporary execution-based baseline on the same 34-bug subset — for instance, the catch rate of a hand-annotated jaxtyping/beartype harness with explicit shape annotations, or `torch.compile(dynamic=True)` graph-break + recompile guard counts on the same minimal repros — so the head-to-head is not the only comparison?
- The Dynamo-guard inclusion lemma's empirical audit reports 13 SHAPE recompiles on the 10 fully end-to-end CNN-type subjects (zero out-of-catalogue), and 0/0 on the 67-module HuggingFace pool because the larger pool produces only INT recompiles. What is the falsifier-non-vacuous denominator if you rerun the 67-pool audit with a transformer-attention dynamic-batch input regime that is known to issue SHAPE guards (e.g., variable sequence lengths under static `head_dim`)? Without that, the 14-module headline carries the entire empirical weight of Theorem 5's necessary direction.

## Scores
Soundness: 2
Presentation: 3
Contribution: 2
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would most cleanly push the overall up is closing the gap between the formal apparatus and the user-visible regime: either a non-trivial unconditional Refuted-Proof rate on the natural-distribution 488-block corpus (e.g., a measured ≥5% RP rate under the free-symbolic-config regime, not just under input-shape contracts) or a contemporary execution-based head-to-head on the 34-bug subset that survives a fragment-fair filter. As it stands the formal calculus, the Lean audit, and the calibrated reporting are solid, but the empirical case for utility on real library source rests on small bug corpora and a 4-year-stale baseline.


Changes   +0 -0
Requests  7.5 Premium (2m 38s)
Tokens    ↑ 878.7k • ↓ 8.0k • 806.9k (cached)
