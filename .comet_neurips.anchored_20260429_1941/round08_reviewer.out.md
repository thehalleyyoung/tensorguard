● Read round08_reviewer.md
  │ .comet_neurips/round08_reviewer.md
  └ 118 lines read

● Check available files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls *.tex *.pdf 2>/dev/null && ls .comet_neurips/ 2>/dev/null | head -30
  └ 32 lines...

● Read round07_reviewer.out.md
  │ .comet_neurips/round07_reviewer.out.md
  └ 133 lines read

● Read round07_improver.out.md
  │ .comet_neurips/round07_improver.out.md
  └ 105 lines read

● Check round 8 state (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls .comet_neurips/round08* 2>&1; ls -la neurips.tex sections_v5 2>&1 |
  │ head -20; wc -l neurips.tex; ls docs/paper/sections_v5/ 2>/dev/null
  └ 16 lines...

● file sizes (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l docs/paper/sections_v5/eval_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex docs/paper/neurips.tex
  └ 5 lines...

● git log (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && git --no-pager log --oneline -20
  └ 21 lines...

● paper history (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && git --no-pager log --oneline --since="2026-04-25" -- docs/paper/
  │ neurips.tex 2>&1 | head -20; echo "---last paper change---"; git --no-pager
  │ log -1 --format="%ai %s" -- docs/paper/sections_v5/eval_v6.tex; git
  │ --no-pager log -1 --format="%ai %s" -- docs/paper/neurips.tex
  └ 2 lines...

● check eval log (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && git --no-pager log --all --oneline -20 --
  │ docs/paper/sections_v5/eval_v6.tex 2>&1; echo "---status---"; git --no-pager
  │ status -s | head -30
  └ 32 lines...

● history (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat .comet_neurips/history.jsonl 2>&1 | tail -30
  └ 8 lines...

● file mtimes (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && stat -f "%m %N" docs/paper/sections_v5/*.tex docs/paper/neurips.tex
  │ 2>/dev/null | sort -rn | head -10; echo "---now---"; date +%s
  └ 13 lines...

● grep eval claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "199\|488\|14/289\|not-analysable\|fragment-coverage\|HuggingFace
  │ \|examples/pytorch\|1/42\|silent-error-positive\|holdout\|McNemar\|Pytea\|55
  │ -module\|custom_op\|SHAPE.*DTYPE.*RANK" docs/paper/sections_v5/eval_v6.tex |
  │ head -50
  └ 51 lines...

● grep for follow-on details (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n
  │ "199\|extractor\|huggingface\|examples/pytorch\|1/42\|silent-error\|55-modul
  │ e\|custom_op\|SHAPE.*DTYPE\|holdout\|AST-pattern\|operator-dispatch\|each
  │ path" docs/paper/sections_v5/eval_v6.tex | head -80
  └ 18 lines...

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1140:1210 (70 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L195:220 (26 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L580:620 (41 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L900:940 (41 lines read)

## Summary
TensorGuard is presented as a static refinement-type checker for PyTorch `nn.Module` forward methods that infers `Tensor{s,g | φ}` signatures from class source, dispatches to Z3, and ships an assume/guarantee discipline at class boundaries. On the curated 60-bug historical corpus it reports 53/60 Refuted-Proof (Wilson [77.8%, 94.2%]); on a 34-row Pytea-fragment-fair head-to-head it reports 32/34 vs 22/34 (McNemar exact p=0.00195); on a pre-registered post-freeze unfiltered N=15 sample it catches 5/15 vs FakeTensorMode 2/15 and Pytea 3/15, reported as directional rather than statistically separable. The shape-transfer table is mechanised in Lean 4 (28/79 handlers, 11/11 previously-axiomatic soundness lemmas closed sorry-free), and `ag_composition_ext` is mechanised over a 17-operator DSL covering the operators that fire on the post-freeze catches (`view`, `reshape`, `conv2d`, `einsum`, `unbind`). The user-visible 488-block headline is 0 unconditional Refuted-Proof under free-symbolic, with a 14/488 (= 14/289 over the analysable subset) input-shape-contract+unbind rerun reported as a fragment-coverage measurement and falsifiability anchor.

## Prior weakness disposition
- [UNRESOLVED] The user-visible default headline on the 488-block real-source corpus is still 0 unconditional Refuted-Proof, and the only path that produces unconditional refutations on this corpus (the unbind+input-shape-contract rerun, 14/488) is its... -- §4.1 still reports the headline as 0/488 unconditional, the 14/488 rerun is still framed as fragment-coverage, and no new statistically separable real-source bug-finding number against either baseline has been added since round 7.
- [UNRESOLVED] The 199/488 "not-analysable" Abstain bucket (§4.1, "Two denominators on the unbind rerun") is attributed to an extractor that strips surrounding class context from those rows. Either fix the extractor and report the rate on a single deno... -- The "Two denominators on the unbind rerun" paragraph still attributes the 199 Abstains to the per-block extractor stripping class context (eval_v6.tex:200--207); no per-class-context re-extraction sample on those 199 rows is reported, and the analysable-subset denominator is still bounded by the same extractor artefact.
- [UNRESOLVED] The Theorem-5 falsifier predicate is still vacuously satisfied on every non-curated population (0 SHAPE/DTYPE/RANK guards on the 55-module audit; N=5 hand-built `torch.library.custom_op` fixtures are the only non-vacuous evaluati... -- The 55-module audit (eval_v6.tex:904--930) still reports 0 SHAPE/DTYPE/RANK guards (72/72 in-contract recompiles classified as Int specialisations), and the only non-vacuous falsifier evaluation remains the N=5 curated `torch.library.custom_op` fixture set.
- [UNRESOLVED] The C3 backward-verifier headline is "8/8 canonical bugs, 0/50 false positives, 500/500 static↔runtime agreement" (intro, C3) on randomly-generated small modules, supplemented by 10/10 real-model agreement and the new 8/8... -- The held-out HuggingFace `examples/pytorch/` audit (eval_v6.tex:1145--1159) is still a 1/42 static AST-grep ground-truth count, with no reported analyser verdict actually run on the 1 silent-error-positive script.
- [UNRESOLVED] The "AST-pattern path alone refutes 53/60, operator-dispatch alone refutes 53/60" over-determination claim (§4.1, "Rule-development holdout") is a strong assertion that the 60-bug corpus does not separate the two reasoning paths. If the ... -- The "Rule-development holdout" paragraph (eval_v6.tex:584--613) still reports only the aggregate 53/60 from each path in isolation; no per-bug attribution counting how many bugs each path uniquely catches in a difficulty-stratified random subsample has been added.
- [UNRESOLVED] The Pytea modern-subset comparison is still an in-2022-catalogue intersection on N=34 historical bugs; the post-freeze unfiltered N=15 (where the gap is 5/15 vs 3/15, p=0.68) is the only out-of-corpus head-to-head and is not separable ... -- The McNemar p=0.00195 on the N=34 fragment-fair subset is still cited as the load-bearing significance test (eval_v6.tex:280--287), while the post-freeze unfiltered N=15 result against the same baseline (5/15 vs 3/15, p=0.68 at line ~480) remains not separable at α=0.05.

## Strengths
- The Lean side remains well-aligned with the empirical headline: every operator firing on the 5/5 unfiltered post-freeze catches has a sorry-free `applyOpExt_sound_*` lemma in `lean/TensorGuard/AssumeGuaranteeExtended.lean` and lives inside `ag_composition_ext`, so the soundness audit footprint actually covers the ops that produced the catches rather than a disjoint fragment.
- Reporting across regimes is calibrated and consistent: free-symbolic 0/488, contract-rerun 14/488, analysable-subset 14/289 with explicit Wilson intervals; modern-subset 32/34 vs Pytea 22/34 with McNemar and bootstrap CI; unfiltered post-freeze 5/15 with explicit Fisher-exact p=0.39 and p=0.68 vs the two baselines and explicit non-separation language. The paper does not collapse to a single inflated number.
- The TCB exposure / measured-flip pair (F1--F4: exposure 0/0/2/7 on the 60-bug corpus, measured flips 0/0/0/0) combined with the 7/50 best-of mutation-testing kill rate is a more disciplined robustness argument than is typical for this paper genre, and bounds how badly a single AST/Z3/handler fault could degrade the headline.
- The stub-mocked runtime sample (0/25 silent Verifieds on the 371-V subset, eval_v6.tex:1181--1209) plus the 8-HF `gradient_checkpointing_enable` runtime-trainer harness (8/8 RP, 0/8 false-Verified) jointly close the abstention-bounded silent-error loophole on the V-population in question with a measured Wilson interval.

## Weaknesses
- The user-visible 488-block bug-finding headline is still 0 unconditional Refuted-Proof, and no new statistically-separable real-source result against either FakeTensorMode or Pytea has been added since the prior round. The paper's load-bearing real-source bug-finding evidence still rests on N=10 (`tab:real-bugs`, 7/10 at ≥0.99) and the unfiltered N=15 post-freeze sample (5/15, Fisher exact p=0.39 vs FakeTensorMode and p=0.68 vs Pytea), neither of which is separable at α=0.05 against the relevant baseline. The 14/488 (and 14/289) numbers are still framed as fragment-coverage rather than as a bug-finding headline.
- The 199/488 "not-analysable" Abstain bucket (eval_v6.tex:195--213) is still attributed to an extractor that strips surrounding class context, and no per-class-context re-extraction rate over a uniform random sample of those 199 rows is reported. The 14/289 = 4.84% analysable-subset rate therefore remains bounded by an unmeasured extractor artefact rather than by an honest single denominator.
- The Theorem-5 falsifier predicate is still vacuously satisfied on every non-curated population (0 SHAPE/DTYPE/RANK guards on the 55-module audit at eval_v6.tex:917--921; N=5 hand-built `torch.library.custom_op` fixtures are the only non-vacuous evaluation). Either extend the audit to a long-tail population large enough to observe at least one in-contract SHAPE/DTYPE/RANK recompile on a real (non-custom-op) module, or weaken the empirical-instantiation language of `thm:dynamo-corr`; the falsification predicate has only ever been exercised on a corpus the authors themselves wrote.
- The C3 held-out HuggingFace `examples/pytorch/` audit (eval_v6.tex:1145--1159) reports `1/42` silent-error-positive scripts as a static AST-grep ground-truth count, but does not report what verdict the analyser actually returns on the 1 positive script. If the analyser is silent-Verified on it, the held-out worst-case rate is 1/42 = 2.4%; if it Refutes or Abstains, that is materially different. Run the verifier on the 1 positive script and report the verdict.
- The "Rule-development holdout" paragraph (eval_v6.tex:584--613) still reports only the aggregate 53/60 for each path in isolation. A strictly-held-out per-bug random partition reporting how many of the 60 bugs each path uniquely catches (vs. catches in common) is needed to substantiate the over-determination claim; otherwise the 53/60 number is consistent with corpus-design properties (each bug recognisable by either a simple AST pattern or a single operator rule) and does not isolate the operator-rule contribution.
- The McNemar exact p=0.00195 from §4.1 (eval_v6.tex:280--287) continues to be the load-bearing significance test cited at the abstract level, but the only out-of-corpus head-to-head against the same baseline (post-freeze N=15) is not separable from Pytea at α=0.05 (5/15 vs 3/15, p=0.68). Either the abstract should not lean on the in-2022-catalogue p=0.00195 as the headline against Pytea, or a fresh out-of-corpus matched-pair sample large enough to power an α=0.05 test (the paper's own pre-registration cites N≈56) needs to land.

## Questions
- For the 199 "not-analysable" rows in the 14/488 unbind rerun: what is the per-class-context re-extraction rate on a uniform random sample of ~30--50 of them, so the reader can verify the 14/289=4.84% rate is not biased by which blocks the extractor happens to strip?
- On the 1/42 silent-error-positive HuggingFace `examples/pytorch/` script, what verdict does the analyser actually return? If it is silent-Verified, the held-out worst-case rate is 1/42; if it is Refuted-Proof or Abstain, the number reported in the held-out audit is materially different.
- For `thm:dynamo-corr`: on a long-tail population of importable real modules large enough to observe at least one in-contract SHAPE/DTYPE/RANK recompile, does the predicate still hold? What is the smallest such non-vacuous denominator the paper can report on a non-custom-op population?
- For the rule-development holdout (eval_v6.tex:584--613): on a strictly held-out per-bug random partition, do the AST-pattern path and the operator-dispatch path independently catch the same per-bug subset, or different bugs that just sum to 53? What is the per-path unique-catch count?
- Does the pre-registered N≈56 second-wave post-freeze head-to-head against Pytea exist yet, and if so what is the matched-pair count? The current text references the pre-registration (eval_v6.tex:484--489) but the paper still relies on the in-2022-catalogue McNemar as its headline significance test.

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 3
Overall: 7

## Borderline reasons
The single change that would push my score up by one point is either (a) a non-vacuous evaluation of the Theorem-5 falsification predicate on a real (non-curated) module population — i.e. extending the 55-module audit until it observes at least one in-contract SHAPE/DTYPE/RANK recompile on a non-custom-op module and reporting the catalogue-membership rate on those events — or (b) the verifier verdict on the 1/42 silent-error-positive HuggingFace `examples/pytorch/` script, which would convert the held-out audit from a static count into an actual end-to-end held-out worst-case rate.


Changes   +0 -0
Requests  7.5 Premium (2m 44s)
Tokens    ↑ 601.7k • ↓ 9.1k • 564.9k (cached)
