● Read round12_reviewer.md
  │ .comet_neurips/round12_reviewer.md
  └ 119 lines read

● Read paper intro (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat neurips.tex 2>/dev/null | head -100 || cat docs/paper/neurips.tex
  │ 2>/dev/null | head -100
  └ 90 lines...

● List paper sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls docs/paper/sections_v5/ 2>/dev/null || ls docs/paper/ 2>/dev/null ||
  │ ls .
  └ 12 lines...

● Read introduction (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/sections_v5 && cat intro_v6.tex
  └ 136 lines...

● Read evaluation section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/sections_v5 && cat eval_v6.tex
  └ 18 lines...

● Read eval section head (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/sections_v5 && head -300 eval_v6.tex
  └ 301 lines...

● Read calculus section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/sections_v5 && cat calculus_v6.tex | head -200
  └ 172 lines...

● Find robustness data (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/sections_v5 && grep -n
  │ "mutation\|kill\|robustness\|conv2d\|einsum\|7/50\|50 bug" eval_v6.tex |
  │ head -30
  └ 31 lines...

● Find fragment-fair comparison (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/sections_v5 && grep -n
  │ "torch\.compile\|FakeTensor\|34/34\|32/34\|head-to-head\|Pytea" eval_v6.tex
  │ | head -40
  └ 41 lines...

● Find Lean/formal verification claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/sections_v5 && grep -n "Lean\|lean\|sorry\|mechanis" eval_v6.tex |
  │ head -40
  └ 31 lines...

● Check Lean files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/ && ls lean/TensorGuard/ 2>/dev/null || true
  └ 8 lines...

● Find cross-family evidence claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "Llama\|cross.family\|num_heads\|6 module\|one.*bug\|synthetic"
  │ docs/paper/sections_v5/eval_v6.tex | head -30
  └ 18 lines...

● Find backward verifier claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "backward\|0/25\|1/42\|2/8\|alias\|held.out\|false.verif"
  │ docs/paper/sections_v5/eval_v6.tex | head -30
  └ 27 lines...

● Read limitations section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat docs/paper/sections_v5/limconc_v6.tex
  └ 187 lines...

● Find formal assurance surface (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "36/185\|V/CV\|Lean.audited.*verdict\|touched.*Lean"
  │ docs/paper/sections_v5/eval_v6.tex | head -20
  └ 2 lines...

● Read cross-family section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '326,380p' docs/paper/sections_v5/eval_v6.tex
  └ 56 lines...

● Find cross-family evaluation details (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "Llama\|cross.family\|26 module\|20 module\|5 Qwen\|Gemma\|Phi"
  │ docs/paper/sections_v5/eval_v6.tex | head -30
  └ 21 lines...

● Read Lean audit section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '1229,1320p' docs/paper/sections_v5/eval_v6.tex
  └ 93 lines...

● Check test files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls tests/ && ls benchmarks/ 2>/dev/null || true
  └ 59 lines...

● Find Dynamo lemma claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "fragment_v8\|Dynamo\|surrogate\|thm:dynamo"
  │ docs/paper/sections_v5/eval_v6.tex | head -20
  └ 21 lines...

● Read comparison section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '490,560p' docs/paper/sections_v5/eval_v6.tex
  └ 72 lines...

● Read real bug corpus section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '600,680p' docs/paper/sections_v5/eval_v6.tex
  └ 82 lines...

● List tables in eval section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "Table\|tabular\|tab:" docs/paper/sections_v5/eval_v6.tex | head
  │ -30
  └ 31 lines...

## Summary

TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that infers tensor shapes and gradient-flow flags from class source without instantiation. It defines a type `Tensor{s,g|φ}` over a Z3-decidable LIA+Div+BMul language, pairs it with an assume/guarantee discipline at module boundaries, and emits five verdict classes (Verified, Refuted-Proof, Contract-Violation, Library-Warn, Abstain). On a curated 60-bug corpus the tool returns Refuted-Proof on 53/60 (88.3%); on the fragment-fair N=34 Pytea head-to-head it achieves 32/34 vs. 25/34 (McNemar p=0.016). A Lean 4 rule audit covers 28 of 79 handlers with 11/11 soundness lemmas closed sorry-free; the composition theorem is mechanised on a 17-operator DSL. On the 488-block real-library-source corpus, zero unconditional RP verdicts are produced under the user-visible free-symbolic-config regime; this limitation is stated in the abstract.

## Prior weakness disposition

- [PARTIAL] The paper's practical utility on natural-distribution real library source remains weak. Table 1 and Section 4.1 still show 0/488 unconditional RP on the user-visible free-symbolic-config regime... -- The rerun under an input-shape contract yields 15/488 RP and the paper now contains a detailed LW→RP candidate table with three measured flips, but the user-visible headline remains 0/488 and the gap between the curated-corpus 88.3% and real-library 0% is not bridged.
- [RESOLVED] The strongest contemporary baseline on the fragment-fair executable subset is actually `torch.compile`/FakeTensor at 34/34, above TensorGuard's 32/34. This does not invalidate the paper's class-source setting... -- The paper now reports torch.compile at 34/34 explicitly with inapplicability context (481/488 real blocks have no instantiatable module), and per-bug verdicts are released.
- [UNRESOLVED] The formal-assurance surface is still narrow relative to the user-visible claims. Section 4.4 reports only 36/185 V/CV verdicts on the 488-block corpus as touching only Lean-audited or pen-and-paper handlers... -- The paper still reports 36/185 (≈19%) of V/CV verdicts touching only the in-soundness footprint; 84% of verdicts on real code still rely on 44 tested-only handlers in the TCB.
- [PARTIAL] The robustness story is improved but still not strong at the analyser level: the best-of-union mutation-kill rate remains 7/50, and the stronger conv2d/einsum numbers come from targeted, load-bearing handler experiments... -- Targeted handler kill rates are now reported (conv2d 20/38=53%, einsum similarly extended), but the broad analyser-level best-of-union rate stays at 7/50=14%.
- [PARTIAL] The cross-family evidence remains thin. The Llama 2/3 result is six modules with one synthetic bug fixture, and one of the two RP outcomes is a division guard on `num_heads`... -- Extended to 26 modules across 5 decoder families with 5 distinct bug classes, which is a real improvement; however, every non-Llama family bug is a deliberately-injected mismatch, and the LlamaAttention RP remains a conservative division-guard refutation rather than a caught field bug.
- [UNRESOLVED] The backward-verifier deployment story is still bounded by small, selective audits: 0/25 on a subset of 371 Verified rows, 1/42 held-out HF training scripts, and 2/8 false-verified on the worst-case alias/checkpoint family... -- Numbers are identical; 2/8=25% false-verified rate on the worst-case construct family is unchanged and the held-out positive sample remains 0/25 (no false positives), which together leave the evaluation inconclusive.

## Strengths

- **Exceptional calibration discipline.** The paper front-loads its limitations in the abstract (0/488 RP on real library source, 2/8=25% backward-verifier false-verified rate on worst-case constructs, LW coverage ceiling) rather than burying them. This level of calibration is rare and should be recognised as a positive signal about the research process.
- **Well-scoped formal contributions.** The Lean 4 audit covers 28/79 handlers with 11/11 sorry-free soundness lemmas, the composition theorem is mechanised on a 17-operator DSL, and the boundary between mechanised and non-mechanised scope is precisely documented in the handler-soundness table. The permList_compose correction (unconditional statement was false at boundary, in-range restatement used) is a concrete example of the audit doing real work.
- **Principled fragment-fair comparison.** The Pytea head-to-head is correctly restricted to the N=34 intersection, McNemar exact test is reported, the torch.compile 34/34 counterpoint is stated prominently, and per-bug verdicts are released with the implementation—leaving no ambiguity about comparison protocol.
- **Post-freeze generalisation harness.** The catalogue-freeze mechanism (commit hash recorded) and the N=6 post-freeze held-out repros provide a genuine falsifiability anchor; 3/6 RP at ≥0.99 confidence without rule edits is a real signal about the generality of the catalogue.
- **Useful LW→RP candidate table.** The 12-row falsifiability table, annotated with which single rule addition would flip each LW to RP, is independently checkable and three entries have since been measured-flipped, demonstrating that the table is not decorative.

## Weaknesses

- **The 0/488 RP gap remains the dominant empirical fact about practical utility.** The paper responds with the 15/488 RP rerun (under an input-shape contract) and the 53/60 curated-corpus number, but the curated corpus was filtered to ≤40-line CPU repros that raise a cited RuntimeError—a selection that systematically biases toward the fragment that TG handles best. Without a quantitative account of what fraction of real-world shape bugs meet this filter, 53/60 cannot be interpreted as an estimate of real-world recall.
- **Mutation-kill rate 7/50 at the analyser level is inadequate validation for a tool making soundness claims.** The paper employs four mutation classes (comparison flip, boolean-op flip, arithmetic-op swap, small-integer constant +1) but excludes boolean-op flips on defensive guards. The surviving boolean-op flips are explained as sitting on guards "whose companion check is still present," but this explanation is not independently verifiable from the paper. A soundness claimant should report kill rates under a comprehensive mutation operator set, not a subset chosen post-hoc.
- **The Lean audit proves rules, not the analyser.** The abstract and introduction's prominent mention of Lean 4 mechanisation risks misleading readers: the AST extractor, analyser implementation, backward verifier, and Z3 dispatch are explicitly not mechanised. On the 488-block corpus, 84% of V/CV verdicts traverse at least one tested-only handler. The TCB fault-injection scan (four faults, upper-bound exposure counts) is a valuable sanity check but does not constitute robustness evidence equivalent to, e.g., a test suite with oracle labels on the full tested-only handler set.
- **Cross-family evaluation uses only synthetic/deliberately-injected bugs.** All five decoder families (Qwen2, Mistral, Gemma, Phi-3) contribute zero caught real bugs; each family's "refuted" module is an author-constructed mismatch variant. The LlamaAttention Refuted-Proof is a conservative division-guard refutation (num_heads could be zero) rather than a field bug catch. The cross-family section therefore measures whether TG can be made to fire on deliberately broken code, not whether it catches bugs in the wild.
- **The Dynamo-guard correspondence (C4) is labelled a "theorem" in the contributions but is empirically audited on 14 modules, 4 of which use a forward-signature surrogate.** The necessary-direction-only scope is documented, but placing a one-directional empirical claim in the contributions list alongside mechanised theorems overstates its epistemic status and may anchor readers' expectations of the tool's runtime applicability.

## Questions

- The 60-bug corpus is filtered to ≤40-line CPU repros that raise a cited RuntimeError. Of the 53 Refuted-Proof catches, how many would remain analysable (i.e., stay in the TG fragment) if the module source were extracted verbatim from upstream using the same `inspect.getsource` pipeline used for the 488-block corpus, without manual simplification? This number would bridge the curated-corpus and real-library-source evaluations.
- Table 1 shows Pytea at N/A=435 but torch.fx/ShapeProp and FakeTensorMode at N/A=481. What accounts for the 46-block gap: does Pytea succeed in analysing 46 blocks that the execution-based tools cannot trace? If so, what does Pytea return on those 46 blocks, and is that comparison reported anywhere?
- The boolean-op flip mutation class is partially excluded because surviving flips "sit on guards whose companion check is still present." Please exhibit one concrete example of such a surviving flip and confirm that the companion check materially prevents the bug from propagating to a verdict change—i.e., that the exclusion is not a retroactive criterion that inflates the effective kill rate.
- The post-freeze corpus has N=6 bugs. Can the authors report how many additional fix-PRs were mined that were excluded from the set, and for what reasons (e.g., multi-file changes, requires instantiation, outside fragment)? This would allow readers to assess selection bias in the post-freeze evaluation.
- The `permList_compose` lemma was originally stated unconditionally and is false at the boundary; the corrected in-range form is used. Section 4.4 states that no TG forward-rule path emits an out-of-range permList composition, but this is an assertion about the Python analyser (which is in the TCB), not a Lean proof. Is there a test or an invariant maintained by the analyser that enforces this, and if so where is it documented?

## Scores

Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons

Demonstrating even 10–20 unconditional Refuted-Proof catches on the 488-block real-library-source corpus under a calling convention that a practitioner could plausibly supply (e.g., config values drawn from published HuggingFace checkpoint configs, not hand-synthesised) would directly address the 0/488 headline and move the paper from a tool that proves its claims on curated benchmarks to one with demonstrated practical utility; that single addition would push the overall score to 7.


Changes   +0 -0
Requests  1 Premium (3m 4s)
Tokens    ↑ 512.1k • ↓ 8.7k • 450.9k (cached)
