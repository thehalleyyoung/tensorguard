● Read round13_reviewer.md
  │ .comet_neurips/round13_reviewer.md
  └ 118 lines read

● check paper artifacts (shell)
  │ ls -la neurips.pdf neurips.tex 2>&1; ls docs/paper/sections_v5/ 2>&1 | head
  │ -50
  └ 34 lines...

● extract paper text (shell)
  │ cd docs/paper && pdftotext neurips.pdf - 2>/dev/null | head -400
  └ 401 lines...

● more paper text (shell)
  │ cd docs/paper && pdftotext neurips.pdf - 2>/dev/null | sed -n '400,900p'
  └ 502 lines...

● more text (shell)
  │ cd docs/paper && pdftotext neurips.pdf - 2>/dev/null | sed -n '900,1500p'
  └ 22 lines...

✗ more text (shell)
  │ cd docs/paper && pdftotext neurips.pdf - 2>/dev/null > /tmp/paper.txt && wc
  │ -l /tmp/paper.txt && sed -n '900,1500p' /tmp/paper.txt
  └ Permission denied and could not request permission from user

● save paper (shell)
  │ cd docs/paper && pdftotext neurips.pdf - 2>/dev/null > paper.txt && wc -l
  │ paper.txt
  └ 2 lines...

● Read paper.txt
  │ docs/paper/paper.txt
  └ L341:700 (360 lines read)

● Read paper.txt
  │ docs/paper/paper.txt
  └ L1500:2200 (457 lines read)

● Read paper.txt
  │ docs/paper/paper.txt
  └ L1957:2600 (491 lines read)

● find table 2 context (shell)
  │ cd docs/paper && grep -n "Table 2\|curated\|≤40\|≤ 40\|filtered to"
  │ paper.txt | head -30
  └ 11 lines...

● Read paper.txt
  │ docs/paper/paper.txt
  └ L955:1100 (146 lines read)

● Read paper.txt
  │ docs/paper/paper.txt
  └ L2440:2700 (260 lines read)

## Summary
TENSORGUARD is a no-execution refinement-type checker for `nn.Module` forward methods that statically verifies symbolic shape and a flat gradient-flag refinement (`{has_grad, no_grad, ⊤}`) from class source via Z3, with five-way verdicts (V/RP/CV/LW/Abstain) and an assume/guarantee discipline at the class boundary. On a curated 60-bug corpus (≤40-line CPU repros) it returns RP on 53/60; on a 488-block real-source corpus it produces 0 RP, 128 CV, 78 LW under synthesised contracts (collapsing to 34 V / 0 RP / 206 LW / 248 A under a free-symbolic-config regime), and on an N=34 fragment-fair subset of the bug corpus it beats Pytea 32/34 vs 25/34 (McNemar p=0.0156). A Lean 4 artifact mechanises 28 of 79 shape handlers and a 17-operator assume/guarantee composition lemma `ag_composition_ext` sorry-free; analyser, AST extractor, backward verifier, and Z3 dispatch remain TCB. A "necessary-direction" Dynamo-guard inclusion (Theorem 5) is empirically audited on 14 modules (9 CNNs end-to-end, 4 transformer surrogates) plus an N=5 adversarial custom-op corpus.

## Prior weakness disposition
- [PARTIAL] The 0/488 RP gap remains the dominant empirical fact about practical utility -- The paper now frames the 0-RP free-symbolic surface as a "fragment-coverage measurement, not a bug-finding result" and exhibits a 12-block per-row LW→RP ceiling with a single named missing rule each, plus a 32/34 vs 25/34 fragment-fair head-to-head and a 7/10 upstream-faithful real-PR corpus, but the natural-distribution practical-utility number on real library source is still 0 unconditional RP, so the underlying empirical fact is reframed rather than improved.
- [PARTIAL] Mutation-kill rate 7/50 at the analyser level is inadequate validation -- The paper adds per-handler targeted enumerations (conv2d 53%, einsum 100%, broadcasting 33%, view/reshape 40%; union 60% on comparison-flip and arithmetic-swap classes) and structurally classifies the 43 surviving multi-corpus mutants, but the analyser-wide union rate stays at 7/50 = 14%, and the per-handler kill rates on the two load-bearing handlers (view/reshape, broadcasting) remain in the 33–40% range.
- [RESOLVED] The Lean audit proves rules, not the analyser -- The abstract and Section 4.4 now state explicitly that "Lean checks the rule table, not the analyser" and list AST extractor, analyser implementation, backward verifier, and Z3 dispatch as TCB; the 11/11 previously-axiomatic soundness lemmas are closed sorry-free, the 17-operator `ag_composition_ext` is added with per-operator `applyOpExt_sound_*` lemmas, and a four-fault TCB exposure scan + measured-flip rerun bounds the residual TCB risk.
- [UNRESOLVED] Cross-family evaluation uses only synthetic/deliberately-injected bugs -- The cross-family corpus is expanded to 26 modules across Llama/Qwen2/Mistral/Gemma/Phi-3, but every refute on these families is still an author-constructed mismatch variant (intermediate_size, GeGLU width, fused-projection chunk-count); zero real upstream cross-family bugs are caught.
- [RESOLVED] The Dynamo-guard correspondence (C4) is labelled a "theorem" but empirically audited on 14 modules, 4 of which use a forward-signature surrogate -- C4 is now explicitly restated as "an exploratory Dynamo-guard inclusion lemma (Theorem 5, necessary direction only)"; the CNN-only restriction (10 fully end-to-end CNN subjects, 13/13 SHAPE recompiles in catalogue) is the headline, the 4 transformer surrogates are documented as a scope limit, and an N=5 adversarial custom-op corpus non-vacuously evaluates the falsification predicate.

## Strengths
- The five-way verdict taxonomy (V/RP/CV/LW/Abstain) plus per-block CV-realisability triage (118/128 single-default-witnessed, 0/128 unwitnessed) and AST-extractor-vs-oracle cross-validation (140/140 subset) provide an unusually disciplined honesty layer for a tool making soundness claims.
- The Lean 4 audit goes meaningfully beyond a token "we mechanised something": 28 shape rules and an operator-agnostic 17-operator composition lemma `ag_composition_ext` close sorry-free, with a Python byte-mirror agreeing with torch 2.9.1 on 28,000/28,000 in-fragment samples and a non-trivial off-envelope boundary check.
- The 32/34 vs 25/34 fragment-fair head-to-head against Pytea with McNemar p=0.0156 and a per-row matched-pair contingency table is a solid, well-controlled comparison against the closest no-execution Z3-based baseline.
- The 488-block corpus is non-trivially diverse (Kast=406 distinct AST skeletons, Khandler=345 handler-call clusters, 369/406 singletons), which substantially weakens the usual "real corpus is a few near-duplicates" critique.
- Backward-verifier scope is honestly bounded: the runtime-trainer harness on 8 positives gives a worst-case 2/8 = 25% false-verified rate on parameter-sharing/checkpointing constructs, and this is propagated into the contribution and limitations rather than buried.

## Weaknesses
- The headline pitch in the abstract — "TENSORGUARD is a no-execution refinement-type checker that verifies tensor shapes and gradient flow statically from class source" — sits awkwardly against the user-visible regime on the 488-block corpus, which produces zero unconditional RP. A reader who only sees the abstract and Table 1 will reasonably read "53/60 RP plus 32/34 vs 25/34" as the practical bug-finding number, when the underlying real-library-source bug-finding number is 0/488 RP under the regime a non-author would actually run. The Headline paragraph (Section 4.1) is candid about this, but the abstract is not.
- The cross-family decoder evaluation (Llama/Qwen2/Mistral/Gemma/Phi-3, 26 modules) still finds zero real cross-family bugs: 5/5 RPs are deliberately broken `*_buggy` variants the authors constructed, and the LlamaAttention division-guard RP is a conservative refutation, not a real-bug catch. As a generalisation argument, this remains an injected-bug study, and it should either be dropped from the contribution or supplemented with at least one real upstream bug from one of those four non-Llama families.
- The 488-block Verified count rests on a synthesised symbolic-config envelope; under the free-symbolic regime 23 of the 57 Verifieds collapse to Abstain. Theorem 2's clause (i) "Verified ⇒ no shape mismatch at any reduction" is therefore guaranteed only relative to the synthesised assume_M, and the 34 surviving Verifieds are the larger, more structurally complex blocks. A single quantitative number on what fraction of the 34 free-symbolic-Verified survivors actually exercise a Lean-audited handler chain only (not just "touch one Lean handler") would directly tell a reader how many block-corpus Verifieds are mechanised end-to-end; right now the 11/57 figure is "touch only the audited footprint", which still permits the audited handler to be the trivial step.
- Theorem 5's empirical surface remains thin where it matters most. The CNN-only restriction (10 subjects, 13 SHAPE recompiles) is now the headline for the falsification predicate, while exactly 1 of the 4 transformer subjects is end-to-end without surrogate. For a paper whose introduction names HuggingFace transformer modules as the target population, "1 transformer block audited end-to-end" is a small base for a theorem; the 55- and 67-module larger pools produce 0 SHAPE/DTYPE/RANK in-contract guards (everything is INT) and so cannot evaluate the falsifier at all. A targeted transformer-block audit at non-surrogate scale (even 3–5 modules) would be more informative than further INT-only denominator audits.
- The analyser-wide mutation-kill rate is still 7/50 = 14% at the union of three corpora, with 43 survivors of which only 18 are claimed to be on verdict-emitting paths. The per-handler targeted extension is welcome, but the natural reading of a union-14% rate on the analyser core is that the test corpora — including the 60-bug headline corpus — exercise a narrow slice of the analyser's branching, which structurally limits how much weight the soundness story can put on the implementation surviving in practice.
- Section 4.1's Table 5 reports that on the real 488-block + 60-bug aggregate every analyser knob (CEGAR, device, phase, gradient-flow, low-conf) leaves verdicts unchanged, with discrimination only on a hand-designed 25-case stress benchmark constructed so each feature would discriminate. The 8/8 backward verifier case-study and 500/500 random-grammar agreement are then carrying the gradient-flow contribution alone on real distributions; a single number for "fraction of real-corpus blocks on which the gradient-flow handler ever fires non-vacuously" is missing and would directly substantiate (or refute) C3 on natural data.
- The 60-bug corpus filter (≤40-line self-contained CPU repro raising the cited RuntimeError) selects strongly for blocks within TG's fragment. The 88.3% RP rate on this corpus is therefore an in-fragment ceiling, not a population estimate; the paper should report what fraction of the candidate keyword-query hits were dropped by the ≤40-line / self-contained filter so a reader can place the 53/60 inside an end-to-end yield.

## Questions
- Of the 34 free-symbolic-Verified blocks, how many use only operators in the 28 Lean-audited + 7 pen-and-paper handler set across the entire forward body (not "touch one such handler")? Please give the integer.
- For the four non-Llama decoder families (Qwen2, Mistral, Gemma, Phi-3), do you have any real upstream issue-tracker bug (open or closed) that TG's catalogue can reach end-to-end? If yes, what is the RP/CV verdict? If no, please say so explicitly in the cross-family paragraph.
- What is the keyword-query yield before the ≤40-line / self-contained / "raises the cited RuntimeError" filter that produced the 60-bug corpus? A single (initial_hits, retained_60) pair would let readers compute the in-fragment retention rate.
- For the 12-block "missing rule per block" LW→RP table in Section 4.1, how many blocks are predicted RP versus measured RP after the rule is implemented? The paper marks 4 as "(measured)"; what is the timeline / commitment to verify the remaining 8 in a follow-up?
- On the gradient-flow contribution (C3): on the 488-block + 60-bug aggregate, on how many blocks does the backward verifier non-vacuously fire (i.e. emit a grad-flag-related Bug, not just process the AST)? A number close to zero would make C3 effectively a stress-only contribution on real distributions.
- For Theorem 5 on transformer blocks: is there a single non-CNN nn.Module from torchvision/timm/transformers for which the full instantiated module fits inside end-to-end constraint solving and the falsification predicate evaluates non-vacuously to false? If so, please report it as the transformer-direction headline; if not, please say so plainly.

## Scores
Soundness: 3
Presentation: 2
Contribution: 2
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would push this from a borderline reject to a borderline accept is one real, non-injected upstream bug caught end-to-end in any of Qwen2/Mistral/Gemma/Phi-3 (or any non-Llama, non-torchvision/timm transformer family) at the live nn.Module class level under the user-visible default regime. As it stands, the paper's empirical generalisation claim beyond the curated 60-bug corpus and the upstream-faithful 10-PR corpus rests on author-constructed `*_buggy` variants, and that is the gap most directly responsible for the 5 rather than a 6.


Changes   +0 -0
Requests  0 Premium (4m 0s)
Tokens    ↑ 736.3k • ↓ 7.2k • 673.0k (cached)
