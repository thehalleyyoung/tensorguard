● Read round04_reviewer.md
  │ .comet_neurips/round04_reviewer.md
  └ 118 lines read

● list repo (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ | head -50 && echo "---" && ls /Users/halleyyoung/Documents/div/mathdivergen
  │ ce/halley-labs/tensorguard/.comet_neurips/ 2>/dev/null | head
  └ 52 lines...

● size (shell)
  │ wc -l /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguar
  │ d/neurips.txt /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/te
  │ nsorguard/neurips.tex
  └ 4 lines...

● Read neurips.tex
  └ 94 lines read

● list sections (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/s
  │ ections_v5/ 2>/dev/null
  └ 1 line...

● find sections (shell)
  │ find /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ -name "intro_v6*" -o -name "sections_v5" 2>/dev/null | head
  └ 4 lines...

● list v5 (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/sections_v5/
  └ 12 lines...

● Read neurips.txt
  └ L1:200 (199 lines read)

● search C5 (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "C5\|three
  │ knobs\|device-consistency\|gradient-flow\|low-confidence" neurips.txt | head
  │ -30
  └ 11 lines...

● Read neurips.txt
  └ L1220:1340 (121 lines read)

● Read neurips.txt
  └ L112:125 (13 lines read)

● search pytea (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "matched-pair\|Pytea modern-subset\|machine-readable
  │ artifact\|reproducibility appendix\|34-row" neurips.txt | head -30
  └ 15 lines...

● Read neurips.txt
  └ L955:985 (31 lines read)

● search numbers (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "25/34\|22/34\|0\.0156\|0\.00195" neurips.txt | head -30
  └ 8 lines...

✗ verify pdf (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls -la neurips.pdf neurips.txt neurips.tex && pdftotext -layout
  │ neurips.pdf /tmp/np.txt 2>&1 | head && grep -c "22/34\|25/34" /tmp/np.txt
  └ Permission denied and could not request permission from user

● check times (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && stat -f "%Sm %z %N" neurips.pdf neurips.txt neurips.tex 2>&1
  └ 4 lines...

● contents (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat neurips.pdf neurips.tex 2>&1; echo "---"; head -3 neurips.txt
  └ 14 lines...

● size (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -c neurips.pdf neurips.tex
  └ 4 lines...

● unwitnessed (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "unwitnessed\|assume_M" neurips.txt | head -30
  └ 4 lines...

● Read neurips.txt
  └ L700:730 (31 lines read)

● search lean+ast (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "AST
  │ extractor\|handler_scope_per_block\|Lean-or-pen-paper\|11/57\|25/128\|AST
  │ oracle\|extractor.*oracle\|extractor.*validat" neurips.txt | head -40
  └ 7 lines...

● Read neurips.txt
  └ L1670:1730 (61 lines read)

● Read neurips.txt
  └ L1730:1830 (101 lines read)

● Read neurips.txt
  └ L1900:2000 (101 lines read)

● search oracle (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n -i
  │ "oracle\|extract.*correctness\|extractor.*audit\|assume_M.*correct\|ast.*dif
  │ ferential\|ground.truth.*assume\|hand.audit.*assume\|extract.*manual"
  │ neurips.txt | head -30
  └ 2 lines...

● extractor (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n -i "extractor\|hand.check\|witness" neurips.txt | head -40
  └ 30 lines...

● per-bug (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "Pytea modern-subset matched-pair\|34-row\|per-bug\|bug-by-bug"
  │ neurips.txt | head
  └ 1 line...

● contingency (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "matched.pair\|contingency\|Pytea-protocol\|appendix.*table"
  │ neurips.txt | head -20
  └ 5 lines...

● Read neurips.txt
  └ L3490:3560 (71 lines read)

● tables (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "Table 6\|Table 7\|Table 8\|Pytea
  │ modern\|per-bug\|Refute\|Refuted\|Verified" neurips.txt | grep -i
  │ "table\|pytea" | head -30
  └ 19 lines...

● repro appendix (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "machine-readable\|reproducibility appendix\|Section J\|appendix
  │ J\|appendix [A-Z]" neurips.txt | head -30
  └ 14 lines...

● repro dir (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "reproducibility/" neurips.txt | head -20; echo ---; ls
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/repr
  │ oducibility/ | head -30
  └ 32 lines...

● pytea files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/r
  │ eproducibility && ls | grep -i "pytea\|mcnemar\|matched\|34"
  └ 14 lines...

● view tables (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/r
  │ eproducibility && head -50 pytea_mcnemar_per_bug.md
  │ ast_extractor_oracle_validation.md
  └ 89 lines...

● check per-bug (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "bug_003\|bug_018\|bug_039\|both_refute\|tg_only" neurips.txt |
  │ head -10
  └ 1 line...

● check oracle paper (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -ni "oracle\|cross.valid\|over.extract\|second.*ast" neurips.txt |
  │ head -10
  └ 1 line...

● tested-only (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "tested-only\|44 tested\|48 tested" neurips.txt | head
  └ 8 lines...

● Read neurips.txt
  └ L20:26 (6 lines read)

## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` source that statically verifies tensor shapes and a static `requires_grad` flag without instantiating or tracing the module. The paper formalises a calculus `Tensor{s, g | φ}` over Z3-decidable shape predicates plus a flat grad lattice, gives an assume/guarantee discipline at the `nn.Module` boundary, and mechanises a 13-operator composition lemma plus 11/11 previously-axiomatic soundness lemmas (sorry-free) on a 28/79-handler subset in Lean 4. Empirically, it claims 53/60 Refuted-Proof on a curated historical bug corpus, 32/34 vs Pytea 22/34 on a fragment-fair modern subset (McNemar exact p=0.00195), and a directional 5/15 vs FakeTensorMode 2/15 vs Pytea 3/15 on a pre-registered post-freeze real-PR sample (explicitly reported as not separable at α=0.05). On a 488-block real-source corpus the user-visible free-symbolic regime returns 0 unconditional RP, framed as a fragment-coverage measurement, with 36/185 in-soundness verdicts touching only Lean-or-pen-paper handlers. A preliminary Dynamo-guard inclusion lemma (necessary direction only) is empirically audited on 14 modules.

## Prior weakness disposition
- [UNRESOLVED] C5 wording in the Introduction contradicts the paper's own ablation data... -- C5 (lines 113–117) still reads "a per-feature stress benchmark on 25 targeted cases (Table 5) reporting that, **on the real corpora**, only three knobs ... move verdicts," while §4.2 (lines 1308–1311) explicitly says "On the 488-block + 60-bug aggregate corpus the per-feature ablation is a flat line"; the contradiction is now between intro and §4.2 instead of within §4.2, but it is still in the paper.
- [PARTIAL] Pytea per-bug table claimed but absent from the compiled artifact... -- the 34-row contingency table now exists at `reproducibility/pytea_mcnemar_per_bug.md` (25 both / 7 TG-only / 0 Pytea-only / 2 neither), but it is still not embedded in the PDF's appendix; the paper only refers to "the reproducibility appendix" without reproducing the table where the McNemar/CI claim is made.
- [PARTIAL] Inconsistent "unwitnessed" terminology... -- §4.1 (lines 706–711) now distinguishes "Zero `assume_M` is unwitnessed" (no contradictory empty contract) from "10 unwitnessed/non-witnessed rows" (default `*Config()` omits one declared sym-attr) by adding the joint-realisability framing, but the word "unwitnessed" is still reused for two different predicates within the same paragraph.
- [RESOLVED] Lean footprint covers a small fraction of deployed verdicts... -- the paper now states the footprint explicitly ("11 of the 57 Verified verdicts and 25 of the 128 CV verdicts touch only handlers in this set (36/185 in-soundness verdicts in total), while 33+72=105/185 touch at least one of the 48 tested-only handlers", lines 1684–1690) and labels the remainder a "TCB obligation rather than as an in-theorem result"; this is an honest disclosure, not a fix to coverage.
- [PARTIAL] AST extractor untrusted and unvalidated against an independent oracle... -- `reproducibility/ast_extractor_oracle_validation.py` now exists and reports `symbolic_config_attrs ⊆ oracle config refs` at 140/140 across the in-repo corpora, but the paper itself never mentions this oracle, never reports its numbers, and continues to label the AST extractor as "not audited" (line 124).

## Strengths
- The verdict taxonomy is genuinely calibrated. The paper consistently distinguishes RP vs CV vs LW vs Abstain vs Verified, and the headline in the abstract that the 488-block free-symbolic regime gives "0 unconditional RP" is reported up front as a fragment-coverage measurement rather than buried.
- The Lean side is honest about its scope. Closing 11/11 previously-axiomatic soundness lemmas sorry-free, exporting the operator registry as JSON so the analyser cannot silently reference an undeclared op, and printing per-handler {Lean / pen-and-paper / tested-only} in Table 7 is a meaningful artifact compared to most papers in this area.
- The matched-pair Pytea comparison on the modern subset is set up correctly: identical operator-catalogue restriction enforced on both tools at verification time, and the McNemar exact statistic with a paired-bootstrap CI lower bound at +14.7 pp is the right test for the design (b=10, c=0).
- The TCB fault-injection scan (F1–F4) with both exposure ceilings and measured RP→Verified flips is a more credible robustness check than any of the comparable PL/ML papers cited.

## Weaknesses
- The C5 contradiction flagged a round ago is still in the introduction. C5 (lines 112–117) attributes the "only three knobs move verdicts" claim to "the real corpora", but §4.2 (lines 1308–1311) says the real-corpus per-feature ablation is *flat*, with the three-knob result coming from a hand-built 25-case stress benchmark that was "constructed so that each feature would discriminate." A reader of the contributions list will form a false impression of what was demonstrated on real code; the correct fix is to localise C5 to the stress benchmark.
- The Pytea matched-pair claim still rests on a 34-row table that is not in the compiled PDF. The paper writes "the membership table is committed at ." (line 982) and "the 32/34 figure is reproduced at verification time ... by\n\nwhich AST-screens each repro" (lines 971–972) — both have empty `\href`/`\cite` targets in the rendered text. A reviewer cannot inspect the per-bug breakdown without leaving the paper, and the broken cross-references undermine the artifact in its own right.
- The AST extractor — explicitly identified as the component synthesising `assume_M` and as the load-bearing TCB component for *every* CV verdict — is now actually validated against an independent simple-AST oracle in `reproducibility/ast_extractor_oracle_validation.py` (140/140 ⊆-config agreement), but the paper does not surface this. The validation result, the corpora it covers (113-fixture + 10-real-public + 6 post-freeze + 9 unfiltered = 140 classes), and its scope (soundness direction only, scalar-attrs only 63/140 ⊆) belong in the body where the AST extractor is named in the TCB list (lines 22, 124, 1673, 1917, 2024). Without that, the audit exists but does not retire the prior reviewer's concern at the level of the paper.
- The headline real-source bug-finding evidence is weak. The 488-block result is 0 unconditional RP (now framed as a fragment measurement); the 5/15 post-freeze unfiltered comparison is "not statistically separable from either baseline at α=0.05" (line 18). What is left is a curated 60-bug corpus and a fragment-fair Pytea subset that, by construction, biases toward operators TensorGuard's catalogue covers. The claim that this checker provides a useful real-world bug-finding signal beyond curated examples is therefore not yet demonstrated; the paper is honest about this but the evidence base does not support a strong contribution claim under C3/C5.
- Theorem 5 (the Dynamo-guard inclusion lemma) is "preliminary, necessary-direction only" with empirical audit on 14 modules of which 4 transformer blocks use a "documented forward-signature surrogate because their full instantiation exceeds constraint solving at this scale" (lines 109–111). For a result that ties the contribution to a widely-deployed PyTorch component, "9 CNN blocks fully + 4 transformer surrogates" is a small base; the paper should either expand this audit (e.g. ≥ 30 modules, no surrogate on ≥ 10 transformer blocks) or downgrade the framing of C4 below "preliminary" to "exploratory."
- The grad-flag lattice's silent-error caveat (Section 6) is bounded by an AST-grep sweep returning 0/2,908 renamed-attribute hits, 1/42 silent-error-positive on training scripts, and 0/8 false-verified on a runtime trainer harness — this is good. But the held-out runtime sweep is on 8 well-known HF heads, and the 2,908-file sweep uses a fixed regex union; an adversarial author who renames `self.X = self.Y.weight` through a helper function evades the regex. The "≤ 12% prevalence" ceiling comes from this regex, not from a semantic alias check, and the paper should either widen the regex or weaken the ceiling claim.

## Questions
- Will you rewrite C5 in the Introduction to scope the "three knobs move verdicts" claim to the *stress benchmark*, removing the phrase "on the real corpora," so it agrees with the §4.2 statement that the real-corpus ablation is flat?
- Can you embed the 34-row `pytea_mcnemar_per_bug` contingency table directly in the appendix (or as a single table figure), and fix the empty `\href`/`\cite` targets at lines 972 and 982?
- Can you add a paragraph to §4.4 reporting the AST-extractor oracle-validation numbers (`extractor ⊆ oracle` 140/140 on `symbolic_config_attrs`, the 113+10+6+9 corpus breakdown, and the explicit scalar-attrs gap of 63/140 ⊆) and explain what the residual scalar-attr non-coverage implies for the soundness of the 128 CV verdicts?
- Can you report a Theorem 5 audit that does *not* rely on the forward-signature surrogate for any transformer block, and quote how many of the audited modules are non-surrogate, before keeping C4 in the contributions list?
- The grad-flag silent-error AST-grep covers five literal patterns; can you give a worst-case bound on what fraction of renamed-attribute aliasing escapes a literal-pattern regex (e.g. by comparing the regex hit count against a small dynamic per-parameter `.data_ptr()` aliasing check on the 8-model runtime harness)?
- On the post-freeze real-PR sample (5/15 vs FakeTensorMode 2/15 vs Pytea 3/15): can you state the Fisher exact p-values explicitly in the body alongside the counts, and report whether the directional gap survives a Benjamini–Hochberg correction across the three pairwise tests?

## Scores
Soundness: 3
Presentation: 2
Contribution: 2
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would push this to a 6 is removing the C5 contradiction in the Introduction *and* embedding the per-bug Pytea contingency table plus the AST-extractor oracle-validation numbers directly in the compiled PDF — those are three small surgical edits that would convert claims currently parked in `reproducibility/` into evidence a reviewer can inspect inside the paper. A second change that would push it further is a non-surrogate Theorem 5 audit on ≥ 10 transformer blocks; without that, C4 should be downgraded from "preliminary result" to "exploratory."

Round: 4


Changes   +0 -0
Requests  7.5 Premium (4m 58s)
Tokens    ↑ 2.0m • ↓ 15.4k • 1.9m (cached)
