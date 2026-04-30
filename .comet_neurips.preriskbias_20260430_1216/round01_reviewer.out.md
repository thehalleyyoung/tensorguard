● Read round01_reviewer.md
  │ .comet_neurips/round01_reviewer.md
  └ 136 lines read

● List paper files (shell)
  │ ls -la neurips.pdf docs/paper/ 2>&1 | head -50
  └ 51 lines...

● List paper sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls docs/paper/sections_v5/ && wc -l neurips.tex docs/paper/neurips.tex
  │ 2>/dev/null
  └ 15 lines...

● Extract paper text (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext docs/paper/neurips.pdf - 2>/dev/null | head -400
  └ 401 lines...

● Page 2 of paper (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext docs/paper/neurips.pdf - 2>/dev/null | sed -n '400,900p'
  └ 502 lines...

● Page 3 (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext docs/paper/neurips.pdf - 2>/dev/null | sed -n '900,1500p'
  └ 22 lines...

✗ save text (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext docs/paper/neurips.pdf /tmp/neurips.txt 2>/dev/null && wc -l
  │ /tmp/neurips.txt
  └ Permission denied and could not request permission from user

✗ save text (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext docs/paper/neurips.pdf /tmp/copilot-neurips.txt 2>&1 && wc -l
  │ /tmp/copilot-neurips.txt
  └ Permission denied and could not request permission from user

✗ save text in tmp dir (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && mkdir -p /tmp/cpr && pdftotext docs/paper/neurips.pdf /tmp/cpr/n.txt 2>&1
  │ && wc -l /tmp/cpr/n.txt
  └ Permission denied and could not request permission from user

● save text in repo (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext docs/paper/neurips.pdf docs/paper/neurips_extracted.txt && wc
  │ -l docs/paper/neurips_extracted.txt
  └ 2 lines...

● Read neurips_extracted.txt
  │ docs/paper/neurips_extracted.txt
  └ L900:1500 (481 lines read)

● Read neurips_extracted.txt
  │ docs/paper/neurips_extracted.txt
  └ L1500:2200 (501 lines read)

● Read neurips_extracted.txt
  │ docs/paper/neurips_extracted.txt
  └ L2200:2900 (492 lines read)

● Inspect Lean (shell)
  │ cd
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ && ls && grep -rn "sorry" --include="*.lean" 2>/dev/null | head -30
  └ 17 lines...

● Check for actual sorry usage (shell)
  │ cd
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ && grep -rn "^\s*sorry\|:= sorry\| sorry$\|sorry --" --include="*.lean"
  │ 2>/dev/null | head -20
  └ 1 line...

## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` source that statically verifies tensor shapes and a coarse `requires_grad`/severed-tape flag, dispatching obligations to Z3. The paper introduces a refinement calculus `Tensor{s, g | φ}` with an assume/guarantee discipline at the class boundary and partial Lean 4 mechanisation of the operator-rule table (28 of 79 handlers sorry-free, plus 16 pen-and-paper, with 35 tested-only). It claims compositional soundness on a 17-operator DSL fragment and a one-directional inclusion to TorchDynamo's specialiser bits. Empirically it reports 53/60 RP on a curated bug corpus, 32/34 vs Pytea 25/34 on a fragment-fair head-to-head (McNemar p=0.0156), 7/7 on naturally-occurring HF transformers PR bugs, and 0/488 unconditional RP under the user-visible free-symbolic regime on a real-source 488-block corpus. The implementation, AST extractor, backward verifier, and Z3 dispatch are explicitly in the trusted computing base.

## Prior weakness disposition
(none — first round)

## Strengths
- The Lean 4 artefact is real and substantive: a `grep` for `sorry` across `lean/TensorGuard/` returns no live admits, the file headers attest sorry-freeness, and the operator-agnostic composition theorem (`ag_composition_ext`) is structured to take per-operator `applyOpExt_sound_*` witnesses, which is the right factoring for a mechanised compositional result.
- The verdict taxonomy (V / RP / CV / LW / ABSTAIN) with Theorem 2 covering only RP and CV is a discipline that most shape-checking papers do not impose. ABSTAIN as a first-class outcome is correctly used to keep the soundness statement honest.
- The fragment-fair Pytea head-to-head (N=34 with explicit AST membership predicate, full per-bug contingency, and a noted upstream-no-commits-since-2022 fact) is a clean comparator design; reporting the matched-pair structure (b=7, c=0) lets the reader audit the McNemar claim.
- The TCB fault-injection scan (F1–F4 with both exposure ceilings and measured RP→V flips) is the kind of empirical TCB envelope that a theorist usually has to reconstruct themselves; doing it in-paper is a real soundness contribution.

## Weaknesses
- **Theorem 1 over-promises relative to its own sketch.** The statement quantifies over "any operator in Cat" where Cat is "the catalogue of Table 8" (i.e. all 79 handlers), but the sketch admits only 28 are Lean-audited and 16 are pen-and-paper, with "the 35 tested-only handlers are not covered by the soundness theorem." A theorem whose conclusion is provably *not* established for 35/79 of its quantification range is not a theorem; it is a conjecture that holds on a sub-fragment. Either restrict Cat in the statement to `Cat_sound = audited ∪ pen_and_paper` (44 ops) or weaken the conclusion to "for every operator in `Cat_sound`."
- **Theorem 2 has the same internal contradiction.** The proof is said to "reduce (i),(ii) to per-operator preservation lemmas already covered by the Lean rule audit (Section 4.4)," but Section 4.4 covers 28/79. The reduction to "per-operator preservation lemmas" therefore is not actually closed for the 35 tested-only operators on which Verify can return V/RP. A reader cannot tell whether RP verdicts touching tested-only handlers are inside or outside Theorem 2.
- **Theorem 4 (monotonicity) cites a "rely/guarantee axiom of fresh refutation witnesses needed to make Theorem 4 hold" deferred to Section E.** A theorist needs that axiom stated where the theorem is. Citing an ungrounded axiom whose form the reader has not seen turns the theorem into "Thm 4 holds modulo whatever we needed to assume." Pen-and-paper proofs in Section C / Section E need to be exhibited inline or at minimum named with their hypotheses, not folded into a forward-reference.
- **The 16 "pen-and-paper" handlers occupy a non-trivial slice of the soundness story but their proofs are not in the main theorem hierarchy in any auditable form.** The contributions claim parity between Lean-audited and pen-and-paper rows when computing the "62/185 in-soundness footprint," but a pen-and-paper sketch is not equivalent to a closed Lean lemma — particularly for rules like `view`, `reshape`, `einsum`, and `unbind`, whose soundness obligations involve nontrivial integer-arithmetic reasoning (divisibility, floor division). The paper should either close them in Lean or stop conflating the two scopes in the headline 62/185.
- **The AST extractor cross-validation does not retire the TCB concern it claims to retire.** The "independent oracle" is "built only from Python's standard `ast` module" and enumerates the same surface features (literal `<config>.<attr>` reads, `<self.attr>=<const>` writes). Two implementations of the same specification, written by the same team, cannot bound systematic-design error in either; this is an over-approximation comparison, not a soundness audit. The claim that the audit "retires the prior concern that the synthesised assume_M could be the unaudited link in the soundness chain" is unsupported.
- **Theorem 5 (Dynamo correspondence) is reported as a theorem but proved by inspection against a single PyTorch release.** The "proof reduces rule-by-rule to PyTorch 2.9.1's specialiser bits"; a moving target whose correspondence is re-checked against each release is a *measurement*, not a theorem in the sense a theorist uses the word. Either downgrade Theorem 5 to "Empirical Correspondence (audited at SHA pinned in §F)" or supply a structural argument independent of a particular Dynamo release.
- **The headline `0/488` unconditional RP under the user-visible regime substantially undercuts the bug-finding narrative.** The 53/60 number on which the abstract leans is on a curated corpus mined by 20+ keyword searches and filtered by four exclusion rules to retain `60` from `1,087` initial hits; under rule (iv) alone the model returns `0/113` RP on a slice the authors themselves call "the operationally correct behaviour" of the front-end. Aggregated, the bug-finding contribution rests on a small, hand-curated, and explicitly scope-restricted dataset; this is not by itself a Soundness deduction, but it tightens what Theorem 2 + the headline can jointly justify.
- **Constants and assumptions in the typing rules are under-specified.** `T-VIEW(-1)` requires `Q | P` but the rule does not state what happens when several axes are `-1` (the BNF `s_bar` shows a single `-1` slot but the side condition is silent on multiple `-1`s, an explicit Python error). `T-MATMUL` quantifies broadcast batch shapes via `broadcast(B̄, B̄′)=C̄` without defining the broadcast relation in the body. A theorist needs the definitions of `broadcast`, `is_on_tape`, and the LIA∪Div∪BMul fragment's exact decision procedure on the same page as the rules that depend on them.

## Questions
- For Theorems 1 and 2: please restate the conclusion so that the operator quantification range is exactly the union of mechanised + pen-and-paper rules, and provide a separate sub-statement (clearly marked as an empirically-supported conjecture, not a theorem) for verdicts touching the 35 tested-only handlers.
- For the 16 pen-and-paper handlers (and especially `view`/`reshape`/`einsum`/`unbind` which appear on the post-freeze real-PR catches): can you exhibit the proof for at least one nontrivial case (e.g. `reshape` with negative-one and divisibility) inline in Section C, with all assumed lemmas named?
- The cited "axiom of fresh refutation witnesses" needed for Theorem 4: what is its precise statement, and is it discharged anywhere (Lean, paper, or reference)?
- For Theorem 5: what would be required to upgrade it from a SHA-pinned correspondence to a structural result (e.g. by abstracting Dynamo's specialiser interface)? Absent that, would the authors agree to relabel it as "Empirical Correspondence"?
- The AST extractor audit compares two implementations against each other. Can you provide a third, semantics-grounded check (e.g. instrumenting `ast.NodeVisitor` over an externally-curated set of HF model classes with hand-labelled `assume_M` ground truth) on, say, 20 modules, to break the same-team / same-spec circularity?
- The `T-VIEW(-1)` rule: what is the rule when the user's `s̄` contains zero or more than one `-1`? And on the divisibility witness `Q | P`, is `Q` allowed to be `0` (e.g. through a config-symbolic dim reduced to 0 by a degenerate envelope)?

## Scores
Soundness: 2
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would push the score up by one point is a clean restatement of Theorems 1 and 2 with the operator-quantification range restricted to the (Lean-audited ∪ Lean-closed-pen-and-paper) handlers, plus one or two of the 16 pen-and-paper proofs (especially `view`/`reshape`/`einsum`) actually closed in Lean. That single edit would convert the current "theorem with a 35/79 unsupported tail" into a real, scope-honest soundness statement and make the empirical Lean-footprint columns of Section 4.4 directly underwrite the headline.

Round: 1


Changes   +0 -0
Requests  7.5 Premium (3m 13s)
Tokens    ↑ 806.4k • ↓ 7.0k • 743.5k (cached)
