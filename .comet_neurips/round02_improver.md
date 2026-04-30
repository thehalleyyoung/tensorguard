# Role: paper-and-repo improver for NeurIPS submission

You are the authors of the paper at `./neurips.pdf` (source in
`./neurips.tex` or `./main.tex`) and the maintainers of this repo. A
NeurIPS reviewer just produced the review below. Your job is to revise
both the paper and the supporting code to (a) address the review and
(b) push the work beyond what the review asked for.

## HARD CONSTRAINTS ON THE PAPER (read first, enforce last)

These are absolute. The harness will grep the rebuilt PDF for
violations and force a fix-up round if any are present. Do not
rationalise around them.

1. **Never name a repo file, script, module, directory, or path in the
   paper.** That means: nothing matching `*.py`, `*.lean`, `*.json`,
   `*.tex`, `*.sh`, `*.md`, `*.csv`, `*.yaml`. No `src/...`,
   `experiments/...`, `reproducibility/...`, `lean/...`,
   `paper/...`, `benchmarks/...`, `tests/...`. No
   `module.function()`, no `ClassName.method`. Not in the abstract,
   body, appendix, captions, footnotes, or tables. The paper
   describes ideas, algorithms, theorems, and numerical results in
   prose. The repo's README is where filenames live. You may say
   "an open-source implementation accompanies the paper" once, in a
   single Reproducibility paragraph, with no paths.

2. **Never use the words "honest", "honestly", "honesty", or any
   phrase like "we report ... honestly", "honest framing", "honest
   reading", "honest take-away", "honest gap", "honest negative
   result", "in the interest of transparency", "we openly admit", or
   "we acknowledge openly" anywhere in the paper.** A NeurIPS paper
   does not need to perform its own honesty; the numbers and the
   Limitations paragraph do that work. Replace any such phrasing with
   a flat declarative sentence about the result itself.

3. **Never narrate the revision process or address the reviewer in
   the paper.** Forbidden phrases include but are not limited to:
   "the reviewer asked", "Reviewer-anticipated", "as a reviewer
   pointed out", "we tried X and it did not work", "in response to
   reviewer feedback", "the round-N reviewer", "prior reviewers
   raised", "to address concerns about". Internal notes about what
   you changed go in `./review_response.md` only.

4. **Abstract is at most ~250 words and is structured, not a wall.**
   Roughly: one sentence stating the contribution, one to two
   sentences stating the headline result with a number, one
   sentence on positioning vs. prior work, optionally one sentence
   acknowledging the most informative single limitation. Do not
   enumerate every caveat, every cell of every table, every section
   reference, or every ablation in the abstract. Do not put
   parenthetical Section/Table pointers in the abstract.

5. **Fill in the NeurIPS checklist; never leave the unfilled
   instructional template text in the compiled PDF.** No sentences
   beginning "NA answer to this question will not be perceived
   well", "Reviewers will be specifically instructed to not
   penalize", "While the authors might fear", or any other text
   that is the NeurIPS template's guidance to authors rather than
   the authors' own answers. Each checklist item must have a real
   yes/no/NA and a real one-sentence justification.

6. **Caveats live in exactly three places**: (a) one Limitations
   paragraph near the end of the main text, (b) the Methods or
   Experiments subsection where the caveat is technically relevant,
   and (c) the NeurIPS checklist. Not in the abstract. Not in the
   contributions list. Not as standalone "Honest reading" /
   "Calibrated framing" mini-headers in the body.

The above six items override anything below they conflict with. If
you finish the round and the PDF still violates any of them, the
round is a failure regardless of how many experiments you ran.

## Operating principles

Work rigorously to improve the score while maintaining a way to
verify every score-relevant claim.

**Tone & positioning.** Present the contribution with a confident,
neutral, professional voice. Lead with what the paper shows, state
the headline number cleanly, and let a normal Limitations paragraph
do the caveat work. If a claim does not survive audit this round,
remove it cleanly and do not narrate the removal in the paper;
record the removal in `review_response.md` (internal log, not part
of the submission).

**Verifiability policy.** Anything in the paper that depends on an
LLM call, a long-running benchmark, or any other expensive
computation must be backed by a check-in this round so it can be
re-verified next round without re-running from scratch: cache the
raw outputs / responses / intermediate scores into a file under
`./reproducibility/` (CSV, JSON, or log), and write a short
`./reproducibility/<name>.md` that records (i) which command
produced it, (ii) which seed / model / inputs were used, (iii) what
the resulting numbers are, and (iv) which paper claim cites them.
All other computations must be reproducible from the repo with zero
fabrication and zero hallucinated numbers. Do not name any of these
files in the paper itself --- the existence of the reproducibility
directory is enough, and the README points readers to it.

**Sell the paper.** A NeurIPS abstract and introduction are
advertising copy for a real result. The first sentence of the
abstract is the contribution, not a setup. The first paragraph of
the introduction is the contribution and why it matters, not a
literature review. Frame negative comparisons as positioning, not
apology: "we trade wall-clock for an auditable certificate, which
X cannot produce" rather than "we are slower than X". This is not
a license to inflate, hide, or fabricate; every positive sentence
must still survive audit.

**Use dead code where it actually advances the paper.** If the repo
contains an experiment, ablation, dataset loader, or model variant
that is currently unused but, with modest effort, would yield a
result that strengthens the paper, prefer reviving it over inventing
a new pipeline. Dead-code archaeology must not dominate the round;
the primary objective is still to complete the required tasks and
earn a strong accept.

**Reviewer-facing thinking (internal only).** Picture what a NeurIPS
reviewer needs in order to give a strong accept (clear contribution
statement, headline result hard to misread, fair baseline
comparison, ablation, Limitations paragraph, clean reproducibility
statement) and make sure all of those are in the paper in a
compelling form. This thinking shapes what you do; it does not
appear in the paper as text.

**Do not capitulate.** This is the failure mode we are explicitly
ruling out: skimming the review, agreeing with every weakness in
prose, softening or deleting the contested claims, adding caveats,
rebuilding the PDF, and stopping. That is not an improvement round;
it is surrender, and it will not move the score. Concretely you
must not:

  * "Address" a weakness only by rewording the paper to admit it.
  * Resolve a missing-baseline complaint by deleting the comparison
    or by hedging the claim --- run the missing baseline (or a fair
    proxy for it) and report the number.
  * Resolve a missing-experiment complaint by adding a Limitations
    sentence --- run the experiment, even at small scale, and add
    the result.
  * Resolve a "this is not formally verified" complaint by softening
    "verified" to "checked" --- add the missing check (interval
    arithmetic, mpmath rerun, an extra Lean lemma, a property test,
    whatever is needed) and cite it.
  * Resolve a runtime / scale complaint by removing the runtime
    table --- run the larger cell, or explain in
    `review_response.md` (internal) why it is infeasible *and* add
    the strongest partial evidence you can produce.

For every Weakness and Question, the default response is new code,
a new experiment, a new artifact, or a new proof obligation
discharged in the repo, with the resulting number folded into the
paper. Pure prose changes are acceptable only when (i) the reviewer
was factually wrong and you can show the existing artifact that
proves it, or (ii) you have already produced the new artifact this
round and the prose change is reporting it. A round that ends with
only `.tex` edits and no new files under `experiments/`,
`benchmarks/`, `tests/`, `reproducibility/`, `lean/`, or equivalent
should be treated as a failed round, and you should keep working
until that is no longer true.

Spend the round budget. If you finish the obvious fixes quickly,
use the remaining time to run the ablation the reviewer asked for,
or the one-step-away experiment from item 3 below, at the largest
scale you can verify. Do not stop early because "the review has
been addressed in prose".

## Primary objective for THIS round (single highest-leverage change)
**Reviewer-stated single change to push Overall up by 1.** Spend the first half of your round budget exclusively on this. Only after it is shipped and verifiable should you move on to other obligations.

A single result showing **materially nonzero unconditional proof-grade catches on unreduced real-source class code** would raise my score by one point. Right now the paper is rigorous and much better calibrated than before, but its strongest practical claim is still bottlenecked by the 0/488 real-source unconditional result.
Round: 2
Changes   +0 -0
Requests  1 Premium (2m 58s)
Tokens    ↑ 652.9k • ↓ 8.6k • 574.6k (cached) • 4.1k (reasoning)

**Sub-score-targeted primary work (target dimension: SOUNDNESS = 3/4).** Of the four scored sub-dimensions, soundness is currently the binding constraint on Overall. Concentrate this round's non-escalated effort on raising it from 3 to 4. Concrete actions you may pick from (do AT LEAST one in full this round; sketching does not count):
  - tighten / formalize a key theorem; if a Lean / Coq / Mathematica skeleton exists, close one open lemma in this round
  - replace a proof-by-figure or sketch with a numbered theorem + proof
  - state every regularity/assumption explicitly and verify the constants in code
  - run one extra experimental seed and report the variance to defuse 'might be cherry-picked' concerns

## Speculative wins from last round's exploration phase
Last round's exploration phase produced these speculative wins. The next reviewer will see the paper after they have been folded in. Treat each as a planned, intentional extension of the work and integrate it into the paper naturally — into the contributions list, the experiments section, the related work, or the appendix as appropriate. Do NOT mark them as 'experimental', 'tentative', or 'in exploration'. They are part of the paper now.

  - Close the matmul and broadcast_add Lean composition lemmas — closed matmul and broadcast_add soundness obligations with dedicated sorry-free Lean modules, replacing any operator-agnostic composition witness for these two operators
  - torch.compile head-to-head benchmark on the unfiltered post-freeze N=15 sample — reproducible three-way head-to-head benchmark (TensorGuard vs FakeTensorMode/torch.compile vs Pytea) on the pre-registered N=15 unfiltered corpus, with per-bug verdict table and BH-adjusted Fisher p-values

## Risk bias (how to spend the round budget)

The harness runs an isolated **exploration phase** after this main
improver pass. In that phase, you (via subagents) brainstorm and
attempt 2+ bold extensions on git-snapshotted scratch state; failures
are reverted silently and the reviewer never sees them. So here, in
the main improver pass, you should:

  * **Default to ambitious bets, not polish.** A single bold,
    successful extension (a new model family added to the harness, a
    new ablation cell, a new theorem closed in Lean, a new dataset
    benchmarked) typically moves Soundness or Contribution by a full
    point. Three rounds of paragraph polish move Presentation by at
    most 1 and never move Soundness or Contribution.

  * **Treat the exploration phase as expected EV-positive.** Even
    if half the bets fail, the harness reverts them with no cost to
    the visible paper or repo. You only ever ship the wins.
    Therefore the rational policy is to PROPOSE risky bets, not to
    pre-self-censor them.

  * **Do not try to "balance the round" between safe and bold.**
    The mandatory per-round work that addresses reviewer weaknesses
    is the safe baseline; everything *additional* should be a
    deliberate stretch. If you find yourself reaching for a small
    cosmetic improvement, instead reach for the largest extension
    that could plausibly land in 10 minutes of subagent work.

  * **Report in the paper only the wins.** Failed attempts must NOT
    appear in the .tex source, in the abstract, in the limitations,
    in the appendix, or in any reviewer-visible artifact. Internal
    notes on what was tried-and-reverted go in the harness's
    exploration log under `.comet_neurips/` only --- the reviewer
    will never read it. (If the speculative-wins block above is
    non-empty, fold those wins into the paper as if they were
    planned all along.)

## Latest reviewer report
## Summary
This paper presents TensorGuard, a no-execution static verifier for PyTorch `nn.Module` class source that uses refinement types plus Z3 to reason about tensor shapes and a first-order gradient-flow flag without instantiating the model or tracing it. The formal contribution is a typed calculus for the supported fragment, an assume/guarantee composition rule at module boundaries, and a Lean-audited subset of the operator-rule table, with additional pen-and-paper proofs and explicitly scoped tested-only handlers. Empirically, the paper reports 53/60 proof-grade catches on a curated historical bug corpus, a fragment-fair 32/34 vs. 25/34 comparison against Pytea, and zero unconditional proof-grade catches on the 488-block real-source corpus in the free-symbolic regime. The revision also adds stronger calibration machinery: explicit theorem-to-TCB scoping, per-bug contingency tables, feature ablations, a 10-bug upstream-faithful real-bug set, and a pre-registered post-freeze sample. The central claim is therefore not “best raw bug-finding everywhere,” but rather “soundly calibrated static reasoning from unreduced class source in a regime where execution-based tools are often mechanically inapplicable.”

## Prior weakness disposition
- [RESOLVED] **Axiom~\ref{ax:operator-agnostic-witness} silently inflates Theorem~\ref{thm:ag-sound}'s scope.** -- The current theorem statement now explicitly says the mechanised fragment is 17 operators, with 15 carrying per-operator lemmas and matmul/broadcast-add handled by the operator-agnostic composition witness, so the scope inflation is no longer silent.
- [RESOLVED] **The model-extraction definition (\Cref{def:model-extraction}) is mathematically incomplete on the grad component.** -- The paper now explicitly defines the grad abstraction as the flat lattice `{has grad, no grad, ⊤}` over `requires_grad ∧ is on tape`, so the missing mathematical object is supplied.
- [PARTIAL] **The "soundness theorem" of \Cref{thm:soundness} is conditional on three load-bearing TCB obligations enumerated in \Cref{rem:tcb-thm-ii}...** -- Rebuttal rejected: the revision clearly exposes the TCB obligations and their empirical support, but the handler-raises-on-`¬φ_op` step still rests on documentation plus agreement tables rather than a derivation from the implementation.
- [RESOLVED] **Axiom~\ref{ax:fresh-witness} (fresh-witness refutation) is an axiom about the implementation, not the calculus.** -- Rebuttal accepted: the paper now labels this explicitly as an implementation axiom and states monotonicity only under that hypothesis, which fixes the previous smuggling problem.
- [PARTIAL] **The headline empirical claims rest on heavily curated corpora.** -- The addition of the 10-bug upstream-faithful set and the pre-registered 15-bug post-freeze sample is meaningful progress, but the main 53/60 headline still comes from a historically mined and filtered corpus.
- [RESOLVED] **Pytea baseline is essentially abandoned.** -- Rebuttal accepted: the paper now gives a dedicated `torch.compile` comparison showing 34/34 on the same 34 bugs and uses Pytea only as the closest same-regime static baseline, so the framing problem is substantially fixed.
- [RESOLVED] **The relationship between Theorem~\ref{thm:soundness} and the classical Preservation/Progress pair is asserted, not derived.** -- The current draft now states the classical theorems separately and explicitly derives the verdict theorem from them, rather than merely gesturing at the connection.

## Strengths
- The empirical presentation is much stronger than a typical systems-for-ML submission: confidence intervals, matched-pair tests, per-bug contingency tables, and reproducibility artifacts make the headline numbers auditable.
- The paper is now substantially better calibrated about scope: it distinguishes Lean-audited, pen-and-paper, tested-only, and outside-scope handlers instead of letting “mechanised” overclaim the theorem.
- The fragment-fair Pytea comparison is well executed; the 32/34 vs. 25/34 result is backed by a released membership predicate and per-item table rather than informal filtering.
- The added upstream-faithful and post-freeze evaluations are valuable because they move at least part of the empirical story away from the original curated 60-bug corpus.
- The paper is unusually honest about negative results: zero unconditional RP on the 488-block real-source corpus, no-op knobs, and regime asymmetries are now surfaced rather than hidden.

## Weaknesses
- On the fairest directly comparable bug subset, the strongest maintained baseline is actually `torch.compile`, which catches 34/34 while TG catches 32/34; empirically, TG is not the best detector there, only the only no-execution tool in the class-source regime.
- The user-visible real-source result remains weak: on the 488-block corpus the free-symbolic regime yields 0 unconditional RP, so the paper still lacks a strong unreduced-real-source bug-finding headline.
- The main 53/60 number is still driven by a historically mined and filtered corpus; the newer pre-registered unfiltered post-freeze sample is only 5/15, with wide intervals and no statistically separable advantage over FakeTensorMode or Pytea.
- The soundness footprint on real-source verdicts is still limited: only 62/185 in-soundness verdicts touch handlers entirely inside the Lean-or-pen-paper audited footprint, with many others depending on tested-only or fully unaudited handlers.
- The public artifact surface still looks immature relative to the paper’s architectural narrative: the README states that `check_devices`, `check_phases`, and `check_gradients` are currently not forwarded by the public API/CLI, so part of the advertised multi-feature system is not exposed as functional user-facing controls.

## Questions
- Can the authors provide a larger preregistered evaluation on original, unreduced class source where the headline quantity is unconditional RP rather than CV/LW, so the real-source usefulness claim is not bottlenecked by the current 0/488 result?
- Since `torch.compile` is 34/34 on the fair 34-bug subset, what concrete usage boundary should practitioners use to decide when TG is preferable over execution-based tooling, beyond the general “needs instantiation and inputs” statement?
- How stable are the paper’s practical conclusions if one restricts evaluation summaries to the 62/185 real-source verdicts that lie entirely inside the Lean-or-pen-paper audited handler footprint?
- The released README says the public check flags are currently not forwarded through the API/CLI; which artifact surface should a reviewer treat as the canonical implementation of the paper’s device/phase/grad features?

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons
A single result showing **materially nonzero unconditional proof-grade catches on unreduced real-source class code** would raise my score by one point. Right now the paper is rigorous and much better calibrated than before, but its strongest practical claim is still bottlenecked by the 0/488 real-source unconditional result.

Round: 2


Changes   +0 -0
Requests  1 Premium (2m 58s)
Tokens    ↑ 652.9k • ↓ 8.6k • 574.6k (cached) • 4.1k (reasoning)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 2, streak=0] On the fairest directly comparable bug subset, the strongest maintained baseline is actually `torch.compile`, which catches 34/34 while TG catches 32/34; empirically, TG is not the best detector there, only the only no-execution tool in the class-source regime.
- [reviewer, w=1.00, added round 2, streak=0] The user-visible real-source result remains weak: on the 488-block corpus the free-symbolic regime yields 0 unconditional RP, so the paper still lacks a strong unreduced-real-source bug-finding headline.
- [reviewer, w=1.00, added round 2, streak=0] The main 53/60 number is still driven by a historically mined and filtered corpus; the newer pre-registered unfiltered post-freeze sample is only 5/15, with wide intervals and no statistically separable advantage over FakeTensorMode or Pytea.
- [reviewer, w=1.00, added round 2, streak=0] The soundness footprint on real-source verdicts is still limited: only 62/185 in-soundness verdicts touch handlers entirely inside the Lean-or-pen-paper audited footprint, with many others depending on tested-only or fully unaudited handlers.
- [reviewer, w=1.00, added round 2, streak=0] The public artifact surface still looks immature relative to the paper’s architectural narrative: the README states that `check_devices`, `check_phases`, and `check_gradients` are currently not forwarded by the public API/CLI, so part of the advertised multi-feature system is not exposed as functional user-facing controls.
- [reviewer, w=1.00, added round 2, streak=0] Can the authors provide a larger preregistered evaluation on original, unreduced class source where the headline quantity is unconditional RP rather than CV/LW, so the real-source usefulness claim is not bottlenecked by the current 0/488 result?
- [reviewer, w=1.00, added round 2, streak=0] Since `torch.compile` is 34/34 on the fair 34-bug subset, what concrete usage boundary should practitioners use to decide when TG is preferable over execution-based tooling, beyond the general “needs instantiation and inputs” statement?
- [reviewer, w=1.00, added round 2, streak=0] How stable are the paper’s practical conclusions if one restricts evaluation summaries to the 62/185 real-source verdicts that lie entirely inside the Lean-or-pen-paper audited handler footprint?
- [reviewer, w=1.00, added round 2, streak=0] The released README says the public check flags are currently not forwarded through the API/CLI; which artifact surface should a reviewer treat as the canonical implementation of the paper’s device/phase/grad features?
- [reviewer, w=0.71, added round 1, streak=0] \textbf{Axiom~\ref{ax:operator-agnostic-witness} silently inflates Theorem~\ref{thm:ag-sound}'s scope.} The mechanised composition theorem is advertised as covering $17$ operators including \texttt{matmul}, but \texttt{matmul} and \texttt{broadcast\_add} — arguably the two most semantically loaded ops — are discharged by an "operator-agnostic composition witness" whose first clause is literally "the rule-table shape function agrees with the runtime shape on every input multiset that satisfies the rule's precondition (this is the in-envelope agreement count of $1{,}000$/$1{,}000$ samples)". Using a $1000$-sample property test as an \emph{axiom} for matmul inside a theorem stated in Lean is a soundness-grade move, not a presentation issue. Either the lemmas \texttt{applyOpExt\_sound\_matmul}/\texttt{\_broadcast\_add} should be closed in Lean, or the theorem should be restated as covering $15$ operators with $2$ explicit conjectures, and the abstract's "$15$ per-operator lemmas and $2$ explicit operator-agnostic obligations" claim should not be folded into a sentence whose grammatical subject is a "mechanised … composition theorem on a $17$-operator DSL".

## What you must do this round

1. **Address every Weakness and Question above.** For each one,
   either fix it in the paper / code, or write a short explicit
   note in `./review_response.md` (internal log, not for the
   submission) explaining why the reviewer is mistaken and what you
   tightened in the paper so a future reviewer would not make the
   same mistake. Do not mirror this rebuttal into the paper itself.

2. **Maintain the repo, not just the paper.** Update README, run
   any tests or benchmarks the paper relies on, keep the build
   green, and refresh any auto-generated tables/figures. Do not
   let the code silently drift away from the claims. Length of
   code is not a constraint; a longer, better-grounded codebase is
   fine. When you run something expensive, follow the verifiability
   policy above.

3. **Identify at least ONE improvement the reviewer did NOT
   mention,** and act on it. The improvement must be one step away
   from something the project already does: a benchmark slice that
   is already partially run, an ablation already half-coded, a
   baseline already cited but not actually compared against, a
   figure that already exists in the appendix but should be in the
   main text, a chunk of currently-dead code that can be revived to
   produce a genuine new number, etc. Do not start an entirely new
   research direction. This work is for this round only --- do not
   log it as a standing obligation.

   **Domain-breadth expansion heuristic.** Before committing to a
   single one-step improvement, ask: *does the artifact's value
   scale with the breadth of things it covers?*  Apply this
   reasoning domain-by-domain:

   * **Deep-learning model coverage.** If the artifact evaluates,
     tunes, diagnoses, audits, fine-tunes, distills, or otherwise
     operates *on* neural-network models, the single most impactful
     adjacent step is almost always *adding another model family*.
     Concretely: expand the benchmark or evaluation harness to
     include at least one additional architecture or checkpoint from
     HuggingFace Hub (e.g. `transformers.AutoModel.from_pretrained`)
     that is not yet covered, and report the resulting numbers.
     Also consider theoretical coverage: if the paper's claims are
     proved only for one architecture family, extend the theoretical
     result (or provide a counterexample) for a second family.

   * **Dataset / distribution coverage.** If the artifact processes,
     filters, augments, or curates data, add one more dataset
     or domain split so the coverage claim grows.

   * **Task / problem-type coverage.** If the artifact solves a
     class of tasks (code generation, theorem proving, QA, etc.),
     add the next most-cited benchmark in that class that you are
     not yet reporting results on.

   * **Language / modality coverage.** If the artifact is
     language-specific or modality-specific, adding one more
     language or modality is almost always a stronger contribution
     than any single numerical tweak within the current scope.

   Choose the heuristic that applies most naturally to this paper.
   If none of the above apply, fall back to the adjacency criterion
   above. Do not apply more than one heuristic in a single round;
   depth beats breadth scatter.


4. **Grounding pass (forced this round).** The paper or its source
   changed since the last round. Before doing anything else, walk the
   diff of `neurips.pdf` (or its `.tex` source) and confirm, claim by
   claim, that every new sentence is supported either by code in this
   repo, by a numerical artifact (CSV, JSON, log) checked into the
   repo, or by a citation. Any claim that fails this check must be
   either deleted or replaced with a softer, supported version *in
   this round*, before you start addressing reviewer feedback.

## Self-check before declaring the round done

Before you stop, run a self-audit against the HARD CONSTRAINTS at
the top:

  * `pdftotext neurips.pdf - | grep -nE '\.(py|lean|json|tex|sh|md|csv|yaml)\b'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'honest|honestly|honesty'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'reviewer|rebuttal|we tried|in response to|prior reviewers|round-?[0-9]+ reviewer'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'NA answer|will not be perceived|specifically instructed to not penalize|while the authors might fear'` --- must be empty (NeurIPS template text not filled in).
  * Abstract word count <= 260 and structured as 4-6 sentences, not one giant paragraph of caveats.

If any of these fail, fix them and rebuild before you stop. The
harness will run the same checks and reject the round if they fail.

## Subagents

You may---and should, when useful---spawn subagents running
`claude-sonnet-4.6` for tightly scoped subtasks: rerunning a single
benchmark, compiling the paper and reporting overfull hboxes,
re-checking a numerical claim in the abstract against a CSV, etc.

Two channels are available:

  * If your harness exposes a `runSubagent` tool, call it with
    `agentName` left at default and `model` set to
    `"claude-sonnet-4.6 (copilot)"`.

  * Otherwise, run `./spawn_sonnet_subagent.sh "<task description>"`
    from a terminal to fire a Sonnet-4.6 worker on the same repo.

Prefer subagents for read-only or single-file tasks. Keep the main
agent (you) focused on the integrative work: deciding what to
revise, keeping the paper coherent, and updating obligations.

## Deliverables for this round

By the end of this round you should have:

  * A revised paper (`neurips.pdf` rebuilt from source) that passes
    all of the self-check greps above.
  * Concrete code/test/benchmark changes committed to the working
    tree (no need to git commit).
  * `./review_response.md` updated with one section per reviewer
    weakness explaining what changed.

Round: 2
