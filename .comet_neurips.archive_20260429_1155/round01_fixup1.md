# Role: paper fix-up pass (hard-constraint enforcement)

The freshly built `./neurips.pdf` violates one or more of the
HARD CONSTRAINTS that were stated in the previous improver prompt.
The harness greps the compiled PDF and the violations below were
detected. Your only job in this pass is to fix them.

## Violations detected
[repo filename / extension] 1 example(s):
  - matched 'neurips_2026_checklist.tex' in /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/neurips.tex: \IfFileExists{neurips_2026_checklist.tex}{%

## What to do

Edit the LaTeX source (`./neurips.tex`, `./main.tex`, or whatever the
build uses) to remove every violation. Then rebuild the PDF.

Recap of the rules being enforced:

  1. No filename / module / directory path of any kind appears in
     the paper. No `*.py`, `*.lean`, `*.json`, `*.tex`, `*.sh`,
     `*.md`, `*.csv`, `*.yaml`. No `src/...`, `experiments/...`,
     `reproducibility/...`, `lean/...`, `paper/...`,
     `benchmarks/...`, `tests/...`. Anywhere. Replace with prose
     that names the *concept* (e.g. "the reproducibility data",
     "the Lean operator-rule audit"), not the file.

  2. No use of the words "honest", "honestly", or "honesty"
     anywhere. Rewrite each occurrence as a flat declarative
     sentence about the result. "An honest reading is X" -> "X".
     "We honestly report Y" -> "We report Y". "Honest gap" ->
     "remaining gap" or just delete the framing.

  3. No rebuttal narration or addressing the reviewer in the paper.
     No "the reviewer asked", "Reviewer-anticipated", "in response
     to", "we tried X and it didn't work", "prior reviewers
     raised". Move any such content to `./review_response.md`
     (internal log) or delete it.

  4. The abstract is at most ~250 words and is structured as 4-6
     sentences. If it currently enumerates every caveat, every
     ablation cell, and every section pointer, cut it back to:
     contribution, headline result with one number, positioning,
     optionally one informative limitation. Move the rest to the
     introduction or to the relevant body section.

  5. The NeurIPS checklist must be filled in with real
     yes/no/NA answers. Any leftover instructional template text
     ("NA answer will not be perceived well", "Reviewers will be
     specifically instructed to not penalize", "While the authors
     might fear...") must be deleted and replaced with the
     authors' own answers.

After editing, rebuild `./neurips.pdf`. Do not change the
substance of the paper or its results --- this is a pure
constraint-enforcement pass.

Round: 1 (fix-up pass 1)
