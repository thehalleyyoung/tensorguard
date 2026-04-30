● Read round01_rebuttal.prompt.md
  │ .comet_neurips/round01_rebuttal.prompt.md
  └ 78 lines read

● list paper sections (shell)
  │ ls docs/paper/sections_v5/ 2>/dev/null | head -40
  └ 32 lines...

● find relevant references (shell)
  │ grep -l "Theorem\|theorem" docs/paper/sections_v5/*.tex | head -5; echo ---;
  │ grep -n "ag_composition\|applyOpExt_sound\|62/185\|Cat_sound\|audited"
  │ docs/paper/sections_v5/*.tex | head -30
  └ 37 lines...

● find theorem statements (shell)
  │ grep -n "Theorem\b\|theorem\b\|\\\\begin{theorem}\|Cat\b\|\\\\Cat"
  │ docs/paper/sections_v5/fragment_v8.tex
  │ docs/paper/sections_v5/subject_reduction_v8.tex 2>/dev/null | head -40
  └ 25 lines...

● Read fragment_v8.tex
  │ docs/paper/sections_v5/fragment_v8.tex
  └ L30:75 (44 lines read)

● find extractor oracle text (shell)
  │ grep -n "assume_M\|hand-labelled\|hand-labeled\|oracle\|extractor"
  │ docs/paper/sections_v5/eval_v6.tex | head -30
  └ 19 lines...

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L140:180 (41 lines read)

### Rebuttal of weakness: Theorem 1 over-promises relative to its own sketch.
The Theorem 1 statement is read in isolation; the very next paragraph (\textbf{Boundary precision}) already restricts the implication "verdict = \textsc{Verified} $\Rightarrow$ no shape mismatch" to clause (ii): "the operator being in $\mathrm{Cat}$ with a Lean-audited or pen-and-paper soundness proof," and clause (e) explicitly excludes "coverage of the 35 tested-only handlers (Table \ref{tab:handler-soundness})." The proof sketch likewise states verbatim "the 35 tested-only handlers are not covered by the soundness theorem." So $\mathrm{Cat}_{\mathrm{sound}} = \mathrm{audited}\cup\mathrm{pen}$-and-paper $= 44$ operators is already the operative quantification range in the paper as written; the catalogue $\mathrm{Cat}$ is the syntactic dispatch domain, not the soundness-quantification domain. The fix the reviewer asks for is a one-line edit folding the boundary clause into the theorem statement itself, not a recovery of a missing 35/79 of "the proof"; no proof is missing because none was claimed.

### Rebuttal of weakness: Theorem 5 (Dynamo correspondence) is reported as a theorem but proved by inspection
Theorem 5 is already audited at a pinned commit: the paper says "we pin the freeze SHA in \Cref{app:dynamo-proof} and mark the statement as audited against \texttt{torch.\_dynamo.symbolic\_convert}." The directionality is also already weakened in the paper to a one-sided inclusion ("inclusion is necessary-direction only, audited on $17$ modules" per Section 1), not a bidirectional structural equivalence. The substantive theorem-content of Theorem 5 is the rule-by-rule mapping from TG's refinements to Dynamo's specialiser bits at a frozen SHA, and that mapping is what is actually used by Theorem 2's CV verdicts; Dynamo evolution between releases changes only the audit witness, not the statement. A relabel from "Theorem" to "Audited Correspondence (SHA-pinned)" is a presentation change that does not move the soundness story; the underlying claim is already SHA-bracketed.

### Rebuttal of weakness: The AST extractor cross-validation does not retire the TCB concern
The independent oracle is not "the same specification re-implemented"; it is the strictly weaker spec "literal \texttt{<config>.<attr>} reads in \texttt{\_\_init\_\_} plus literal scalar writes," derived from Python's standard \texttt{ast} grammar, not from TG's $\mathit{assume}_M$ spec. The audited direction is one-sided (extractor $\subseteq$ oracle on $\mathit{symbolic\_config\_attrs}$, $140/140$, with \emph{zero over-extractions}), which is exactly the soundness direction for the caller-rely contract: any over-extraction would manifest as the extractor naming a config attribute that the upstream class does not literally read, and that class of error is bounded to $0/140$ across the $113$ config-attribute fixtures, the $10$-bug upstream-faithful corpus, the $15$-bug post-freeze corpus, and the $15$-bug unfiltered corpus. The $63/140$ scalar-attr gap runs in the opposite (rely-strengthening) direction and is documented as such in the paper. The retired concern is precisely the soundness-direction one; a third semantics-grounded oracle would re-audit the same one-sided inclusion that is already at $0$ over-extractions.


Changes   +0 -0
Requests  7.5 Premium (1m 3s)
Tokens    ↑ 292.0k • ↓ 3.1k • 269.4k (cached)
