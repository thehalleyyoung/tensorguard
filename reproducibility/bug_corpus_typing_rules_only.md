# Typing-rules-only RP on the 60-bug corpus (round-4 Q5)

Reviewer Q5: report the 60-bug RP count when the analyser is
restricted to the typing rules whose soundness is asserted by
Theorem 2 -- with the constraint-based shape back-end, the
AST-pattern intent-bug analyser, and the per-operator
handler dispatch all disabled.

| configuration | RP | silent | abstain | err |
|---|---|---|---|---|
| (i) full pipeline | 53 | 7 | 0 | 0 |
| (ii) intent-bug disabled | 53 | 0 | 0 | 7 |
| (iii) intent-bug + handlers disabled | 53 | 0 | 0 | 7 |
| (iv) typing-rules only | 0 | 53 | 0 | 7 |

The (iv) row is the answer to the reviewer's question.  The drop from (iii) to (iv) isolates the contribution of the constraint-based shape back-end, separating it from the rule-table fragment that Theorem 2 covers.

Run with `python3.11 reproducibility/bug_corpus_typing_rules_only.py`.
