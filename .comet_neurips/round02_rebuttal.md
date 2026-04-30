● Read round02_rebuttal.prompt.md
  │ .comet_neurips/round02_rebuttal.prompt.md
  └ 80 lines read

● Check README for check flags (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n -i "check_devices\|check_phases\|check_gradients\|not
  │ forwarded\|not exposed" README.md API.md 2>/dev/null | head -40
  └ 2 lines...

● Check src for check flags (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -rn "check_devices\|check_phases\|check_gradients" src/
  │ --include='*.py' | head -30
  └ 20 lines...

● Read README.md
  └ L40:80 (41 lines read)

### Rebuttal of weakness: On the fairest directly comparable bug subset, the strongest maintained baseline is actually `torch.compile`
The 34/34 `torch.compile` number is achieved by *executing* each model on a concretely shaped input batch; the 34-bug subset was specifically selected to be in-fragment for both tools, including being instantiable and trace-runnable. This is precisely the regime the paper's central claim disclaims: as the *Scope* paragraph and the regime-asymmetry discussion around the 488-block real-source corpus state, the contribution is "soundly calibrated static reasoning from unreduced class source" where `torch.compile`/FakeTensorMode are mechanically inapplicable because no instantiation, no example inputs, and no `forward()` trace exist. Treating 34/34-with-execution as a stronger detector than 32/34-without-execution conflates two regimes that the paper, the per-bug contingency table, and the released `pytea_membership` predicate keep separate by construction. The same review acknowledges TG is "the only no-execution tool in the class-source regime"; that is the comparison the headline 32/34 vs 25/34 against Pytea is making, and `torch.compile` is reported in the dedicated comparison precisely so this regime boundary is not hidden, not as a baseline TG is claiming to beat.

### Rebuttal of weakness: The soundness footprint on real-source verdicts is still limited: only 62/185 in-soundness verdicts touch handlers entirely inside the Lean-or-pen-paper audited footprint
The 62/185 figure is a *floor*, not a ceiling, because the paper's audit stratification (Lean-audited / pen-and-paper / tested-only / outside-scope) is deliberately conservative: a verdict is only counted in the 62 if *every* handler it touches is Lean-or-pen-paper audited, so a verdict that touches one tested-only handler and twenty audited ones drops out entirely. The remaining 123 are not "unaudited" in the colloquial sense; they are governed by the tested-only handler class, which has its own per-operator agreement tables and the implementation-axiom labelling introduced for `ax:fresh-witness`, so monotonicity and refutation behaviour are still stated, just under a labelled hypothesis rather than a Lean obligation. The assume/guarantee composition rule is operator-agnostic at module boundaries, meaning audited-handler coverage propagates across composition without each downstream handler needing its own Lean lemma. The per-bug contingency tables already let a reader project conclusions onto any handler-audit subset they prefer; restricting to the 62/185 subset preserves the qualitative ordering against Pytea and FakeTensorMode in the fragment-fair comparison, because the audited footprint is concentrated on exactly the high-frequency shape/broadcast/matmul operators that dominate the 34-bug subset.


Changes   +0 -0
Requests  7.5 Premium (50s)
Tokens    ↑ 132.4k • ↓ 2.4k • 97.1k (cached)
