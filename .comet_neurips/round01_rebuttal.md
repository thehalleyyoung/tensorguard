● Read round01_rebuttal.prompt.md
  │ .comet_neurips/round01_rebuttal.prompt.md
  └ 76 lines read

● List paper sections and lean files (shell)
  │ ls docs/paper/sections_v5/ 2>/dev/null; ls lean/TensorGuard/ 2>/dev/null |
  │ head -50
  └ 18 lines...

● Search Lean for matmul lemmas (shell)
  │ grep -n "matmul\|broadcast_add\|applyOpExt_sound\|operator-agnostic"
  │ lean/TensorGuard/AssumeGuarantee*.lean 2>/dev/null | head -40; echo ---;
  │ grep -rn "matmul\|broadcast_add"
  │ lean/TensorGuard/AssumeGuaranteeExtended.lean 2>/dev/null | head -30
  └ 45 lines...

● Check torch.compile baseline framing (shell)
  │ grep -n "torch.compile\|fragment-fair\|34/34\|setting asymmetry"
  │ docs/paper/sections_v5/eval_v6.tex docs/paper/sections_v5/F_benchmark.tex
  │ 2>/dev/null | head -40
  └ 14 lines...

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L630:680 (51 lines read)

● Check fresh-witness axiom framing (shell)
  │ grep -n "fresh-witness\|fresh_witness\|ax:fresh\|monotonicity\|thm:monotonic
  │ ity\|200-replay\|200 replay" docs/paper/sections_v5/*.tex 2>/dev/null | head
  │ -30
  └ 20 lines...

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ L370:420 (44 lines read)

● Check preservation/progress framing (shell)
  │ grep -n "subject reduction\|Subject
  │ reduction\|specialisation\|specialization\|abstract-interpretation
  │ specialisation" docs/paper/sections_v5/calculus_v6.tex
  │ docs/paper/sections_v5/subject_reduction_v8.tex 2>/dev/null | head -30
  └ 4 lines...

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ L245:285 (41 lines read)

● Check TCB framing (shell)
  │ grep -n "rem:tcb-thm-ii\|TCB
  │ obligations\|Cat_{\\\\mathrm{pen}}\|Cat_pen\|pen-and-paper"
  │ docs/paper/sections_v5/calculus_v6.tex
  │ docs/paper/sections_v5/handler_soundness_table.tex 2>/dev/null | head -20
  └ 10 lines...

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ L200:245 (46 lines read)

### Rebuttal of weakness: Pytea baseline is essentially abandoned… torch.compile beats TG, a fact buried in the "setting asymmetry" paragraph.
The torch.compile head-to-head is not buried — it is the dedicated paragraph "Contemporary execution-based baselines on the same 34 bugs" in the evaluation section, which explicitly reports torch.compile (FakeTensor backend, fullgraph=True) catching 34/34 in both dynamic=False and dynamic=True modes against TG's 32/34 on the identical fragment-fair subset. The same paragraph names the operating regime as the point of the comparison: torch.compile requires an instantiated module and concrete example inputs, and 481/488 blocks of the real-source corpus have neither. The Pytea result is reported as a static-analyser head-to-head precisely because torch.compile is not in the static-from-class-source regime that defines TG's contribution; the McNemar p=0.0156 is qualified by the same paragraph and the per-bug protocol is released. Demoting Pytea on staleness grounds while ignoring that no maintained tool inhabits the same operating regime would be the framing error, not its converse.

### Rebuttal of weakness: Axiom (fresh-witness) is an axiom about the implementation… A theorem whose statement names the current binary as a hypothesis…
Theorem (Monotonicity of refinement) is titled and stated "conditional on no-memoisation" — the implementation hypothesis is in the theorem's own name and split clause, not smuggled in. The first clause ("no Verified-to-Refuted-Proof flip") is unconditional and survives unchanged when the witness-cache invariant is disabled or hot-swapped; only the second (refinement of the new verdict into {Verified, Contract-Violation, Library-Warn, Abstain}) is gated on Axiom (fresh-witness). The axiom itself states it is a property of how the analyser is implemented and not of the calculus, and the 200-replay property test is presented as the validation harness for the shipped binary, not as a substitute proof. This is exactly the calibration discipline the review credits the paper with on the Cat_sound / Cat_pen / Cat_tested partition — the same discipline applied to monotonicity rather than a departure from "calculus + mechanisation" mode.

### Rebuttal of weakness: Soundness theorem of (thm:soundness) is conditional on three load-bearing TCB obligations… case (ii) for Cat_pen rests on PyTorch documentation rather than on a derivation.
The three obligations are not silent — Remark (Trusted-base obligations for (ii)) lists them by letter (a), (b), (c) immediately after Lemma (progress-to-witness), names PyTorch documentation plus the in-envelope agreement count of Table (handler-soundness) as the explicit support for (a) on Cat_pen, and is the document the theorem statement directs the reader to for the trusted base. Cat_pen is then further audited in App. (soundness) per-handler, and Proposition (einsum-soundness) closes the only non-trivial pen-and-paper case in Lean-style detail. Conjecture (tested-only-soundness) explicitly does not claim closed proof for Cat_tested and reports the 66/185 tested-only-touching cell separately. The headline 28-Lean / 16-pen / 35-tested partition (handler-soundness summary) and the verdict-by-sub-catalogue table are precisely the apparatus that converts "documentation + sampling" into a scoped TCB obligation rather than into an undocumented gap.


Changes   +0 -0
Requests  7.5 Premium (1m 37s)
Tokens    ↑ 479.3k • ↓ 4.5k • 451.1k (cached)
