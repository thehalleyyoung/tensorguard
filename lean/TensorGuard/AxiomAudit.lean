/-
  Axiom audit for the TensorGuard soundness proofs.

  `#print axioms` lists exactly the axioms a declaration transitively depends
  on.  A `sorry` (or any unfinished proof) shows up as `sorryAx`.  The Python
  regression test `tests/test_lean_soundness.py` builds this file and asserts
  that the emitted axiom set for each core soundness theorem is contained in the
  trusted kernel set {propext, Classical.choice, Quot.sound} and, in particular,
  never contains `sorryAx`.  This is the machine-checked witness behind the
  README's "sorry-free" claim and behind Step 87 (soundness of the core transfer
  functions for the declared fragment).
-/
import TensorGuard

-- Shape transfer-function soundness (single-shape DSL)
#print axioms TensorGuard.applyOp_sound_linear
#print axioms TensorGuard.applyOp_sound_view
#print axioms TensorGuard.applyOp_sound_broadcast_add

-- Extended operator set
#print axioms TensorGuard.applyOpExt_sound_matmul
#print axioms TensorGuard.applyOpExt_sound_transpose
#print axioms TensorGuard.applyOpExt_sound_permute
#print axioms TensorGuard.applyOpExt_sound_sum_reduce

-- V5 operator rules (cross-entropy, argmax, …)
#print axioms TensorGuard.V5.applyOp_sound_cross_entropy
#print axioms TensorGuard.V5.applyOp_sound_argmax

-- Operator-agnostic composition witnesses (matmul / broadcast_add)
#print axioms TensorGuard.MatmulSound.matmul_contraction_sound
#print axioms TensorGuard.BroadcastAddSound.broadcast_add_total

-- Reduced-product transfer functions (Step 126): reductions are reductive and
-- the product meet is a component-wise lower bound.
#print axioms TensorGuard.RP.reduceTagNul_reductive
#print axioms TensorGuard.RP.reduceNulTag_reductive
#print axioms TensorGuard.RP.reduce_reductive
#print axioms TensorGuard.RP.pmeet_le_left
#print axioms TensorGuard.RP.pmeet_le_right
-- Reduced-product monotonicity (Step 127): the meet is monotone and the
-- reduction is monotone on the canonical (consistent) sublattice.
#print axioms TensorGuard.RP.pmeet_mono
#print axioms TensorGuard.RP.reduce_mono_consistent
-- Reduced-product γ-concretization soundness (Step 128): γ is monotone, the
-- meet is exact, and the reduction preserves concretization.
#print axioms TensorGuard.RP.gamma_mono
#print axioms TensorGuard.RP.pmeet_gamma
#print axioms TensorGuard.RP.reduce_gamma

-- Shape-CEGAR termination & tight iteration bound (Steps 129–130): a productive
-- run terminates inside a finite predicate universe and obeys
-- iterations ≤ 1 + |discovered predicates|.
#print axioms TensorGuard.Cegar.length_le_lsum
#print axioms TensorGuard.Cegar.cegar_iter_bound
#print axioms TensorGuard.Cegar.cegar_terminates
#print axioms TensorGuard.Cegar.tight_below_naive

-- Known-unsoundness U2 closed (Step 132): the SAFE-on-infeasible terminal
-- decision is fixed. The new decision abstains on infeasible refinements and is
-- sound under the feasible-branch guarantee; the old decision is unsound.
#print axioms TensorGuard.CegarU2.decideNew_safeSound
#print axioms TensorGuard.CegarU2.decideOld_unsound
#print axioms TensorGuard.CegarU2.fix_abstains_on_infeasible
#print axioms TensorGuard.CegarU2.fix_keeps_safe_when_feasible
#print axioms TensorGuard.CegarU2.old_always_safe

-- Known-unsoundness U1 re-audited (Step 133): the verifiable-fragment boundary
-- is mode-dependent. `sound` mode abstains on every fragment violation and is
-- sound (sound_safeSound); `balanced`/`heuristic` may report SAFE on an
-- out-of-fragment construct hiding a real bug (balanced_unsound,
-- heuristic_unsound). The modes agree in-fragment and differ exactly on a
-- fragment violation.
#print axioms TensorGuard.FragmentU1.sound_safeSound
#print axioms TensorGuard.FragmentU1.balanced_unsound
#print axioms TensorGuard.FragmentU1.heuristic_unsound
#print axioms TensorGuard.FragmentU1.modes_agree_in_fragment
#print axioms TensorGuard.FragmentU1.modes_differ_iff_violation

-- Non-shape transfer functions (Step 134): device / dtype / phase / gradient
-- algebras each abstain on unknown operands (no false positive) and are
-- refutation-sound (a flagged bug witnesses a genuine runtime error); the
-- reduced product over the four inherits both properties.
#print axioms TensorGuard.DevDtype.devBug_no_false_positive
#print axioms TensorGuard.DevDtype.devBug_refutation_sound
#print axioms TensorGuard.DevDtype.cuda_then_cpu_roundtrip
#print axioms TensorGuard.DevDtype.pinMemory_preserves
#print axioms TensorGuard.DevDtype.dtMatmulBug_no_false_positive
#print axioms TensorGuard.DevDtype.dtMatmulBug_refutation_sound
#print axioms TensorGuard.DevDtype.dtFloatParamBug_refutation_sound
#print axioms TensorGuard.DevDtype.dtElementwise_never_bug
#print axioms TensorGuard.DevDtype.dtPromote_comm
#print axioms TensorGuard.DevDtype.dtPromote_idem
#print axioms TensorGuard.DevDtype.phaseBug_count_gt_one
#print axioms TensorGuard.DevDtype.phaseBug_eval_tracking_safe
#print axioms TensorGuard.DevDtype.phaseBug_refutation_sound
#print axioms TensorGuard.DevDtype.gradBrokenBug_refutation_sound
#print axioms TensorGuard.DevDtype.productBug_false_iff
#print axioms TensorGuard.DevDtype.productBug_refutation_sound

-- SMT-encoding faithfulness (Step 135): the enum-equality constraint the
-- verifier hands Z3 for device/phase/gradient checks is UNSAT iff the pinned
-- endpoints differ, so the solver's verdict coincides exactly with the abstract
-- `*Bug` predicate.
#print axioms TensorGuard.SmtEncoding.sat_iff_eq
#print axioms TensorGuard.SmtEncoding.unsat_iff_ne
#print axioms TensorGuard.SmtEncoding.unsat_sound
#print axioms TensorGuard.SmtEncoding.eq_is_sat
#print axioms TensorGuard.SmtEncoding.device_smt_matches_devBug
#print axioms TensorGuard.SmtEncoding.phase_smt_unsat_iff_ne
#print axioms TensorGuard.SmtEncoding.grad_smt_unsat_iff_ne

-- Cross-domain (shape × device) encoding faithfulness (Step 136): a transfer op
-- preserves shape exactly (device free); a non-transfer op preserves device
-- exactly (shape free), so the solver flags a cross-domain violation iff the
-- preserved component changed.
#print axioms TensorGuard.CrossDomain.transfer_sat_iff_shape_eq
#print axioms TensorGuard.CrossDomain.transfer_unsat_iff_shape_ne
#print axioms TensorGuard.CrossDomain.transfer_device_free
#print axioms TensorGuard.CrossDomain.nontransfer_sat_iff_dev_eq
#print axioms TensorGuard.CrossDomain.nontransfer_unsat_iff_dev_ne
#print axioms TensorGuard.CrossDomain.nontransfer_shape_free
#print axioms TensorGuard.CrossDomain.branch_selects_preserved

-- Dtype promotion lattice laws (Step 137): the elementwise promotion join is
-- associative and unknown-absorbing (with comm/idem already audited), making it
-- a well-defined semilattice join — justifying the order-independent,
-- never-flagging elementwise dtype transfer.
#print axioms TensorGuard.DevDtype.dtPromote_assoc
#print axioms TensorGuard.DevDtype.dtPromote_unknown_absorbs_left
#print axioms TensorGuard.DevDtype.dtPromote_unknown_absorbs_right

-- Gradient-flow chain transfer (Step 138): the requires_grad bit propagated
-- through a forward — keep is identity, detach/no_grad reset to false
-- (absorbing), reattach sets true; the run is compositional and, on the
-- reattach-free fragment, true iff the input required grad and no reset
-- intervened.
#print axioms TensorGuard.GradFlow.keep_id
#print axioms TensorGuard.GradFlow.detach_false
#print axioms TensorGuard.GradFlow.reattach_true
#print axioms TensorGuard.GradFlow.reset_absorbing
#print axioms TensorGuard.GradFlow.gradRun_append
#print axioms TensorGuard.GradFlow.run_after_reset
#print axioms TensorGuard.GradFlow.run_noReattach_true_iff

-- Device-placement chain transfer (Step 140): the device tag propagated through
-- a forward — keep is identity, .to(cpu)/.to(accel) move (absorbing, last wins);
-- the run is compositional, ends at the last move's target, and a binary op is
-- valid iff both operands share a device tag (cross-device always flagged).
#print axioms TensorGuard.DevicePlacement.keep_id
#print axioms TensorGuard.DevicePlacement.move_absorbing
#print axioms TensorGuard.DevicePlacement.devRun_append
#print axioms TensorGuard.DevicePlacement.run_after_move
#print axioms TensorGuard.DevicePlacement.run_ends_at_target
#print axioms TensorGuard.DevicePlacement.run_noMove_id
#print axioms TensorGuard.DevicePlacement.binValid_refl
#print axioms TensorGuard.DevicePlacement.binValid_iff_eq
#print axioms TensorGuard.DevicePlacement.cpu_accel_invalid
#print axioms TensorGuard.DevicePlacement.chain_binValid_iff

-- Train/eval phase chain transfer (Step 141): the training bit propagated
-- through a sequence of mode setters — keep is identity, .train()/.eval() set
-- (absorbing, last wins); the run is compositional, ends at the last setter's
-- value, and is preserved on the setter-free fragment.
#print axioms TensorGuard.PhaseFlow.keep_id
#print axioms TensorGuard.PhaseFlow.setTrain_true
#print axioms TensorGuard.PhaseFlow.setEval_false
#print axioms TensorGuard.PhaseFlow.setter_absorbing
#print axioms TensorGuard.PhaseFlow.phaseRun_append
#print axioms TensorGuard.PhaseFlow.run_after_setter
#print axioms TensorGuard.PhaseFlow.run_ends_at_value
#print axioms TensorGuard.PhaseFlow.run_noSetter_id

-- Reduction rank-transfer chain (Step 142): keepdim=True preserves rank,
-- keepdim=False lowers it by one (truncated); the run is compositional, monotone
-- non-increasing, equals input rank minus the number of keepdim=False reductions
-- (closed form), and is exact on no-underflow chains.
#print axioms TensorGuard.RankTransfer.keep_id
#print axioms TensorGuard.RankTransfer.drop_pred
#print axioms TensorGuard.RankTransfer.step_le
#print axioms TensorGuard.RankTransfer.rankRun_append
#print axioms TensorGuard.RankTransfer.rankRun_eq_sub_countDrop
#print axioms TensorGuard.RankTransfer.rankRun_le
#print axioms TensorGuard.RankTransfer.rankRun_allKeep
#print axioms TensorGuard.RankTransfer.rankRun_exact
