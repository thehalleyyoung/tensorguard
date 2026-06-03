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

-- CEGAR predicate-record serialization (Step 241): the Lean mirror covers the
-- actual Python schema-v1 record keys and `PredicateKind.name` tags, append-only
-- serialized histories are monotone, and replaying jointly infeasible serialized
-- predicates abstains instead of reporting SAFE.
#print axioms TensorGuard.CegarSerialized.python_record_keys_match_v1
#print axioms TensorGuard.CegarSerialized.python_kind_names_cover_v1
#print axioms TensorGuard.CegarSerialized.serialized_append_preserves_membership
#print axioms TensorGuard.CegarSerialized.serialized_append_length_mono
#print axioms TensorGuard.CegarSerialized.serialized_history_step_prefix
#print axioms TensorGuard.CegarSerialized.serialized_infeasible_abstains
#print axioms TensorGuard.CegarSerialized.serialized_feasible_safe
#print axioms TensorGuard.CegarSerialized.decideSerialized_safeSound
#print axioms TensorGuard.CegarSerialized.infeasible_serialized_safeSound_any_bug

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

-- Dtype matmul encoding faithfulness (Step 146): the dtype-equality formula is
-- UNSAT iff the dtypes differ, and for known dtypes coincides with dtMatmulBug.
#print axioms TensorGuard.SmtEncoding.dtype_smt_unsat_iff_ne
#print axioms TensorGuard.SmtEncoding.dtype_smt_matches_dtMatmulBug
#print axioms TensorGuard.SmtEncoding.dtype_same_sat

-- Shape/dtype SMT encoding faithfulness (Step 242): the formulas handed to Z3
-- for broadcast compatibility, reshape divisibility, split/chunk partition
-- reconstruction, and dtype promotion are satisfiable exactly when the
-- corresponding Lean transfer guard accepts; UNSAT is exactly the modeled error.
#print axioms TensorGuard.SmtEncoding.broadcast_smt_sat_iff_bcDim_some
#print axioms TensorGuard.SmtEncoding.broadcast_smt_unsat_iff_bcDim_none
#print axioms TensorGuard.SmtEncoding.broadcast_smt_unsat_iff_incompatible
#print axioms TensorGuard.SmtEncoding.divisibility_smt_sat_iff_reshapeValid
#print axioms TensorGuard.SmtEncoding.divisibility_smt_unsat_iff_invalid
#print axioms TensorGuard.SmtEncoding.partition_smt_sat_iff_sum_eq
#print axioms TensorGuard.SmtEncoding.partition_smt_matches_splitSectionsValid
#print axioms TensorGuard.SmtEncoding.partition_smt_unsat_iff_mismatch
#print axioms TensorGuard.SmtEncoding.dtype_promote_smt_sat_iff
#print axioms TensorGuard.SmtEncoding.dtype_promote_smt_unsat_iff_mismatch
#print axioms TensorGuard.SmtEncoding.dtype_promote_chain_smt_sat_iff
#print axioms TensorGuard.SmtEncoding.dtype_promote_chain_smt_unsat_iff_mismatch

-- Sparse-layout constructor invariants (Step 236): COO/CSR/CSC/BSR/BSC accepted
-- constructor models materialize to the requested dense shape; checked examples
-- cover batched and blocked layouts plus invariant and dense-tail rejections.
#print axioms TensorGuard.SparseLayouts.dense_materialization_shape_sound
#print axioms TensorGuard.SparseLayouts.mkAccepted_dense_shape_sound
#print axioms TensorGuard.SparseLayouts.coo234_accepts
#print axioms TensorGuard.SparseLayouts.csr23_accepts
#print axioms TensorGuard.SparseLayouts.csc23_accepts
#print axioms TensorGuard.SparseLayouts.bsr43_accepts
#print axioms TensorGuard.SparseLayouts.bsc23_accepts
#print axioms TensorGuard.SparseLayouts.batched_csr_accepts
#print axioms TensorGuard.SparseLayouts.batched_bsr_accepts
#print axioms TensorGuard.SparseLayouts.coo234_toDense_shape
#print axioms TensorGuard.SparseLayouts.csr23_toDense_shape
#print axioms TensorGuard.SparseLayouts.csc23_toDense_shape
#print axioms TensorGuard.SparseLayouts.bsr43_toDense_shape
#print axioms TensorGuard.SparseLayouts.bsc23_toDense_shape
#print axioms TensorGuard.SparseLayouts.batched_csr_toDense_shape
#print axioms TensorGuard.SparseLayouts.batched_bsr_toDense_shape
#print axioms TensorGuard.SparseLayouts.csr_bad_compressed_length_rejected
#print axioms TensorGuard.SparseLayouts.csc_bad_compressed_length_rejected
#print axioms TensorGuard.SparseLayouts.bsr_bad_row_divisibility_rejected
#print axioms TensorGuard.SparseLayouts.bsr_bad_column_divisibility_rejected
#print axioms TensorGuard.SparseLayouts.compressed_dense_tail_mismatch_rejected

-- Recurrent hidden-state contracts (Step 237): RNN/GRU/LSTM output and h_n/c_n
-- shape transformations preserve batch/sequence layout, select the right
-- `batch_first` state axis, double only the output feature/state depth for
-- bidirectional layers, and keep projected LSTM c_n on hidden_size.
#print axioms TensorGuard.RecurrentRule.batch_first_output_preserves_layout
#print axioms TensorGuard.RecurrentRule.time_major_output_preserves_layout
#print axioms TensorGuard.RecurrentRule.batch_first_state_selects_dim0
#print axioms TensorGuard.RecurrentRule.time_major_state_selects_dim1
#print axioms TensorGuard.RecurrentRule.bidirectional_output_feature_doubles
#print axioms TensorGuard.RecurrentRule.bidirectional_state_depth_doubles
#print axioms TensorGuard.RecurrentRule.lstm_cell_state_uses_hidden_size_under_projection
#print axioms TensorGuard.RecurrentRule.gru_cell_state_rejected
#print axioms TensorGuard.RecurrentRule.rnn_cell_state_rejected
#print axioms TensorGuard.RecurrentRule.projected_bilstm_output_shape
#print axioms TensorGuard.RecurrentRule.projected_bilstm_h_state_shape
#print axioms TensorGuard.RecurrentRule.projected_bilstm_c_state_shape
#print axioms TensorGuard.RecurrentRule.time_major_bigru_output_shape
#print axioms TensorGuard.RecurrentRule.time_major_bigru_h_state_shape
#print axioms TensorGuard.RecurrentRule.unbatched_rnn_output_shape
#print axioms TensorGuard.RecurrentRule.unbatched_rnn_h_state_shape
#print axioms TensorGuard.RecurrentRule.wrong_input_size_rejected
#print axioms TensorGuard.RecurrentRule.bad_rank_rejected

-- Whole-module straight-line subject reduction (Step 238): local
-- complete/sound transfer functions compose across an entire straight-line
-- module, preserving well-formed positive shapes for all intermediates and
-- final outputs.  Concrete theorem-shaped MLP/CNN/indexing/attention programs
-- pin execution of representative real-code families.
#print axioms TensorGuard.SubjectReduction.step_subject_reduction
#print axioms TensorGuard.SubjectReduction.exec_subject_reduction
#print axioms TensorGuard.SubjectReduction.whole_module_subject_reduction
#print axioms TensorGuard.SubjectReduction.program_outputs_have_positive_shapes
#print axioms TensorGuard.SubjectReduction.mlp_exec_shape
#print axioms TensorGuard.SubjectReduction.cnn_head_exec_shape
#print axioms TensorGuard.SubjectReduction.indexing_exec_shape
#print axioms TensorGuard.SubjectReduction.attention_exec_shape

-- Path-sensitive conditional subject reduction (Step 239): supported branch
-- points check both paths and require equal joined output environments before
-- downstream code can run; unsupported branch points cannot produce a SAFE
-- environment, so sound mode conservatively abstains instead of silently
-- verifying tensor-value control flow.
#print axioms TensorGuard.SubjectReduction.envJoin_preserves_wf
#print axioms TensorGuard.SubjectReduction.condExec_some_iff_supported_join
#print axioms TensorGuard.SubjectReduction.sound_safe_implies_supported_branch
#print axioms TensorGuard.SubjectReduction.unsupported_branch_cannot_silently_safe
#print axioms TensorGuard.SubjectReduction.supported_conditional_subject_reduction
#print axioms TensorGuard.SubjectReduction.conditional_then_program_subject_reduction
#print axioms TensorGuard.SubjectReduction.conditional_join_exec_shape
#print axioms TensorGuard.SubjectReduction.conditional_tail_exec_shape
#print axioms TensorGuard.SubjectReduction.divergent_branch_join_rejected
#print axioms TensorGuard.SubjectReduction.unsupported_branch_abstains_example

-- Bounded-loop / ModuleList subject reduction (Step 240): statically resolved
-- ModuleList/Sequential and literal-range loops execute only within the
-- configured unroll limit; over-budget and unsupported loops abstain, while
-- successful bounded unrolls preserve well-formed environments.
#print axioms TensorGuard.SubjectReduction.exec_program_list_subject_reduction
#print axioms TensorGuard.SubjectReduction.bounded_unroll_exec_some_implies_supported_and_within_limit
#print axioms TensorGuard.SubjectReduction.modulelist_beyond_unroll_limit_abstains
#print axioms TensorGuard.SubjectReduction.static_range_beyond_unroll_limit_abstains
#print axioms TensorGuard.SubjectReduction.unsupported_loop_cannot_silently_safe
#print axioms TensorGuard.SubjectReduction.bounded_unroll_subject_reduction
#print axioms TensorGuard.SubjectReduction.modulelist_unroll_exec_shape
#print axioms TensorGuard.SubjectReduction.static_range_unroll_exec_shape
#print axioms TensorGuard.SubjectReduction.modulelist_beyond_limit_rejected
#print axioms TensorGuard.SubjectReduction.static_range_beyond_limit_rejected
#print axioms TensorGuard.SubjectReduction.unsupported_loop_abstains_example

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

-- Dtype-promotion chain transfer (Step 143): the multi-operand promotion fold is
-- compositional, an upper bound of every operand, order-independent, and
-- unknown-absorbing along the chain.
#print axioms TensorGuard.DtypePromoteChain.promoteRun_append
#print axioms TensorGuard.DtypePromoteChain.promoteRun_ge_acc
#print axioms TensorGuard.DtypePromoteChain.promoteRun_ge_elem
#print axioms TensorGuard.DtypePromoteChain.promoteRun_swap
#print axioms TensorGuard.DtypePromoteChain.promoteRun_unknown

-- Broadcast dim-chain transfer (Step 144): per-dimension broadcasting is
-- commutative, 1 is a two-sided identity, the compatible size is the max, the
-- rule flags iff genuinely incompatible, and the chain fold is compositional /
-- none-absorbing.
#print axioms TensorGuard.BroadcastChain.bcDim_comm
#print axioms TensorGuard.BroadcastChain.bcDim_one_left
#print axioms TensorGuard.BroadcastChain.bcDim_one_right
#print axioms TensorGuard.BroadcastChain.bcDim_self
#print axioms TensorGuard.BroadcastChain.bcDim_compat_max
#print axioms TensorGuard.BroadcastChain.bcDim_none_iff
#print axioms TensorGuard.BroadcastChain.bcRun_append
#print axioms TensorGuard.BroadcastChain.bcRun_none
#print axioms TensorGuard.BroadcastChain.bcRun_ones

-- Contiguity transfer under transpose/contiguous chain (Step 145): compositional,
-- .contiguous() erases history, transpose is an involution, keep-only preserves.
#print axioms TensorGuard.ContigFlow.ctgRun_append
#print axioms TensorGuard.ContigFlow.run_cons_contig
#print axioms TensorGuard.ContigFlow.run_after_contig
#print axioms TensorGuard.ContigFlow.run_transpose_involution
#print axioms TensorGuard.ContigFlow.run_allKeep
-- Concrete layout algebra (Step 235): row-major stride recurrence, canonical
-- non-degenerate channels-last NCHW strides, storage-offset preservation, and
-- CHW-tail viewability/refutation cases.
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.contigLayout_is_contiguous
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.rowMajorCHWTail_viewable
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.rowMajorCHWTail_view_layout
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.transpose01_preserves_storageOffset
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.permuteNCHWtoNHWC_preserves_storageOffset
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.rowMajor4_strides
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.rowMajor4_view_tail
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.channelsLast4_is_canonical_channels_last
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.channelsLast4_not_row_major_contiguous
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.channelsLast4_tail_not_viewable
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.channelsLast4_view_tail_rejected
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.narrowChannel_rowMajor_offset
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.narrowChannel_channelsLast_offset
#print axioms TensorGuard.ContigFlow.LayoutAlgebra.canonicalChannelsLast4_degenerateC_abstains

-- Flatten shape rule (Step 147): numel preservation, rank law, full flatten =
-- [numel], flattened dim = product of spanned sizes.
#print axioms TensorGuard.Flatten.prod_append
#print axioms TensorGuard.Flatten.prod_flatten
#print axioms TensorGuard.Flatten.length_flatten
#print axioms TensorGuard.Flatten.flatten_full
#print axioms TensorGuard.Flatten.flatten_singleton
#print axioms TensorGuard.Flatten.flatten_dim_value

-- Concatenation (torch.cat) shape rule (Step 148): compatibility iff non-axis
-- dims coincide, axis additivity, numel additivity, commutative/associative axis
-- sum, zero-length identity.
#print axioms TensorGuard.CatRule.prod_append
#print axioms TensorGuard.CatRule.catValid_iff
#print axioms TensorGuard.CatRule.catValid_pre_mismatch
#print axioms TensorGuard.CatRule.catAxis_value
#print axioms TensorGuard.CatRule.prod_cat
#print axioms TensorGuard.CatRule.cat_axis_comm
#print axioms TensorGuard.CatRule.cat_assoc
#print axioms TensorGuard.CatRule.cat_zero_right

-- nn.Embedding shape + index-range rule (Step 149): rank +1, trailing dim =
-- embedding_dim, numel scaling, index guard passes iff every index in range,
-- range monotonicity.
#print axioms TensorGuard.EmbeddingRule.prod_append
#print axioms TensorGuard.EmbeddingRule.emb_rank
#print axioms TensorGuard.EmbeddingRule.emb_trailing
#print axioms TensorGuard.EmbeddingRule.emb_prefix
#print axioms TensorGuard.EmbeddingRule.emb_numel
#print axioms TensorGuard.EmbeddingRule.idxValid_iff
#print axioms TensorGuard.EmbeddingRule.allValid_iff
#print axioms TensorGuard.EmbeddingRule.outOfRange_flagged
#print axioms TensorGuard.EmbeddingRule.allValid_mono

-- Reshape/view -1-inference rule (Step 150): inferred dim = numel / prod(known),
-- numel preserved under validity, admitted iff prod(known) positive and divides
-- numel, non-divisible flagged.
#print axioms TensorGuard.ReshapeInfer.prod_append
#print axioms TensorGuard.ReshapeInfer.reshapeValid_iff
#print axioms TensorGuard.ReshapeInfer.reshapeValid_imp_dvd
#print axioms TensorGuard.ReshapeInfer.nondivisible_flagged
#print axioms TensorGuard.ReshapeInfer.inferDim_spec
#print axioms TensorGuard.ReshapeInfer.prod_reshape_valid
#print axioms TensorGuard.ReshapeInfer.reshape_infer_position

-- nn.Linear shape rule (Step 151): rank preserved, last dim = out_features,
-- prefix preserved, numel scaling, in_features guard.
#print axioms TensorGuard.LinearRule.lin_rank
#print axioms TensorGuard.LinearRule.lin_last
#print axioms TensorGuard.LinearRule.lin_prefix
#print axioms TensorGuard.LinearRule.lin_numel
#print axioms TensorGuard.LinearRule.linValid_iff
#print axioms TensorGuard.LinearRule.mismatch_flagged

-- nn.Conv2d spatial rule (Step 152): identity, stride-1 form, monotonicity,
-- padded-input upper bound, positive-output guard, 4-D shape assembly.
#print axioms TensorGuard.Conv2d.convOut_identity
#print axioms TensorGuard.Conv2d.convOut_stride_one
#print axioms TensorGuard.Conv2d.convOut_mono
#print axioms TensorGuard.Conv2d.convOut_le
#print axioms TensorGuard.Conv2d.convOut_pos
#print axioms TensorGuard.Conv2d.conv2d_rank
#print axioms TensorGuard.Conv2d.conv2d_channels

-- nn.MaxPool2d/AvgPool2d spatial rule (Step 153): identity, stride-1 form,
-- monotonicity, upper bound, positive-output guard, channel preservation.
#print axioms TensorGuard.Pool2d.poolOut_identity
#print axioms TensorGuard.Pool2d.poolOut_stride_one
#print axioms TensorGuard.Pool2d.poolOut_mono
#print axioms TensorGuard.Pool2d.poolOut_le
#print axioms TensorGuard.Pool2d.poolOut_pos
#print axioms TensorGuard.Pool2d.pool2d_channels_preserved

-- nn.ConvTranspose1d length rule (Step 154): identity, no-pad form,
-- monotonicity, upsampling lower bound, shape laws.
#print axioms TensorGuard.ConvTranspose.ctOut_identity
#print axioms TensorGuard.ConvTranspose.ctOut_no_pad
#print axioms TensorGuard.ConvTranspose.ctOut_mono
#print axioms TensorGuard.ConvTranspose.ctOut_ge
#print axioms TensorGuard.ConvTranspose.ct_channels

-- nn.LayerNorm rule (Step 155): shape & numel preservation, suffix length,
-- suffix match, trailing-mismatch refutation.
#print axioms TensorGuard.LayerNormRule.ln_preserves
#print axioms TensorGuard.LayerNormRule.ln_numel
#print axioms TensorGuard.LayerNormRule.ln_length
#print axioms TensorGuard.LayerNormRule.ln_suffix_match
#print axioms TensorGuard.LayerNormRule.ln_mismatch_flagged

-- nn.PixelShuffle rule (Step 156): numel preservation, channel divisibility,
-- recovered channels, divisibility refutation.
#print axioms TensorGuard.PixelShuffle.ps_numel
#print axioms TensorGuard.PixelShuffle.ps_divisible
#print axioms TensorGuard.PixelShuffle.ps_cout
#print axioms TensorGuard.PixelShuffle.psValid_iff
#print axioms TensorGuard.PixelShuffle.ps_construct_valid

-- nn.AdaptiveAvgPool2d rule (Step 157): target-size exactness, batch/channel
-- preservation, rank, idempotence.
#print axioms TensorGuard.AdaptivePool.ap_spatial_h
#print axioms TensorGuard.AdaptivePool.ap_spatial_w
#print axioms TensorGuard.AdaptivePool.ap_channels
#print axioms TensorGuard.AdaptivePool.ap_idempotent

-- nn.Unflatten rule (Step 158): numel preservation under validity, rank law,
-- flatten/unflatten inverse, size guard.
#print axioms TensorGuard.Unflatten.unflatten_numel
#print axioms TensorGuard.Unflatten.unflatten_rank
#print axioms TensorGuard.Unflatten.unflatten_then_flatten
#print axioms TensorGuard.Unflatten.unflattenValid_iff
#print axioms TensorGuard.Unflatten.size_mismatch_flagged

-- nn.BatchNorm rule (Step 159): shape & numel preservation, feature guard,
-- channel index.
#print axioms TensorGuard.BatchNormRule.bn_preserves
#print axioms TensorGuard.BatchNormRule.bn_numel
#print axioms TensorGuard.BatchNormRule.featValid_iff
#print axioms TensorGuard.BatchNormRule.feat_mismatch_flagged
#print axioms TensorGuard.BatchNormRule.bn_channel_index

-- nn.Conv1d spatial rule (Step 160): identity, stride-1 form, monotonicity,
-- upper bound, positive-output guard, shape laws.
#print axioms TensorGuard.Conv1d.convOut_identity
#print axioms TensorGuard.Conv1d.convOut_stride_one
#print axioms TensorGuard.Conv1d.convOut_mono
#print axioms TensorGuard.Conv1d.convOut_le
#print axioms TensorGuard.Conv1d.convOut_pos
#print axioms TensorGuard.Conv1d.conv1d_channels

-- einops decomposition and rearrange axis-bijection rules (Step 229):
-- grouped-axis divisibility, inferred sub-axis reconstruction/product
-- preservation, and no drop/add/duplicate named-axis rearranges.
#print axioms TensorGuard.Einops.prod_append
#print axioms TensorGuard.Einops.decompValid_iff
#print axioms TensorGuard.Einops.decompValid_imp_dvd
#print axioms TensorGuard.Einops.nondivisible_decomposition_flagged
#print axioms TensorGuard.Einops.inferSubaxis_spec
#print axioms TensorGuard.Einops.decomposedGroup_product
#print axioms TensorGuard.Einops.inferSubaxis_position
#print axioms TensorGuard.Einops.axisBijection_iff_counts
#print axioms TensorGuard.Einops.axisBijection_refl
#print axioms TensorGuard.Einops.axisBijection_sym
#print axioms TensorGuard.Einops.axisBijection_trans
#print axioms TensorGuard.Einops.adjacent_swap_axis_bijection
#print axioms TensorGuard.Einops.dropped_axis_not_bijection
#print axioms TensorGuard.Einops.added_axis_not_bijection
#print axioms TensorGuard.Einops.duplicated_axis_not_bijection

-- SDPA batch/head/mask broadcasting and scoped GQA rules (Step 230): ordinary
-- leading-dimension broadcast, mask broadcast against score tensors, and
-- enable_gqa head divisibility/output-shape caveat.
#print axioms TensorGuard.SDPA.bcDim_none_iff
#print axioms TensorGuard.SDPA.bcShape_same
#print axioms TensorGuard.SDPA.bcShape_suffix_same
#print axioms TensorGuard.SDPA.standard_output_shape
#print axioms TensorGuard.SDPA.standard_output_rank
#print axioms TensorGuard.SDPA.standard_equal_leads
#print axioms TensorGuard.SDPA.mask_exact_valid
#print axioms TensorGuard.SDPA.mask_trailing_valid
#print axioms TensorGuard.SDPA.gqaHeadsValid_iff
#print axioms TensorGuard.SDPA.gqa_key_repetition_count
#print axioms TensorGuard.SDPA.gqa_value_repetition_count
#print axioms TensorGuard.SDPA.gqa_nondivisible_key_flagged
#print axioms TensorGuard.SDPA.gqa_nondivisible_value_flagged
#print axioms TensorGuard.SDPA.gqa_output_shape
#print axioms TensorGuard.SDPA.gqa_output_rank
#print axioms TensorGuard.SDPA.gqa_output_uses_query_heads
#print axioms TensorGuard.SDPA.gqa_prefix_broadcast_required

-- Chunk/split partition rules (Step 231): list split specifications are valid
-- iff their sections reconstruct the axis, valid sections re-concatenate to the
-- original shape/numel, and the PyTorch edge cases for uneven chunks, fewer
-- returned chunks, zero axes, empty split sections and invalid section sums are
-- machine-checked.
#print axioms TensorGuard.ChunkSplit.sum_append
#print axioms TensorGuard.ChunkSplit.prod_append
#print axioms TensorGuard.ChunkSplit.splitValid_iff
#print axioms TensorGuard.ChunkSplit.split_list_mismatch_flagged
#print axioms TensorGuard.ChunkSplit.axisConcat_reconstruct
#print axioms TensorGuard.ChunkSplit.splitConcat_shape
#print axioms TensorGuard.ChunkSplit.splitConcat_numel
#print axioms TensorGuard.ChunkSplit.split_int_uneven_example
#print axioms TensorGuard.ChunkSplit.split_int_tail_example
#print axioms TensorGuard.ChunkSplit.split_int_zero_axis_example
#print axioms TensorGuard.ChunkSplit.split_list_with_empty_section_valid
#print axioms TensorGuard.ChunkSplit.split_list_mismatch_example
#print axioms TensorGuard.ChunkSplit.chunk_uneven_example
#print axioms TensorGuard.ChunkSplit.chunk_many_sections_example
#print axioms TensorGuard.ChunkSplit.chunk_fewer_than_requested_example
#print axioms TensorGuard.ChunkSplit.chunk_fewer_than_requested_len
#print axioms TensorGuard.ChunkSplit.chunk_zero_axis_returns_requested_empties
#print axioms TensorGuard.ChunkSplit.split_concat_reconstruct_example
#print axioms TensorGuard.ChunkSplit.chunk_concat_reconstruct_example

-- Named-tensor refine_names / align_to rules (Step 232): existing names are
-- preserved by refine, singleton axes are inserted by align_to for fresh or
-- anonymous targets, duplicate concrete names are rejected, and no-ellipsis
-- align_to rejects unnamed input axes instead of silently dropping them.
#print axioms TensorGuard.NamedTensor.containsNamed_head
#print axioms TensorGuard.NamedTensor.unique_named_duplicate_head
#print axioms TensorGuard.NamedTensor.uniqueNamed_allows_repeated_anon
#print axioms TensorGuard.NamedTensor.refine_existing_name_preserved
#print axioms TensorGuard.NamedTensor.refine_rename_rejected
#print axioms TensorGuard.NamedTensor.refine_demotion_rejected
#print axioms TensorGuard.NamedTensor.refine_duplicate_requested_rejected
#print axioms TensorGuard.NamedTensor.refine_duplicate_current_rejected
#print axioms TensorGuard.NamedTensor.refine_shape_preserved
#print axioms TensorGuard.NamedTensor.refine_fill_anon_example
#print axioms TensorGuard.NamedTensor.refine_preserve_existing_example
#print axioms TensorGuard.NamedTensor.refine_duplicate_names_rejected
#print axioms TensorGuard.NamedTensor.existing_name_dim_preserved
#print axioms TensorGuard.NamedTensor.fresh_name_inserts_singleton
#print axioms TensorGuard.NamedTensor.anon_target_inserts_singleton
#print axioms TensorGuard.NamedTensor.align_names_preserved
#print axioms TensorGuard.NamedTensor.align_duplicate_target_rejected
#print axioms TensorGuard.NamedTensor.align_duplicate_current_rejected
#print axioms TensorGuard.NamedTensor.align_unnamed_input_rejected
#print axioms TensorGuard.NamedTensor.align_permute_example
#print axioms TensorGuard.NamedTensor.align_singleton_insert_example
#print axioms TensorGuard.NamedTensor.align_anon_target_insert_example
#print axioms TensorGuard.NamedTensor.align_missing_name_rejected
#print axioms TensorGuard.NamedTensor.align_duplicate_names_rejected

-- grid_sample / affine_grid shape rules (Step 233): rank-4 2-D and rank-5
-- 3-D sampler variants are distinguished, grid coordinate sizes select the
-- variant, input spatial axes / affine output sizes must be positive, empty
-- grid_sample output grids remain legal, and affine theta matrices are tied to
-- the requested sampler dimension.
#print axioms TensorGuard.GridSample.gridSample2DValid_iff
#print axioms TensorGuard.GridSample.gridSample3DValid_iff
#print axioms TensorGuard.GridSample.gridSample2D_valid_link
#print axioms TensorGuard.GridSample.gridSample3D_valid_link
#print axioms TensorGuard.GridSample.gridSample2D_invalid_link
#print axioms TensorGuard.GridSample.gridSample3D_invalid_link
#print axioms TensorGuard.GridSample.gridSample2D_output_shape
#print axioms TensorGuard.GridSample.gridSample3D_output_shape
#print axioms TensorGuard.GridSample.gridSample2D_output_rank
#print axioms TensorGuard.GridSample.gridSample3D_output_rank
#print axioms TensorGuard.GridSample.gridSample_wrong_input_rank_rejected
#print axioms TensorGuard.GridSample.gridSample_grid_rank_mismatch_rejected
#print axioms TensorGuard.GridSample.gridSample2D_coord_dim_flagged
#print axioms TensorGuard.GridSample.gridSample3D_coord_dim_flagged
#print axioms TensorGuard.GridSample.gridSample2D_zero_height_flagged
#print axioms TensorGuard.GridSample.gridSample2D_zero_width_flagged
#print axioms TensorGuard.GridSample.gridSample3D_zero_depth_flagged
#print axioms TensorGuard.GridSample.gridSample3D_zero_height_flagged
#print axioms TensorGuard.GridSample.gridSample3D_zero_width_flagged
#print axioms TensorGuard.GridSample.gridSample2D_batch_mismatch_flagged
#print axioms TensorGuard.GridSample.gridSample_accepts_empty_output_grid
#print axioms TensorGuard.GridSample.affineGrid2DValid_iff
#print axioms TensorGuard.GridSample.affineGrid3DValid_iff
#print axioms TensorGuard.GridSample.affineGrid2D_valid_link
#print axioms TensorGuard.GridSample.affineGrid3D_valid_link
#print axioms TensorGuard.GridSample.affineGrid2D_invalid_link
#print axioms TensorGuard.GridSample.affineGrid3D_invalid_link
#print axioms TensorGuard.GridSample.affineGrid2D_output_shape
#print axioms TensorGuard.GridSample.affineGrid3D_output_shape
#print axioms TensorGuard.GridSample.affineGrid2D_output_rank
#print axioms TensorGuard.GridSample.affineGrid3D_output_rank
#print axioms TensorGuard.GridSample.affineGrid_size_rank_rejected
#print axioms TensorGuard.GridSample.affineGrid_theta_rank_rejected
#print axioms TensorGuard.GridSample.affineGrid2D_theta_rows_flagged
#print axioms TensorGuard.GridSample.affineGrid2D_theta_cols_flagged
#print axioms TensorGuard.GridSample.affineGrid3D_theta_rows_flagged
#print axioms TensorGuard.GridSample.affineGrid3D_theta_cols_flagged
#print axioms TensorGuard.GridSample.affineGrid2D_size_batch_positive_required
#print axioms TensorGuard.GridSample.affineGrid2D_size_channel_positive_required
#print axioms TensorGuard.GridSample.affineGrid2D_size_height_positive_required
#print axioms TensorGuard.GridSample.affineGrid2D_size_width_positive_required
#print axioms TensorGuard.GridSample.affineGrid3D_size_depth_positive_required
#print axioms TensorGuard.GridSample.affineGrid_theta_batch_positive_required
#print axioms TensorGuard.GridSample.affineGrid2D_batch_mismatch_flagged

-- Distribution batch/event/log_prob rules (Step 234): constructor shape
-- algebra for Normal, Categorical, MultivariateNormal, Independent, plus the
-- identity/reshape TransformedDistribution fragment and event-rank reduction in
-- log_prob.
#print axioms TensorGuard.Distributions.bcDim_same
#print axioms TensorGuard.Distributions.bcDim_one_left
#print axioms TensorGuard.Distributions.bcDim_one_right
#print axioms TensorGuard.Distributions.bcDim_incompatible_example
#print axioms TensorGuard.Distributions.broadcast_example
#print axioms TensorGuard.Distributions.broadcast_incompatible_example
#print axioms TensorGuard.Distributions.normal_broadcast_output
#print axioms TensorGuard.Distributions.normal_bad_broadcast_rejected
#print axioms TensorGuard.Distributions.categorical_batch_drops_category_dim
#print axioms TensorGuard.Distributions.categorical_empty_rank_rejected
#print axioms TensorGuard.Distributions.categorical_zero_categories_rejected
#print axioms TensorGuard.Distributions.mvn_batch_event_output
#print axioms TensorGuard.Distributions.mvn_matrix_square_rejected
#print axioms TensorGuard.Distributions.mvn_event_mismatch_rejected
#print axioms TensorGuard.Distributions.mvn_batch_broadcast_rejected
#print axioms TensorGuard.Distributions.independent_moves_batch_to_event
#print axioms TensorGuard.Distributions.independent_preserves_when_zero
#print axioms TensorGuard.Distributions.independent_too_many_rejected
#print axioms TensorGuard.Distributions.normal_logProb_broadcasts_value
#print axioms TensorGuard.Distributions.categorical_logProb_drops_no_event
#print axioms TensorGuard.Distributions.mvn_logProb_drops_event_dim
#print axioms TensorGuard.Distributions.logProb_bad_value_broadcast_rejected
#print axioms TensorGuard.Distributions.reshapeShape_output
#print axioms TensorGuard.Distributions.reshapeShape_wrong_suffix_rejected
#print axioms TensorGuard.Distributions.reshapeShape_numel_mismatch_rejected
#print axioms TensorGuard.Distributions.transformed_identity_preserves_shape
#print axioms TensorGuard.Distributions.transformed_reshape_event_shape
#print axioms TensorGuard.Distributions.transformed_reshape_reinterprets_batch
#print axioms TensorGuard.Distributions.transformed_composed_reshape_identity
#print axioms TensorGuard.Distributions.transformed_wrong_domain_rejected
