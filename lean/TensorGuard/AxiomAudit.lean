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
