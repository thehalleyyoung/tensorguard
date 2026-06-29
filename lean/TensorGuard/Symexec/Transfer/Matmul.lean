/-
TensorGuard.Symexec.Transfer.Matmul

Refutation soundness for the matmul contracted-dimension check
(`_check_matmul`, SymBugKind.MATMUL_DIM_MISMATCH).  For `A @ B` with
`A : (.., m, k1)` and `B : (.., k2, n)`, the engine reports a bug exactly when
`k1`, `k2` are both known and unequal.  We prove this never fires on a
runnable product and always exhibits a concrete witness.
-/
import TensorGuard.Symexec.Core

namespace TensorGuard
namespace Symexec
namespace Matmul

open Dim Core

/-- The runtime precondition for `A @ B`: the contracted dims agree. -/
def InnerOk (k1 k2 : Nat) : Prop := k1 = k2

/-- The engine's check: the two contraction dims are known-unequal. -/
def fires (kA kB : Dim) : Bool := disagreeFires kA kB

theorem conservative_left (kB : Dim) : fires .unk kB = false :=
  disagree_conservative_left kB

theorem conservative_right (kA : Dim) : fires kA .unk = false :=
  disagree_conservative_right kA

/-- **Refutation soundness.**  If the matmul check fires, every concretization
of the contraction dims violates `InnerOk` (the product is unrunnable). -/
theorem matmul_refute {kA kB : Dim} (h : fires kA kB = true) :
    ∀ k1 k2, dimModels k1 kA → dimModels k2 kB → ¬ InnerOk k1 k2 := by
  intro k1 k2 h1 h2
  exact disagree_sound h k1 k2 h1 h2

/-- **Certified counterexample.** -/
theorem matmul_witness {kA kB : Dim} (h : fires kA kB = true) :
    ∃ k1 k2, dimModels k1 kA ∧ dimModels k2 kB ∧ ¬ InnerOk k1 k2 :=
  disagree_witness h

end Matmul
end Symexec
end TensorGuard
