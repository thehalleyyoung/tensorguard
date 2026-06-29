/-
TensorGuard.Symexec.Soundness

The whole-program soundness theorem for the symbolic-execution engine
(SYMEXEC_100_STEPS Step 92): **a report implies a real failing concretization
exists.**

We package each `TensorGuard.Symexec.Transfer.*` check as an instance of the
uniform `Sem.DimPairCheck` / `Sem.UnaryCheck` framework, then prove:

* `report_sound` — for a straight-line program (a list of checking steps over an
  abstract store), if the engine reports a bug then some step fires and there is
  a concrete store, modeled by the abstract store, on which the corresponding
  operation genuinely fails;
* `report_no_false_positive` — equivalently, the engine never reports on a
  program all of whose concretizations are well-typed.

Together with the per-check `sound`/`witness` obligations (machine-checked in
the transfer modules) and the merge/widening soundness of `Symexec.Store`, this
is the Lean witness behind the engine's zero-false-positive guarantee.

Pure Lean 4 core (no mathlib).
-/

import TensorGuard.Symexec.Semantics
import TensorGuard.Symexec.Transfer.Matmul
import TensorGuard.Symexec.Transfer.Broadcast
import TensorGuard.Symexec.Transfer.Reshape
import TensorGuard.Symexec.Transfer.CatStack
import TensorGuard.Symexec.Transfer.Linear
import TensorGuard.Symexec.Transfer.UnpackArity
import TensorGuard.Symexec.Transfer.Einsum
import TensorGuard.Symexec.Transfer.AxisOOB
import TensorGuard.Symexec.Transfer.IndexOOB
import TensorGuard.Symexec.Transfer.DivZero
import TensorGuard.Symexec.Transfer.NegativeDim

namespace TensorGuard
namespace Symexec
namespace Soundness

open Dim Sem

/-! ## Every transfer is a sound check -/

def matmulCheck : DimPairCheck :=
  { fires := Matmul.fires, Ok := Matmul.InnerOk,
    sound := @Matmul.matmul_refute, witness := @Matmul.matmul_witness }

def broadcastCheck : DimPairCheck :=
  { fires := Broadcast.fires, Ok := Broadcast.BroadcastOk,
    sound := @Broadcast.broadcast_refute, witness := @Broadcast.broadcast_witness }

def reshapeCheck : DimPairCheck :=
  { fires := Reshape.fires, Ok := Reshape.ReshapeOk,
    sound := @Reshape.refute, witness := @Reshape.witness }

def catStackCheck : DimPairCheck :=
  { fires := CatStack.fires, Ok := CatStack.MustMatchOk,
    sound := @CatStack.refute, witness := @CatStack.witness }

def linearCheck : DimPairCheck :=
  { fires := Linear.fires, Ok := Linear.FeatureOk,
    sound := @Linear.refute, witness := @Linear.witness }

def unpackArityCheck : DimPairCheck :=
  { fires := UnpackArity.fires, Ok := UnpackArity.ArityOk,
    sound := @UnpackArity.refute, witness := @UnpackArity.witness }

def einsumCheck : DimPairCheck :=
  { fires := Einsum.fires, Ok := Einsum.LabelOk,
    sound := @Einsum.refute, witness := @Einsum.witness }

def axisCheck : DimPairCheck :=
  { fires := AxisOOB.fires, Ok := AxisOOB.AxisOk,
    sound := @AxisOOB.refute, witness := @AxisOOB.witness }

def indexCheck : DimPairCheck :=
  { fires := IndexOOB.fires, Ok := IndexOOB.IndexOk,
    sound := @IndexOOB.refute, witness := @IndexOOB.witness }

def divZeroCheck : UnaryCheck :=
  { fires := DivZero.fires, Ok := DivZero.DivisorOk,
    sound := @DivZero.refute, witness := @DivZero.witness }

/-- The catalogue of binary checks the engine runs. -/
def allBinaryChecks : List DimPairCheck :=
  [matmulCheck, broadcastCheck, reshapeCheck, catStackCheck, linearCheck,
   unpackArityCheck, einsumCheck, axisCheck, indexCheck]

/-! ## Whole-program soundness -/

/-- A straight-line program: a list of checking steps over the store. -/
abbrev Program := List Step

/-- The engine reports a bug iff some step fires under the store. -/
def reports (p : Program) (σ : Store) : Bool :=
  p.any (fun s => s.fires σ)

/-- **Step 92 — report implies a real failing concretization.**  If the engine
reports on program `p` under abstract store `σ`, then there is a step that fired
and a pair of concrete sizes — modeled by the store at that step's operands — on
which the checked operation's precondition genuinely fails.  Hence the report is
backed by an actual, reproducible failing execution. -/
theorem report_sound (p : Program) (σ : Store) (h : reports p σ = true) :
    ∃ s ∈ p, ∃ x y,
      dimModels x (σ s.lhs) ∧ dimModels y (σ s.rhs) ∧ ¬ s.check.Ok x y := by
  rw [reports, List.any_eq_true] at h
  obtain ⟨s, hmem, hf⟩ := h
  exact ⟨s, hmem, step_certificate s σ hf⟩

/-- **No false positive.**  Contrapositive view: if *every* step's precondition
holds on the concrete sizes a model assigns, the engine does not report.  More
precisely, a fired step rules out any concretization satisfying its
precondition. -/
theorem report_no_false_positive (s : Step) (σ : Store) (c : CStore)
    (hσ : StoreModels c σ) (hf : s.fires σ = true) :
    ¬ s.check.Ok (c s.lhs) (c s.rhs) :=
  step_local_sound s σ c hσ hf

/-- The single-check specialization, stated directly for the matmul detector as
a sanity anchor: a fired matmul check has a concrete contraction-dim
counterexample. -/
theorem matmul_report_has_witness {a b : Dim}
    (h : matmulCheck.fires a b = true) :
    ∃ x y, dimModels x a ∧ dimModels y b ∧ ¬ matmulCheck.Ok x y :=
  pair_certificate matmulCheck h

/-- And the engine cannot report a matmul bug on operands that admit equal
contraction dims (no false positive at the leaf). -/
theorem matmul_no_false_positive {a b : Dim}
    (h : matmulCheck.fires a b = true) :
    ¬ ∃ x y, dimModels x a ∧ dimModels y b ∧ matmulCheck.Ok x y :=
  pair_no_false_positive matmulCheck h

/-- Division-by-zero, the unary anchor. -/
theorem divzero_no_false_positive {a : Dim} (h : divZeroCheck.fires a = true) :
    ¬ ∃ x, dimModels x a ∧ divZeroCheck.Ok x :=
  unary_no_false_positive divZeroCheck h

end Soundness
end Symexec
end TensorGuard
