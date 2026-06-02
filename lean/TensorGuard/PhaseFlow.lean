/-
TensorGuard train/eval phase transfer semantics, machine-checked in Lean 4
(Step 141).

Several TensorGuard bug classes are *phase-dependent*: a `Dropout`/`BatchNorm`
module behaves differently under `.train()` vs `.eval()`, and the verifier
tracks the `training` bit a module carries through a sequence of mode setters.
This file models the **phase transfer function over a chain of mode setters**:

  * a phase-agnostic op (`keep`, e.g. constructing or forwarding) propagates the
    incoming `training` bit unchanged;
  * `setTrain` (`.train()`) forces the bit to `true`;
  * `setEval` (`.eval()`) forces the bit to `false`.

`phaseRun b0 ops` folds the per-op transfer over a chain.  We prove the laws the
verifier relies on: `keep` is the identity, a setter is absorbing (last setter
wins), the run decomposes over concatenation, the outgoing bit equals the last
explicit setter (or the input bit when none intervened).

The companion test `tests/test_phase_flow_transfer.py` replays each chain on a
**real `nn.Module`** (and its children, since `.train()`/`.eval()` recurse) and
asserts `module.training` equals `phaseRun`, so the proved transfer holds against
the live torch module-mode machinery.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace PhaseFlow

/-- One step of the phase transfer. -/
inductive PhaseOp
  | keep      -- phase-agnostic op: propagate the incoming bit
  | setTrain  -- `.train()`: training := true
  | setEval   -- `.eval()`:  training := false
  deriving DecidableEq, Repr

/-- Per-op transfer on the `training` bit. -/
def phaseStep (b : Bool) : PhaseOp → Bool
  | PhaseOp.keep     => b
  | PhaseOp.setTrain => true
  | PhaseOp.setEval  => false

/-- Whether an op *sets* the phase (resets the bit irrespective of its input). -/
def isSetter : PhaseOp → Bool
  | PhaseOp.keep => false
  | _ => true

/-- The bit a setter forces (only meaningful when `isSetter` is true). -/
def setterValue : PhaseOp → Bool
  | PhaseOp.setTrain => true
  | PhaseOp.setEval  => false
  | PhaseOp.keep     => false  -- unused

/-- Fold the transfer over a chain of ops. -/
def phaseRun (b0 : Bool) : List PhaseOp → Bool
  | [] => b0
  | op :: rest => phaseRun (phaseStep b0 op) rest

/- ===================================================================== -/
/- 1. Per-op laws                                                        -/
/- ===================================================================== -/

theorem keep_id (b : Bool) : phaseStep b PhaseOp.keep = b := rfl
theorem setTrain_true (b : Bool) : phaseStep b PhaseOp.setTrain = true := rfl
theorem setEval_false (b : Bool) : phaseStep b PhaseOp.setEval = false := rfl

/-- A setter is **absorbing**: its output bit does not depend on the input
    (this is what makes `.train()`/`.eval()` last-wins). -/
theorem setter_absorbing (op : PhaseOp) (h : isSetter op = true) (b b' : Bool) :
    phaseStep b op = phaseStep b' op := by
  cases op <;> simp_all [isSetter, phaseStep]

/-- A setter outputs exactly its declared value, independent of input. -/
theorem setter_hits_value (op : PhaseOp) (h : isSetter op = true) (b : Bool) :
    phaseStep b op = setterValue op := by
  cases op <;> simp_all [isSetter, phaseStep, setterValue]

/- ===================================================================== -/
/- 2. Chain laws                                                         -/
/- ===================================================================== -/

/-- The run decomposes over concatenation (compositionality of the transfer). -/
theorem phaseRun_append (b0 : Bool) (xs ys : List PhaseOp) :
    phaseRun b0 (xs ++ ys) = phaseRun (phaseRun b0 xs) ys := by
  induction xs generalizing b0 with
  | nil => rfl
  | cons op rest ih => simp [phaseRun, ih]

/-- **Last setter wins**: appending a setter makes the whole prefix irrelevant —
    the result is exactly that setter's value, independent of `b0`. -/
theorem run_after_setter (b0 b0' : Bool) (xs : List PhaseOp)
    (op : PhaseOp) (h : isSetter op = true) :
    phaseRun b0 (xs ++ [op]) = phaseRun b0' (xs ++ [op]) := by
  rw [phaseRun_append, phaseRun_append]
  have : phaseStep (phaseRun b0 xs) op = phaseStep (phaseRun b0' xs) op :=
    setter_absorbing op h _ _
  simp [phaseRun, this]

/-- The result of a chain ending in a setter equals that setter's value. -/
theorem run_ends_at_value (b0 : Bool) (xs : List PhaseOp)
    (op : PhaseOp) (h : isSetter op = true) :
    phaseRun b0 (xs ++ [op]) = setterValue op := by
  rw [phaseRun_append]
  simpa [phaseRun] using setter_hits_value op h (phaseRun b0 xs)

/-- **Characterization for the setter-free fragment**: with no setter the
    outgoing bit is exactly the input bit (phase is preserved). -/
theorem run_noSetter_id (b0 : Bool) (xs : List PhaseOp)
    (h : ∀ op ∈ xs, isSetter op = false) :
    phaseRun b0 xs = b0 := by
  induction xs generalizing b0 with
  | nil => rfl
  | cons op rest ih =>
    have hop : isSetter op = false := h op (List.mem_cons_self _ _)
    have hrest : ∀ o ∈ rest, isSetter o = false :=
      fun o ho => h o (List.mem_cons_of_mem _ ho)
    cases op
    · simp only [phaseRun, phaseStep]; exact ih b0 hrest
    · simp [isSetter] at hop
    · simp [isSetter] at hop

end PhaseFlow
end TensorGuard
