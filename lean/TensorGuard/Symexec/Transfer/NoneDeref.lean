/-
TensorGuard.Symexec.Transfer.NoneDeref

Refutation soundness for the None-propagation check
(`_report_none_deref` / `_report_unpack`, SymBugKind.NONE_PROPAGATION):
dereferencing (attribute access / call / indexing) or tuple-unpacking a value
that is `None` raises `AttributeError`/`TypeError` at runtime.  The engine
reports a bug only when the use site's value is *positively pinned* to the
abstract `None` (a known, non-⊤ `NoneVal`); a value that is ⊤ (e.g. it first
passed through an opaque call) abstains rather than guessing.

Carries a tiny three-valued nullability abstraction (`NoneAbs`).
-/
import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec
namespace NoneDeref

/-- A three-valued nullability abstraction: known `None`, known non-`None`,
or ⊤ (unknown). -/
inductive NoneAbs
  | isNone  : NoneAbs
  | notNone : NoneAbs
  | unk     : NoneAbs
  deriving DecidableEq, Repr

/-- Concretization membership: a concrete "is this value `None`?" boolean. -/
def models (c : Bool) : NoneAbs → Prop
  | .isNone  => c = true
  | .notNone => c = false
  | .unk     => True

/-- The runtime precondition for a safe dereference/unpack: the value is not
`None`. -/
def NotNoneOk (isNone : Bool) : Prop := isNone = false

/-- The engine's check: the value is positively known to be the abstract
`None`. -/
def fires (v : NoneAbs) : Bool :=
  match v with
  | .isNone => true
  | _       => false

/-- **Conservativity.** ⊤ (unknown) abstains. -/
theorem conservative_unk : fires .unk = false := rfl

/-- **Conservativity.** A known non-`None` value abstains. -/
theorem conservative_notNone : fires .notNone = false := rfl

/-- **Refutation soundness.** When the check fires, every concretization is
`None`, so the dereference/unpack precondition fails (no false positive). -/
theorem refute {v : NoneAbs} (h : fires v = true) :
    ∀ isNone, models isNone v → ¬ NotNoneOk isNone := by
  cases v with
  | notNone => simp [fires] at h
  | unk     => simp [fires] at h
  | isNone  =>
    intro isNone hm
    simp only [models] at hm
    subst hm
    simp [NotNoneOk]

/-- **Certified counterexample.** A fired check exhibits a concrete `None`
value violating the precondition. -/
theorem witness {v : NoneAbs} (h : fires v = true) :
    ∃ isNone, models isNone v ∧ ¬ NotNoneOk isNone := by
  cases v with
  | notNone => simp [fires] at h
  | unk     => simp [fires] at h
  | isNone  => exact ⟨true, by simp [models], by simp [NotNoneOk]⟩

end NoneDeref
end Symexec
end TensorGuard
