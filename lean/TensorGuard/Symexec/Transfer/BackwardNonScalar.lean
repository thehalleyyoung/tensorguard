/-
TensorGuard.Symexec.Transfer.BackwardNonScalar

Refutation soundness for the non-scalar ``.backward()`` check
(`_check_backward`, SymBugKind.BACKWARD_ON_NONSCALAR): ``tensor.backward()``
with no ``gradient`` argument requires a scalar output — a non-scalar raises a
``RuntimeError`` ("grad can be implicitly created only for scalar outputs").

The engine reports a bug only when (a) ``requires_grad`` is positively known
``true`` (otherwise a *different* "does not require grad" error masks this one),
(b) the element count is a *known* natural number `≠ 1`, and (c) no ``gradient``
argument was supplied.  Any unknown input abstains.

Combines a three-valued boolean abstraction (`Tri`) for the ``requires_grad``
and "gradient supplied" flags with a `Nat` element-count abstraction.
-/
import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec
namespace BackwardNonScalar

/-- A three-valued boolean-flag abstraction. -/
inductive Tri
  | yes : Tri
  | no  : Tri
  | unk : Tri
  deriving DecidableEq, Repr

def triModels (c : Bool) : Tri → Prop
  | .yes => c = true
  | .no  => c = false
  | .unk => True

/-- A natural-number element-count abstraction. -/
inductive NumelAbs
  | known : Nat → NumelAbs
  | unk   : NumelAbs
  deriving DecidableEq, Repr

def numelModels (c : Nat) : NumelAbs → Prop
  | .known n => c = n
  | .unk     => True

/-- The runtime precondition for an implicit ``.backward()`` to succeed: the
tensor requires grad, and either a gradient was supplied or the output is a
single-element (scalar) tensor. -/
def BackwardOk (rg grad : Bool) (numel : Nat) : Prop :=
  rg = true ∧ (grad = true ∨ numel = 1)

/-- The engine's check: ``requires_grad`` known ``true``, no gradient supplied
(known ``false``), and a known element count `≠ 1`. -/
def fires (rg grad : Tri) (numel : NumelAbs) : Bool :=
  match rg, grad, numel with
  | .yes, .no, .known n => decide (n ≠ 1)
  | _,    _,   _        => false

/-- **Conservativity.** An unknown ``requires_grad`` flag abstains. -/
theorem conservative_rg_unk (grad : Tri) (numel : NumelAbs) :
    fires .unk grad numel = false := by
  cases grad <;> cases numel <;> rfl

/-- **Conservativity.** An unknown element count abstains. -/
theorem conservative_numel_unk (rg grad : Tri) :
    fires rg grad .unk = false := by
  cases rg <;> cases grad <;> rfl

/-- **Conservativity.** A supplied gradient abstains. -/
theorem conservative_grad_yes (rg : Tri) (numel : NumelAbs) :
    fires rg .yes numel = false := by
  cases rg <;> cases numel <;> rfl

/-- **Refutation soundness.** When the check fires, every concretization has
``requires_grad = true``, ``grad = false`` and element count `≠ 1`, so the
implicit-``.backward()`` precondition fails. -/
theorem refute {a b : Tri} {v : NumelAbs} (h : fires a b v = true) :
    ∀ rg grad n, triModels rg a → triModels grad b → numelModels n v →
      ¬ BackwardOk rg grad n := by
  cases a <;> cases b <;> cases v <;> simp [fires] at h
  rename_i n0
  intro rg grad n hrg hgrad hn
  simp only [triModels] at hrg hgrad
  simp only [numelModels] at hn
  subst hrg; subst hgrad; subst hn
  simp only [BackwardOk]
  rintro ⟨-, hb⟩
  rcases hb with hb | hb
  · exact (Bool.false_ne_true hb)
  · exact h hb

/-- **Certified counterexample.** -/
theorem witness {a b : Tri} {v : NumelAbs} (h : fires a b v = true) :
    ∃ rg grad n, triModels rg a ∧ triModels grad b ∧ numelModels n v ∧
      ¬ BackwardOk rg grad n := by
  cases a <;> cases b <;> cases v <;> simp [fires] at h
  rename_i n0
  refine ⟨true, false, n0, by simp [triModels], by simp [triModels],
    by simp [numelModels], ?_⟩
  simp only [BackwardOk]
  rintro ⟨-, hb⟩
  rcases hb with hb | hb
  · exact (Bool.false_ne_true hb)
  · exact h hb

end BackwardNonScalar
end Symexec
end TensorGuard
