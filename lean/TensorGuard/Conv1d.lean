/-
TensorGuard `nn.Conv1d` spatial shape-rule, machine-checked in Lean 4 (Step 160).

`nn.Conv1d` maps `(N, C_in, L) → (N, C_out, L')` with

    L' = ⌊(L + 2·pad − dilation·(kernel−1) − 1) / stride⌋ + 1
       = (L + 2·pad − eff) / stride + 1,   eff = dilation·(kernel−1) + 1,

exactly as `src/model_checker.py::_propagate_conv1d` computes (Python `//`), plus
the channel transform `C_in ↦ C_out`.  This file models the integer spatial map
and the 3-D shape assembly and proves identity, the stride-1 closed form,
monotonicity in the input length, the padded-input upper bound, the
positive-output guard, and the shape laws (rank 3, batch kept, channels set to
`C_out`).

The companion test `tests/test_conv1d_rule.py` replays the rule on **real**
`nn.Conv1d` modules over a kernel/stride/padding/dilation grid.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace Conv1d

theorem div_mono {a b k : Nat} (h : a ≤ b) : a / k ≤ b / k := by
  rcases Nat.eq_zero_or_pos k with hk | hk
  · subst hk; simp
  · rw [Nat.le_div_iff_mul_le hk]
    exact Nat.le_trans (Nat.div_mul_le_self a k) h

def eff (kernel dilation : Nat) : Nat := dilation * (kernel - 1) + 1

/-- Spatial output length: `(L + 2·pad − eff) / stride + 1`. -/
def convOut (input pad eff stride : Nat) : Nat :=
  (input + 2 * pad - eff) / stride + 1

theorem convOut_identity (input : Nat) (h : 1 ≤ input) :
    convOut input 0 1 1 = input := by
  unfold convOut; simp [Nat.div_one]; omega

theorem convOut_stride_one (input pad e : Nat) :
    convOut input pad e 1 = input + 2 * pad - e + 1 := by
  unfold convOut; simp [Nat.div_one]

theorem convOut_mono (i₁ i₂ pad e stride : Nat) (h : i₁ ≤ i₂) :
    convOut i₁ pad e stride ≤ convOut i₂ pad e stride := by
  unfold convOut
  have h1 : i₁ + 2 * pad - e ≤ i₂ + 2 * pad - e :=
    Nat.sub_le_sub_right (Nat.add_le_add_right h _) e
  exact Nat.add_le_add_right (div_mono h1) 1

theorem convOut_le (input pad e stride : Nat)
    (he : 1 ≤ e) (hv : e ≤ input + 2 * pad) :
    convOut input pad e stride ≤ input + 2 * pad := by
  unfold convOut
  have hdiv : (input + 2 * pad - e) / stride ≤ input + 2 * pad - e := Nat.div_le_self _ _
  have : (input + 2 * pad - e) + 1 ≤ input + 2 * pad := by omega
  exact Nat.le_trans (Nat.add_le_add_right hdiv 1) this

theorem convOut_pos (input pad e stride : Nat) : 1 ≤ convOut input pad e stride := by
  unfold convOut; exact Nat.le_add_left 1 _

def conv1dShape (n cout lOut : Nat) : List Nat := [n, cout, lOut]

theorem conv1d_rank (n cout lOut : Nat) : (conv1dShape n cout lOut).length = 3 := rfl
theorem conv1d_batch (n cout lOut : Nat) : (conv1dShape n cout lOut).head? = some n := rfl
theorem conv1d_channels (n cout lOut : Nat) : (conv1dShape n cout lOut).get? 1 = some cout := rfl

end Conv1d
end TensorGuard
