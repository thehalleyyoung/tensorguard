/-
TensorGuard `nn.PixelShuffle` shape-rule, machine-checked in Lean 4 (Step 156).

`nn.PixelShuffle(r)` maps `(N, C·r², H, W) → (N, C, H·r, W·r)`: it requires the
channel count be divisible by `r²` and rearranges those channels into space.
This is exactly what `src/model_checker.py::_propagate_pixel_shuffle` computes
(channel divisibility guard + `C ↦ C/r²`, `H ↦ H·r`, `W ↦ W·r`).

We model the channel count as `C·r²` and prove:

  * **numel preservation** (`ps_numel`): the periodic-shuffle output has the same
    number of elements as the input — the rearrangement neither creates nor drops
    elements;
  * **channel divisibility** (`ps_divisible`, `ps_cout`): `r² ∣ (C·r²)` and the
    recovered output channel count is exactly `C` (for `r > 0`);
  * **divisibility refutation** (`psValid_iff`): the layer is applicable iff
    `r² ∣ C_in` — the soundness direction behind the "not divisible by r²" alarm.

The companion test `tests/test_pixelshuffle_rule.py` replays the rule on **real**
`nn.PixelShuffle` modules and confirms a non-divisible channel count makes torch
**raise** exactly when the Lean guard flags it.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace PixelShuffle

/-- numel of `(N, C·r², H, W)`. -/
def inNumel (n c r h w : Nat) : Nat := n * (c * (r * r)) * h * w

/-- numel of `(N, C, H·r, W·r)`. -/
def outNumel (n c r h w : Nat) : Nat := n * c * (h * r) * (w * r)

/-- **Numel preservation**: pixel-shuffle is a pure rearrangement. -/
theorem ps_numel (n c r h w : Nat) : inNumel n c r h w = outNumel n c r h w := by
  unfold inNumel outNumel
  simp [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]

/-- **Channel divisibility**: the input channel count `C·r²` is divisible by
    `r²`. -/
theorem ps_divisible (c r : Nat) : (r * r) ∣ (c * (r * r)) :=
  ⟨c, by rw [Nat.mul_comm]⟩

/-- **Recovered channels**: dividing the input channels `C·r²` by `r²` recovers
    exactly `C` (for `r > 0`). -/
theorem ps_cout (c r : Nat) (hr : 0 < r) : (c * (r * r)) / (r * r) = c := by
  have : 0 < r * r := Nat.mul_pos hr hr
  exact Nat.mul_div_cancel c this

/-- Applicability guard: pixel-shuffle is admitted iff `r² ∣ C_in`. -/
def psValid (cin r : Nat) : Bool := cin % (r * r) == 0

theorem psValid_iff (cin r : Nat) (hr : 0 < r) :
    psValid cin r = true ↔ (r * r) ∣ cin := by
  unfold psValid
  rw [beq_iff_eq]
  exact Nat.dvd_iff_mod_eq_zero.symm

/-- A channel count divisible by `r²` passes the guard. -/
theorem ps_construct_valid (c r : Nat) : psValid (c * (r * r)) r = true := by
  unfold psValid
  rw [beq_iff_eq]
  exact Nat.mul_mod_left c (r * r)

end PixelShuffle
end TensorGuard
