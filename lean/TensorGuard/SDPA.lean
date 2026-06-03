/-
TensorGuard scaled-dot-product attention (SDPA) shape rules, machine-checked in
Lean 4 (Step 230).

`src/sdpa_verify.py::verify_sdpa` models the PyTorch contract for
`F.scaled_dot_product_attention`:

  * query/key embedding dimensions must agree;
  * ordinary SDPA right-aligns and broadcasts the leading batch/head dimensions;
  * an attention mask must broadcast against the score tensor
    `(..., L_q, L_k)`;
  * with explicit `enable_gqa=True`, key/value are repeated on the `-3` head
    axis, so each key/value head count must divide the query head count.  That
    scoped GQA head axis is not ordinary broadcast; the output uses query heads.

The companion test `tests/test_sdpa_lean_conformance.py` generates concrete
cases from these theorem shapes and compares TensorGuard with live PyTorch.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace SDPA

/- ===================================================================== -/
/- 1. Right-aligned broadcast shapes                                      -/
/- ===================================================================== -/

/-- Per-dimension PyTorch/NumPy broadcast; `none` marks incompatibility. -/
def bcDim (a b : Nat) : Option Nat :=
  if a = 1 then some b
  else if b = 1 then some a
  else if a = b then some a
  else none

/-- Broadcast two reversed shapes.  Once one side is exhausted, the remaining
    reversed dimensions are copied because missing leading dimensions behave as
    singleton dimensions. -/
def bcRev : List Nat → List Nat → Option (List Nat)
  | [], [] => some []
  | xs, [] => some xs
  | [], ys => some ys
  | x :: xs, y :: ys =>
      match bcDim x y, bcRev xs ys with
      | some z, some rest => some (z :: rest)
      | _, _ => none

/-- Right-aligned shape broadcast. -/
def bcShape (xs ys : List Nat) : Option (List Nat) :=
  Option.map List.reverse (bcRev xs.reverse ys.reverse)

/-- Broadcast of three leading shape tuples, used for q/k/v. -/
def bc3 (xs ys zs : List Nat) : Option (List Nat) :=
  match bcShape xs ys with
  | none => none
  | some xy => bcShape xy zs

theorem bcDim_one_left (a : Nat) : bcDim 1 a = some a := by
  unfold bcDim
  simp

theorem bcDim_one_right (a : Nat) : bcDim a 1 = some a := by
  unfold bcDim
  by_cases h : a = 1 <;> simp [h]

theorem bcDim_self (a : Nat) : bcDim a a = some a := by
  unfold bcDim
  by_cases h : a = 1 <;> simp [h]

/-- A dimension broadcast is rejected exactly for genuinely incompatible sizes. -/
theorem bcDim_none_iff (a b : Nat) :
    bcDim a b = none ↔ (a ≠ 1 ∧ b ≠ 1 ∧ a ≠ b) := by
  unfold bcDim
  by_cases ha : a = 1 <;> by_cases hb : b = 1 <;> by_cases hab : a = b <;>
    simp_all

theorem bcRev_same (xs : List Nat) : bcRev xs xs = some xs := by
  induction xs with
  | nil => rfl
  | cons x rest ih =>
      simp [bcRev, bcDim_self, ih]

theorem bcRev_nil_right (xs : List Nat) : bcRev xs [] = some xs := by
  cases xs <;> rfl

theorem bcShape_same (xs : List Nat) : bcShape xs xs = some xs := by
  unfold bcShape
  rw [bcRev_same xs.reverse]
  simp

theorem bcRev_append_same_left (xs extra : List Nat) :
    bcRev (xs ++ extra) xs = some (xs ++ extra) := by
  induction xs with
  | nil => simpa using bcRev_nil_right extra
  | cons x rest ih =>
      simp [bcRev, bcDim_self, ih]

/-- A shorter suffix shape broadcasts against an equal trailing suffix. -/
theorem bcShape_suffix_same (pre suffix : List Nat) :
    bcShape (pre ++ suffix) suffix = some (pre ++ suffix) := by
  unfold bcShape
  rw [List.reverse_append]
  rw [bcRev_append_same_left suffix.reverse pre.reverse]
  simp [List.reverse_append]

/- ===================================================================== -/
/- 2. Ordinary SDPA output and mask rules                                  -/
/- ===================================================================== -/

/-- Ordinary SDPA output, after q/k/v leading dimensions have broadcast. -/
def standardOutput? (qLead kLead vLead : List Nat)
    (lq eq ek ev : Nat) : Option (List Nat) :=
  if eq = ek then
    match bc3 qLead kLead vLead with
    | none => none
    | some lead => some (lead ++ [lq, ev])
  else none

/-- Score tensor shape used by attention masks. -/
def scoreShape (lead : List Nat) (lq lk : Nat) : List Nat :=
  lead ++ [lq, lk]

def maskValid (lead : List Nat) (lq lk : Nat) (mask : List Nat) : Bool :=
  (bcShape (scoreShape lead lq lk) mask).isSome

theorem standard_output_shape (qLead kLead vLead lead : List Nat)
    (lq eq ev : Nat) (hbc : bc3 qLead kLead vLead = some lead) :
    standardOutput? qLead kLead vLead lq eq eq ev = some (lead ++ [lq, ev]) := by
  simp [standardOutput?, hbc]

theorem standard_output_rank (lead : List Nat) (lq ev : Nat) :
    (lead ++ [lq, ev]).length = lead.length + 2 := by
  simp

theorem standard_equal_leads (lead : List Nat) (lq eq ev : Nat) :
    standardOutput? lead lead lead lq eq eq ev = some (lead ++ [lq, ev]) := by
  have hbc : bc3 lead lead lead = some lead := by
    simp [bc3, bcShape_same]
  exact standard_output_shape lead lead lead lead lq eq ev hbc

theorem mask_exact_valid (lead : List Nat) (lq lk : Nat) :
    maskValid lead lq lk (scoreShape lead lq lk) = true := by
  unfold maskValid
  rw [bcShape_same (scoreShape lead lq lk)]
  rfl

theorem mask_trailing_valid (lead : List Nat) (lq lk : Nat) :
    maskValid lead lq lk [lq, lk] = true := by
  unfold maskValid scoreShape
  rw [bcShape_suffix_same lead [lq, lk]]
  rfl

/- ===================================================================== -/
/- 3. Scoped grouped-query attention (GQA) caveat                          -/
/- ===================================================================== -/

/-- Explicit GQA accepts concrete head counts exactly when key/value heads are
    positive divisors of query heads. -/
def gqaHeadsValid (hq hk hv : Nat) : Bool :=
  decide (0 < hk) && decide (hk ∣ hq) &&
    decide (0 < hv) && decide (hv ∣ hq)

/-- GQA broadcasts only the batch prefix; the output head count is query heads. -/
def gqaOutput? (qBatch kBatch vBatch : List Nat)
    (hq hk hv lq eq ek ev : Nat) : Option (List Nat) :=
  if eq = ek then
    if gqaHeadsValid hq hk hv then
      match bc3 qBatch kBatch vBatch with
      | none => none
      | some batch => some (batch ++ [hq, lq, ev])
    else none
  else none

theorem gqaHeadsValid_iff (hq hk hv : Nat) :
    gqaHeadsValid hq hk hv = true ↔
      (0 < hk ∧ hk ∣ hq ∧ 0 < hv ∧ hv ∣ hq) := by
  unfold gqaHeadsValid
  simp [Bool.and_eq_true, decide_eq_true_eq, and_assoc]

theorem gqa_key_repetition_count (hq hk hv : Nat)
    (h : gqaHeadsValid hq hk hv = true) :
    hk * (hq / hk) = hq := by
  have hvld := (gqaHeadsValid_iff hq hk hv).mp h
  exact Nat.mul_div_cancel' hvld.2.1

theorem gqa_value_repetition_count (hq hk hv : Nat)
    (h : gqaHeadsValid hq hk hv = true) :
    hv * (hq / hv) = hq := by
  have hvld := (gqaHeadsValid_iff hq hk hv).mp h
  exact Nat.mul_div_cancel' hvld.2.2.2

theorem gqa_nondivisible_key_flagged (hq hk hv : Nat) (h : ¬ hk ∣ hq) :
    gqaHeadsValid hq hk hv = false := by
  unfold gqaHeadsValid
  simp [h]

theorem gqa_nondivisible_value_flagged (hq hk hv : Nat) (h : ¬ hv ∣ hq) :
    gqaHeadsValid hq hk hv = false := by
  unfold gqaHeadsValid
  simp [h]

theorem gqa_output_shape (qBatch kBatch vBatch batch : List Nat)
    (hq hk hv lq eq ev : Nat)
    (hheads : gqaHeadsValid hq hk hv = true)
    (hbc : bc3 qBatch kBatch vBatch = some batch) :
    gqaOutput? qBatch kBatch vBatch hq hk hv lq eq eq ev =
      some (batch ++ [hq, lq, ev]) := by
  simp [gqaOutput?, hheads, hbc]

theorem gqa_output_rank (batch : List Nat) (hq lq ev : Nat) :
    (batch ++ [hq, lq, ev]).length = batch.length + 3 := by
  simp

theorem gqa_output_uses_query_heads (batch : List Nat) (hq lq ev : Nat) :
    (batch ++ [hq, lq, ev]).get? batch.length = some hq := by
  induction batch with
  | nil => rfl
  | cons _ rest ih => simpa using ih

theorem gqa_prefix_broadcast_required (qBatch kBatch vBatch : List Nat)
    (hq hk hv lq eq ev : Nat)
    (hheads : gqaHeadsValid hq hk hv = true)
    (hbc : bc3 qBatch kBatch vBatch = none) :
    gqaOutput? qBatch kBatch vBatch hq hk hv lq eq eq ev = none := by
  simp [gqaOutput?, hheads, hbc]

end SDPA
end TensorGuard
