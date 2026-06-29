/-
TensorGuard.WeightsSound

Machine-checked soundness for the **weights-layer certifier**
(`src/symexec/weights.py`, even_more.md quantum leap), roadmap step 2.

Three independent guarantees, mirroring the three families of `WeightsFinding`
the certifier rules out, are formalised and proved here with no `sorry` and only
the trusted kernel axioms (audited in `AxiomAudit.lean`):

* **Storage** — if the tensor byte-ranges *tile* the data buffer `[0, L)`
  (contiguous, gap-free, in order), then no two distinct tensors share a byte
  (`storage_overlap` is impossible) and every byte is covered exactly once
  (`storage_gap` / `storage_undercovered` are impossible).  This is the formal
  content of `_validate_storage`.

* **Finiteness** — the IEEE all-ones-exponent bit-class scan (`_scan_nonfinite`)
  is a *decision procedure* for non-finiteness: it reports no finding iff every
  scanned word is finite, and when it does fire it exhibits a non-finite witness.

* **Contract** — if a checkpoint satisfies a (possibly *partial*) `name → shape`
  contract, then `load_state_dict(strict=True)` cannot raise a missing-key or
  size-mismatch error, and a partial contract's findings only ever concern keys
  the contract actually required (extra checkpoint tensors are never flagged —
  the soundness of `_check_contract(..., partial=True)`).

Pure Lean 4 core (no Mathlib); reasoning by structural induction + `omega`.
-/

namespace TensorGuard
namespace Weights

/-! ## 1. Storage tiling ⇒ no aliasing, perfect coverage -/
namespace Storage

/-- A byte range `[begin, end)` carried by a tensor in the data buffer. -/
abbrev Range := Nat × Nat

/-- Byte `k` lies in range `r` iff `r.begin ≤ k < r.end`. -/
def inRange (k : Nat) (r : Range) : Prop := r.1 ≤ k ∧ k < r.2

/-- `tiledFrom c rs L` : laid out in order, the ranges `rs` exactly tile `[c, L)` —
each range begins where the previous one ended (no gap, no overlap) and the last
ends at `L`.  This is precisely the success condition of `_validate_storage`
(cursor advances contiguously from `0` and finishes at the buffer length). -/
def tiledFrom : Nat → List Range → Nat → Prop
  | c, [], L => c = L
  | c, r :: rs, L => r.1 = c ∧ c ≤ r.2 ∧ tiledFrom r.2 rs L

/-- Every tiled range starts at or after the running cursor. -/
theorem tiled_lb {rs : List Range} {c L : Nat}
    (h : tiledFrom c rs L) : ∀ r ∈ rs, c ≤ r.1 := by
  induction rs generalizing c with
  | nil => intro r hr; cases hr
  | cons hd tl ih =>
    intro r hr
    simp only [tiledFrom] at h
    obtain ⟨hhd, hce, htail⟩ := h
    rcases List.mem_cons.mp hr with h1 | h2
    · subst h1; omega
    · have := ih htail r h2; omega

/-- The running cursor never exceeds the buffer length. -/
theorem tiled_cursor_le {rs : List Range} {c L : Nat}
    (h : tiledFrom c rs L) : c ≤ L := by
  induction rs generalizing c with
  | nil => simp only [tiledFrom] at h; omega
  | cons hd tl ih =>
    simp only [tiledFrom] at h
    obtain ⟨hhd, hce, htail⟩ := h
    have := ih htail; omega

/-- Every tiled range ends at or before the buffer length. -/
theorem tiled_ub {rs : List Range} {c L : Nat}
    (h : tiledFrom c rs L) : ∀ r ∈ rs, r.2 ≤ L := by
  induction rs generalizing c with
  | nil => intro r hr; cases hr
  | cons hd tl ih =>
    intro r hr
    simp only [tiledFrom] at h
    obtain ⟨hhd, hce, htail⟩ := h
    rcases List.mem_cons.mp hr with h1 | h2
    · subst h1; exact tiled_cursor_le htail
    · exact ih htail r h2

/-- **No aliasing.**  In a tiled layout, any byte is contained in *at most one*
range — distinct tensors never share storage (`storage_overlap` is refuted). -/
theorem tiled_no_alias {rs : List Range} {c L : Nat}
    (h : tiledFrom c rs L) :
    ∀ k, ∀ r1 ∈ rs, ∀ r2 ∈ rs, inRange k r1 → inRange k r2 → r1 = r2 := by
  induction rs generalizing c with
  | nil => intro k r1 hr1; cases hr1
  | cons hd tl ih =>
    intro k r1 hr1 r2 hr2 hk1 hk2
    simp only [tiledFrom] at h
    obtain ⟨hhd, hce, htail⟩ := h
    have key : ∀ r ∈ tl, inRange k hd → inRange k r → False := by
      intro r hr hkhd hkr
      have hlb := tiled_lb htail r hr      -- hd.2 ≤ r.1
      simp only [inRange] at hkhd hkr
      omega                                -- k < hd.2 ≤ r.1 ≤ k
    rcases List.mem_cons.mp hr1 with e1 | e1 <;>
      rcases List.mem_cons.mp hr2 with e2 | e2
    · subst e1; subst e2; rfl
    · subst e1; exact (key r2 e2 hk1 hk2).elim
    · subst e2; exact (key r1 e1 hk2 hk1).elim
    · exact ih htail k r1 e1 r2 e2 hk1 hk2

/-- **Perfect coverage.**  In a tiled layout every byte of `[c, L)` is covered by
some range — there are no unreferenced gaps and no undercoverage
(`storage_gap` / `storage_undercovered` are refuted). -/
theorem tiled_total {rs : List Range} {c L : Nat}
    (h : tiledFrom c rs L) :
    ∀ k, c ≤ k → k < L → ∃ r ∈ rs, inRange k r := by
  induction rs generalizing c with
  | nil =>
    intro k hck hkL
    simp only [tiledFrom] at h; omega
  | cons hd tl ih =>
    intro k hck hkL
    simp only [tiledFrom] at h
    obtain ⟨hhd, hce, htail⟩ := h
    by_cases hlt : k < hd.2
    · exact ⟨hd, List.mem_cons_self .., by simp only [inRange]; omega⟩
    · obtain ⟨r, hr, hkr⟩ := ih htail k (by omega) hkL
      exact ⟨r, List.mem_cons_of_mem _ hr, hkr⟩

end Storage

/-! ## 2. Finiteness bit-class scan is a decision procedure -/
namespace Finite

/-- The runtime non-finiteness predicate: an IEEE float word is NaN/Inf iff its
exponent field (the `mask` bits at offset `shift`) is all ones.  This is exactly
`(word >> exp_shift) & exp_mask == exp_mask` from `_scan_nonfinite`. -/
def isNonFinite (mask shift w : Nat) : Prop := (w >>> shift) &&& mask = mask

instance (mask shift w : Nat) : Decidable (isNonFinite mask shift w) := by
  unfold isNonFinite; exact inferInstance

/-- The scanner: fires iff some word in the tensor is non-finite. -/
def scan (mask shift : Nat) : List Nat → Bool
  | [] => false
  | w :: ws => decide (isNonFinite mask shift w) || scan mask shift ws

@[simp] theorem scan_nil (mask shift : Nat) : scan mask shift [] = false := rfl

/-- **Soundness of a clean scan.**  If the scan reports no finding, then *every*
word is finite — the certifier never misses a NaN/Inf. -/
theorem scan_sound {mask shift : Nat} {ws : List Nat}
    (h : scan mask shift ws = false) : ∀ w ∈ ws, ¬ isNonFinite mask shift w := by
  induction ws with
  | nil => intro w hw; cases hw
  | cons hd tl ih =>
    simp only [scan, Bool.or_eq_false_iff, decide_eq_false_iff_not] at h
    obtain ⟨hhd, htl⟩ := h
    intro w hw
    rcases List.mem_cons.mp hw with e | e
    · subst e; exact hhd
    · exact ih htl w e

/-- **Witness on a firing scan.**  When the scan fires it exhibits a concrete
non-finite element (a certified counterexample). -/
theorem scan_refute {mask shift : Nat} {ws : List Nat}
    (h : scan mask shift ws = true) : ∃ w ∈ ws, isNonFinite mask shift w := by
  induction ws with
  | nil => simp at h
  | cons hd tl ih =>
    simp only [scan, Bool.or_eq_true, decide_eq_true_eq] at h
    rcases h with hhd | htl
    · exact ⟨hd, List.mem_cons_self .., hhd⟩
    · obtain ⟨w, hw, hwnf⟩ := ih htl
      exact ⟨w, List.mem_cons_of_mem _ hw, hwnf⟩

/-- **Completeness on finite input.**  If every word is finite the scan does not
fire — no false positives. -/
theorem all_finite_no_fire {mask shift : Nat} {ws : List Nat}
    (h : ∀ w ∈ ws, ¬ isNonFinite mask shift w) : scan mask shift ws = false := by
  induction ws with
  | nil => rfl
  | cons hd tl ih =>
    simp only [scan]
    have hhd : decide (isNonFinite mask shift hd) = false :=
      decide_eq_false (h hd (List.mem_cons_self ..))
    have htl : scan mask shift tl = false :=
      ih (fun w hw => h w (List.mem_cons_of_mem _ hw))
    simp [hhd, htl]

end Finite

/-! ## 3. Contract satisfaction ⇒ no missing key / no shape mismatch -/
namespace Contract

/-- A tensor shape. -/
abbrev Shape := List Nat
/-- A contract / checkpoint entry: an (encoded) name and a shape. -/
abbrev Entry := Nat × Shape

/-- Look up a name in the checkpoint, returning its shape if present. -/
def lookupShape : List Entry → Nat → Option Shape
  | [], _ => none
  | (n, s) :: rest, k => if n = k then some s else lookupShape rest k

/-- Keys the contract requires that are *absent* from the checkpoint — exactly
the `contract_missing_key` set under `load_state_dict(strict=True)`. -/
def missing : List Entry → List Entry → List Nat
  | [], _ => []
  | (n, _) :: rest, hv =>
      (match lookupShape hv n with | none => [n] | some _ => []) ++ missing rest hv

/-- Required keys present but with the wrong shape — the `contract_shape_mismatch`
set (`RuntimeError: size mismatch for <key>`). -/
def mismatch : List Entry → List Entry → List Nat
  | [], _ => []
  | (n, s) :: rest, hv =>
      (match lookupShape hv n with
        | some s' => if s' = s then [] else [n]
        | none => []) ++ mismatch rest hv

/-- The checkpoint satisfies the contract: every required `(name, shape)` is
present with exactly that shape. -/
def Satisfied (req hv : List Entry) : Prop :=
  ∀ p ∈ req, lookupShape hv p.1 = some p.2

/-- **No missing key.**  A satisfied contract has an empty missing set, so
`strict=True` loading cannot raise a missing-key error. -/
theorem satisfied_no_missing {req hv : List Entry}
    (h : Satisfied req hv) : missing req hv = [] := by
  induction req with
  | nil => rfl
  | cons hd tl ih =>
    obtain ⟨n, s⟩ := hd
    have hlk : lookupShape hv n = some s := h (n, s) (List.mem_cons_self ..)
    have htl : Satisfied tl hv := fun p hp => h p (List.mem_cons_of_mem _ hp)
    simp only [missing, hlk, ih htl, List.nil_append]

/-- **No shape mismatch.**  A satisfied contract has an empty mismatch set, so
`strict=True` loading cannot raise a size-mismatch error. -/
theorem satisfied_no_mismatch {req hv : List Entry}
    (h : Satisfied req hv) : mismatch req hv = [] := by
  induction req with
  | nil => rfl
  | cons hd tl ih =>
    obtain ⟨n, s⟩ := hd
    have hlk : lookupShape hv n = some s := h (n, s) (List.mem_cons_self ..)
    have htl : Satisfied tl hv := fun p hp => h p (List.mem_cons_of_mem _ hp)
    simp [mismatch, hlk, ih htl]

/-- **Partiality soundness.**  Every key a *partial* contract flags as missing was
actually required by the contract — a partial contract never flags an extra
checkpoint tensor (`contract_unexpected_key` is intentionally not emitted, the
soundness of `_check_contract(partial=True)`). -/
theorem missing_in_req {req hv : List Entry} :
    ∀ k ∈ missing req hv, ∃ s, (k, s) ∈ req := by
  induction req with
  | nil => intro k hk; cases hk
  | cons hd tl ih =>
    obtain ⟨n, s⟩ := hd
    intro k hk
    simp only [missing] at hk
    rcases List.mem_append.mp hk with hhead | htail
    · -- head contributes [n] only when n is absent
      cases hlk : lookupShape hv n with
      | none =>
        simp only [hlk] at hhead
        rcases List.mem_singleton.mp hhead with e
        exact ⟨s, e ▸ List.mem_cons_self ..⟩
      | some s' => simp only [hlk] at hhead; cases hhead
    · obtain ⟨s', hmem⟩ := ih k htail
      exact ⟨s', List.mem_cons_of_mem _ hmem⟩

/-- Likewise every shape-mismatch key was a required key. -/
theorem mismatch_in_req {req hv : List Entry} :
    ∀ k ∈ mismatch req hv, ∃ s, (k, s) ∈ req := by
  induction req with
  | nil => intro k hk; cases hk
  | cons hd tl ih =>
    obtain ⟨n, s⟩ := hd
    intro k hk
    simp only [mismatch] at hk
    rcases List.mem_append.mp hk with hhead | htail
    · cases hlk : lookupShape hv n with
      | none => simp only [hlk] at hhead; cases hhead
      | some s' =>
        simp only [hlk] at hhead
        by_cases he : s' = s
        · simp only [he, if_pos] at hhead; cases hhead
        · simp only [if_neg he] at hhead
          rcases List.mem_singleton.mp hhead with e
          exact ⟨s, e ▸ List.mem_cons_self ..⟩
    · obtain ⟨s', hmem⟩ := ih k htail
      exact ⟨s', List.mem_cons_of_mem _ hmem⟩

end Contract

end Weights
end TensorGuard
