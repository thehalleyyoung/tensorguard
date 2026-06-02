/-
TensorGuard reduced-product transfer functions, machine-checked in Lean 4
(Step 126).

`Soundness.lean` / `Extended.lean` / `V5OperatorRules.lean` model the *operator*
transfer functions (shape typing rules). This file models the orthogonal axis:
the **reduced product** abstract domain that `src/domains/product.py` implements,
where several sub-domains exchange information through *reductions* so the
product is strictly more precise than the independent (direct) product.

We model the central reduced product exercised by Step 118 — the Type-tag ×
Nullity product — and define its full set of transfer functions:

* the two component lattices (`Tag`, `Nullity`) with their order and meet;
* the product value `PVal`, its product order `ple` and product meet `pmeet`;
* the inter-domain reductions `reduceTagNul` (Type-tag → Nullity) and
  `reduceNulTag` (Nullity → Type-tag), mirroring `TypeTagToNullityReduction`
  and `NullityToTypeTagReduction`, and their composition `reduce`.

Step 126's obligation is that this model is well-formed: the reductions are
*total* and *reductive* (they only ever sharpen, never coarsen, an abstract
value — `reduce p ⊑ p`), and the product meet is a genuine greatest-lower-bound
component-wise. Monotonicity (Step 127) and γ-soundness (Step 128) build on the
definitions established here.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace RP

/-! ## Nullity lattice: ⊥ ⊑ {null, notnull} ⊑ ⊤ -/

inductive Nullity
  | bot      -- unreachable
  | null     -- definitely None
  | notnull  -- definitely not None
  | top      -- unknown
  deriving DecidableEq, Repr

namespace Nullity

/-- Order as a Boolean test: `leb a b` iff a ⊑ b (a is at least as precise). -/
def leb : Nullity → Nullity → Bool
  | .bot,     _        => true
  | _,        .top     => true
  | .null,    .null    => true
  | .notnull, .notnull => true
  | _,        _        => false

/-- Greatest lower bound (meet). Conflicting facts meet to ⊥. -/
def meet : Nullity → Nullity → Nullity
  | .bot, _ => .bot
  | _, .bot => .bot
  | .top, y => y
  | x, .top => x
  | .null, .null => .null
  | .notnull, .notnull => .notnull
  | .null, .notnull => .bot
  | .notnull, .null => .bot

end Nullity

/-! ## Type-tag abstraction.

The Python domain is a powerset of type names. For the Type-tag ↔ Nullity
reduction only two predicates matter: can the value be `NoneType`, and can it be
some non-`None` type? We model the tag set by exactly those two bits, which is a
faithful (Galois) abstraction of the powerset with respect to nullity. -/

structure Tag where
  mayNone : Bool   -- NoneType ∈ tags
  mayOther : Bool  -- some non-NoneType tag ∈ tags
  deriving DecidableEq, Repr

namespace Tag

/-- Tag order: subset of possibilities (fewer possibilities = more precise). -/
def leb (a b : Tag) : Bool :=
  (!a.mayNone || b.mayNone) && (!a.mayOther || b.mayOther)

/-- Tag meet: intersection of possibilities. -/
def meet (a b : Tag) : Tag :=
  ⟨a.mayNone && b.mayNone, a.mayOther && b.mayOther⟩

end Tag

/-! ## Product value and its order / meet -/

structure PVal where
  tag : Tag
  nul : Nullity
  deriving DecidableEq, Repr

namespace PVal

/-- Product order: component-wise. -/
def ple (a b : PVal) : Bool :=
  a.tag.leb b.tag && a.nul.leb b.nul

/-- Product (component-wise) meet. -/
def pmeet (a b : PVal) : PVal :=
  ⟨Tag.meet a.tag b.tag, Nullity.meet a.nul b.nul⟩

end PVal

/-! ## Inter-domain reductions (the transfer functions of the reduced product) -/

/-- Type-tag → Nullity, mirroring `TypeTagToNullityReduction`:
    * tag is exactly `{NoneType}` ⇒ the value is definitely null;
    * `NoneType ∉ tag`           ⇒ the value is definitely not null.
    Otherwise nullity is left unchanged. The new nullity is *met* with the old
    so the step can only sharpen. -/
def reduceTagNul (p : PVal) : PVal :=
  if p.tag.mayNone && !p.tag.mayOther then
    ⟨p.tag, Nullity.meet p.nul .null⟩
  else if !p.tag.mayNone then
    ⟨p.tag, Nullity.meet p.nul .notnull⟩
  else
    p

/-- Nullity → Type-tag, mirroring `NullityToTypeTagReduction`:
    * definitely null     ⇒ tag must be `{NoneType}` (drop other tags);
    * definitely not null ⇒ tag cannot contain `NoneType` (drop NoneType). -/
def reduceNulTag (p : PVal) : PVal :=
  match p.nul with
  | .null    => ⟨⟨p.tag.mayNone, false⟩, p.nul⟩
  | .notnull => ⟨⟨false, p.tag.mayOther⟩, p.nul⟩
  | _        => p

/-- One reduction pass: apply both directions. -/
def reduce (p : PVal) : PVal :=
  reduceNulTag (reduceTagNul p)

/-! ## Step 126 obligations: the model is well-formed.

`leb`/`ple` are reflexive (totality of the order), the product meet is a lower
bound of both arguments, and every reduction is *reductive*: it returns a value
below its input, i.e. it only ever sharpens information. -/

theorem nul_leb_refl (n : Nullity) : n.leb n = true := by
  cases n <;> rfl

theorem tag_leb_refl (t : Tag) : t.leb t = true := by
  cases t with
  | mk a b => cases a <;> cases b <;> rfl

theorem ple_refl (p : PVal) : p.ple p = true := by
  simp [PVal.ple, tag_leb_refl, nul_leb_refl]

/-- The nullity meet is below its left argument. -/
theorem nul_meet_le_left (a b : Nullity) : (Nullity.meet a b).leb a = true := by
  cases a <;> cases b <;> rfl

theorem nul_meet_le_right (a b : Nullity) : (Nullity.meet a b).leb b = true := by
  cases a <;> cases b <;> rfl

theorem tag_meet_le_left (a b : Tag) : (Tag.meet a b).leb a = true := by
  cases a with
  | mk a1 a2 => cases b with
    | mk b1 b2 => cases a1 <;> cases a2 <;> cases b1 <;> cases b2 <;> rfl

theorem tag_meet_le_right (a b : Tag) : (Tag.meet a b).leb b = true := by
  cases a with
  | mk a1 a2 => cases b with
    | mk b1 b2 => cases a1 <;> cases a2 <;> cases b1 <;> cases b2 <;> rfl

/-- The product meet is a lower bound of its left argument. -/
theorem pmeet_le_left (a b : PVal) : (PVal.pmeet a b).ple a = true := by
  simp [PVal.pmeet, PVal.ple, tag_meet_le_left, nul_meet_le_left]

/-- The product meet is a lower bound of its right argument. -/
theorem pmeet_le_right (a b : PVal) : (PVal.pmeet a b).ple b = true := by
  simp [PVal.pmeet, PVal.ple, tag_meet_le_right, nul_meet_le_right]

/-- **Type-tag → Nullity is reductive.** -/
theorem reduceTagNul_reductive (p : PVal) : (reduceTagNul p).ple p = true := by
  unfold reduceTagNul
  cases p with
  | mk t n =>
    cases t with
    | mk mn mo =>
      cases mn <;> cases mo <;>
        simp [PVal.ple, tag_leb_refl] <;>
        cases n <;> rfl

/-- **Nullity → Type-tag is reductive.** -/
theorem reduceNulTag_reductive (p : PVal) : (reduceNulTag p).ple p = true := by
  unfold reduceNulTag
  cases p with
  | mk t n =>
    cases t with
    | mk mn mo =>
      cases n <;>
        simp [PVal.ple, nul_leb_refl] <;>
        cases mn <;> cases mo <;> rfl

/-- `ple` is transitive (needed to chain the two reductive steps). -/
theorem nul_leb_trans {a b c : Nullity}
    (h1 : a.leb b = true) (h2 : b.leb c = true) : a.leb c = true := by
  cases a <;> cases b <;> cases c <;> simp_all [Nullity.leb]

theorem tag_leb_trans {a b c : Tag}
    (h1 : a.leb b = true) (h2 : b.leb c = true) : a.leb c = true := by
  cases a with | mk a1 a2 => cases b with | mk b1 b2 => cases c with
    | mk c1 c2 =>
      cases a1 <;> cases a2 <;> cases b1 <;> cases b2 <;> cases c1 <;> cases c2 <;>
        simp_all [Tag.leb]

theorem ple_trans {a b c : PVal}
    (h1 : a.ple b = true) (h2 : b.ple c = true) : a.ple c = true := by
  simp only [PVal.ple, Bool.and_eq_true] at *
  exact ⟨tag_leb_trans h1.1 h2.1, nul_leb_trans h1.2 h2.2⟩

/-- **The full reduction is reductive.** A complete reduced-product reduction
    pass returns an abstract value below its input: it never loses precision and
    never adds spurious concrete points. This is the structural soundness
    obligation for Step 126. -/
theorem reduce_reductive (p : PVal) : (reduce p).ple p = true := by
  unfold reduce
  exact ple_trans (reduceNulTag_reductive _) (reduceTagNul_reductive p)

end RP
end TensorGuard
