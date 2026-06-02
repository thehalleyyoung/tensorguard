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

/-- One reduction pass: apply both directions, then **canonicalise** — if either
    component collapsed to the unreachable bottom (an empty type-tag or a ⊥
    nullity, which a reduction produces exactly when the two sub-domains carry
    contradictory facts), return the single canonical bottom value. This keeps
    `reduce` both reductive and monotone on the canonical sublattice. -/
def botPV : PVal := ⟨⟨false, false⟩, .bot⟩

def isBot (p : PVal) : Bool :=
  (!p.tag.mayNone && !p.tag.mayOther) ||
    (match p.nul with | .bot => true | _ => false)

def reduce (p : PVal) : PVal :=
  if isBot (reduceNulTag (reduceTagNul p)) then botPV
  else reduceNulTag (reduceTagNul p)

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

/-- The canonical bottom is below every abstract value. -/
theorem botPV_le (p : PVal) : botPV.ple p = true := by
  cases p with
  | mk t n => cases t with
    | mk a b => cases a <;> cases b <;> cases n <;> rfl

/-- **The full reduction is reductive.** A complete reduced-product reduction
    pass returns an abstract value below its input: it never loses precision and
    never adds spurious concrete points. This is the structural soundness
    obligation for Step 126. -/
theorem reduce_reductive (p : PVal) : (reduce p).ple p = true := by
  unfold reduce
  split
  · exact botPV_le p
  · exact ple_trans (reduceNulTag_reductive _) (reduceTagNul_reductive p)

/-! ## Step 127 obligations: monotonicity of the transfer functions.

A reduced-product operator is only sound to iterate to a fixed point if it is
*monotone*: refining an input can never coarsen an output. Two facts hold over
this finite lattice and are decided by exhaustive case analysis.

* The lattice **meet** (used by the worklist/fixpoint engine) is monotone in
  both arguments — unconditionally.
* The **reduction** `reduce` is monotone on *consistent* abstract values. A
  reduction need not be monotone on ill-formed values (e.g. an empty type-tag,
  which already denotes ⊥, paired with a non-⊥ nullity): those are exactly the
  non-canonical points the reduced product never materialises. `consistent`
  pins down the canonical sublattice, and on it monotonicity holds. -/

/-- The nullity meet is monotone in both arguments. -/
theorem nul_meet_mono {a a' b b' : Nullity}
    (ha : a.leb a' = true) (hb : b.leb b' = true) :
    (Nullity.meet a b).leb (Nullity.meet a' b') = true := by
  cases a <;> cases a' <;> cases b <;> cases b' <;> revert ha hb <;> decide

/-- The tag meet is monotone in both arguments. -/
theorem tag_meet_mono {a a' b b' : Tag}
    (ha : a.leb a' = true) (hb : b.leb b' = true) :
    (Tag.meet a b).leb (Tag.meet a' b') = true := by
  cases a with | mk a1 a2 => cases a' with | mk a1' a2' =>
  cases b with | mk b1 b2 => cases b' with | mk b1' b2' =>
  cases a1 <;> cases a2 <;> cases a1' <;> cases a2' <;>
    cases b1 <;> cases b2 <;> cases b1' <;> cases b2' <;>
    revert ha hb <;> decide

/-- The product meet is monotone in both arguments (the fixpoint engine's join
    of facts is order-preserving). -/
theorem pmeet_mono {a a' b b' : PVal}
    (ha : a.ple a' = true) (hb : b.ple b' = true) :
    (PVal.pmeet a b).ple (PVal.pmeet a' b') = true := by
  simp only [PVal.ple, PVal.pmeet, Bool.and_eq_true] at *
  exact ⟨tag_meet_mono ha.1 hb.1, nul_meet_mono ha.2 hb.2⟩

/-- A *consistent* (canonical) abstract value: the type-tag is non-empty (the
    empty tag already denotes ⊥) and the nullity is not the unreachable ⊥. These
    are the values the reduced product actually materialises. -/
def consistent (p : PVal) : Bool :=
  (p.tag.mayNone || p.tag.mayOther) &&
    (match p.nul with | .bot => false | _ => true)

/-- **The full reduction pass is monotone on consistent values.** Refining a
    canonical abstract value can never coarsen the result of a reduction pass —
    the property that makes the reduced-product fixpoint iteration sound. -/
theorem reduce_mono_consistent {a b : PVal}
    (hca : consistent a = true) (hcb : consistent b = true)
    (h : a.ple b = true) : (reduce a).ple (reduce b) = true := by
  cases a with | mk ta na => cases ta with | mk an ao =>
  cases b with | mk tb nb => cases tb with | mk bn bo =>
  cases an <;> cases ao <;> cases na <;> cases bn <;> cases bo <;> cases nb <;>
    revert hca hcb h <;> decide

/-! ## Step 128 obligations: γ-concretization soundness.

We make the abstraction's *meaning* explicit by a concretization γ into the
concrete value domain `CVal = {none, obj}` (is the runtime value `None`, or some
non-`None` object?) and prove the soundness facts that justify calling this a
sound abstract domain:

* **γ is monotone**: a more precise abstract value denotes fewer concrete values
  (`gamma_mono`);
* **the meet is exact**: γ(a ⊓ b) = γ(a) ∩ γ(b) (`pmeet_gamma`);
* **the reduction is concretization-preserving**: γ(reduce p) = γ(p)
  (`reduce_gamma`). This is the crucial soundness property — a reduction only
  removes abstract *imprecision*, never a concrete runtime behaviour. -/

inductive CVal
  | cnone  -- the runtime value is None
  | cobj   -- the runtime value is some non-None object
  deriving DecidableEq, Repr

/-- Does the type-tag admit this concrete value? -/
def tagAllows (t : Tag) : CVal → Bool
  | .cnone => t.mayNone
  | .cobj  => t.mayOther

/-- Does the nullity admit this concrete value? -/
def nulAllows : Nullity → CVal → Bool
  | .bot,     _      => false
  | .null,    .cnone => true
  | .null,    .cobj  => false
  | .notnull, .cnone => false
  | .notnull, .cobj  => true
  | .top,     _      => true

/-- γ as a membership test: `c ∈ γ(p)` iff both components admit `c`. -/
def mem (c : CVal) (p : PVal) : Bool :=
  tagAllows p.tag c && nulAllows p.nul c

/-- **γ is monotone**: a value below `b` denotes a subset of `b`'s concretization. -/
theorem gamma_mono {a b : PVal} {c : CVal}
    (h : a.ple b = true) (hc : mem c a = true) : mem c b = true := by
  cases c <;>
    (cases a with | mk ta na => cases ta with | mk an ao =>
     cases b with | mk tb nb => cases tb with | mk bn bo =>
     cases an <;> cases ao <;> cases na <;> cases bn <;> cases bo <;> cases nb <;>
       revert h hc <;> decide)

/-- **The meet is exact**: γ(a ⊓ b) = γ(a) ∩ γ(b). -/
theorem pmeet_gamma (a b : PVal) (c : CVal) :
    mem c (PVal.pmeet a b) = (mem c a && mem c b) := by
  cases c <;>
    (cases a with | mk ta na => cases ta with | mk an ao =>
     cases b with | mk tb nb => cases tb with | mk bn bo =>
     cases an <;> cases ao <;> cases na <;> cases bn <;> cases bo <;> cases nb <;>
       decide)

/-- **The reduction preserves concretization**: γ(reduce p) = γ(p). A reduction
    pass removes only abstract imprecision; every concrete runtime value
    admitted before is still admitted after (and none is added). -/
theorem reduce_gamma (p : PVal) (c : CVal) : mem c (reduce p) = mem c p := by
  cases c <;>
    (cases p with | mk t n => cases t with | mk mn mo =>
     cases mn <;> cases mo <;> cases n <;> decide)

end RP
end TensorGuard
