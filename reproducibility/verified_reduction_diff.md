# Step 131 — Differential test of Python reduced-product reduction vs Lean-extracted verified checker

Reference: `lean/TensorGuard/ReducedProduct.lean (extracted to reproducibility/lean_reduction_extracted.txt)`.

- Abstract values audited: **16**
- Agree with verified reduction: **14**
- Consistent (reachable) inputs: **7**, of which agree: **7**
- Python a sound over-approximation: **16** of 16

## Findings

- `python_matches_verified_on_all_consistent_inputs`: **True**
- `python_is_sound_overapproximation_on_all_inputs`: **True**
- `divergences_only_on_contradictory_unreachable_inputs`: **True**

On every reachable (consistent) abstract value the Python reduction is byte-identical to the verified Lean reduction. On all sixteen values it is a sound over-approximation (gamma_python contains gamma_lean). The only divergences are on contradictory, unreachable inputs (empty concretization), where the verified model collapses to bottom but the Python rule overwrites the conflicting nullity rather than proving unreachability -- a precision gap, never an unsoundness.

## Per-input diff

| input | lean reduce | python reduce | agree | consistent | γ(lean) | γ(python) | python ⊇ |
|---|---|---|---|---|---|---|---|
| `00:bot` | `00:bot` | `00:bot` | True | False | {} | {} | True |
| `00:null` | `00:bot` | `00:bot` | True | False | {} | {} | True |
| `00:notnull` | `00:bot` | `00:bot` | True | False | {} | {} | True |
| `00:top` | `00:bot` | `00:bot` | True | False | {} | {} | True |
| `01:bot` | `00:bot` | `00:bot` | True | False | {} | {} | True |
| `01:null` | `00:bot` | `01:notnull` | False | False | {} | {cobj} | True |
| `01:notnull` | `01:notnull` | `01:notnull` | True | True | {cobj} | {cobj} | True |
| `01:top` | `01:notnull` | `01:notnull` | True | True | {cobj} | {cobj} | True |
| `10:bot` | `00:bot` | `00:bot` | True | False | {} | {} | True |
| `10:null` | `10:null` | `10:null` | True | True | {cnone} | {cnone} | True |
| `10:notnull` | `00:bot` | `10:null` | False | False | {} | {cnone} | True |
| `10:top` | `10:null` | `10:null` | True | True | {cnone} | {cnone} | True |
| `11:bot` | `00:bot` | `00:bot` | True | False | {} | {} | True |
| `11:null` | `10:null` | `10:null` | True | True | {cnone} | {cnone} | True |
| `11:notnull` | `01:notnull` | `01:notnull` | True | True | {cobj} | {cobj} | True |
| `11:top` | `11:top` | `11:top` | True | True | {cnone,cobj} | {cnone,cobj} | True |
