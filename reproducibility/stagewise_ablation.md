# Stage-wise ablation stack (Step 253)

This artifact isolates TensorGuard's major verification stages with live code paths: extraction, abstract domains, cross-domain reductions, CEGAR, third-party stubs, and proof-backed versus heuristic rule policy.

## Headline

| stage | live check | outcome |
| --- | --- | --- |
| extraction | graph has 2 layers / 3 steps and catches the seeded shape bug | True |
| abstract domains | each verification domain loses its own detection when ablated; phase stays diagnostic-only | True / True |
| cross-domain reductions | every registered reduction has a witness and refines its input | True / True |
| CEGAR | refined-contract diagnoses rise from 0 to 16 at depth 1 | True |
| stubs | registering `FancyBlock` turns an opaque clean model into a SAFE `STUB` model and catches bad contracts | True |
| proof rules | `torch.relu` is SAFE in sound mode while heuristic `torch.unique` abstains with a heuristic reason | True / True |

## Per-domain ablation

| domain | mode | full caught | ablated caught | tags |
| --- | --- | --- | --- | --- |
| shape | report_filter | True | False | SHAPE-INCOMPATIBLE, SHAPE-INCOMPATIBLE |
| dtype | report_filter | True | False | DTYPE-ERROR |
| device | runtime_flag | True | False | DEVICE-MISMATCH, DEVICE-MISMATCH |
| gradient | runtime_flag | True | False | GRADIENT-BROKEN |
| phase | diagnostic_flag | False | False | - |

## Per-reduction witnesses

| reduction class | rule | changed components | default engine? |
| --- | --- | --- | --- |
| NullityToTypeTagReduction | NullityToTypeTag | type_tag | True |
| NumericToNullityReduction | NumericToNullity | nullity | True |
| TruthinessReduction | Truthiness(truthy=True) | interval, type_tag, nullity | False |
| TypeTagToNullityReduction | TypeTagToNullity | nullity | True |
| TypeTagToNumericReduction | TypeTagToNumeric | interval | True |
