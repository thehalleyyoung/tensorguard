# Reduced product vs independent domains: precision gain (Step 118)

10 labeled single-variable programs run through the *real* `ProductInterpreter` twice -- with the production `ReductionEngine` (reduced product) and with an empty reduction set (independent product) -- and cross-checked against a CPython execution oracle.

## Headline

- false null-deref warnings under the independent product: **7**
- false null-deref warnings under the reduced product: **0**
- spurious warnings the reduction eliminates (oracle-confirmed safe): **7**
- reduced product strictly more precise: **True**
- real null derefs missed by the reduced product (recall loss): **0**
- reduced value ⊑ independent value on every program (γ(reduced) ⊆ γ(independent)): **True**

## Per-scenario

| id | family | oracle null-deref | independent warns | reduced warns | precision gain | refinement |
| --- | --- | --- | --- | --- | --- | --- |
| guard_int | guarded_precise | False | True | False | True | True |
| guard_float | guarded_precise | False | True | False | True | True |
| guard_str | guarded_precise | False | True | False | True | True |
| guard_bool | guarded_precise | False | True | False | True | True |
| guard_list | guarded_precise | False | True | False | True | True |
| guard_dict | guarded_precise | False | True | False | True | True |
| guard_tuple | guarded_precise | False | True | False | True | True |
| unguarded_maybe | genuine_null | True | True | True | False | True |
| unguarded_maybe_2 | genuine_null | True | True | True | False | True |
| definitely_null | definite_null | True | True | True | False | True |
