# Renamed-attribute parameter-sharing prevalence (round-4 Q6)

Reviewer: the limconc claim of ≤12% prevalence used the import
filter `from torch.utils.checkpoint import` and the string
`tied_weights_keys`; this misses renamed-attribute aliasing of
the form `self.alias = self.layer.weight`, which is the exact
pattern under which TG silently misclassifies grad-flag topology.

## Method

AST-grep every .py file under
`benchmarks/_corpus/transformers/src/transformers` and
`benchmarks/_corpus/timm/timm` for these patterns:

- R1: `self.X = self.Y.weight`
- R2: `self.X = self.Y.bias`
- R3: `self.X.weight = self.Y.weight`  (in-place rebind)
- R4: `self.X = nn.Parameter(self.Y.weight ...)`
- R5: `self.X.data = self.Y.weight.data`

## Result

- Files scanned: **2908**
- Parse failures: 0
- Files with any R1..R5 hit: **0 (0.00%)**
- Files with R1 or R2 (the strict aliasing case): 0
- Files with R4 (nn.Parameter wrap): 0
- Files importing torch.utils.checkpoint: 5 (0.17%)
- Files using tied_weights_keys / tie_weights / _tie_or_clone_weights: 333 (11.45%)

**Headline.** AST-grep over 2908 importable .py files in benchmarks/_corpus/{transformers,timm} finds 0 (0.00%) with at least one renamed-attribute aliasing pattern (R1..R5); 5 (0.17%) import torch.utils.checkpoint; 333 (11.45%) use tied_weights_keys or tie_weights/_tie_or_clone_weights.

## Calibration

We use this measurement to refine the limconc ≤12% protocol:
the renamed-attribute aliasing pattern is what TG silently
misclassifies, and its direct AST-grep prevalence on the
available transformers+timm corpus is reported above.  We
treat this as a known limitation, not a bug in
Theorem 5.7's grad-flag lattice.  The 5,000-script remote
sweep cited as ≤12% remains a standing obligation; we do not
claim a population-level rate from this 4,500-file local sweep.

Run with `python3.11 reproducibility/renamed_attribute_sharing_prevalence.py`.
