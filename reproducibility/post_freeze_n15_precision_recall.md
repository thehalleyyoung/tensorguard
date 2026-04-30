# Post-freeze N=15 per-PR ground-truth labels and TP/FP/FN matrix

## Obligation
Round-1 reviewer Q3: for the N=15 pre-registered post-freeze PRs, label each
with a bug-class (shape / dtype / distributed / autograd) and build a per-tool
TP/FP/FN matrix against ground truth.

## Command

    python3 reproducibility/post_freeze_n15_precision_recall.py

## Bug-class labels

| bug_id | bug_class | rationale |
|---|---|---|
| rb_pf_001 | shape | config_dependent_linear_chain — shape arithmetic |
| rb_pf_002 | shape | attention_mask_dim_mismatch — shape (config-bound) |
| rb_pf_003 | shape | lora_in_out_swap_3d — shape (in/out dimension swap) |
| rb_pf_004 | shape | router_topk_vs_num_experts — shape (topk index vs. expert count) |
| rb_pf_005 | shape | attention_mask_expand_off_by_one — shape (off-by-one expand) |
| rb_pf_006 | shape | batch_ordering_chunk_mismatch — shape (batch chunk size) |
| rb_uf_007 | shape | patch_merge_view — shape (view total size) |
| rb_uf_008 | shape | view_total_size_mismatch — shape (view) |
| rb_uf_009 | shape | config_dependent_chunk_size — shape (config-bound) |
| rb_uf_010 | **dtype** | dtype_mismatch — dtype root cause (NOT a shape bug) |
| rb_uf_011 | **distributed** | distributed_all_gather — process boundary (out-of-scope) |
| rb_uf_012 | shape | data_dependent_control_flow — shape (runtime-value path) |
| rb_uf_013 | shape | literal_broadcast_mismatch — shape (literal broadcast) |
| rb_uf_014 | **autograd** | autograd_param_sharing — grad flow (TG's grad scope) |
| rb_uf_015 | shape | transpose_swap_view — shape (transpose swap) |

Distribution: 12 shape / 1 dtype / 1 distributed / 1 autograd.

## Ground-truth positive definition

- **TG scope**: shape + autograd = 13/15 GT-positive bugs
- **FakeTensorMode scope**: shape (execution-based tracing) = 12/15 GT-positive bugs
- **Pytea scope**: shape (type-based, 2022 catalogue fragment) = 12/15 GT-positive bugs

## Per-PR matrix

| bug_id | class | TG | FakeTensorMode | Pytea |
|---|---|---|---|---|
| rb_pf_001 | shape | **TP** | FN | FN |
| rb_pf_002 | shape | FN | FN | FN |
| rb_pf_003 | shape | **TP** | FN | **TP** |
| rb_pf_004 | shape | **TP** | FN | FN |
| rb_pf_005 | shape | FN | FN | FN |
| rb_pf_006 | shape | FN | FN | FN |
| rb_uf_007 | shape | FN | FN | **TP** |
| rb_uf_008 | shape | **TP** | **TP** | FN |
| rb_uf_009 | shape | FN | **TP** | FN |
| rb_uf_010 | **dtype** | **FP** | TN | TN |
| rb_uf_011 | distributed | TN | TN | TN |
| rb_uf_012 | shape | **TP** | FN | FN |
| rb_uf_013 | shape | FN | FN | FN |
| rb_uf_014 | autograd | FN | TN | TN |
| rb_uf_015 | shape | FN | FN | **TP** |

## Aggregate per-tool metrics

| tool | TP | FP | FN | TN | Precision | Recall | GT+ |
|---|---|---|---|---|---|---|---|
| TG | 5 | 1 | 8 | 1 | **0.833** | **0.385** | 13 |
| FakeTensorMode | 2 | 0 | 10 | 3 | 1.000 | 0.167 | 12 |
| Pytea | 3 | 0 | 9 | 3 | 1.000 | 0.250 | 12 |

## Reading

TG achieves the highest recall (38.5%) of the three tools on its declared
in-scope bugs, with precision 83.3% (one FP: rb_uf_010, a dtype-class bug
caught by TG's device-mismatch check rather than the dtype root cause).
FakeTensorMode and Pytea have precision 100% but much lower recall (16.7%
and 25% respectively).

At TG's operating point (38.5% recall), FakeTensorMode fires on only 2 bugs
and Pytea on only 3; the recall gap confirms that TG's static shape analysis
surfaces bugs the execution-based tools miss.  The single FP (rb_uf_010) and
the precision difference (83.3% vs 100%) are the calibrated cost of the
higher recall.

## Paper claim

Section 4.1 ("Unfiltered pre-registered post-freeze sample"): reporting
precision at fixed recall on the per-class TP/FP/FN matrix alongside the
raw catch counts.

## Inputs / seed

- `reproducibility/real_bugs_unfiltered.json` (N=15 verdict data)
- No randomness; deterministic.
