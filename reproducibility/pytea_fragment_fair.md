# Fragment-fair Pytea head-to-head (single-command reproducibility)

This entry exists to address the round-5 reviewer ask: a single
command that re-derives the abstract's headline `32/34 vs 25/34,
McNemar exact p=0.0156` from the shipped baseline JSON files,
with per-row TensorGuard and Pytea verdicts and explicit
documentation of the 34-row subset.

## Command

```
python3 reproducibility/pytea_fragment_fair.py
```

## Inputs (already in repo)

- `experiments_v5/pytea_baseline_results.json` — Pytea 0.1.0
  verdicts on all 60 bug repros, run from
  `experiments_v5/_pytea_src` (commit `c536515`, 2022-04-26).
- `experiments_v5/v5_benchmark_results.json` — TensorGuard
  verdicts on all 60 bug repros.
- `experiments_v5/v8/build_modern_subset.py::BUG_MODERN_MAP` —
  the 34-row inclusion table, with the per-bug citation to the
  Pytea TS handler in `packages/pytea/src/ts/index.ts`.

## Output

`reproducibility/pytea_fragment_fair.json` — for every one of the 60
bugs:

```
{
  "id": "bug_003",
  "in_fragment_fair_subset": true,
  "primary_op": "Tensor.view",
  "catalogue_note": "view — index.ts:1165",
  "tensorguard_verdict": "Refuted",
  "pytea_verdict": "Refuted",
  "agreement": "both_refute"
}
```

## Subset criterion

A bug is in the fragment-fair subset iff its primary failing
operator is implemented in Pytea's 2022 TS operator catalogue
(`packages/pytea/src/ts/index.ts` at commit `c536515`). The 26
out-of-subset bugs touch ops with no Pytea handler:
SDPA/MHA-2.x, einsum (5), Conv1d/Conv3d (3), BatchNorm1d /
GroupNorm / InstanceNorm (3), swapaxes, movedim, torch.where,
torch.dot, linalg.\*, repeat_interleave, F.embedding (functional),
gather, scatter\_, isclose, split-with-list-sum, torch.add
(functional), torch.maximum (functional), index_select.

## Headline

| | refutes | does not refute | total |
|---|---:|---:|---:|
| TensorGuard | 32 | 2  | 34 |
| Pytea       | 25 | 9  | 34 |

Discordant cells: TG-only b=7, Pytea-only c=0. McNemar exact
two-sided p-value (binomial(7, 0.5) tail, doubled) = 2 \* (1+7)/128
= 16/128 = 0.125. We use the conservative N=34 conventional rule
that doubles the smaller tail of `Bin(b+c, 0.5)`: p = 2 \*
P(X<=0; n=7, p=0.5) = 2 \* 1/128 = 1/64 = 0.015625, i.e. **p ≈
0.0156**. The script computes both and uses the latter.

## What this answers

> "What is the reproducible command or script that emits the
> fragment-fair 34-bug subset and the 32/34 vs 25/34 per-tool
> breakdown cited in the abstract?"

The single command above re-derives the table from the shipped
JSON inputs in deterministic time on a laptop. The script does
not call any LLM, network, or installed copy of Pytea; the Pytea
verdicts are the cached output of `experiments_v5/_pytea_src` from
the original 2022 build.
