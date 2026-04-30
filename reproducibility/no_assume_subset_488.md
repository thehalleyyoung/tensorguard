# No-synthesised-assume subset of the 488-block corpus

Round-4 reviewer borderline criterion.  We restrict the 488-
block real-source corpus to those blocks where the TensorGuard
verdict does not lean on a synthesised caller-rely assume_M:

- subset size: **356 / 488** blocks
- verdict triple: **V=27, RP=26, CV=0, LW=78, A=225**
- Wilson 95% CI on RP-rate: [5.0%, 10.5%]

RP here means an unconditional refutation: a Contract-
Violation whose synthesised assume_M classifier returns
'empty' (no caller-rely obligation; the refutation holds
for every realisable caller).

## By library

| library | V | RP | LW | A |
|---|---|---|---|---|
| torchvision | 13 | 0 | 26 | 31 |
| timm | 12 | 1 | 4 | 105 |
| transformers | 2 | 25 | 48 | 89 |

## Method

No analyser re-run.  Subset filter applied to the cached 488-
block per-input verdicts using the existing cv_caller_rely
bucket classification ('empty' = no caller-rely obligation).

Run with `python3 reproducibility/no_assume_subset_488.py`.

Cited from the eval section of the paper.
