# Post-freeze second-wave power calculation

## Command

```
python3.11 reproducibility/postfreeze_second_wave_power.py
```

## Question

On the post-freeze unfiltered surface (N=15, TG 5/15, FT 2/15, Pytea 3/15), what is the smallest second-wave N (additional items from the frozen 2026-04-08 GitHub-search query) at which the union (N=15 + N_new) yields Fisher exact p<0.05 on at least one pairwise comparison (TG vs FT or TG vs Pytea), assuming the second wave matches the observed point estimates?

## Result

| Threshold | Smallest N_new | Total N |
|---|---|---|
| Fisher p<0.05 (either pair) | 26 | 41 |
| Fisher p<0.05 (TG vs FT)    | 26 | 41 |
| Fisher p<0.05 (TG vs Pytea) | 77 | 92 |
| Fisher p<0.01 (either pair) | 53 | 68 |
| BF10 >= 10 (either pair)    | 56 | 71 |

## Paper claim closed

Round-4 reviewer W4/Q5 asks whether a pre-registered N>=30 second wave is feasible before camera-ready, and on the observed point estimates what is the smallest second-wave N at which the union yields Fisher p<0.05.  This artefact answers the second question; whether the wave is feasible before camera-ready is recorded in the internal review response log.
