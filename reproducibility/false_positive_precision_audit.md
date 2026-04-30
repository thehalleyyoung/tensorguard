# False-positive precision audit across all four corpora

## Command

```bash
python3 reproducibility/false_positive_precision_audit.py
```

## Per-corpus fire and FP count

| Corpus | Fires | TP | FP | FP ids |
|---|---|---|---|---|
| 60-bug historical corpus | 53 | 53 | 0 | — |
| 10-real-bug upstream faithfulness corpus | 8 | 8 | 0 | — |
| N=15 post-freeze unfiltered corpus | 6 | 5 | 1 | rb_uf_010 |
| **Union** | **67** | **66** | **1** | rb_uf_010 |

## Union precision (Wilson 95% CI)

- Point estimate: 66/67 = 0.9851
- Wilson 95% CI: [0.9202, 0.9974]

## Interpretation

The single FP across all four corpora is **rb_uf_010** (dtype-root-cause
bug caught by TG's device-mismatch heuristic rather than a shape violation).
Every other RP verdict in the union of 67 fires is a confirmed
true positive.  The union precision is 66/67 with
Wilson 95% CI [0.920, 0.997].

## Paper claim (Q6)

Round-2 Q6 asks whether rb_uf_010 is the only FP across all corpora.
This artefact confirms: yes, 1 FP in the union of 67
RP fires.  The corresponding Wilson 95% precision interval is
[0.920, 0.997].
