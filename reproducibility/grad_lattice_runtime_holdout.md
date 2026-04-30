# Grad-flag false-verified-rate vs runtime p.grad != None (round 5 rewrite)

## Command

```
python3.11 reproducibility/grad_lattice_runtime_holdout.py
```

## Held-out positive sample (round 5 rewrite)

This artefact replaces the round-4 version, which fed raw HuggingFace head-class source via ``inspect.getsource(model.__class__)`` and produced a vacuous 0/8 because TG could not parse the truncated source. This rewrite uses 10 *self-contained* ``nn.Module`` subclasses (8 positives + 2 clean negative controls) that exercise the same constructs in a form TG can actually parse.

## Result

| Metric | Value |
|---|---|
| Subjects (total) | 10 |
| Positives | 8 |
| Negative controls | 2 |
| Positives with `[GRADIENT-OUT-OF-FRAGMENT]` Refuted-Proof | 6/8 |
| Positive false-verified (TG SAFE+VERIFIED on a runtime positive) | 2/8 |
| Negative-control SAFE+VERIFIED (specificity) | 2/2 |
| Negative-control false-out-of-fragment | 0/2 |

## Per-subject

| name | kind | runtime_ok | tg_verdict | n_bugs | grad_oof | false_verified |
|---|---|---|---|---|---|---|
| ResidualMLP_checkpoint | checkpoint | True | UNSAFE | 1 | True | False |
| TwoLayerCNN_checkpoint | checkpoint | True | UNSAFE | 1 | True | False |
| GatedTransformerBlock_checkpoint | checkpoint | True | UNSAFE | 1 | True | False |
| SequentialMLP_checkpoint_sequential | checkpoint_sequential | True | UNSAFE | 1 | True | False |
| InlineCheckpointToggle_gc | gc_enable | True | UNSAFE | 1 | True | False |
| HfStyleEnableToggle_gc | gc_enable | True | UNSAFE | 1 | True | False |
| TiedEmbeddingLMHead_tied | tied_weights | True | SAFE | 0 | False | True |
| RenamedSharedLinear_tied | tied_weights | True | SAFE | 0 | False | True |
| CleanMLP_negative_control | clean | True | SAFE | 0 | False | False |
| CleanConvBNReLU_negative_control | clean | True | SAFE | 0 | False | False |

## Paper claim closed

Round-5 reviewer W3 / Q2 noted that the prior version of this artefact reported `0/8 false-verified` on subjects whose first TG bug was the parser-failure marker `No nn.Module subclass found in source`, making the rate vacuous.  This rewrite ships self-contained `nn.Module` subjects that TG actually parses. The grad-lattice out-of-fragment detector (`[GRADIENT-OUT-OF-FRAGMENT]`) fires on 6/8 positives and the measured false-verified-rate is 2/8.  Negative-control specificity is 2/2.
