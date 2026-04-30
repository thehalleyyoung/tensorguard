# Grad-flag silent-error: HF Trainer/accelerate held-out audit

## Command

```
git clone --depth 1 https://github.com/huggingface/transformers.git \
    .tmp_hf_examples_repo
python3.11 reproducibility/grad_lattice_hf_trainer_holdout.py
```

## Population

- 42 PyTorch training scripts under `examples/pytorch/` of `huggingface/transformers` (master).
- Disjoint from the 16-module Track-E fixture and the 2,908-file model-definition source sweep.

## Exposure counts

| Construct | Hits / 42 |
|---|---|
| G1 `torch.utils.checkpoint(...)` | 0 |
| G2 `gradient_checkpointing_enable()` | 1 |
| G3 `accelerator.prepare(model, ...)` | 17 |
| G4 `requires_grad = False` (well-handled) | 1 |
| G5 `tie_weights / _tie_or_clone_weights` (well-handled) | 4 |
| G6 renamed-attribute parameter sharing | 0 |

## Silent-error positives (G1 v G2 v G6)

**1/42** (2.4%) training scripts trigger at least one grad-lattice silent-error construct (gradient checkpointing or renamed-attribute parameter sharing).  G3 (`accelerator.prepare`) is reported separately because DDP grad reduction does *not* break the first-order grad-flag lattice.

## Worst-case false-verified rate

<= 1/42 (2.4%) on this held-out population.  Combined with the 6/6 ABSTAIN result on the held-out positive `backward_param_sharing_audit` sample, this discriminates the headline <=12% prevalence claim from a pessimistic 25%.

## Paper claim closed

Round-3 reviewer W4/Q4 asked for a held-out audit on a different population than the 16 importable Track-E modules.  The HF Trainer/accelerate corpus is disjoint from both the Track-E fixture and the model-definition source sweep, and the static-construct exposure here, combined with the 6/6 ABSTAIN result on the held-out positive sample, bounds the false-verified rate above by 1/42.
