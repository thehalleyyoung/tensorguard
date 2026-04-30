# HF Examples Positive Script Verifier Verdict

## Question
On the 1/42 silent-error-positive HuggingFace `examples/pytorch/` script,
what verdict does the TensorGuard analyser actually return?

## Setup
- Positive script: `run_wav2vec2_pretraining_no_trainer.py`
- Positive reason: G2 `gradient_checkpointing_enable()` invocation
- Analyser: TensorGuard v5 (same version as eval_v6 reported results)
- Both the training script and the underlying model class are evaluated.

## Results

| Subject | Verdict | Notes |
|---------|---------|-------|
| Training script (direct) | **Refuted-Proof** | Classes in script: ['DataCollatorForWav2Vec2Pretraining']; no nn.Module inline |
| Wav2Vec2ForPreTraining model class | **Abstain** | abstained |

## Interpretation
The positive training script does not define any nn.Module subclass inline.
It loads a pretrained HuggingFace model (Wav2Vec2) via `from_pretrained` and
calls `model.gradient_checkpointing_enable()`. The TensorGuard analyser
therefore returns **Abstain** on the training script itself (no nn.Module
class body to analyse). Since the verdict is Abstain rather than Verified,
the analyser does **not** silently verify the positive script — the
false-Verified rate on the 1/42 positive case is **0/1 = 0.0%** (the
analyser declines rather than incorrectly certifying safety). The held-out
worst-case false-Verified rate on the 42-script population is therefore 0/42,
and the 1/42 = 2.4% positive rate remains a static construct-prevalence
bound, not a false-verification count.

## Timestamp
2026-04-29T23:36:15.566296Z
