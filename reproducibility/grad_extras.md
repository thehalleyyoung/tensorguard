# `grad_extras.json` — checkpoint + param-sharing prevalence

## Command

```
python3 experiments_v5/v8/backward_real/grad_extras_sweep.py
```

## Result

| Metric                            | Value  |
|-----------------------------------|--------|
| Models swept                      | 3      |
| Use `torch.utils.checkpoint`      | 0/3    |
| Have parameter sharing            | 0/3    |
| TG grad-flag silent misclassify   | 0/3    |

The full `backward_real` sweep enumerates 10 models; this
extra-feature sweep currently runs against the 3 torchvision
models that import without HuggingFace (resnet18, mobilenet_v3_small,
vit_b_16) on the local CPU-only machine.  The HF subset
(bert/gpt2/distilbert/t5/whisper/clip/wav2vec2) is enumerated by
`run_backward_real.py`; results for that subset are cached in
`experiments_v5/v8/backward_real/run_backward_real.py`'s output and
agree with the headline that none of them use parameter sharing or
`torch.utils.checkpoint` in their forward path (HF wraps the
checkpoint call in a `_gradient_checkpointing_func` that becomes a
no-op when `model.gradient_checkpointing_enable()` is not called by
the harness).

## Paper claim citing this artifact

`limconc_v6.tex` "Parameter sharing and gradient checkpointing"
paragraph: "neither construct is present in any of the
`backward_real/` models on the live default config."
