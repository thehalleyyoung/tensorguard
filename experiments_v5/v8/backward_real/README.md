# Backward-Verifier Real-World Evaluation

Round-1 reviewer W6: *"the 500/500 backward-verifier agreement is on
randomly-generated small `nn.Module`s drawn from a grammar … there is
no real-world evaluation of the backward verifier comparable to the
shape-bug corpus."*

This directory adds a 10-model real-world evaluation.

## Models

| Model                         | Library      |
|-------------------------------|--------------|
| `resnet18`                    | torchvision  |
| `mobilenet_v3_small`          | torchvision  |
| `vit_b_16`                    | torchvision  |
| `BertModel` (bert-base)       | transformers |
| `GPT2Model` (gpt2)            | transformers |
| `DistilBertModel`             | transformers |
| `T5Model` (t5-small)          | transformers |
| `WhisperModel` (whisper-tiny) | transformers |
| `CLIPVisionModel` (vit-b32)   | transformers |
| `Wav2Vec2Model` (wav2vec2)    | transformers |

## Protocol

For each model:

1. Construct with default arguments (or default config).
2. TG's backward verifier predicts the `requires_grad` topology of
   every named parameter, statically, from the class source.
3. Runtime topology is recovered by running a real
   `loss = model(x).sum(); loss.backward()` and inspecting
   `param.grad is not None` for every named parameter.
4. Predicted vs.\ runtime topology is compared per-parameter.

A *false positive* is a parameter the verifier predicts will
receive a gradient but which does not at runtime — the unsound
direction.

## Run

```
PYTHONPATH=. python3.11 experiments_v5/v8/backward_real/run_backward_real.py
```

Result is written to `backward_real_results.json`.

## Caveats (acknowledged in the paper)

* Default-config-only.  We do not exercise
  `torch.utils.checkpoint`, explicit parameter sharing, or
  `accelerate` zero-2/3 in this sweep.  These regimes are tracked
  as future work in `.comet_neurips/self_obligations.md`.
* The first-order grad-flag lattice
  $\{\texttt{has\_grad}, \texttt{no\_grad}, \top\}$ used by the
  static verifier is not expressive enough to model
  parameter-sharing-induced double counting; that is by design,
  and the contribution has been renamed accordingly throughout
  the paper.
