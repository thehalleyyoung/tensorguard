# torch.compile / FakeTensorMode baseline on 7 natural HF bugs

## Motivation

Reviewer round-18 question: how do execution-based baselines fare on the 7 naturally-occurring HuggingFace bugs that TensorGuard catches at 7/7? The 7 bugs are minimal `nn.Module` repros taken from public upstream fix-PRs/issues across Llama, Qwen2, Mistral, and Phi-3 (see `cross_family_natural_bugs.md` for citations and bug descriptions).

## Method

For each bug we instantiate the module with default constructor args and the input shapes used in the TG run, then attempt:
1. `torch.compile(mod, fullgraph=True, dynamic=False)` followed by an invocation on `torch.zeros(...)` inputs.
2. `FakeTensorMode()` invocation on `torch.zeros(...)` inputs.
A baseline is recorded as **catching** the bug if and only if the tracer/eager-fake invocation raises an exception traceable to the shape/dimension mismatch the upstream PR fixes.

## Per-bug results

| Bug | TG | torch.compile | FakeTensorMode |
|---|---|---|---|
| `Qwen2AttentionHeadMismatch` | RP | caught (RuntimeError) | caught (RuntimeError) |
| `MistralGQAProjectionMismatch` | RP | caught (RuntimeError) | caught (AssertionError) |
| `Phi3FusedQKVSlice` | RP | caught (RuntimeError) | caught (AssertionError) |
| `LlamaIntermediateSizeMismatch` | RP | caught (RuntimeError) | caught (AssertionError) |
| `MistralAttentionOutputHeads` | RP | caught (RuntimeError) | caught (AssertionError) |
| `Qwen2ReshapeTargetMismatch` | RP | caught (RuntimeError) | caught (AssertionError) |
| `LlamaAttentionHeadCount` | RP | caught (RuntimeError) | caught (AssertionError) |

## Summary

- TensorGuard:    7/7
- torch.compile:  7/7
- FakeTensorMode: 7/7

## Paper claims cited by this artifact

- Eval section: torch.compile / FakeTensorMode catch-rate on the 7-bug naturally-occurring HuggingFace cross-family corpus.
