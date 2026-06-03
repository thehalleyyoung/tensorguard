# Deployment latency and memory budgets

Release gates run TensorGuard before and after export/compile. The committed artifact stores deterministic latency/memory budgets; `make deployment-budgets-gate` measures real wall-clock latency and verifier-stage memory live.

## Gate matrix

| Model | Pipeline | Phase | Backend | Latency budget (s) | Memory budget (MB) |
|-------|----------|-------|---------|--------------------|--------------------|
| `tiny_mlp_classifier` | compile | after | `torch.dynamo` | 35.0 | 320.0 |
| `tiny_mlp_classifier` | compile | before | `fx` | 8.0 | 80.0 |
| `tiny_mlp_classifier` | export | after | `torch.export` | 18.0 | 192.0 |
| `tiny_mlp_classifier` | export | before | `fx` | 6.0 | 96.0 |
| `tiny_vision_classifier` | compile | after | `torch.dynamo` | 35.0 | 320.0 |
| `tiny_vision_classifier` | compile | before | `fx` | 8.0 | 80.0 |
| `tiny_vision_classifier` | export | after | `torch.export` | 18.0 | 192.0 |
| `tiny_vision_classifier` | export | before | `fx` | 6.0 | 96.0 |

## Per-backend Pareto budget curves

### `fx`

| Profile | Latency budget (s) | Memory budget (MB) | Use |
|---------|--------------------|--------------------|-----|
| `interactive` | 6.0 | 96.0 | pre-export local feedback |
| `memory_capped` | 8.0 | 80.0 | pre-compile constrained CI runner |

### `torch.dynamo`

| Profile | Latency budget (s) | Memory budget (MB) | Use |
|---------|--------------------|--------------------|-----|
| `fast_compile` | 24.0 | 420.0 | warm compile-capable host |
| `compile_cold_start` | 35.0 | 320.0 | default cold compile release gate |

### `torch.export`

| Profile | Latency budget (s) | Memory budget (MB) | Use |
|---------|--------------------|--------------------|-----|
| `fast_export` | 12.0 | 240.0 | release smoke on beefy runners |
| `balanced` | 18.0 | 192.0 | default post-export release gate |
