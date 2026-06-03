# Good first operator issues

This queue is generated from `operator_confidence_table.json` and
`proof_footprint_manifest.json`. It prioritizes low-confidence or
lightly-evidenced operators so contributor work improves the verifier's
actual trust surface rather than a hand-maintained wishlist.

| Operator | Difficulty | Confidence | Proof footprint | Labels |
| --- | --- | --- | --- | --- |
| `torch.einsum` | beginner | `heuristic` | `heuristic` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-einsum` |
| `torch.multinomial` | beginner | `heuristic` | `heuristic` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-multinomial` |
| `torch.unique` | beginner | `heuristic` | `heuristic` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-unique` |
| `torch.argsort` | beginner | `sound` | `tested_only_rule` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-argsort` |
| `torch.bernoulli` | beginner | `sound` | `tested_only_rule` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-bernoulli` |
| `torch.cdist` | beginner | `sound` | `tested_only_rule` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-cdist` |
| `torch.fft.fft` | beginner | `sound` | `tested_only_rule` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-fft-fft` |
| `torch.fft.fft2` | beginner | `sound` | `tested_only_rule` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-fft-fft2` |
| `torch.fft.ifft` | beginner | `sound` | `tested_only_rule` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-fft-ifft` |
| `torch.fft.ifft2` | beginner | `sound` | `tested_only_rule` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-fft-ifft2` |
| `torch.fft.irfft` | beginner | `sound` | `tested_only_rule` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-fft-irfft` |
| `torch.fft.rfft` | beginner | `sound` | `tested_only_rule` | `good first issue`, `operator-coverage`, `tensorguard`, `operator:torch-fft-rfft` |

## Copyable issue bodies

### Good first operator: upgrade `torch.einsum` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.einsum`.

- Current confidence: `heuristic`
- Current proof footprint: `heuristic`
- Rationale: Output shape depends on runtime values or is approximated generically; best-effort, neither sound nor complete in general.
- Starting evidence: `src/proof_footprint.py`, `tests/test_operator_confidence.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.

### Good first operator: upgrade `torch.multinomial` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.multinomial`.

- Current confidence: `heuristic`
- Current proof footprint: `heuristic`
- Rationale: Output shape depends on runtime values or is approximated generically; best-effort, neither sound nor complete in general.
- Starting evidence: `src/proof_footprint.py`, `tests/test_operator_confidence.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.

### Good first operator: upgrade `torch.unique` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.unique`.

- Current confidence: `heuristic`
- Current proof footprint: `heuristic`
- Rationale: Output shape depends on runtime values or is approximated generically; best-effort, neither sound nor complete in general.
- Starting evidence: `src/proof_footprint.py`, `tests/test_operator_confidence.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.

### Good first operator: upgrade `torch.argsort` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.argsort`.

- Current confidence: `sound`
- Current proof footprint: `tested_only_rule`
- Rationale: Structural op whose output shape is an exact function of the input shapes and static integer arguments; enforced soundly.
- Starting evidence: `tests/test_graph_compiler.py`, `tests/test_index_value_ops_precise.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.

### Good first operator: upgrade `torch.bernoulli` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.bernoulli`.

- Current confidence: `sound`
- Current proof footprint: `tested_only_rule`
- Rationale: Structural op whose output shape is an exact function of the input shapes and static integer arguments; enforced soundly.
- Starting evidence: `tests/test_graph_compiler.py`, `tests/test_operator_confidence.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.

### Good first operator: upgrade `torch.cdist` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.cdist`.

- Current confidence: `sound`
- Current proof footprint: `tested_only_rule`
- Rationale: Structural op whose output shape is an exact function of the input shapes and static integer arguments; enforced soundly.
- Starting evidence: `tests/test_graph_compiler.py`, `tests/test_operator_confidence.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.

### Good first operator: upgrade `torch.fft.fft` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.fft.fft`.

- Current confidence: `sound`
- Current proof footprint: `tested_only_rule`
- Rationale: FFT family: exact, well-defined output-shape rule (e.g. rfft maps the last dim n -> n//2 + 1) enforced soundly.
- Starting evidence: `tests/test_complex_verify.py`, `tests/test_operator_confidence.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.

### Good first operator: upgrade `torch.fft.fft2` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.fft.fft2`.

- Current confidence: `sound`
- Current proof footprint: `tested_only_rule`
- Rationale: FFT family: exact, well-defined output-shape rule (e.g. rfft maps the last dim n -> n//2 + 1) enforced soundly.
- Starting evidence: `tests/test_complex_verify.py`, `tests/test_operator_confidence.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.

### Good first operator: upgrade `torch.fft.ifft` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.fft.ifft`.

- Current confidence: `sound`
- Current proof footprint: `tested_only_rule`
- Rationale: FFT family: exact, well-defined output-shape rule (e.g. rfft maps the last dim n -> n//2 + 1) enforced soundly.
- Starting evidence: `tests/test_complex_verify.py`, `tests/test_operator_confidence.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.

### Good first operator: upgrade `torch.fft.ifft2` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.fft.ifft2`.

- Current confidence: `sound`
- Current proof footprint: `tested_only_rule`
- Rationale: FFT family: exact, well-defined output-shape rule (e.g. rfft maps the last dim n -> n//2 + 1) enforced soundly.
- Starting evidence: `tests/test_complex_verify.py`, `tests/test_operator_confidence.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.

### Good first operator: upgrade `torch.fft.irfft` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.fft.irfft`.

- Current confidence: `sound`
- Current proof footprint: `tested_only_rule`
- Rationale: FFT family: exact, well-defined output-shape rule (e.g. rfft maps the last dim n -> n//2 + 1) enforced soundly.
- Starting evidence: `tests/test_complex_verify.py`, `tests/test_operator_confidence.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.

### Good first operator: upgrade `torch.fft.rfft` transfer evidence

### Goal
Improve TensorGuard's transfer-function evidence for `torch.fft.rfft`.

- Current confidence: `sound`
- Current proof footprint: `tested_only_rule`
- Rationale: FFT family: exact, well-defined output-shape rule (e.g. rfft maps the last dim n -> n//2 + 1) enforced soundly.
- Starting evidence: `tests/test_complex_verify.py`, `tests/test_operator_confidence.py`

### Acceptance checklist
1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.
2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.
3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.
4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.
