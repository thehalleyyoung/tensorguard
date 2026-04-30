# Reproducibility scripts

This directory contains scripts and JSON artifacts that support
independent verification of the claims in the TensorGuard paper.

## Quick start

```bash
# Reproduce the pen-and-paper handler classification certificate
python3 reproducibility/classify_pen_and_paper_handlers.py

# Run the test suite for the classification
pytest tests/test_pen_and_paper_classification.py -x
```

## Pen-and-paper handler audit (`pen-paper-audit`)

Run from the repository root:

```bash
python3 reproducibility/classify_pen_and_paper_handlers.py && \
pytest tests/test_pen_and_paper_classification.py -x && \
python3 -c "
import json
d = json.load(open('reproducibility/pen_and_paper_classification.json'))
assert len(d) == 13 and all(r['class'] in ('T-Identity', 'T-Broadcast') for r in d)
print('pen-paper-audit: PASS —', len(d), 'handlers certified')
"
```

This command:
1. Inspects each of the 13 pen-and-paper soundness handlers via Python AST
   pattern matching, classifying each as **T-Identity** (shape-preserving
   or deterministic single-input transform) or **T-Broadcast** (output shape
   determined by broadcasting multiple input shapes).
2. Emits `reproducibility/pen_and_paper_classification.json` with one record
   per handler: `{handler, class, evidence_lines, sha}`.
3. Asserts via the pytest suite that all 13 handlers are classified and that
   no handler has `class == "unknown"`.

### Classification summary

| Class | Handlers |
|-------|----------|
| T-Identity | relu, gelu, silu, tanh, sigmoid, softmax, detach, flatten, pad, reduce |
| T-Broadcast | elementwise_binary, where, einsum |

### Artefact

`reproducibility/pen_and_paper_classification.json` — the generated
certificate; each record contains:

- `handler` — handler name as it appears in
  `experiments_v5/handler_soundness_scope.json`
- `class` — `"T-Identity"` or `"T-Broadcast"`
- `evidence_lines` — 1-based source line numbers in the relevant module
  where the shape rule is implemented
- `sha` — first 16 hex digits of the SHA-256 digest of the source module,
  so the certificate is tied to a specific revision of the source
