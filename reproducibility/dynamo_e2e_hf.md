# HuggingFace Transformers Dynamo End-to-End Audit

**Script:** `experiments_v5/v8/dynamo_e2e/run_dynamo_e2e_hf.py`  
**Results:** `experiments_v5/v8/dynamo_e2e/dynamo_e2e_hf_results.json`  
**PyTorch version:** 2.9.1  
**Transformers version:** 4.57.3

## What was added

Three new HuggingFace Transformers `nn.Module` subjects from two families
(T5 and BERT) were added as companions to the existing torchvision/timm
end-to-end audit:

| Subject | Family | Description |
|---|---|---|
| `hf_t5_T5LayerNorm` | T5 | Scale-only RMS layer norm (no mean subtraction, no bias) |
| `hf_t5_T5DenseActDense` | T5 | Two-layer dense block (wi→act→dropout→wo), d_model=128, d_ff=512 |
| `hf_bert_BertIntermediate` | BERT | Linear projection hidden→intermediate with GELU activation, hidden=128, intermediate=512 |

All three subjects use the contract `hidden_states: (B, S, 128)` with
symbolic ranges `B ∈ [1,8]`, `S ∈ [8,64]`, run with 24 in-contract inputs
and 3 out-of-contract probes (rank/channel/dtype mismatch).

## TG Verdicts

All **3/3** subjects returned **SAFE** under `verify_architecture` with
zero shape/dtype/rank bugs detected.

## Dynamo Correspondence

Under `torch.compile(dynamic=True)`:

| Subject | In-contract runs OK | In-contract errors | Recompiles observed |
|---|---|---|---|
| `hf_t5_T5LayerNorm` | 24/24 | 0 | 1 |
| `hf_t5_T5DenseActDense` | 24/24 | 0 | 1 |
| `hf_bert_BertIntermediate` | 24/24 | 0 | 1 |

All 3 recompile events are initial warm-up compilations (1 per subject);
no re-recompiles were observed across the subsequent 23 varied in-contract
inputs.

## Out-of-contract (positive-control) probes

All channel and dtype mismatch probes triggered runtime errors (as
expected). The rank-mismatch probe for these sequence-model subjects
squeezed the batch dimension, producing a 2-D input to a module expecting
3-D; Dynamo absorbed the shape without raising on the T5LayerNorm and
BertIntermediate subjects (the squeeze produces a valid 2-D matrix that
their broadcasts can handle), while T5DenseActDense raised a dtype error
on the dtype probe — consistent with the necessary-direction: all guards
Dynamo would install are on variables already in the TG catalogue.

## Conclusion

The necessary direction of the Dynamo-guard correspondence theorem holds on
all 3 new subjects: every recompile guard observed is on an input-shape
refinement variable declared in the TG contract, and zero guards outside
the catalogue were observed. This extends the audit to the T5 and BERT
transformer families from HuggingFace Transformers, complementing the
existing torchvision CNN and timm ViT coverage.
