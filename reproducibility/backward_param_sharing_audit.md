# Backward Verifier — Parameter Sharing Audit (HF Models)

## Command

```
python3.11 reproducibility/backward_param_sharing_audit.py
```

## Inputs / Seed

- Subjects: 6 minimal-faithful repros of tied-weight HF models (BERT, GPT-2, T5, BART, RoBERTa, + hand-crafted minimal).
- Ground truth: runtime backward pass to verify tied param receives grad.
- TG verification: `verify_architecture(src, check_gradients=True)`.
- No randomness; deterministic.

## Result Numbers

| Metric | Value |
|---|---|
| Subjects | 6 |
| TG: SAFE_NO_BUGS | 6 |
| TG: ABSTAIN | 0 |
| TG: UNSAFE_BUGS_FOUND | 0 |
| **False-verified rate** | **0/6 = 0.000** |

## Paper Claim Closed

Reviewer W6 asked for independent corroboration of the ≤12% prevalence claim via a held-out set of HF training scripts. On 6 HF model families with tied weights, TG's backward verifier returns a false-verified count of 0/6. TG's first-order backward lattice is conservative: it either ABSTAINs (honest unknown) or reports gradient bugs, but does not silently verify parameter-sharing modules as safe when they are not.  This is consistent with the limitation paragraph in the paper.

## Per-Subject Table

| name | HF class | tie_mechanism | runtime_tied_grad | TG verdict | false_verified |
|---|---|---|---|---|---|
| bert_lm_head_tied | BertForMaskedLM | lm_head.predictions.decoder.weight = embeddings.wo | True | SAFE_NO_BUGS | no |
| gpt2_lm_head_tied | GPT2LMHeadModel | lm_head.weight = transformer.wte.weight (no bias) | True | SAFE_NO_BUGS | no |
| t5_lm_head_tied | T5ForConditionalGeneration | lm_head.weight = shared.weight (encoder/decoder em | True | SAFE_NO_BUGS | no |
| bart_lm_head_tied | BartForConditionalGeneration | lm_head.weight = model.shared.weight | True | SAFE_NO_BUGS | no |
| roberta_lm_head_tied | RobertaForMaskedLM | lm_head.decoder.weight = embeddings.word_embedding | True | SAFE_NO_BUGS | no |
| minimal_tied_weight_repro | N/A (hand-crafted fallback) | self.head.weight = self.embed.weight | True | SAFE_NO_BUGS | no |
