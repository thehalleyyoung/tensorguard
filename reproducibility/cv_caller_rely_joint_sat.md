# CV assume_M Joint Satisfiability Audit

## Command

```
python3.11 reproducibility/cv_caller_rely_joint_sat.py
```

## Inputs / Seed

- Source: `reproducibility/cv_caller_rely.json` (90 symbolic-config-only rows)
- Subsample: 30 rows, seed=0
- Config instantiation: default `*Config()` constructor (documented defaults)

## Result Numbers

| Metric | Value |
|---|---|
| Sampled rows | 30 |
| Excluded (no config class found) | 0 |
| Attempted | 30 |
| **Jointly satisfied** | **24/30** |
| Jointly-sat fraction | 0.800 |

## Paper Claim Closed

Reviewer W3/Q2 objected that the existing witness audit only shows each assume_M clause is satisfied by *some* config, not that the *conjunction* holds for a *single* config.  This audit explicitly checks the conjunction on the default *Config() constructor for each sampled block.  Result: 24/30 blocks have their full assume_M conjunction satisfied by the documented default constructor config.

## Per-Row Results

| id | config_class | n_clauses | jointly_sat | note |
|---|---|---|---|---|
| transformers__FalconForSequenceClassification__53d | FalconConfig | 2 | ✓ | all 2 clauses satisfied by FalconConfig() |
| transformers__GPTNeoXForCausalLM__303baee1 | GPTNeoXConfig | 2 | ✓ | all 2 clauses satisfied by GPTNeoXConfig() |
| transformers__GPT2Block__5a2da440 | GPT2Config | 1 | ✓ | all 1 clauses satisfied by GPT2Config() |
| transformers__DistilBertForMaskedLM__074cc05b | DistilBertConfig | 2 | ✓ | all 2 clauses satisfied by DistilBertConfig() |
| transformers__AlbertForMultipleChoice__3d721e99 | AlbertConfig | 2 | ✓ | all 2 clauses satisfied by AlbertConfig() |
| transformers__RobertaForSequenceClassification__38 | RobertaConfig | 1 | ✓ | all 1 clauses satisfied by RobertaConfig() |
| transformers__FalconModel__15321937 | FalconConfig | 4 | ✓ | all 4 clauses satisfied by FalconConfig() |
| transformers__CLIPForImageClassification__53204b83 | CLIPConfig | 1 | ✓ | all 1 clauses satisfied by CLIPConfig() |
| transformers__RobertaForQuestionAnswering__e34c038 | RobertaConfig | 2 | ✓ | all 2 clauses satisfied by RobertaConfig() |
| transformers__GemmaModel__a5eac09f | GemmaConfig | 3 | ✓ | all 3 clauses satisfied by GemmaConfig() |
| transformers__OPTForSequenceClassification__5365eb | OPTConfig | 2 | ✓ | all 2 clauses satisfied by OPTConfig() |
| transformers__WhisperForConditionalGeneration__ed4 | WhisperConfig | 3 | ✓ | all 3 clauses satisfied by WhisperConfig() |
| transformers__RobertaModel__e903ada9 | RobertaConfig | 2 | ✗ | one or more clauses FAILED for RobertaConfig() |
| transformers__T5Stack__3c5b9f6a | T5Config | 2 | ✓ | all 2 clauses satisfied by T5Config() |
| transformers__DistilBertForSequenceClassification_ | DistilBertConfig | 3 | ✓ | all 3 clauses satisfied by DistilBertConfig() |
| transformers__BloomForQuestionAnswering__7994d60e | BloomConfig | 1 | ✓ | all 1 clauses satisfied by BloomConfig() |
| transformers__T5EncoderModel__1c87733d | T5Config | 2 | ✓ | all 2 clauses satisfied by T5Config() |
| transformers__BartForQuestionAnswering__11e5505f | BartConfig | 2 | ✓ | all 2 clauses satisfied by BartConfig() |
| transformers__AlbertForTokenClassification__a514ef | AlbertConfig | 2 | ✓ | all 2 clauses satisfied by AlbertConfig() |
| transformers__LlamaDecoderLayer__fd4ea02c | LlamaConfig | 1 | ✓ | all 1 clauses satisfied by LlamaConfig() |
| transformers__CLIPModel__13d2dfa6 | CLIPConfig | 3 | ✓ | all 3 clauses satisfied by CLIPConfig() |
| transformers__OPTForQuestionAnswering__b4351cc4 | OPTConfig | 1 | ✓ | all 1 clauses satisfied by OPTConfig() |
| transformers__GPT2ForTokenClassification__abb65812 | GPT2Config | 3 | ✗ | one or more clauses FAILED for GPT2Config() |
| transformers__MistralModel__a8738ef2 | MistralConfig | 3 | ✗ | one or more clauses FAILED for MistralConfig() |
| transformers__RobertaForMultipleChoice__0324395b | RobertaConfig | 2 | ✓ | all 2 clauses satisfied by RobertaConfig() |
| transformers__OPTDecoderLayer__24a4c9cc | OPTConfig | 4 | ✓ | all 4 clauses satisfied by OPTConfig() |
| transformers__DebertaForMaskedLM__bdf0c81c | DebertaConfig | 1 | ✓ | all 1 clauses satisfied by DebertaConfig() |
| transformers__BloomForTokenClassification__057bf6a | BloomConfig | 3 | ✗ | one or more clauses FAILED for BloomConfig() |
| transformers__LlamaModel__3fb478fa | LlamaConfig | 3 | ✗ | one or more clauses FAILED for LlamaConfig() |
| transformers__AlbertModel__f4b637f2 | AlbertConfig | 3 | ✗ | one or more clauses FAILED for AlbertConfig() |
