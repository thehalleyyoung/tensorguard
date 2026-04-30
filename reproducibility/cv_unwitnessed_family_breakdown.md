# CV unwitnessed family decomposition (Round-2)

Source: `cv_caller_rely_joint_sat_full128.json` (128 rows, 10 unwitnessed).

All 10 unwitnessed CVs come from a single library (`transformers`) and a
single bucket (`symbolic-config-only`); zero come from `torchvision` or
`timm`. Every unwitnessed entry has the same note ("one or more
sym_attr clauses failed"). The unwitnessed family histogram is:

| Family    | Count | Module |
|-----------|-------|--------|
| gpt2      | 2     | `GPT2ForTokenClassification`, `GPT2Model` |
| bert      | 1     | `BertModel` |
| llama     | 1     | `LlamaModel` |
| mistral   | 1     | `MistralModel` |
| falcon    | 1     | `FalconForTokenClassification` |
| qwen2     | 1     | `Qwen2Model` |
| roberta   | 1     | `RobertaModel` |
| albert    | 1     | `AlbertModel` |
| bloom     | 1     | `BloomForTokenClassification` |

Interpretation: the 10 unwitnessed CVs are uniformly transformer-family
encoder/decoder backbones whose `sym_attr` clauses (symbolic
`hidden_size % num_attention_heads == 0`-style constraints) bind a
configuration variable that does not appear concretely in the
witnessed-config dictionary. They are not concentrated in any one
family; they are a pattern across HF transformer backbones.
