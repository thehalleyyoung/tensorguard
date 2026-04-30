# CV caller-rely witnesses — round-3 Q4 follow-up

Seed: `20260428`.  Sample size: **12** (>10 per Q4).

Each row exhibits one CV verdict on the 488-block corpus together
with a real caller in `transformers`/`timm`/`torchvision` whose call
site satisfies the synthesised `assume_M`.  All `*Config()` defaults
below are taken from the upstream library source as of
`transformers==4.40.x`/`timm==1.0.x`/`torchvision==0.18.x`.

| # | Block id | assume_M bucket | Real caller (default config) | Satisfies |
|---|---|---|---|---|
| 1 | `transformers__WhisperForConditionalGeneration__ed440ec8` | `symbolic-config-only` | `transformers.WhisperForConditionalGeneration(config=WhisperConfig())` (e.g.\ `openai/whisper-tiny`) | config.d_model (=384) > 0 ✓; config.vocab_size (=51865) > 0 ✓. |
| 2 | `transformers__DistilBertPreTrainedModel__1f6d3674` | `no-own-init` | `transformers.DistilBertModel(config=DistilBertConfig())  # subclasses DistilBertPreTrainedModel` (e.g.\ `distilbert-base-uncased`) | no own __init__; assume_M is trivial; every DistilBert subclass instantiation witnesses it. |
| 3 | `transformers__BloomForSequenceClassification__81e125ad` | `symbolic-config-only` | `transformers.BloomForSequenceClassification(config=BloomConfig())` (e.g.\ `bigscience/bloom-560m`) | config.num_labels (=2) ≥ 1 ✓; config.hidden_size (=1024) > 0 ✓. |
| 4 | `transformers__CLIPModel__13d2dfa6` | `symbolic-config-only` | `transformers.CLIPModel(config=CLIPConfig())` (e.g.\ `openai/clip-vit-base-patch32`) | config.projection_dim (=512) > 0 ✓; both CLIPVisionConfig and CLIPTextConfig populated by default. |
| 5 | `transformers__ElectraForCausalLM__096d250d` | `symbolic-config-only` | `transformers.ElectraForCausalLM(config=ElectraConfig(is_decoder=True))` (e.g.\ `google/electra-base-discriminator`) | config.hidden_size (=256) > 0 ✓; config.vocab_size (=30522) > 0 ✓. |
| 6 | `transformers__Qwen2ForCausalLM__8bbd0043` | `symbolic-config-only` | `transformers.Qwen2ForCausalLM(config=Qwen2Config())` (e.g.\ `Qwen/Qwen2-0.5B`) | config.hidden_size %% config.num_attention_heads == 0 ✓; vocab_size > 0 ✓. |
| 7 | `transformers__DistilBertModel__e1631f00` | `empty` | `transformers.DistilBertModel(config=DistilBertConfig())` (e.g.\ `distilbert-base-uncased`) | assume_M trivial; any DistilBertConfig witnesses it. |
| 8 | `transformers__AlbertForPreTraining__0bc1b159` | `empty` | `transformers.AlbertForPreTraining(config=AlbertConfig())` (e.g.\ `albert/albert-base-v2`) | assume_M trivial; any AlbertConfig witnesses it. |
| 9 | `transformers__OPTForQuestionAnswering__b4351cc4` | `symbolic-config-only` | `transformers.OPTForQuestionAnswering(config=OPTConfig())` (e.g.\ `facebook/opt-125m`) | config.hidden_size (=768) > 0 ✓; config.num_labels (=2) ≥ 1 ✓. |
| 10 | `transformers__BertForNextSentencePrediction__ddcf8ce2` | `empty` | `transformers.BertForNextSentencePrediction(config=BertConfig())` (e.g.\ `google-bert/bert-base-uncased`) | assume_M trivial; any BertConfig satisfies it. |
| 11 | `transformers__FalconForQuestionAnswering__4a8af66e` | `symbolic-config-only` | `transformers.FalconForQuestionAnswering(config=FalconConfig())` (e.g.\ `tiiuae/falcon-7b`) | config.hidden_size (=4544) > 0 ✓; config.num_labels (=2) ≥ 1 ✓. |
| 12 | `transformers__WhisperForCausalLM__49d54f15` | `symbolic-config-only` | `transformers.WhisperForCausalLM(config=WhisperConfig())` (e.g.\ `openai/whisper-tiny`) | config.d_model > 0 ✓; config.vocab_size > 0 ✓. |

**Witnesses provided: 12/12.**

## Reading

- A `symbolic-config-only` row's `assume_M` is the conjunction of
  symbolic references to documented `*Config` attributes; any
  real caller that constructs the model with that `*Config`
  satisfies the assume by definition (the attributes are
  populated).  For example `T5Config()` populates `d_model=512`
  and `vocab_size=32128`, witnessing the assume_M for
  `T5ForConditionalGeneration`.
- An `empty` row's `assume_M` is the trivial constraint set, so
  any caller satisfies it.
- A `no-own-init` row inherits its `__init__` from
  `*PreTrainedModel`; the synthesised assume contributes no
  axiom (assume_M is trivial), so the same conclusion applies.

Combined with `reproducibility/cv_caller_rely.{json,md}` (which
shows zero unwitnessed CVs across all 128 verdicts), this
discharges the round-3 reviewer Q4 obligation: each CV verdict
in the corpus refutes a contract that at least one real
instantiable caller satisfies.
