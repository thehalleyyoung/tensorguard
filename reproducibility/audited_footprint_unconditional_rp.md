# Audited-footprint unconditional RP catches (488 corpus)

Of the 26 unconditional RP
catches in the no-synthesised-assume subset of the 488-block
corpus, **5** fire through a handler chain
that is entirely Lean-audited or pen-and-paper sound, with
no tested-only or uncovered handler in the detected set.

Footprint breakdown:

- `touches_tested_only`: 10
- `uncovered_only`: 6
- `lean_or_pp_only`: 5
- `no_handlers_detected`: 5

## Per-catch

| id | library | loc | footprint | handlers |
|---|---|---:|---|---|
| `timm__VisionTransformerDistilled__032a9d92` | timm | 97 | lean_or_pp_only | cat,expand,linear,view |
| `transformers__AlbertForMaskedLM__444a57ae` | transformers | 104 | touches_tested_only | argmax,cross_entropy,embed,linear,view,where |
| `transformers__AlbertForPreTraining__0bc1b159` | transformers | 106 | touches_tested_only | cross_entropy,embed,linear,unsqueeze,view |
| `transformers__BartDecoderWrapper__145bd043` | transformers | 12 | no_handlers_detected |  |
| `transformers__BartForSequenceClassification__7b1a4896` | transformers | 149 | touches_tested_only | cross_entropy,squeeze,sum,to,view |
| `transformers__BartModel__27886110` | transformers | 158 | uncovered_only | sqrt |
| `transformers__BertForMaskedLM__f07c0986` | transformers | 107 | touches_tested_only | cat,cross_entropy,view |
| `transformers__BertForNextSentencePrediction__ddcf8ce2` | transformers | 98 | touches_tested_only | cross_entropy,view |
| `transformers__BertForPreTraining__fb4fc487` | transformers | 103 | touches_tested_only | cross_entropy,view |
| `transformers__BertLMHeadModel__ec7a8f20` | transformers | 93 | no_handlers_detected |  |
| `transformers__BloomPreTrainedModel__4a772e64` | transformers | 28 | lean_or_pp_only | embed,layer_norm,linear |
| `transformers__DebertaForSequenceClassification__f3f064d8` | transformers | 110 | touches_tested_only | cross_entropy,dropout,expand,gather,linear,mean,squeeze,sum,to,view |
| `transformers__DebertaModel__d4dcd33c` | transformers | 105 | no_handlers_detected |  |
| `transformers__DistilBertFlashAttention2__9971185e` | transformers | 99 | uncovered_only | reshape,to,view |
| `transformers__DistilBertModel__e1631f00` | transformers | 140 | uncovered_only | embed,to |
| `transformers__ElectraForPreTraining__74bdacae` | transformers | 99 | lean_or_pp_only | squeeze,view |
| `transformers__FalconFlashAttention2__c64fa866` | transformers | 107 | uncovered_only | reshape,to,transpose |
| `transformers__FalconPreTrainedModel__55936532` | transformers | 40 | lean_or_pp_only | embed,layer_norm,linear |
| `transformers__GPT2PreTrainedModel__7452c372` | transformers | 44 | uncovered_only | embed,layer_norm,linear,sqrt |
| `transformers__OPTModel__bb5f30a8` | transformers | 60 | no_handlers_detected |  |
| `transformers__RobertaForCausalLM__bd93edfc` | transformers | 122 | uncovered_only | to |
| `transformers__RobertaForMaskedLM__6e029774` | transformers | 91 | touches_tested_only | cross_entropy,to,view |
| `transformers__T5ForSequenceClassification__9276522e` | transformers | 166 | touches_tested_only | cross_entropy,squeeze,sum,to,view |
| `transformers__ViTForMaskedImageModeling__062602cb` | transformers | 113 | touches_tested_only | contiguous,conv2d,permute,pixel_shuffle,reshape,sum,unsqueeze |
| `transformers__WhisperDecoderWrapper__f16eb33b` | transformers | 19 | no_handlers_detected |  |
| `transformers__WhisperModel__2cb724a8` | transformers | 190 | lean_or_pp_only | expand |

Reproduce with `python3 reproducibility/audited_footprint_unconditional_rp.py`.
