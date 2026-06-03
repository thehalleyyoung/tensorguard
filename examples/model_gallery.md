# TensorGuard model gallery

Twenty-five pasteable PyTorch `nn.Module` cases. Each row includes a clean verification path, a paired caught bug, and the minimal command/config users can copy.

| # | Model | Family | Caught bug | Copy command |
|---:|---|---|---|---|
| 1 | `mlp_classifier_head` | tabular | Second Linear expects 30 features, but the previous layer produces 20. | `tensorguard verify mlp_classifier_head.py -s x=batch,10` |
| 2 | `residual_projection_head` | vision | Second Linear expects 18 features, but the previous layer produces 24. | `tensorguard verify residual_projection_head.py -s x=batch,12` |
| 3 | `transformer_ffn_block` | attention | Second Linear expects 40 features, but the previous layer produces 32. | `tensorguard verify transformer_ffn_block.py -s x=batch,16` |
| 4 | `bert_pooler_head` | nlp | Second Linear expects 96 features, but the previous layer produces 64. | `tensorguard verify bert_pooler_head.py -s x=batch,768` |
| 5 | `vit_patch_classifier` | vision | Second Linear expects 96 features, but the previous layer produces 128. | `tensorguard verify vit_patch_classifier.py -s x=batch,64` |
| 6 | `unet_time_mlp` | diffusion | Second Linear expects 48 features, but the previous layer produces 64. | `tensorguard verify unet_time_mlp.py -s x=batch,32` |
| 7 | `gan_discriminator_head` | generative | Second Linear expects 40 features, but the previous layer produces 50. | `tensorguard verify gan_discriminator_head.py -s x=batch,100` |
| 8 | `vae_latent_decoder` | generative | Second Linear expects 32 features, but the previous layer produces 40. | `tensorguard verify vae_latent_decoder.py -s x=batch,20` |
| 9 | `recommender_dense_tower` | recommender | Second Linear expects 21 features, but the previous layer produces 28. | `tensorguard verify recommender_dense_tower.py -s x=batch,14` |
| 10 | `speech_ctc_projection` | speech | Second Linear expects 128 features, but the previous layer produces 160. | `tensorguard verify speech_ctc_projection.py -s x=batch,80` |
| 11 | `rl_policy_head` | reinforcement | Second Linear expects 24 features, but the previous layer produces 34. | `tensorguard verify rl_policy_head.py -s x=batch,17` |
| 12 | `q_value_estimator` | reinforcement | Second Linear expects 33 features, but the previous layer produces 44. | `tensorguard verify q_value_estimator.py -s x=batch,22` |
| 13 | `metric_learning_embedder` | retrieval | Second Linear expects 72 features, but the previous layer produces 96. | `tensorguard verify metric_learning_embedder.py -s x=batch,48` |
| 14 | `siamese_projection_head` | retrieval | Second Linear expects 54 features, but the previous layer produces 72. | `tensorguard verify siamese_projection_head.py -s x=batch,36` |
| 15 | `contrastive_text_head` | multimodal | Second Linear expects 192 features, but the previous layer produces 256. | `tensorguard verify contrastive_text_head.py -s x=batch,128` |
| 16 | `image_caption_bridge` | multimodal | Second Linear expects 96 features, but the previous layer produces 128. | `tensorguard verify image_caption_bridge.py -s x=batch,256` |
| 17 | `tabnet_decision_head` | tabular | Second Linear expects 45 features, but the previous layer produces 60. | `tensorguard verify tabnet_decision_head.py -s x=batch,30` |
| 18 | `graph_node_classifier` | graph | Second Linear expects 63 features, but the previous layer produces 84. | `tensorguard verify graph_node_classifier.py -s x=batch,42` |
| 19 | `time_series_forecaster` | forecasting | Second Linear expects 36 features, but the previous layer produces 48. | `tensorguard verify time_series_forecaster.py -s x=batch,24` |
| 20 | `anomaly_detector_head` | monitoring | Second Linear expects 27 features, but the previous layer produces 36. | `tensorguard verify anomaly_detector_head.py -s x=batch,18` |
| 21 | `autofix_regression_head` | tabular | Second Linear expects 17 features, but the previous layer produces 22. | `tensorguard verify autofix_regression_head.py -s x=batch,11` |
| 22 | `adapter_bottleneck` | fine-tuning | Second Linear expects 12 features, but the previous layer produces 10. | `tensorguard verify adapter_bottleneck.py -s x=batch,40` |
| 23 | `lora_merge_probe` | fine-tuning | Second Linear expects 20 features, but the previous layer produces 16. | `tensorguard verify lora_merge_probe.py -s x=batch,64` |
| 24 | `optimizer_resume_probe` | training | Second Linear expects 13 features, but the previous layer produces 18. | `tensorguard verify optimizer_resume_probe.py -s x=batch,9` |
| 25 | `serving_schema_head` | serving | Second Linear expects 20 features, but the previous layer produces 30. | `tensorguard verify serving_schema_head.py -s x=batch,15` |

## Example snippet

Each manifest row contains `clean_source` and `buggy_source`. The clean source executes in eager PyTorch; the buggy source is expected to return `UNSAFE` under TensorGuard.

