# AST-frontend parse-success SLA (real-world model corpus)

Static ingestion of 35 self-contained, real-world-style PyTorch model sources through TensorGuard's source/AST frontend (`extract_computation_graph`).

Parse success: **35 of 35** models lowered to a non-empty computation graph without crashing (rate 1.000). Across the corpus, 163 of 163 extracted steps are operators reasoned about precisely; 0 are soundly abstracted as unsupported (Step 34) and 0 statements were isolated (Step 43).

| Model | Parsed | Steps | Isolated | Unsupported |
|-------|--------|-------|----------|-------------|
| `annotated_forward_jaxtyping` | yes | 2 | 0 | 0 |
| `autoencoder` | yes | 5 | 0 | 0 |
| `avgpool_flatten_classifier` | yes | 5 | 0 | 0 |
| `bilinear_fusion` | yes | 5 | 0 | 0 |
| `conv_stack_helper_method` | yes | 4 | 0 | 0 |
| `convtranspose_decoder` | yes | 3 | 0 | 0 |
| `deep_sequential_resnet` | yes | 9 | 0 | 0 |
| `docstring_spec_model` | yes | 2 | 0 | 0 |
| `dropout_regularized` | yes | 4 | 0 | 0 |
| `dynamic_control_flow` | yes | 3 | 0 | 0 |
| `embedding_pool_classifier` | yes | 5 | 0 | 0 |
| `functional_relu_chain` | yes | 6 | 0 | 0 |
| `groupnorm_block` | yes | 4 | 0 | 0 |
| `gru_seq2vec` | yes | 4 | 0 | 0 |
| `inheritance_super_forward` | yes | 3 | 0 | 0 |
| `instancenorm_block` | yes | 4 | 0 | 0 |
| `layernorm_mlp_head` | yes | 3 | 0 | 0 |
| `lstm_tagger` | yes | 4 | 0 | 0 |
| `mlp_classifier` | yes | 4 | 0 | 0 |
| `mlp_mixer_block` | yes | 5 | 0 | 0 |
| `modulelist_loop` | yes | 3 | 0 | 0 |
| `multibranch_concat` | yes | 5 | 0 | 0 |
| `nested_helper_methods` | yes | 4 | 0 | 0 |
| `pointwise_depthwise` | yes | 3 | 0 | 0 |
| `reshape_view_model` | yes | 4 | 0 | 0 |
| `residual_mlp` | yes | 6 | 0 | 0 |
| `resnet_basic_block` | yes | 9 | 0 | 0 |
| `self_attention_qkv` | yes | 7 | 0 | 0 |
| `sequential_features` | yes | 3 | 0 | 0 |
| `siamese_two_inputs` | yes | 5 | 0 | 0 |
| `simple_cnn` | yes | 7 | 0 | 0 |
| `transformer_encoder_block` | yes | 8 | 0 | 0 |
| `two_layer_tanh` | yes | 4 | 0 | 0 |
| `unet_encoder` | yes | 6 | 0 | 0 |
| `vit_patch_embed` | yes | 5 | 0 | 0 |
