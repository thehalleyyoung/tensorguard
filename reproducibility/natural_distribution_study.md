# Natural-distribution coverage study

On a sample of **29** clean, idiomatic, public-style architectures across **14** families (attention, autoencoder, cnn, conditioning, embedding, gating, generative, mlp, multibranch, norm, recurrent, sequential, siamese, upsample), we measure how often the verifier returns a *decided* verdict rather than abstaining, and how often it false-alarms on this clean natural distribution.

| mode | decided | abstained | coverage [95% CI] | false alarms |
| --- | --- | --- | --- | --- |
| sound | 29 | 0 | 1.0 [0.883, 1.0] | 0 |
| balanced | 29 | 0 | 1.0 [0.883, 1.0] | 0 |
| heuristic | 29 | 0 | 1.0 [0.883, 1.0] | 0 |

- full coverage (zero abstention) in every mode: **True**
- zero false alarms in every mode: **True**

Each model is clean by construction (it executes under eager PyTorch), so every UNSAFE verdict would be a false alarm; none occur.
