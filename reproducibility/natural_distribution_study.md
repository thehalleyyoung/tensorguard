# Natural-distribution coverage study

On a sample of **174** clean, idiomatic, public-style model instances across **14** families and **9** public-repo strata (attention, autoencoder, cnn, conditioning, embedding, gating, generative, mlp, multibranch, norm, recurrent, sequential, siamese, upsample), we measure how often the verifier returns a *decided* verdict rather than abstaining, and how often it false-alarms on this clean natural distribution.

| mode | decided | abstained | coverage [95% CI] | false alarms | FP upper bound (95%) | top abstention causes |
| --- | --- | --- | --- | --- | --- | --- |
| sound | 174 | 0 | 1.0 [0.9784, 1.0] | 0 | 0.0216 | none: 0 |
| balanced | 174 | 0 | 1.0 [0.9784, 1.0] | 0 | 0.0216 | none: 0 |
| heuristic | 174 | 0 | 1.0 [0.9784, 1.0] | 0 | 0.0216 | none: 0 |

- full coverage (zero abstention) in every mode: **True**
- zero false alarms in every mode: **True**
- source policy: compact redistributable motif reimplementations, not vendored third-party source files

Each model is clean by construction (it executes under eager PyTorch), so every UNSAFE verdict would be a false alarm; none occur.
