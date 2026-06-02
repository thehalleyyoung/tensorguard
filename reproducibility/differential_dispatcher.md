# Differential testing vs the live torch dispatcher

We generate **2000** random PyTorch modules (seed `20240601`) across **5** families (cat, conv, convflat, mlp, reshape), with dimensions chosen so that adjacent-layer boundaries are compatible or not by chance. Each module is judged twice: by a real eager-PyTorch forward pass (ground truth: clean vs raises) and by TensorGuard's sound-mode verdict.

- ground truth: **1235** clean, **765** raise
- verdicts: **1235** SAFE, **765** UNSAFE, **0** UNKNOWN (abstain)

Agreement matrix (verdict rows x ground-truth columns):

| verdict \\ torch | clean | raises |
| --- | --- | --- |
| SAFE | 1235 | 0 |
| UNSAFE | 0 | 765 |
| UNKNOWN | 0 | 0 |

- **soundness violations** (SAFE but torch raises): **0** (rate 0.0, 95% CI upper 0.005)
- **false alarms** (UNSAFE but torch clean): **0** (rate 0.0, 95% CI upper 0.0031)
- on the **2000** decided (non-abstained) modules, TensorGuard agrees with torch on **2000** -- perfect agreement: **True**

Every decided verdict matches the live dispatcher: no random module is ever proved SAFE while torch rejects it (zero soundness violations), and no clean module is ever rejected (zero false alarms). Abstention is the only permitted form of disagreement.
