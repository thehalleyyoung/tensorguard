# Pytea modern-subset 34-bug contingency table

Source: `reproducibility/pytea_mcnemar_per_bug.json`
Generator: inline command in `reproducibility/pytea_mcnemar_per_bug.md`

**Tally** (n=34):
- both refute: 25
- TG only: 7
- Pytea only: 0
- neither: 2

McNemar pair structure:
- a = both = 25
- b = TG-only (TG refute, Pytea not) = 7
- c = Pytea-only (Pytea refute, TG not) = 0
- d = neither = 2

| # | bug id | primary op | TG (enforced) | Pytea | agreement |
|---|--------|------------|---------------|-------|-----------|
| 1 | bug_003 | Tensor.view | Refuted | Refuted | both_refute |
| 2 | bug_004 | Tensor.view | Refuted | Refuted | both_refute |
| 3 | bug_005 | broadcast (param + tensor) | Refuted | Refuted | both_refute |
| 4 | bug_006 | nn.CrossEntropyLoss | Verified | Verified | neither |
| 5 | bug_007 | F.conv2d (dtype check) | Verified | Verified | neither |
| 6 | bug_008 | nn.Conv2d | Refuted | Refuted | both_refute |
| 7 | bug_009 | nn.Linear | Refuted | Refuted | both_refute |
| 8 | bug_010 | Tensor.view | Refuted | Refuted | both_refute |
| 9 | bug_011 | broadcast (a + b) | Refuted | Refuted | both_refute |
| 10 | bug_014 | Tensor.transpose | Refuted | Refuted | both_refute |
| 11 | bug_015 | nn.BatchNorm2d | Refuted | Refuted | both_refute |
| 12 | bug_016 | nn.Embedding | Refuted | Verified | tg_only |
| 13 | bug_018 | matmul (@ operator) | Refuted | N/A | tg_only |
| 14 | bug_019 | Tensor.reshape | Refuted | Refuted | both_refute |
| 15 | bug_020 | torch.cat | Refuted | Refuted | both_refute |
| 16 | bug_027 | torch.bmm | Refuted | Refuted | both_refute |
| 17 | bug_028 | Tensor.view (wildcard -1) | Refuted | Refuted | both_refute |
| 18 | bug_034 | nn.Embedding | Refuted | Verified | tg_only |
| 19 | bug_035 | nn.ConvTranspose2d | Refuted | Refuted | both_refute |
| 20 | bug_037 | flatten + view | Refuted | Refuted | both_refute |
| 21 | bug_038 | torch.stack | Refuted | Refuted | both_refute |
| 22 | bug_039 | F.softmax | Refuted | Verified | tg_only |
| 23 | bug_040 | matmul (@ operator, batched) | Refuted | N/A | tg_only |
| 24 | bug_042 | nn.CrossEntropyLoss | Refuted | Refuted | both_refute |
| 25 | bug_043 | nn.MSELoss | Refuted | Refuted | both_refute |
| 26 | bug_044 | nn.NLLLoss | Refuted | Verified | tg_only |
| 27 | bug_045 | nn.Conv2d | Refuted | Refuted | both_refute |
| 28 | bug_047 | Tensor.unsqueeze | Refuted | Refuted | both_refute |
| 29 | bug_049 | F.layer_norm | Refuted | Refuted | both_refute |
| 30 | bug_053 | Tensor.expand | Refuted | Refuted | both_refute |
| 31 | bug_056 | nn.Conv2d | Refuted | Refuted | both_refute |
| 32 | bug_057 | torch.mm | Refuted | Refuted | both_refute |
| 33 | bug_063 | Tensor.view (post-transpose) | Refuted | Verified | tg_only |
| 34 | bug_068 | nn.MaxPool2d | Refuted | Refuted | both_refute |
