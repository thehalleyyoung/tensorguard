# TensorGuard Bug Corpus Manifest (N=60)

**Provenance:** 60 historical PyTorch shape bugs mined from the pytorch/pytorch issue tracker; each has a self-contained ≤40-line CPU repro that raises the cited RuntimeError. TG verdicts use the v7 three-way refuted taxonomy (REFUTED_PROOF when a bug-corpus item is refuted, meaning TensorGuard successfully caught the bug; VERIFIED means TensorGuard reported safe — a silent miss; ABSTAIN means analysis was inconclusive; NA means no result recorded). Of 60 bugs: 60 are in-fragment (shape errors expressible in TensorGuard's symbolic fragment) and 0 are out-of-fragment (require torch.compile, custom autograd, external kernels, or similar). TG caught 56 bugs (REFUTED_PROOF), silently missed 4 (VERIFIED), and abstained on 0.


| ID | Category | In-fragment | TG verdict | Source |
|----|----------|-------------|------------|--------|
| bug_001 | attention_dim | yes | VERIFIED | [177482](https://github.com/pytorch/pytorch/issues/177482) |
| bug_002 | broadcasting | yes | VERIFIED | [174985](https://github.com/pytorch/pytorch/issues/174985) |
| bug_003 | view_reshape_total_size | yes | REFUTED_PROOF | [176375](https://github.com/pytorch/pytorch/issues/176375) |
| bug_004 | view_reshape_total_size | yes | REFUTED_PROOF | [174379](https://github.com/pytorch/pytorch/issues/174379) |
| bug_005 | broadcasting | yes | REFUTED_PROOF | [179573](https://github.com/pytorch/pytorch/issues/179573) |
| bug_006 | other | yes | VERIFIED | [174181](https://github.com/pytorch/pytorch/issues/174181) |
| bug_007 | other | yes | VERIFIED | [180548](https://github.com/pytorch/pytorch/issues/180548) |
| bug_008 | conv_channel_mismatch | yes | REFUTED_PROOF | [179931](https://github.com/pytorch/pytorch/issues/179931) |
| bug_009 | linear_inout_mismatch | yes | REFUTED_PROOF | [179789](https://github.com/pytorch/pytorch/issues/179789) |
| bug_010 | view_reshape_total_size | yes | REFUTED_PROOF | [177691](https://github.com/pytorch/pytorch/issues/177691) |
| bug_011 | broadcasting | yes | REFUTED_PROOF | [179568](https://github.com/pytorch/pytorch/issues/179568) |
| bug_012 | attention_dim | yes | REFUTED_PROOF | [178882](https://github.com/pytorch/pytorch/issues/178882) |
| bug_013 | einsum_dim | yes | REFUTED_PROOF | [177611](https://github.com/pytorch/pytorch/issues/177611) |
| bug_014 | transpose_axes | yes | REFUTED_PROOF | [177254](https://github.com/pytorch/pytorch/issues/177254) |
| bug_015 | batchnorm_features | yes | REFUTED_PROOF | [177017](https://github.com/pytorch/pytorch/issues/177017) |
| bug_016 | embedding_index | yes | REFUTED_PROOF | [176446](https://github.com/pytorch/pytorch/issues/176446) |
| bug_017 | conv_channel_mismatch | yes | REFUTED_PROOF | [176231](https://github.com/pytorch/pytorch/issues/176231) |
| bug_018 | linear_inout_mismatch | yes | REFUTED_PROOF | [176230](https://github.com/pytorch/pytorch/issues/176230) |
| bug_019 | view_reshape_total_size | yes | REFUTED_PROOF | [175831](https://github.com/pytorch/pytorch/issues/175831) |
| bug_020 | broadcasting | yes | REFUTED_PROOF | [175683](https://github.com/pytorch/pytorch/issues/175683) |
| bug_022 | einsum_dim | yes | REFUTED_PROOF | [175165](https://github.com/pytorch/pytorch/issues/175165) |
| bug_024 | batchnorm_features | yes | REFUTED_PROOF | [174339](https://github.com/pytorch/pytorch/issues/174339) |
| bug_026 | conv_channel_mismatch | yes | REFUTED_PROOF | [173902](https://github.com/pytorch/pytorch/issues/173902) |
| bug_027 | linear_inout_mismatch | yes | REFUTED_PROOF | [173765](https://github.com/pytorch/pytorch/issues/173765) |
| bug_028 | view_reshape_total_size | yes | REFUTED_PROOF | [173724](https://github.com/pytorch/pytorch/issues/173724) |
| bug_029 | broadcasting | yes | REFUTED_PROOF | [173709](https://github.com/pytorch/pytorch/issues/173709) |
| bug_031 | einsum_dim | yes | REFUTED_PROOF | [173316](https://github.com/pytorch/pytorch/issues/173316) |
| bug_032 | transpose_axes | yes | REFUTED_PROOF | [173171](https://github.com/pytorch/pytorch/issues/173171) |
| bug_033 | batchnorm_features | yes | REFUTED_PROOF | [173157](https://github.com/pytorch/pytorch/issues/173157) |
| bug_034 | embedding_index | yes | REFUTED_PROOF | [172880](https://github.com/pytorch/pytorch/issues/172880) |
| bug_035 | conv_channel_mismatch | yes | REFUTED_PROOF | [172822](https://github.com/pytorch/pytorch/issues/172822) |
| bug_037 | view_reshape_total_size | yes | REFUTED_PROOF | [172739](https://github.com/pytorch/pytorch/issues/172739) |
| bug_038 | broadcasting | yes | REFUTED_PROOF | [172712](https://github.com/pytorch/pytorch/issues/172712) |
| bug_039 | attention_dim | yes | REFUTED_PROOF | [172684](https://github.com/pytorch/pytorch/issues/172684) |
| bug_040 | einsum_dim | yes | REFUTED_PROOF | [172579](https://github.com/pytorch/pytorch/issues/172579) |
| bug_041 | transpose_axes | yes | REFUTED_PROOF | [172529](https://github.com/pytorch/pytorch/issues/172529) |
| bug_042 | other | yes | REFUTED_PROOF | [172419](https://github.com/pytorch/pytorch/issues/172419) |
| bug_043 | other | yes | REFUTED_PROOF | [172386](https://github.com/pytorch/pytorch/issues/172386) |
| bug_044 | other | yes | REFUTED_PROOF | [172374](https://github.com/pytorch/pytorch/issues/172374) |
| bug_045 | conv_channel_mismatch | yes | REFUTED_PROOF | [172364](https://github.com/pytorch/pytorch/issues/172364) |
| bug_047 | view_reshape_total_size | yes | REFUTED_PROOF | [172073](https://github.com/pytorch/pytorch/issues/172073) |
| bug_048 | broadcasting | yes | REFUTED_PROOF | [172064](https://github.com/pytorch/pytorch/issues/172064) |
| bug_049 | attention_dim | yes | REFUTED_PROOF | [172019](https://github.com/pytorch/pytorch/issues/172019) |
| bug_050 | einsum_dim | yes | REFUTED_PROOF | [172014](https://github.com/pytorch/pytorch/issues/172014) |
| bug_051 | other | yes | REFUTED_PROOF | [171994](https://github.com/pytorch/pytorch/issues/171994) |
| bug_052 | other | yes | REFUTED_PROOF | [171948](https://github.com/pytorch/pytorch/issues/171948) |
| bug_053 | transpose_axes | yes | REFUTED_PROOF | [171931](https://github.com/pytorch/pytorch/issues/171931) |
| bug_054 | batchnorm_features | yes | REFUTED_PROOF | [171858](https://github.com/pytorch/pytorch/issues/171858) |
| bug_055 | embedding_index | yes | REFUTED_PROOF | [171853](https://github.com/pytorch/pytorch/issues/171853) |
| bug_056 | conv_channel_mismatch | yes | REFUTED_PROOF | [171852](https://github.com/pytorch/pytorch/issues/171852) |
| bug_057 | linear_inout_mismatch | yes | REFUTED_PROOF | [171850](https://github.com/pytorch/pytorch/issues/171850) |
| bug_058 | view_reshape_total_size | yes | REFUTED_PROOF | [171764](https://github.com/pytorch/pytorch/issues/171764) |
| bug_059 | broadcasting | yes | REFUTED_PROOF | [171704](https://github.com/pytorch/pytorch/issues/171704) |
| bug_060 | attention_dim | yes | REFUTED_PROOF | [171669](https://github.com/pytorch/pytorch/issues/171669) |
| bug_063 | view_reshape_total_size | yes | REFUTED_PROOF | [171622](https://github.com/pytorch/pytorch/issues/171622) |
| bug_064 | broadcasting | yes | REFUTED_PROOF | [171523](https://github.com/pytorch/pytorch/issues/171523) |
| bug_065 | other | yes | REFUTED_PROOF | [171517](https://github.com/pytorch/pytorch/issues/171517) |
| bug_067 | other | yes | REFUTED_PROOF | [170980](https://github.com/pytorch/pytorch/issues/170980) |
| bug_068 | other | yes | REFUTED_PROOF | [170934](https://github.com/pytorch/pytorch/issues/170934) |
| bug_069 | other | yes | REFUTED_PROOF | [170666](https://github.com/pytorch/pytorch/issues/170666) |

## Category tally

| Category | Count |
|----------|-------|
| attention_dim | 5 |
| batchnorm_features | 4 |
| broadcasting | 9 |
| conv_channel_mismatch | 6 |
| einsum_dim | 5 |
| embedding_index | 3 |
| linear_inout_mismatch | 4 |
| other | 11 |
| transpose_axes | 4 |
| view_reshape_total_size | 9 |

## TG verdict tally

| Verdict | Count |
|---------|-------|
| REFUTED_PROOF | 56 |
| VERIFIED | 4 |
