# 60-bug RP with parser-failure marker excluded (round 5 Q3)

## Configurations

  * (A) Full pipeline (operator dispatch + AST-pattern + parser marker)
  * (B) AST-pattern path disabled
  * (C) AST-pattern path disabled AND parser-failure marker excluded
        (rule-driven only)

## Result

| Configuration | RP / 60 | parser-marker-only | rule-driven |
|---|---|---|---|
| (A) full pipeline | 53 | 53 | 0 |
| (B) AST-pattern path disabled | 53 | 53 | 0 |
| (C) AST-pattern disabled + parser-marker excluded (rule-driven only) | 0 | -- | 0 |

## Plain reading

The rule-driven symbolic analyser catches **0 / 60 = 0.0%** of the 60-bug corpus once both the AST-pattern path and the parser-failure marker are removed.  The high headline number on the curated corpus is therefore powered substantially by the AST-pattern path (which contributes 0 extra catches) and by the parser-failure marker (which contributes 53 catches under (B)).  The calculus is the correctness substrate that justifies which catches are sound, but the recognition of a buggy fragment routinely goes through one of the other two paths on this corpus.

## Per-bug detail

| id | cat | full | full_parser_only | ast_dis | ast_dis_parser_only | first_real_bug_under_(B) |
|---|---|---|---|---|---|---|
| bug_001 | attention_dim | Verified | False | Verified | False |  |
| bug_002 | broadcasting | Verified | False | Verified | False |  |
| bug_003 | view_reshape_total_size | Verified | False | Verified | False |  |
| bug_004 | view_reshape_total_size | Verified | False | Verified | False |  |
| bug_005 | broadcasting | Verified | False | Verified | False |  |
| bug_006 | other | Verified | False | Verified | False |  |
| bug_007 | other | Verified | False | Verified | False |  |
| bug_008 | conv_channel_mismatch | Refuted | True | Refuted | True |  |
| bug_009 | linear_inout_mismatch | Refuted | True | Refuted | True |  |
| bug_010 | view_reshape_total_size | Refuted | True | Refuted | True |  |
| bug_011 | broadcasting | Refuted | True | Refuted | True |  |
| bug_012 | attention_dim | Refuted | True | Refuted | True |  |
| bug_013 | einsum_dim | Refuted | True | Refuted | True |  |
| bug_014 | transpose_axes | Refuted | True | Refuted | True |  |
| bug_015 | batchnorm_features | Refuted | True | Refuted | True |  |
| bug_016 | embedding_index | Refuted | True | Refuted | True |  |
| bug_017 | conv_channel_mismatch | Refuted | True | Refuted | True |  |
| bug_018 | linear_inout_mismatch | Refuted | True | Refuted | True |  |
| bug_019 | view_reshape_total_size | Refuted | True | Refuted | True |  |
| bug_020 | broadcasting | Refuted | True | Refuted | True |  |
| bug_022 | einsum_dim | Refuted | True | Refuted | True |  |
| bug_024 | batchnorm_features | Refuted | True | Refuted | True |  |
| bug_026 | conv_channel_mismatch | Refuted | True | Refuted | True |  |
| bug_027 | linear_inout_mismatch | Refuted | True | Refuted | True |  |
| bug_028 | view_reshape_total_size | Refuted | True | Refuted | True |  |
| bug_029 | broadcasting | Refuted | True | Refuted | True |  |
| bug_031 | einsum_dim | Refuted | True | Refuted | True |  |
| bug_032 | transpose_axes | Refuted | True | Refuted | True |  |
| bug_033 | batchnorm_features | Refuted | True | Refuted | True |  |
| bug_034 | embedding_index | Refuted | True | Refuted | True |  |
| bug_035 | conv_channel_mismatch | Refuted | True | Refuted | True |  |
| bug_037 | view_reshape_total_size | Refuted | True | Refuted | True |  |
| bug_038 | broadcasting | Refuted | True | Refuted | True |  |
| bug_039 | attention_dim | Refuted | True | Refuted | True |  |
| bug_040 | einsum_dim | Refuted | True | Refuted | True |  |
| bug_041 | transpose_axes | Refuted | True | Refuted | True |  |
| bug_042 | other | Refuted | True | Refuted | True |  |
| bug_043 | other | Refuted | True | Refuted | True |  |
| bug_044 | other | Refuted | True | Refuted | True |  |
| bug_045 | conv_channel_mismatch | Refuted | True | Refuted | True |  |
| bug_047 | view_reshape_total_size | Refuted | True | Refuted | True |  |
| bug_048 | broadcasting | Refuted | True | Refuted | True |  |
| bug_049 | attention_dim | Refuted | True | Refuted | True |  |
| bug_050 | einsum_dim | Refuted | True | Refuted | True |  |
| bug_051 | other | Refuted | True | Refuted | True |  |
| bug_052 | other | Refuted | True | Refuted | True |  |
| bug_053 | transpose_axes | Refuted | True | Refuted | True |  |
| bug_054 | batchnorm_features | Refuted | True | Refuted | True |  |
| bug_055 | embedding_index | Refuted | True | Refuted | True |  |
| bug_056 | conv_channel_mismatch | Refuted | True | Refuted | True |  |
| bug_057 | linear_inout_mismatch | Refuted | True | Refuted | True |  |
| bug_058 | view_reshape_total_size | Refuted | True | Refuted | True |  |
| bug_059 | broadcasting | Refuted | True | Refuted | True |  |
| bug_060 | attention_dim | Refuted | True | Refuted | True |  |
| bug_063 | view_reshape_total_size | Refuted | True | Refuted | True |  |
| bug_064 | broadcasting | Refuted | True | Refuted | True |  |
| bug_065 | other | Refuted | True | Refuted | True |  |
| bug_067 | other | Refuted | True | Refuted | True |  |
| bug_068 | other | Refuted | True | Refuted | True |  |
| bug_069 | other | Refuted | True | Refuted | True |  |
