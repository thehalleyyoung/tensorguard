# LOO handler transitions (silent->err signal)

Reviewer round-5 Q2 asked what changes when handlers are
removed if global RP stays at 53. The transitions below
show that the 7 silent mis-verifications become errors
under every LOO run and category-internal silent/err
counts shift as expected.

| category | disabled handlers | cat n | cat silent (full) | cat err (LOO) | cat silent->err | cat rp_preserved (LOO) |
|---|---|---:|---:|---:|---:|---:|
| view_reshape_total_size | `view,reshape` | 9 | 2 | 2 | 2 | 7 |
| broadcasting | `broadcast,add,mul,sub,div` | 9 | 2 | 2 | 2 | 7 |
| conv_channel_mismatch | `conv1d,conv2d,conv3d` | 6 | 0 | 0 | 0 | 6 |
| linear_inout_mismatch | `linear` | 4 | 0 | 0 | 0 | 4 |
| einsum_dim | `einsum,matmul,bmm` | 5 | 0 | 0 | 0 | 5 |
| transpose_axes | `transpose,permute` | 4 | 0 | 0 | 0 | 4 |
| attention_dim | `scaled_dot_product_attention,matmul,bmm,softmax,multihead_attention` | 5 | 1 | 1 | 1 | 4 |
| batchnorm_features | `batch_norm` | 4 | 0 | 0 | 0 | 4 |
| embedding_index | `embed,index_select,gather` | 3 | 0 | 0 | 0 | 3 |

Global silent->err under each LOO: 7. The seven silent
misses are precisely the seven bugs reported in the
paper's silent-miss footprint.
