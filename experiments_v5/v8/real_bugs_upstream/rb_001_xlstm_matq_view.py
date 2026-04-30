"""
Upstream-faithful real-bug repro: xLSTM mlstm_chunkwise_parallel_fw_H matQ view
GitHub Issue: https://github.com/huggingface/transformers/issues/43208
Fixed in PR : https://github.com/huggingface/transformers/pull/43209
Repository  : huggingface/transformers
Buggy file  : transformers/models/xlstm/modeling_xlstm.py (pre-#43209)

Mirrors the buggy `MLSTMChunkwiseParallelFwH` class shipped in the
parent commit of the fix PR: scalar config attributes (`num_heads`,
`num_chunks`, `chunk_size`, `dqk`) are bound from a config object in
`__init__` and the buggy reshape is the same `matQ.view(batch_size,
nh, nc, chunk_size, dqk)` line the upstream maintainer patched
(should be `dqk // nc`).
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, num_heads=4, num_chunks=2, chunk_size=64, dqk=192):
        super().__init__()
        self.num_heads = num_heads
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.dqk = dqk

    def forward(self, matQ):
        # matQ shape: (batch, num_heads, num_chunks * chunk_size, dqk)
        batch_size = matQ.shape[0]
        # BUG (pre-#43209): trailing dim is self.dqk; fix divides by num_chunks.
        return matQ.view(
            batch_size,
            self.num_heads,
            self.num_chunks,
            self.chunk_size,
            self.dqk,
        )


INPUT_SHAPES = {"matQ": (2, 4, 2 * 64, 192)}
