"""
Upstream-faithful real-bug repro: xLSTM mlstm_chunkwise_parallel_fw_H matK view
GitHub Issue: https://github.com/huggingface/transformers/issues/43208
Fixed in PR : https://github.com/huggingface/transformers/pull/43209
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

    def forward(self, matK):
        batch_size = matK.shape[0]
        # BUG (pre-#43209): trailing dim should be self.dqk // self.num_chunks.
        return matK.view(
            batch_size,
            self.num_heads,
            self.num_chunks,
            self.chunk_size,
            self.dqk,
        )


INPUT_SHAPES = {"matK": (2, 4, 2 * 64, 192)}
