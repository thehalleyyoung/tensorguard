"""
Real Bug Repro: xLSTM matK view shape mismatch (variable 2 of 2)

GitHub Issue: https://github.com/huggingface/transformers/issues/43208
Fixed in PR:  https://github.com/huggingface/transformers/pull/43209
Repository:   huggingface/transformers

Bug: In mlstm_chunkwise_parallel_fw_H, `matK.view(batch_size, nh, nc, chunk_size, dqk)`
has the same dimension error as matQ -- uses dqk instead of dqk//nc, causing a reshape
incompatibility for any model config where nc > 1 (models smaller than 7B).

Original error from issue:
  RuntimeError: shape '[12, 4, 8, 64, 768]' is invalid for input of size 2359296

Substitution note: Identical configuration to rb_001; matK is the key tensor in the
same chunkwise parallel forward function.
"""
import torch
import torch.nn as nn

# Input: matK of shape (batch_size, nh, nc*chunk_size, dqk) = (2, 4, 128, 192)
INPUT_SHAPES = {"x": (2, 4, 128, 192)}


class BuggyModule(nn.Module):
    def forward(self, x):
        # Bug: uses dqk=384 instead of dqk//nc=192 for the key tensor.
        # 2*4*128*192 = 196608 ≠ 2*4*2*64*384 = 393216
        return x.view(2, 4, 2, 64, 384)
