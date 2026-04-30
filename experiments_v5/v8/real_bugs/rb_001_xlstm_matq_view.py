"""
Real Bug Repro: xLSTM matQ view shape mismatch (variable 1 of 2)

GitHub Issue: https://github.com/huggingface/transformers/issues/43208
Fixed in PR:  https://github.com/huggingface/transformers/pull/43209
Repository:   huggingface/transformers

Bug: In mlstm_chunkwise_parallel_fw_H, `matQ.view(batch_size, nh, nc, chunk_size, dqk)`
uses dqk=384 instead of dqk//nc=192, causing a reshape incompatibility for models with
chunk_size>1 (i.e., models smaller than 7B).

Original error from issue:
  RuntimeError: shape '[12, 4, 8, 64, 768]' is invalid for input of size 2359296
  12*4*8*64*768 = 18874368 ≠ 2359296

Substitution note: Concrete batch=2, nh=4, nc=2, chunk_size=64, dqk=192 derived from
issue reporter's error trace. dqk was incorrectly set to 384 (=192*2, i.e. dqk instead
of dqk//nc), doubling the last dimension.
"""
import torch
import torch.nn as nn

# Input: matQ of shape (batch_size, nh, nc*chunk_size, dqk) = (2, 4, 128, 192)
INPUT_SHAPES = {"x": (2, 4, 128, 192)}


class BuggyModule(nn.Module):
    def forward(self, x):
        # Bug: uses dqk=384 instead of dqk//nc = 192//2 = 96... simplified for repro.
        # With batch=2, nh=4, nc=2, chunk_size=64, the dqk should be 192,
        # but the buggy code passes dqk=384 (not divided by nc=2).
        # 2*4*128*192 = 196608 ≠ 2*4*2*64*384 = 393216
        return x.view(2, 4, 2, 64, 384)
