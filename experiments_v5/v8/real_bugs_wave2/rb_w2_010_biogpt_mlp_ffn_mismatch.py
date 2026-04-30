"""
Upstream-faithful real-bug repro: BioGPT MLP gate projection size mismatch.

GitHub Issue: https://github.com/huggingface/transformers/issues/20841
Buggy file  : transformers/models/biogpt/modeling_biogpt.py
              (BioGptMLP fc1 out_features vs fc2 in_features mismatch)

BioGPT MLP uses fc1 (embed_dim -> ffn_dim) then fc2 (ffn_dim -> embed_dim).
Bug: fc2 expects ffn_dim as input but fc1 produces a different out_features
when the config uses embed_dim*4 but fc2 uses embed_dim*2 as its input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BuggyModule(nn.Module):
    def __init__(self, embed_dim=1024, ffn_embed_dim=4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        # BUG: fc2 input should be ffn_embed_dim (4096) but uses embed_dim*2 (2048)
        wrong_input = embed_dim * 2  # 2048 instead of ffn_embed_dim=4096
        self.fc2 = nn.Linear(wrong_input, embed_dim)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)  # -> (bsz, seq, 4096)
        hidden_states = F.gelu(hidden_states, approximate='tanh')
        # BUG: fc2 expects 2048 input but hidden_states has 4096 features
        hidden_states = self.fc2(hidden_states)
        return hidden_states


INPUT_SHAPES = {"hidden_states": (2, 8, 1024)}
