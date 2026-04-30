import torch
import torch.nn as nn

INPUT_SHAPES = {"hidden_states": (2, 8, 768)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 12
        self.attention_head_size = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)

    def transpose_for_scores(self, x):
        batch_size, seq_len, _ = x.size()
        new_shape = (batch_size, seq_len, 11, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(q, k.transpose(-1, -2))
        return torch.matmul(scores, v)
