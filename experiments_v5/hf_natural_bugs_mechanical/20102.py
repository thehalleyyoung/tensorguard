import torch
import torch.nn as nn

INPUT_SHAPES = {"hidden_states": (2, 8, 768)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(769, 3072)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        x = hidden_states.reshape(batch_size * seq_len, 769)
        return self.dense(x).view(batch_size, seq_len, 3072)
