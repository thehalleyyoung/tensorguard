import torch
import torch.nn as nn

INPUT_SHAPES = {"hidden_states": (2, 8, 3072), "input_tensor": (2, 8, 768)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(3073, 768)
        self.LayerNorm = nn.LayerNorm(768)

    def forward(self, hidden_states, input_tensor):
        batch, seq, _ = hidden_states.shape
        h = hidden_states.reshape(batch * seq, 3073)
        h = self.dense(h).view(batch, seq, 768)
        return self.LayerNorm(h + input_tensor)
