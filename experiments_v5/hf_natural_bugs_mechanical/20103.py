import torch
import torch.nn as nn

INPUT_SHAPES = {"hidden_states": (2, 8, 3072), "input_tensor": (2, 8, 768)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = nn.LayerNorm(768)

    def forward(self, hidden_states, input_tensor):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)
