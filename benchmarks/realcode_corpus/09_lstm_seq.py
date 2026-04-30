# input_shape: (4, 30, 8)
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self, in_size: int = 8, hidden: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(out[:, -1, :])
