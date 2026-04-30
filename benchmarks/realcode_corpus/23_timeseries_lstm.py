# input_shape: (16, 50, 5)
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self, in_features: int = 5, hidden: int = 64, horizon: int = 1):
        super().__init__()
        self.lstm1 = nn.LSTM(in_features, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)
        return self.fc(h[:, -1, :])
