# input_shape: (8, 20, 16)
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self, in_size: int = 16, hidden: int = 32, num_classes: int = 4):
        super().__init__()
        self.rnn = nn.RNN(in_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, h = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)
