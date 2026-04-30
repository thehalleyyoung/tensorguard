# input_shape: (4, 16)
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self, vocab: int = 1000, emb: int = 64, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        e = self.embed(x)
        out, h = self.gru(e)
        return self.head(out)
