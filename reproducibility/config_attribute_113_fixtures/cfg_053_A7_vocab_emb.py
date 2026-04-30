import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, vocab_size=100000, hidden_size=4096, out_vocab=100000):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, out_vocab)
    def forward(self, x):
        return self.head(self.emb(x))
