import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, vocab_size=8192, hidden_size=512, out_vocab=16384):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, out_vocab)
    def forward(self, x):
        return self.head(self.emb(x))
