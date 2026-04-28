"""
L4-gradient stress case 04: detach on embedding before downstream layers.

Target feature: gradient-flow check (L4, B1 type).
Bug: Token embeddings are detached; no gradient flows to self.embed.weight.
The embedding table will never be updated during training.

Expected:
  WITHOUT L4: Verified
  WITH    L4: Refuted (GRADIENT-BROKEN on embedding output)
"""
import torch
import torch.nn as nn


class FrozenEmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int = 1000, d_model: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BUG: detach prevents gradient from reaching self.embed.weight
        emb = self.embed(x).detach()
        h = self.rnn(emb)
        return self.head(h)


FEATURE = "L4_gradient"
INPUT_SHAPES = {"x": ("batch", "seq")}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
