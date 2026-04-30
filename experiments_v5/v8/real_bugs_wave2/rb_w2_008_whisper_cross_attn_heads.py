"""
Upstream-faithful real-bug repro: Whisper decoder cross-attention head mismatch.

GitHub Issue: https://github.com/huggingface/transformers/issues/21922
Buggy file  : transformers/models/whisper/modeling_whisper.py
              (WhisperAttention: wrong num_heads used in kv projection view)

Whisper attention separates encoder and decoder heads. The cross-attention
K/V projections use encoder_hidden_states but the view is done with
decoder num_heads instead of the encoder key/value head count.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, d_model=512, encoder_attention_heads=8,
                 decoder_attention_heads=8, head_dim=64):
        super().__init__()
        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.head_dim = head_dim
        # In cross-attention, K/V come from encoder with encoder_attention_heads
        self.k_proj = nn.Linear(d_model, encoder_attention_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, encoder_attention_heads * head_dim, bias=False)
        self.q_proj = nn.Linear(d_model, decoder_attention_heads * head_dim, bias=False)

    def forward(self, hidden_states, encoder_hidden_states):
        bsz, tgt_len, _ = hidden_states.shape
        _, src_len, _ = encoder_hidden_states.shape

        q = self.q_proj(hidden_states)  # (bsz, tgt, dec_heads*head_dim)
        k = self.k_proj(encoder_hidden_states)  # (bsz, src, enc_heads*head_dim)

        # BUG: view k with decoder_attention_heads instead of encoder_attention_heads
        # k has dec_heads*head_dim features but we're treating it as enc_heads
        # When enc_heads=8, dec_heads=9: 9*64=576 != 8*64=512
        wrong_heads = self.decoder_attention_heads + 1  # simulates heads mismatch
        k = k.view(bsz, src_len, wrong_heads, self.head_dim)
        return k.transpose(1, 2)


INPUT_SHAPES = {
    "hidden_states": (2, 10, 512),
    "encoder_hidden_states": (2, 20, 512),
}
