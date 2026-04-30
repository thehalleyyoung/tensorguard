"""
Upstream-faithful real-bug repro: Wav2Vec2 conv feature extractor output.

GitHub Issue: https://github.com/huggingface/transformers/issues/15003
Buggy file  : transformers/models/wav2vec2/modeling_wav2vec2.py
              (Wav2Vec2FeatureEncoder conv output channel count mismatch)

Wav2Vec2 uses a stack of 1D convolutions for feature extraction.
Bug: the projection from conv output (512 channels) to model hidden
dimension uses wrong input size (uses 256 instead of 512 first-stage output).
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, conv_dim=(512, 512, 512), hidden_size=768):
        super().__init__()
        self.conv_dim = conv_dim
        self.hidden_size = hidden_size
        # Conv layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1 if i == 0 else conv_dim[i-1], conv_dim[i],
                      kernel_size=10 if i == 0 else 3,
                      stride=5 if i == 0 else 2)
            for i in range(len(conv_dim))
        ])
        # BUG: projection uses conv_dim[0] = 512 but should use conv_dim[-1] = 512
        # In original bug: used 256 (an intermediate dim) instead of conv_dim[-1]
        wrong_input_dim = conv_dim[0] // 2  # 256 instead of 512
        self.feature_projection = nn.Linear(wrong_input_dim, hidden_size)

    def forward(self, input_values):
        hidden_states = input_values.unsqueeze(1)  # (batch, 1, seq)
        for conv in self.conv_layers:
            hidden_states = conv(hidden_states)
        # hidden_states: (batch, 512, seq')
        hidden_states = hidden_states.transpose(1, 2)  # (batch, seq', 512)
        # BUG: feature_projection expects 256 but hidden_states has 512 features
        hidden_states = self.feature_projection(hidden_states)
        return hidden_states


INPUT_SHAPES = {"input_values": (2, 16000)}
