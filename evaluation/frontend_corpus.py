"""Curated corpus of self-contained, real-world-style PyTorch model sources.

These mirror the architectural idioms found across the most-starred PyTorch
repositories — CNN backbones, residual blocks, transformer encoders, attention
with q/k/v projections, MLP-mixers, U-Net-style encoders/decoders, RNNs,
models factored into helper methods (interprocedural), models using inheritance
and ``super().forward``, dynamic control flow, third-party blocks behind shape
stubs, and shape-annotated forwards.  Every entry is a standalone source string
so the AST frontend (``extract_computation_graph``) can ingest it without any
external dependency, keeping the parse-success SLA artifact byte-reproducible.

Keeping the corpus here (rather than live-cloning repos) makes the Step 45
stress test deterministic and version-independent: the source frontend is pure
AST and never imports torch, so the published rate is stable across machines.
"""
from __future__ import annotations

from typing import Dict

CORPUS: Dict[str, str] = {}


def _add(name: str, src: str) -> None:
    CORPUS[name] = src


_add("mlp_classifier", """
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
""")

_add("simple_cnn", """
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        x = self.pool(self.c1(x))
        x = self.pool(self.c2(x))
        x = x.mean(dim=(2, 3))
        return self.fc(x)
""")

_add("resnet_basic_block", """
import torch.nn as nn
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)
""")

_add("conv_stack_helper_method", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, padding=1)
        self.head = nn.Linear(32, 100)
    def _features(self, x):
        return self.stem(x)
    def forward(self, x):
        f = self._features(x)
        f = f.mean(dim=(2, 3))
        return self.head(f)
""")

_add("self_attention_qkv", """
import torch.nn as nn
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(512, 512)
        self.k = nn.Linear(512, 512)
        self.v = nn.Linear(512, 512)
        self.out = nn.Linear(512, 512)
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = q + k + v
        return self.out(attn)
""")

_add("transformer_encoder_block", """
import torch.nn as nn
class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.norm1 = nn.LayerNorm(256)
        self.ff1 = nn.Linear(256, 1024)
        self.ff2 = nn.Linear(1024, 256)
        self.norm2 = nn.LayerNorm(256)
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        h = self.ff2(self.ff1(x))
        return self.norm2(x + h)
""")

_add("mlp_mixer_block", """
import torch.nn as nn
class MixerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(512)
        self.token_mlp1 = nn.Linear(196, 256)
        self.token_mlp2 = nn.Linear(256, 196)
        self.norm2 = nn.LayerNorm(512)
        self.chan_mlp1 = nn.Linear(512, 2048)
        self.chan_mlp2 = nn.Linear(2048, 512)
    def forward(self, x):
        y = self.chan_mlp2(self.chan_mlp1(self.norm2(x)))
        return x + y
""")

_add("unet_encoder", """
import torch.nn as nn
import torch.nn.functional as F
class UNetEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Conv2d(3, 64, 3, padding=1)
        self.d2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x1 = F.relu(self.d1(x))
        x = self.pool(x1)
        x2 = F.relu(self.d2(x))
        return x2
""")

_add("lstm_tagger", """
import torch.nn as nn
class Tagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10000, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, 17)
    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        return self.fc(out)
""")

_add("gru_seq2vec", """
import torch.nn as nn
class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.fc = nn.Linear(128, 4)
    def forward(self, x):
        out, h = self.gru(x)
        return self.fc(out)
""")

_add("inheritance_super_forward", """
import torch.nn as nn
class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 64)
    def forward(self, x):
        return self.fc(x)
class Child(Base):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(64, 8)
    def forward(self, x):
        h = super().forward(x)
        return self.head(h)
""")

_add("dynamic_control_flow", """
import torch.nn as nn
class Branchy(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 32)
        self.b = nn.Linear(16, 32)
        self.head = nn.Linear(32, 4)
    def forward(self, x, flag: bool = True):
        if flag:
            h = self.a(x)
        else:
            h = self.b(x)
        return self.head(h)
""")

_add("modulelist_loop", """
import torch.nn as nn
class Stacked(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])
        self.head = nn.Linear(64, 10)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
""")

_add("sequential_features", """
import torch.nn as nn
class SeqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
        )
        self.head = nn.Linear(50, 3)
    def forward(self, x):
        return self.head(self.features(x))
""")

_add("annotated_forward_jaxtyping", """
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
class Annotated(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 8)
    def forward(self, x: Float[Tensor, "batch 32"]):
        return self.fc(x)
""")

_add("nested_helper_methods", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 16)
        self.b = nn.Linear(16, 16)
        self.head = nn.Linear(16, 4)
    def _inner(self, t):
        return self.b(t)
    def _outer(self, t):
        return self._inner(self.a(t))
    def forward(self, x):
        return self.head(self._outer(x))
""")

_add("embedding_pool_classifier", """
import torch.nn as nn
class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(20000, 300)
        self.fc1 = nn.Linear(300, 128)
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        e = self.emb(x)
        e = e.mean(dim=1)
        return self.fc2(self.fc1(e))
""")

_add("convtranspose_decoder", """
import torch.nn as nn
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
    def forward(self, x):
        x = self.up1(x)
        return self.up2(x)
""")

_add("groupnorm_block", """
import torch.nn as nn
import torch.nn.functional as F
class GNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, 3, padding=1)
        self.gn = nn.GroupNorm(8, 32)
    def forward(self, x):
        return F.relu(self.gn(self.conv(x)))
""")

_add("residual_mlp", """
import torch.nn as nn
class ResMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, 10)
    def forward(self, x):
        x = x + self.fc1(x)
        x = x + self.fc2(x)
        return self.head(x)
""")

_add("vit_patch_embed", """
import torch.nn as nn
class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(768)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return self.norm(x)
""")

_add("siamese_two_inputs", """
import torch.nn as nn
class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(64, 32)
        self.head = nn.Linear(32, 1)
    def forward(self, a, b):
        ea = self.enc(a)
        eb = self.enc(b)
        return self.head(ea + eb)
""")

_add("dropout_regularized", """
import torch.nn as nn
class DropNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.drop(self.fc1(x))
        return self.fc2(x)
""")

_add("autoencoder", """
import torch.nn as nn
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Linear(784, 256)
        self.e2 = nn.Linear(256, 64)
        self.d1 = nn.Linear(64, 256)
        self.d2 = nn.Linear(256, 784)
    def forward(self, x):
        z = self.e2(self.e1(x))
        return self.d2(self.d1(z))
""")

_add("multibranch_concat", """
import torch
import torch.nn as nn
class Inception(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Conv2d(64, 16, 1)
        self.b2 = nn.Conv2d(64, 16, 3, padding=1)
        self.b3 = nn.Conv2d(64, 16, 5, padding=2)
    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        return torch.cat([y1, y2, y3], dim=1)
""")

_add("layernorm_mlp_head", """
import torch.nn as nn
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc = nn.Linear(512, 1000)
    def forward(self, x):
        return self.fc(self.norm(x))
""")

_add("functional_relu_chain", """
import torch.nn as nn
import torch.nn.functional as F
class FuncNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
""")

_add("avgpool_flatten_classifier", """
import torch.nn as nn
class PoolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flat(x)
        return self.fc(x)
""")

_add("bilinear_fusion", """
import torch.nn as nn
class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fa = nn.Linear(32, 64)
        self.fb = nn.Linear(48, 64)
        self.head = nn.Linear(64, 5)
    def forward(self, a, b):
        return self.head(self.fa(a) + self.fb(b))
""")

_add("reshape_view_model", """
import torch.nn as nn
class Reshaper(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(48, 10)
    def forward(self, x):
        b = x.size(0)
        x = x.view(b, -1)
        return self.fc(x)
""")

_add("docstring_spec_model", '''
import torch.nn as nn
class Documented(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 8)
    def forward(self, x):
        """Run the head.

        Args:
            x: shape (batch, 64)
        """
        return self.fc(x)
''')

_add("instancenorm_block", """
import torch.nn as nn
import torch.nn.functional as F
class INBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.norm = nn.InstanceNorm2d(16)
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))
""")

_add("deep_sequential_resnet", """
import torch.nn as nn
import torch.nn.functional as F
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(256, 64, 1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 256, 1)
    def forward(self, x):
        identity = x
        out = F.relu(self.c1(x))
        out = F.relu(self.c2(out))
        out = self.c3(out)
        return F.relu(out + identity)
""")

_add("pointwise_depthwise", """
import torch.nn as nn
class DWSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.pw = nn.Conv2d(32, 64, 1)
    def forward(self, x):
        return self.pw(self.dw(x))
""")

_add("two_layer_tanh", """
import torch
import torch.nn as nn
class TanhNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))
""")
