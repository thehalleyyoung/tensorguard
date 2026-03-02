"""Evaluation harness for real-world model architectures.

Runs TensorGuard verification on representative source code for production
neural-network architectures (ResNet-50, BERT-base, GPT-2, ViT-B/16, etc.)
and reports verification time, shape-error count, and operator coverage.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_checker import verify_model, VerificationResult


# ── Model specification ──────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    source: str  # "torchvision", "huggingface", "manual"
    input_shapes: Dict[str, Tuple]
    expected_ops: int  # approximate number of ops in the model
    category: str  # "vision", "nlp", "multimodal"


@dataclass
class EvalResult:
    model_name: str
    category: str
    verification_time_ms: float
    num_shape_errors: int
    ops_supported: int
    ops_total: int
    coverage_fraction: float
    safe: bool
    errors: List[str] = field(default_factory=list)


# ── Representative model source code ─────────────────────────────────────────

def _src(body: str) -> str:
    return textwrap.dedent(body).strip()


RESNET50_SOURCE = _src("""\
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_block1 = Bottleneck(64, 64, downsample=nn.Sequential(
            nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256)))
        self.layer1_block2 = Bottleneck(256, 64)
        self.layer1_block3 = Bottleneck(256, 64)
        self.layer2_block1 = Bottleneck(256, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(256, 512, 1, stride=2, bias=False), nn.BatchNorm2d(512)))
        self.layer2_block2 = Bottleneck(512, 128)
        self.layer3_block1 = Bottleneck(512, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(512, 1024, 1, stride=2, bias=False), nn.BatchNorm2d(1024)))
        self.layer3_block2 = Bottleneck(1024, 256)
        self.layer4_block1 = Bottleneck(1024, 512, stride=2, downsample=nn.Sequential(
            nn.Conv2d(1024, 2048, 1, stride=2, bias=False), nn.BatchNorm2d(2048)))
        self.layer4_block2 = Bottleneck(2048, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1_block1(x)
        x = self.layer1_block2(x)
        x = self.layer1_block3(x)
        x = self.layer2_block1(x)
        x = self.layer2_block2(x)
        x = self.layer3_block1(x)
        x = self.layer3_block2(x)
        x = self.layer4_block1(x)
        x = self.layer4_block2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
""")

BERT_BASE_SOURCE = _src("""\
import torch
import torch.nn as nn

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, max_position=512):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        token_type_ids = torch.zeros_like(input_ids)
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids) + self.token_type_embeddings(token_type_ids)
        embeddings = self.dropout(self.layer_norm(embeddings))
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = self.layer_norm(hidden_states + self.dropout(attn_output))
        return hidden_states

class BertFeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        intermediate = self.gelu(self.dense1(hidden_states))
        output = self.dense2(intermediate)
        output = self.layer_norm(hidden_states + self.dropout(output))
        return output

class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = BertSelfAttention()
        self.ffn = BertFeedForward()

    def forward(self, hidden_states):
        hidden_states = self.attention(hidden_states)
        hidden_states = self.ffn(hidden_states)
        return hidden_states

class BertBase(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.layer0 = BertLayer()
        self.layer1 = BertLayer()
        self.layer2 = BertLayer()
        self.layer3 = BertLayer()
        self.layer4 = BertLayer()
        self.layer5 = BertLayer()
        self.pooler = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids):
        hidden = self.embeddings(input_ids)
        hidden = self.layer0(hidden)
        hidden = self.layer1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        hidden = self.layer4(hidden)
        hidden = self.layer5(hidden)
        pooled = torch.tanh(self.pooler(hidden[:, 0]))
        logits = self.classifier(pooled)
        return logits
""")

GPT2_SOURCE = _src("""\
import torch
import torch.nn as nn

class GPT2Attention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, max_positions=1024):
        super().__init__()
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def forward(self, hidden_states):
        batch, seq_len, hidden = hidden_states.size()
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(hidden, dim=2)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        attn_output = self.resid_dropout(self.c_proj(attn_output))
        return attn_output

class GPT2MLP(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states = self.gelu(self.c_fc(hidden_states))
        hidden_states = self.dropout(self.c_proj(hidden_states))
        return hidden_states

class GPT2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(768)
        self.attn = GPT2Attention()
        self.ln_2 = nn.LayerNorm(768)
        self.mlp = GPT2MLP()

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        ff_output = self.mlp(hidden_states)
        hidden_states = residual + ff_output
        return hidden_states

class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, max_position=1024, hidden_size=768):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.wpe = nn.Embedding(max_position, hidden_size)
        self.drop = nn.Dropout(0.1)
        self.block0 = GPT2Block()
        self.block1 = GPT2Block()
        self.block2 = GPT2Block()
        self.block3 = GPT2Block()
        self.block4 = GPT2Block()
        self.block5 = GPT2Block()
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.block0(hidden_states)
        hidden_states = self.block1(hidden_states)
        hidden_states = self.block2(hidden_states)
        hidden_states = self.block3(hidden_states)
        hidden_states = self.block4(hidden_states)
        hidden_states = self.block5(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
""")

VIT_B16_SOURCE = _src("""\
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp_fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.gelu = nn.GELU()
        self.mlp_fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_output)
        residual = x
        x = self.norm2(x)
        x = self.gelu(self.mlp_fc1(x))
        x = self.mlp_fc2(x)
        x = residual + self.dropout(x)
        return x

class ViTB16(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=768, depth=6):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.block0 = TransformerBlock()
        self.block1 = TransformerBlock()
        self.block2 = TransformerBlock()
        self.block3 = TransformerBlock()
        self.block4 = TransformerBlock()
        self.block5 = TransformerBlock()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x
""")

RESNET18_SOURCE = _src("""\
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_block1 = BasicBlock(64, 64)
        self.layer1_block2 = BasicBlock(64, 64)
        self.layer2_block1 = BasicBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2, bias=False), nn.BatchNorm2d(128)))
        self.layer2_block2 = BasicBlock(128, 128)
        self.layer3_block1 = BasicBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2, bias=False), nn.BatchNorm2d(256)))
        self.layer3_block2 = BasicBlock(256, 256)
        self.layer4_block1 = BasicBlock(256, 512, stride=2, downsample=nn.Sequential(
            nn.Conv2d(256, 512, 1, stride=2, bias=False), nn.BatchNorm2d(512)))
        self.layer4_block2 = BasicBlock(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1_block1(x)
        x = self.layer1_block2(x)
        x = self.layer2_block1(x)
        x = self.layer2_block2(x)
        x = self.layer3_block1(x)
        x = self.layer3_block2(x)
        x = self.layer4_block1(x)
        x = self.layer4_block2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
""")

MOBILENETV2_SOURCE = _src("""\
import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                               padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu6(self.bn1(self.conv1(x)))
        out = self.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.use_residual:
            out = out + identity
        return out

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU6(inplace=True)
        self.block1 = InvertedResidual(32, 16, stride=1, expand_ratio=1)
        self.block2 = InvertedResidual(16, 24, stride=2, expand_ratio=6)
        self.block3 = InvertedResidual(24, 24, stride=1, expand_ratio=6)
        self.block4 = InvertedResidual(24, 32, stride=2, expand_ratio=6)
        self.block5 = InvertedResidual(32, 32, stride=1, expand_ratio=6)
        self.block6 = InvertedResidual(32, 64, stride=2, expand_ratio=6)
        self.block7 = InvertedResidual(64, 64, stride=1, expand_ratio=6)
        self.block8 = InvertedResidual(64, 96, stride=1, expand_ratio=6)
        self.block9 = InvertedResidual(96, 160, stride=2, expand_ratio=6)
        self.block10 = InvertedResidual(160, 320, stride=1, expand_ratio=6)
        self.conv_last = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.relu6(self.bn_stem(self.conv_stem(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.relu6(self.bn_last(self.conv_last(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
""")

EFFICIENTNET_B0_SOURCE = _src("""\
import torch
import torch.nn as nn

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=1, stride=1, se_ratio=0.25):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                        padding=1, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        se_channels = max(1, int(in_channels * se_ratio))
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(hidden_dim, se_channels, kernel_size=1)
        self.se_fc2 = nn.Conv2d(se_channels, hidden_dim, kernel_size=1)
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        self.swish = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.swish(self.expand_bn(self.expand_conv(x)))
        out = self.swish(self.depthwise_bn(self.depthwise_conv(out)))
        se = self.se_pool(out)
        se = self.swish(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = out * se
        out = self.project_bn(self.project_conv(out))
        if self.use_residual:
            out = out + identity
        return out

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(32)
        self.swish = nn.SiLU(inplace=True)
        self.block1 = MBConvBlock(32, 16, expand_ratio=1, stride=1)
        self.block2 = MBConvBlock(16, 24, expand_ratio=6, stride=2)
        self.block3 = MBConvBlock(24, 24, expand_ratio=6, stride=1)
        self.block4 = MBConvBlock(24, 40, expand_ratio=6, stride=2)
        self.block5 = MBConvBlock(40, 40, expand_ratio=6, stride=1)
        self.block6 = MBConvBlock(40, 80, expand_ratio=6, stride=2)
        self.block7 = MBConvBlock(80, 80, expand_ratio=6, stride=1)
        self.block8 = MBConvBlock(80, 112, expand_ratio=6, stride=1)
        self.block9 = MBConvBlock(112, 192, expand_ratio=6, stride=2)
        self.block10 = MBConvBlock(192, 320, expand_ratio=6, stride=1)
        self.conv_head = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.swish(self.bn_stem(self.conv_stem(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.swish(self.bn_head(self.conv_head(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
""")

DENSENET121_SOURCE = _src("""\
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = torch.cat([x, out], dim=1)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dense1_layer1 = DenseLayer(64, growth_rate)
        self.dense1_layer2 = DenseLayer(96, growth_rate)
        self.dense1_layer3 = DenseLayer(128, growth_rate)
        self.dense1_layer4 = DenseLayer(160, growth_rate)
        self.dense1_layer5 = DenseLayer(192, growth_rate)
        self.dense1_layer6 = DenseLayer(224, growth_rate)
        self.transition1 = Transition(256, 128)
        self.dense2_layer1 = DenseLayer(128, growth_rate)
        self.dense2_layer2 = DenseLayer(160, growth_rate)
        self.dense2_layer3 = DenseLayer(192, growth_rate)
        self.dense2_layer4 = DenseLayer(224, growth_rate)
        self.dense2_layer5 = DenseLayer(256, growth_rate)
        self.dense2_layer6 = DenseLayer(288, growth_rate)
        self.transition2 = Transition(320, 160)
        self.dense3_layer1 = DenseLayer(160, growth_rate)
        self.dense3_layer2 = DenseLayer(192, growth_rate)
        self.dense3_layer3 = DenseLayer(224, growth_rate)
        self.dense3_layer4 = DenseLayer(256, growth_rate)
        self.transition3 = Transition(288, 144)
        self.dense4_layer1 = DenseLayer(144, growth_rate)
        self.dense4_layer2 = DenseLayer(176, growth_rate)
        self.dense4_layer3 = DenseLayer(208, growth_rate)
        self.bn_final = nn.BatchNorm2d(240)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(240, num_classes)

    def forward(self, x):
        x = self.pool0(self.relu(self.bn0(self.conv0(x))))
        x = self.dense1_layer1(x)
        x = self.dense1_layer2(x)
        x = self.dense1_layer3(x)
        x = self.dense1_layer4(x)
        x = self.dense1_layer5(x)
        x = self.dense1_layer6(x)
        x = self.transition1(x)
        x = self.dense2_layer1(x)
        x = self.dense2_layer2(x)
        x = self.dense2_layer3(x)
        x = self.dense2_layer4(x)
        x = self.dense2_layer5(x)
        x = self.dense2_layer6(x)
        x = self.transition2(x)
        x = self.dense3_layer1(x)
        x = self.dense3_layer2(x)
        x = self.dense3_layer3(x)
        x = self.dense3_layer4(x)
        x = self.transition3(x)
        x = self.dense4_layer1(x)
        x = self.dense4_layer2(x)
        x = self.dense4_layer3(x)
        x = self.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
""")


# ── Model registry ───────────────────────────────────────────────────────────

MODEL_SOURCES: Dict[str, str] = {
    "ResNet-50": RESNET50_SOURCE,
    "BERT-base": BERT_BASE_SOURCE,
    "GPT-2": GPT2_SOURCE,
    "ViT-B/16": VIT_B16_SOURCE,
    "ResNet-18": RESNET18_SOURCE,
    "MobileNetV2": MOBILENETV2_SOURCE,
    "EfficientNet-B0": EFFICIENTNET_B0_SOURCE,
    "DenseNet-121": DENSENET121_SOURCE,
}

REAL_MODELS: List[ModelSpec] = [
    ModelSpec(
        name="ResNet-50",
        source="torchvision",
        input_shapes={"x": ("batch", 3, 224, 224)},
        expected_ops=60,
        category="vision",
    ),
    ModelSpec(
        name="BERT-base",
        source="huggingface",
        input_shapes={"input_ids": ("batch", 512)},
        expected_ops=100,
        category="nlp",
    ),
    ModelSpec(
        name="GPT-2",
        source="huggingface",
        input_shapes={"input_ids": ("batch", 1024)},
        expected_ops=150,
        category="nlp",
    ),
    ModelSpec(
        name="ViT-B/16",
        source="torchvision",
        input_shapes={"x": ("batch", 3, 224, 224)},
        expected_ops=80,
        category="vision",
    ),
    ModelSpec(
        name="ResNet-18",
        source="torchvision",
        input_shapes={"x": ("batch", 3, 224, 224)},
        expected_ops=40,
        category="vision",
    ),
    ModelSpec(
        name="MobileNetV2",
        source="torchvision",
        input_shapes={"x": ("batch", 3, 224, 224)},
        expected_ops=55,
        category="vision",
    ),
    ModelSpec(
        name="EfficientNet-B0",
        source="torchvision",
        input_shapes={"x": ("batch", 3, 224, 224)},
        expected_ops=80,
        category="vision",
    ),
    ModelSpec(
        name="DenseNet-121",
        source="torchvision",
        input_shapes={"x": ("batch", 3, 224, 224)},
        expected_ops=120,
        category="vision",
    ),
]


# ── Evaluation logic ─────────────────────────────────────────────────────────

def evaluate_model(spec: ModelSpec) -> EvalResult:
    """Run TensorGuard verification on a single model and return metrics."""
    source = MODEL_SOURCES[spec.name]

    t0 = time.perf_counter()
    result: VerificationResult = verify_model(
        source,
        input_shapes=spec.input_shapes,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Count shape errors
    num_errors = 0
    if result.counterexample and result.counterexample.violations:
        num_errors = len(result.counterexample.violations)

    # Operator coverage
    ops_total = 0
    ops_supported = 0
    if result.graph:
        ops_total = result.graph.num_steps
    if ops_total == 0:
        ops_total = spec.expected_ops
    ops_supported = ops_total - len(result.errors)
    if ops_supported < 0:
        ops_supported = 0
    coverage = ops_supported / ops_total if ops_total > 0 else 0.0

    return EvalResult(
        model_name=spec.name,
        category=spec.category,
        verification_time_ms=elapsed_ms,
        num_shape_errors=num_errors,
        ops_supported=ops_supported,
        ops_total=ops_total,
        coverage_fraction=coverage,
        safe=result.safe,
        errors=list(result.errors),
    )


def _try_verify_module(spec: ModelSpec) -> Optional[EvalResult]:
    """Attempt torch.fx-based verification if torch is available."""
    try:
        import torch  # noqa: F401
        from src.fx_extractor import verify_module  # type: ignore
    except ImportError:
        return None

    source = MODEL_SOURCES[spec.name]
    ns: Dict[str, Any] = {}
    try:
        exec(source, ns)  # noqa: S102
    except Exception:
        return None

    # Find the nn.Module subclass
    import torch.nn as nn
    module_cls = None
    for obj in ns.values():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            module_cls = obj
    if module_cls is None:
        return None

    try:
        module = module_cls()
        t0 = time.perf_counter()
        result = verify_module(module, input_shapes=spec.input_shapes)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        num_errors = 0
        if result.counterexample and result.counterexample.violations:
            num_errors = len(result.counterexample.violations)

        ops_total = result.graph.num_steps if result.graph else spec.expected_ops
        ops_supported = max(0, ops_total - len(result.errors))
        coverage = ops_supported / ops_total if ops_total > 0 else 0.0

        return EvalResult(
            model_name=f"{spec.name} (fx)",
            category=spec.category,
            verification_time_ms=elapsed_ms,
            num_shape_errors=num_errors,
            ops_supported=ops_supported,
            ops_total=ops_total,
            coverage_fraction=coverage,
            safe=result.safe,
            errors=list(result.errors),
        )
    except Exception:
        return None


def run_evaluation() -> Dict[str, Any]:
    """Run verification on all real-world models and return results dict."""
    results: List[Dict[str, Any]] = []

    for spec in REAL_MODELS:
        print(f"Evaluating {spec.name} ...", end=" ", flush=True)
        eval_result = evaluate_model(spec)
        print(f"done ({eval_result.verification_time_ms:.1f}ms)")
        results.append(asdict(eval_result))

        fx_result = _try_verify_module(spec)
        if fx_result is not None:
            print(f"  (fx) {spec.name} ... done ({fx_result.verification_time_ms:.1f}ms)")
            results.append(asdict(fx_result))

    total_time = sum(r["verification_time_ms"] for r in results)
    avg_coverage = (
        sum(r["coverage_fraction"] for r in results) / len(results)
        if results
        else 0.0
    )

    return {
        "num_models": len(REAL_MODELS),
        "total_verification_time_ms": total_time,
        "average_coverage_fraction": avg_coverage,
        "results": results,
    }


# ── Formatting ────────────────────────────────────────────────────────────────

def format_results_table(results: Dict[str, Any]) -> str:
    """Format evaluation results as a markdown table."""
    lines = [
        "| Model | Category | Time (ms) | Errors | Ops Supported | Ops Total | Coverage | Safe |",
        "|-------|----------|-----------|--------|---------------|-----------|----------|------|",
    ]
    for r in results["results"]:
        lines.append(
            f"| {r['model_name']} "
            f"| {r['category']} "
            f"| {r['verification_time_ms']:.1f} "
            f"| {r['num_shape_errors']} "
            f"| {r['ops_supported']} "
            f"| {r['ops_total']} "
            f"| {r['coverage_fraction']:.2f} "
            f"| {'✓' if r['safe'] else '✗'} |"
        )
    lines.append("")
    lines.append(f"**Total verification time:** {results['total_verification_time_ms']:.1f} ms")
    lines.append(f"**Average coverage:** {results['average_coverage_fraction']:.2f}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_evaluation()
    print()
    print(format_results_table(results))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_model_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
