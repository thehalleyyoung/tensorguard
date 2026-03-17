"""Tests for real-world PyTorch model code analysis.

Validates TensorGuard against realistic model code: HuggingFace-style
transformers, torchvision-style CNNs, and known shape bug patterns
collected from common mistakes in production ML code.

Clean models must produce no bugs (no false positives).  Buggy models
must have their specific shape errors detected.
"""

from __future__ import annotations

import textwrap
import time

import pytest

from src.api import AnalysisResult, Bug, BugCategory, analyze


# ===================================================================
# Helpers
# ===================================================================


def _analyze_snippet(code: str, filename: str = "<test>") -> AnalysisResult:
    """Analyze a code snippet and return the result."""
    return analyze(textwrap.dedent(code), filename=filename)


def _timed_analyze(code: str, filename: str = "<test>"):
    """Analyze and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = _analyze_snippet(code, filename)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


# ===================================================================
# Real model code snippets — CLEAN (no bugs)
# ===================================================================

# --- HuggingFace-style transformer blocks ---

BERT_ENCODER_LAYER = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, self.num_attention_heads,
                    self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return context_layer
"""

GPT2_ATTENTION = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Attention(nn.Module):
    def __init__(self, n_embd=768, n_head=12, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.size()
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.n_embd, dim=2)

        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_embd)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output
"""

VIT_PATCH_EMBEDDING = """\
import torch
import torch.nn as nn

class ViTPatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x
"""

TRANSFORMER_FFN = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerFFN(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        residual = hidden_states
        x = self.dense1(hidden_states)
        x = F.gelu(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x
"""

# --- torchvision-style CNN blocks ---

RESNET_BASIC_BLOCK = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
"""

VGG_BLOCK = """\
import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        layers = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
"""

EFFICIENTNET_MBCONV = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.expand_conv = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_channels)

        self.dw_conv = nn.Conv2d(mid_channels, mid_channels, 3,
                                 stride=stride, padding=1,
                                 groups=mid_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(mid_channels)

        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Linear(mid_channels, mid_channels // 4)
        self.se_fc2 = nn.Linear(mid_channels // 4, mid_channels)

        self.project_conv = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.silu(self.expand_bn(self.expand_conv(x)))
        out = F.silu(self.dw_bn(self.dw_conv(out)))

        se = self.se_pool(out).flatten(1)
        se = F.silu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        se = se.unsqueeze(-1).unsqueeze(-1)
        out = out * se

        out = self.project_bn(self.project_conv(out))
        if self.use_residual:
            out = out + residual
        return out
"""


# ===================================================================
# Known shape bug patterns
# ===================================================================

BUG_CONV_LINEAR_FLATTEN = """\
import torch
import torch.nn as nn

class BuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        # BUG: wrong flatten dim — should be 64*7*7 = 3136, not 64*14*14
        self.fc = nn.Linear(64 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
"""

BUG_ATTENTION_HEAD_RESHAPE = """\
import torch
import torch.nn as nn

class BuggyAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x):
        batch, seq, _ = x.shape
        qkv = self.qkv(x)
        # BUG: reshaping with wrong num_heads (8 instead of 12)
        qkv = qkv.reshape(batch, seq, 3, 8, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        return q
"""

BUG_TRANSPOSED_BATCH_SEQ = """\
import torch
import torch.nn as nn

class BuggyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x is (batch, seq, d_model) but attn expects (seq, batch, d_model)
        # BUG: passing batch-first tensor to seq-first attention
        attn_out, _ = self.attn(x, x, x)
        # This will silently produce wrong results because batch and seq are swapped
        return self.norm(x + attn_out)
"""

BUG_MISSING_UNSQUEEZE = """\
import torch
import torch.nn as nn

class BuggyBroadcast(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x is (batch, seq, hidden) and weight is (hidden,)
        # BUG: missing unsqueeze — should be weight.unsqueeze(0).unsqueeze(0)
        # for proper broadcast over batch and seq dims
        scaled = x * self.weight
        out = self.linear(scaled)
        # Then try to add a bias of shape (1, hidden) — dimension mismatch
        bias = torch.zeros(1, x.size(-1))
        return out + bias
"""

BUG_WRONG_PERMUTE_MHA = """\
import torch
import torch.nn as nn

class BuggyMHA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self.q_proj(x)
        q = q.view(batch, seq, self.num_heads, self.head_dim)
        # BUG: wrong permute order — should be (0, 2, 1, 3) not (0, 1, 2, 3)
        q = q.permute(0, 1, 2, 3)  # No-op! Doesn't move heads to dim 1
        # Now q is (batch, seq, heads, head_dim) but we treat it as
        # (batch, heads, seq, head_dim) in matmul
        scores = torch.matmul(q, q.transpose(-1, -2))
        return scores
"""


# ===================================================================
# Tests — Clean models (no false positives)
# ===================================================================


@pytest.mark.real_model
class TestCleanTransformerModels:
    """HuggingFace-style transformer blocks should have zero bugs."""

    def test_bert_encoder_clean(self):
        result = _analyze_snippet(BERT_ENCODER_LAYER)
        assert len(result.bugs) == 0, \
            f"BERT encoder: expected 0 bugs, got {result.bugs}"

    def test_gpt2_attention_clean(self):
        result = _analyze_snippet(GPT2_ATTENTION)
        assert len(result.bugs) == 0, \
            f"GPT-2 attention: expected 0 bugs, got {result.bugs}"

    def test_vit_patch_embedding_clean(self):
        result = _analyze_snippet(VIT_PATCH_EMBEDDING)
        assert len(result.bugs) == 0, \
            f"ViT patch embed: expected 0 bugs, got {result.bugs}"

    def test_transformer_ffn_clean(self):
        result = _analyze_snippet(TRANSFORMER_FFN)
        assert len(result.bugs) == 0, \
            f"Transformer FFN: expected 0 bugs, got {result.bugs}"


@pytest.mark.real_model
class TestCleanCNNModels:
    """torchvision-style CNN blocks should have zero bugs."""

    def test_resnet_basic_block_clean(self):
        result = _analyze_snippet(RESNET_BASIC_BLOCK)
        assert len(result.bugs) == 0, \
            f"ResNet block: expected 0 bugs, got {result.bugs}"

    def test_vgg_block_clean(self):
        result = _analyze_snippet(VGG_BLOCK)
        assert len(result.bugs) == 0, \
            f"VGG block: expected 0 bugs, got {result.bugs}"

    def test_efficientnet_mbconv_clean(self):
        result = _analyze_snippet(EFFICIENTNET_MBCONV)
        assert len(result.bugs) == 0, \
            f"EfficientNet MBConv: expected 0 bugs, got {result.bugs}"


# ===================================================================
# Tests — Buggy models (must detect issues)
# ===================================================================


@pytest.mark.real_model
class TestKnownShapeBugs:
    """Known shape bug patterns must be detected."""

    def test_conv_linear_flatten_mismatch(self):
        """Conv2d output flattened to wrong Linear input dimension."""
        result = _analyze_snippet(BUG_CONV_LINEAR_FLATTEN)
        assert result.bug_count > 0, \
            "Should detect Conv→Linear dimension mismatch"

    def test_attention_head_reshape_error(self):
        """Wrong num_heads in reshape causes silent dimension error."""
        result = _analyze_snippet(BUG_ATTENTION_HEAD_RESHAPE)
        assert result.bug_count > 0, \
            "Should detect wrong num_heads in reshape"

    def test_transposed_batch_seq(self):
        """Batch-first tensor passed to seq-first attention."""
        result = _analyze_snippet(BUG_TRANSPOSED_BATCH_SEQ)
        assert result.bug_count > 0, \
            "Should detect batch/seq dimension swap"

    def test_missing_unsqueeze(self):
        """Missing unsqueeze before broadcast operation."""
        result = _analyze_snippet(BUG_MISSING_UNSQUEEZE)
        assert result.bug_count > 0, \
            "Should detect missing unsqueeze for broadcast"

    def test_wrong_permute_mha(self):
        """Wrong permute order in multi-head attention."""
        result = _analyze_snippet(BUG_WRONG_PERMUTE_MHA)
        assert result.bug_count > 0, \
            "Should detect wrong permute order"


# ===================================================================
# Tests — Analysis performance
# ===================================================================


@pytest.mark.real_model
class TestAnalysisPerformance:
    """Analysis should complete within reasonable time for real models."""

    @pytest.mark.parametrize("snippet,name", [
        (BERT_ENCODER_LAYER, "BERT encoder"),
        (GPT2_ATTENTION, "GPT-2 attention"),
        (VIT_PATCH_EMBEDDING, "ViT patch"),
        (TRANSFORMER_FFN, "Transformer FFN"),
        (RESNET_BASIC_BLOCK, "ResNet block"),
        (VGG_BLOCK, "VGG block"),
        (EFFICIENTNET_MBCONV, "EfficientNet MBConv"),
    ])
    def test_analysis_time(self, snippet, name):
        """Each clean model should analyse in < 5 seconds."""
        result, elapsed_ms = _timed_analyze(snippet)
        assert elapsed_ms < 5000.0, \
            f"{name} analysis took {elapsed_ms:.0f}ms (limit: 5000ms)"

    def test_buggy_models_analysis_time(self):
        """Buggy models should also analyse quickly."""
        buggy_snippets = [
            BUG_CONV_LINEAR_FLATTEN,
            BUG_ATTENTION_HEAD_RESHAPE,
            BUG_TRANSPOSED_BATCH_SEQ,
            BUG_MISSING_UNSQUEEZE,
            BUG_WRONG_PERMUTE_MHA,
        ]
        for snippet in buggy_snippets:
            result, elapsed_ms = _timed_analyze(snippet)
            assert elapsed_ms < 5000.0, \
                f"Buggy model analysis took {elapsed_ms:.0f}ms (limit: 5000ms)"


# ===================================================================
# Tests — Metadata quality
# ===================================================================


@pytest.mark.real_model
class TestAnalysisMetadata:
    """Analysis metadata should be populated correctly."""

    def test_functions_analyzed_count(self):
        """Should count the correct number of functions."""
        result = _analyze_snippet(BERT_ENCODER_LAYER)
        # BERT encoder has __init__, transpose_for_scores, forward
        assert result.functions_analyzed >= 2

    def test_guards_harvested(self):
        """Should harvest guards from conditional code."""
        result = _analyze_snippet(BERT_ENCODER_LAYER)
        # The `if attention_mask is not None:` check is a guard
        assert result.guards_harvested >= 1

    def test_duration_recorded(self):
        """Analysis duration should be recorded and positive."""
        result = _analyze_snippet(RESNET_BASIC_BLOCK)
        assert result.duration_ms > 0.0

    def test_lines_analyzed(self):
        """Should report lines analysed."""
        result = _analyze_snippet(GPT2_ATTENTION)
        assert result.lines_analyzed > 0
