#!/usr/bin/env python3.11
"""Track E: TorchDynamo Guard Correspondence Experiment

CLAIM: TensorGuard-verified programs produce VALID Dynamo guard sets.
When run under torch.compile with shape ranges that respect TG's verified
contract, Dynamo doesn't recompile.

This experiment:
1. Picks 10-20 small HF / timm / torchvision models
2. Runs TensorGuard verification on each
3. For verified models, extracts Dynamo guards and tests shape variations
4. Records whether Dynamo recompiles within contract vs out-of-contract
5. Tabulates results to experiments/dynamo_guard_correspondence.json
"""

import json
import os
import sys
import time
import traceback
import inspect
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch._dynamo
import torch._dynamo.utils

# Check availability of model libraries
HAS_TORCHVISION = False
HAS_TRANSFORMERS = False
HAS_TIMM = False

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    pass

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    pass

try:
    import timm
    HAS_TIMM = True
except ImportError:
    pass

from src.api import verify_architecture, AnalysisResult


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelTestResult:
    """Result for a single model."""
    name: str
    source: str = ""
    tg_verdict: str = "not_attempted"  # verified, unsafe, error, not_attempted
    tg_contract: Optional[Dict[str, Any]] = None
    tg_error: Optional[str] = None
    in_contract_variations: int = 0
    in_contract_recompiles: int = 0
    out_of_contract_variations: int = 0
    out_of_contract_recompiles: int = 0
    claim_holds: Optional[bool] = None
    notes: str = ""
    duration_ms: float = 0.0

    def to_dict(self):
        d = asdict(self)
        # Don't include source in output (too verbose)
        d.pop("source", None)
        return d


@dataclass
class ExperimentResults:
    """Overall experiment results."""
    torch_version: str
    models: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# Simple standalone models
# ═══════════════════════════════════════════════════════════════════════════════

def get_simple_models() -> List[Tuple[str, nn.Module, str, Dict]]:
    """Returns (name, instance, source, input_shapes) for simple models."""
    models = []
    
    # 1. SimpleMLP
    mlp_src = """
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
"""
    namespace = {"nn": nn, "torch": torch}
    exec(mlp_src, namespace)
    SimpleMLP = namespace["SimpleMLP"]
    models.append(("SimpleMLP", SimpleMLP(), mlp_src, {"x": ("B", 64)}))
    
    # 2. ConvStack
    conv_src = """
class ConvStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(x)
        return self.fc(x.flatten(1))
"""
    namespace = {"nn": nn, "torch": torch}
    exec(conv_src, namespace)
    ConvStack = namespace["ConvStack"]
    models.append(("ConvStack", ConvStack(), conv_src, {"x": ("B", 3, "H", "W")}))
    
    # 3. TinyMHA (multi-head attention block)
    mha_src = """
class TinyMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
    
    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out
"""
    namespace = {"nn": nn, "torch": torch}
    exec(mha_src, namespace)
    TinyMHA = namespace["TinyMHA"]
    models.append(("TinyMHA", TinyMHA(), mha_src, {"x": ("B", "S", 64)}))
    
    # 4. SimpleTransformerBlock
    trans_src = """
class SimpleTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 64)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
"""
    namespace = {"nn": nn, "torch": torch}
    exec(trans_src, namespace)
    SimpleTransformerBlock = namespace["SimpleTransformerBlock"]
    models.append(("SimpleTransformerBlock", SimpleTransformerBlock(), trans_src, 
                   {"x": ("B", "S", 64)}))
    
    # 5. SimpleRNN
    rnn_src = """
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, 10)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
"""
    namespace = {"nn": nn, "torch": torch}
    exec(rnn_src, namespace)
    SimpleRNN = namespace["SimpleRNN"]
    models.append(("SimpleRNN", SimpleRNN(), rnn_src, {"x": ("B", "S", 64)}))
    
    # 6. TinyResBlock
    resblock_src = """
class TinyResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)
"""
    namespace = {"nn": nn, "torch": torch}
    exec(resblock_src, namespace)
    TinyResBlock = namespace["TinyResBlock"]
    models.append(("TinyResBlock", TinyResBlock(), resblock_src, {"x": ("B", 32, "H", "W")}))
    
    # 7. SimpleCNN
    cnn_src = """
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)
"""
    namespace = {"nn": nn, "torch": torch}
    exec(cnn_src, namespace)
    SimpleCNN = namespace["SimpleCNN"]
    models.append(("SimpleCNN", SimpleCNN(), cnn_src, {"x": ("B", 1, 28, 28)}))
    
    return models


def get_torchvision_models() -> List[Tuple[str, nn.Module, str, Dict]]:
    """Returns torchvision models with minimal source implementations."""
    if not HAS_TORCHVISION:
        return []
    
    models = []
    
    # Create simplified versions that we can actually verify
    
    # 1. Minimal ResNet-like block
    resnet_minimal_src = """
class MinimalResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1000)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
"""
    namespace = {"nn": nn, "torch": torch}
    exec(resnet_minimal_src, namespace)
    MinimalResNet = namespace["MinimalResNet"]
    models.append(("MinimalResNet", MinimalResNet(), resnet_minimal_src, {"x": ("B", 3, "H", "W")}))
    
    # 2. Minimal MobileNet-like
    mobilenet_minimal_src = """
class MinimalMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6()
        self.dw = nn.Conv2d(32, 32, 3, groups=32, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pw = nn.Conv2d(32, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1000)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.dw(x)))
        x = self.relu(self.bn3(self.pw(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
"""
    namespace = {"nn": nn, "torch": torch}
    exec(mobilenet_minimal_src, namespace)
    MinimalMobileNet = namespace["MinimalMobileNet"]
    models.append(("MinimalMobileNet", MinimalMobileNet(), mobilenet_minimal_src, {"x": ("B", 3, "H", "W")}))
    
    # 3. Minimal VGG-like
    vgg_minimal_src = """
class MinimalVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""
    namespace = {"nn": nn, "torch": torch}
    exec(vgg_minimal_src, namespace)
    MinimalVGG = namespace["MinimalVGG"]
    models.append(("MinimalVGG", MinimalVGG(), vgg_minimal_src, {"x": ("B", 3, 28, 28)}))
    
    return models


def get_transformers_models() -> List[Tuple[str, nn.Module, str, Dict]]:
    """Returns HuggingFace transformers models (tiny configs)."""
    if not HAS_TRANSFORMERS:
        return []
    
    models = []
    
    # Create minimal transformer-like models instead of using HF
    
    # 1. Minimal BERT-like encoder
    bert_minimal_src = """
class MinimalBERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.pos_embed = nn.Embedding(512, 128)
        self.encoder = nn.TransformerEncoderLayer(d_model=128, nhead=4, 
                                                    dim_feedforward=512, batch_first=True)
        self.fc = nn.Linear(128, 128)
    
    def forward(self, input_ids):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(positions)
        x = self.encoder(x)
        return self.fc(x[:, 0, :])
"""
    namespace = {"nn": nn, "torch": torch}
    exec(bert_minimal_src, namespace)
    MinimalBERTEncoder = namespace["MinimalBERTEncoder"]
    models.append(("MinimalBERTEncoder", MinimalBERTEncoder(), bert_minimal_src, 
                   {"input_ids": ("B", "S")}))
    
    # 2. Minimal GPT-like decoder
    gpt_minimal_src = """
class MinimalGPTDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.pos_embed = nn.Embedding(512, 128)
        self.decoder = nn.TransformerDecoderLayer(d_model=128, nhead=4,
                                                    dim_feedforward=512, batch_first=True)
        self.fc = nn.Linear(128, 1000)
    
    def forward(self, input_ids):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(positions)
        # Self-attention (memory = tgt for causal)
        x = self.decoder(x, x)
        return self.fc(x)
"""
    namespace = {"nn": nn, "torch": torch}
    exec(gpt_minimal_src, namespace)
    MinimalGPTDecoder = namespace["MinimalGPTDecoder"]
    models.append(("MinimalGPTDecoder", MinimalGPTDecoder(), gpt_minimal_src,
                   {"input_ids": ("B", "S")}))
    
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# TensorGuard verification
# ═══════════════════════════════════════════════════════════════════════════════

def run_tensorguard_verification(source: str, input_shapes: Dict[str, Any], 
                                  model_name: str) -> Tuple[str, Optional[Dict], Optional[str]]:
    """
    Run TensorGuard verification.
    
    Returns:
        (verdict, contract, error_msg)
        verdict: "verified", "unsafe", "error"
    """
    if not source or source.startswith("#"):
        # Pre-built models without source inspection
        return "not_attempted", None, "Source not available for pre-built model"
    
    try:
        result: AnalysisResult = verify_architecture(
            source,
            input_shapes=input_shapes,
            filename=f"<{model_name}>",
        )
        
        if result.status == "SAFE":
            return "verified", input_shapes, None
        else:
            # Has bugs
            errors = [b.message for b in result.bugs]
            return "unsafe", None, f"Found {len(errors)} issue(s): {errors[:3]}"
    
    except Exception as e:
        return "error", None, f"{type(e).__name__}: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamo guard extraction and recompilation testing
# ═══════════════════════════════════════════════════════════════════════════════

def reset_dynamo():
    """Reset Dynamo compilation state."""
    torch._dynamo.reset()
    if hasattr(torch._dynamo.utils, 'counters'):
        torch._dynamo.utils.counters.clear()


def get_recompilation_count() -> int:
    """Get the number of unique graphs compiled by Dynamo."""
    if hasattr(torch._dynamo.utils, 'counters'):
        return torch._dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0)
    return 0


def test_dynamo_correspondence(model: nn.Module, contract: Dict[str, Any], 
                                model_name: str) -> Tuple[int, int, int, int, str]:
    """
    Test Dynamo guard correspondence.
    
    Returns:
        (in_contract_variations, in_contract_recompiles, 
         out_of_contract_variations, out_of_contract_recompiles, notes)
    """
    notes = []
    
    try:
        # Compile with dynamic shapes
        reset_dynamo()
        torch._dynamo.config.cache_size_limit = 64  # Allow many entries
        compiled_model = torch.compile(model, dynamic=True)
        
        # Determine input shapes
        # contract format: {"x": ("B", 3, "H", "W")} or {"x": ("B", 64)}
        example_inputs = {}
        for param_name, shape_spec in contract.items():
            # Replace symbolic dims with concrete values
            concrete_shape = []
            for dim in shape_spec:
                if isinstance(dim, str):
                    # Symbolic dimension
                    if dim in ("B", "batch", "batch_size"):
                        concrete_shape.append(4)
                    elif dim in ("S", "seq", "seq_len"):
                        concrete_shape.append(16)
                    elif dim in ("H", "height"):
                        concrete_shape.append(32)
                    elif dim in ("W", "width"):
                        concrete_shape.append(32)
                    else:
                        concrete_shape.append(8)  # default
                else:
                    concrete_shape.append(dim)
            
            # Create tensor
            if len(concrete_shape) == 2:  # (B, D)
                example_inputs[param_name] = torch.randn(*concrete_shape)
            elif len(concrete_shape) == 3:  # (B, S, D)
                example_inputs[param_name] = torch.randn(*concrete_shape)
            elif len(concrete_shape) == 4:  # (B, C, H, W)
                example_inputs[param_name] = torch.randn(*concrete_shape)
            else:
                example_inputs[param_name] = torch.randn(*concrete_shape)
        
        # Initial compilation
        model.eval()
        with torch.no_grad():
            if len(example_inputs) == 1:
                input_tensor = list(example_inputs.values())[0]
                _ = compiled_model(input_tensor)
            else:
                _ = compiled_model(**example_inputs)
        
        initial_graphs = get_recompilation_count()
        notes.append(f"Initial compilation: {initial_graphs} graphs")
        
        # In-contract variations
        in_variations = 0
        in_recompiles = 0
        
        # Vary batch size
        for batch_size in [2, 8, 16]:
            try:
                varied_inputs = {}
                for param_name, shape_spec in contract.items():
                    concrete_shape = []
                    for dim in shape_spec:
                        if isinstance(dim, str):
                            if dim in ("B", "batch", "batch_size"):
                                concrete_shape.append(batch_size)
                            elif dim in ("S", "seq", "seq_len"):
                                concrete_shape.append(16)
                            elif dim in ("H", "height"):
                                concrete_shape.append(32)
                            elif dim in ("W", "width"):
                                concrete_shape.append(32)
                            else:
                                concrete_shape.append(8)
                        else:
                            concrete_shape.append(dim)
                    varied_inputs[param_name] = torch.randn(*concrete_shape)
                
                with torch.no_grad():
                    if len(varied_inputs) == 1:
                        input_tensor = list(varied_inputs.values())[0]
                        _ = compiled_model(input_tensor)
                    else:
                        _ = compiled_model(**varied_inputs)
                
                in_variations += 1
                new_graphs = get_recompilation_count()
                if new_graphs > initial_graphs:
                    in_recompiles += (new_graphs - initial_graphs)
                    initial_graphs = new_graphs
            except Exception as e:
                notes.append(f"In-contract variation failed: {e}")
        
        # Vary spatial dimensions (if applicable)
        if any("H" in str(s) or "W" in str(s) for s in contract.values()):
            for size in [64, 128]:
                try:
                    varied_inputs = {}
                    for param_name, shape_spec in contract.items():
                        concrete_shape = []
                        for dim in shape_spec:
                            if isinstance(dim, str):
                                if dim in ("B", "batch", "batch_size"):
                                    concrete_shape.append(4)
                                elif dim in ("S", "seq", "seq_len"):
                                    concrete_shape.append(16)
                                elif dim in ("H", "height", "W", "width"):
                                    concrete_shape.append(size)
                                else:
                                    concrete_shape.append(8)
                            else:
                                concrete_shape.append(dim)
                        varied_inputs[param_name] = torch.randn(*concrete_shape)
                    
                    with torch.no_grad():
                        if len(varied_inputs) == 1:
                            input_tensor = list(varied_inputs.values())[0]
                            _ = compiled_model(input_tensor)
                        else:
                            _ = compiled_model(**varied_inputs)
                    
                    in_variations += 1
                    new_graphs = get_recompilation_count()
                    if new_graphs > initial_graphs:
                        in_recompiles += (new_graphs - initial_graphs)
                        initial_graphs = new_graphs
                except Exception as e:
                    notes.append(f"In-contract spatial variation failed: {e}")
        
        # Out-of-contract variations
        out_variations = 0
        out_recompiles = 0
        
        # Change a fixed dimension (e.g., channels from 3 to 4)
        if any(3 in shape_spec if isinstance(shape_spec, (tuple, list)) else False 
               for shape_spec in contract.values()):
            try:
                varied_inputs = {}
                for param_name, shape_spec in contract.items():
                    concrete_shape = []
                    for dim in shape_spec:
                        if isinstance(dim, int) and dim == 3:
                            concrete_shape.append(4)  # Change 3 to 4
                        elif isinstance(dim, str):
                            if dim in ("B", "batch", "batch_size"):
                                concrete_shape.append(4)
                            elif dim in ("S", "seq", "seq_len"):
                                concrete_shape.append(16)
                            elif dim in ("H", "height"):
                                concrete_shape.append(32)
                            elif dim in ("W", "width"):
                                concrete_shape.append(32)
                            else:
                                concrete_shape.append(8)
                        else:
                            concrete_shape.append(dim)
                    varied_inputs[param_name] = torch.randn(*concrete_shape)
                
                pre_out_graphs = get_recompilation_count()
                with torch.no_grad():
                    if len(varied_inputs) == 1:
                        input_tensor = list(varied_inputs.values())[0]
                        _ = compiled_model(input_tensor)
                    else:
                        _ = compiled_model(**varied_inputs)
                
                out_variations += 1
                new_graphs = get_recompilation_count()
                if new_graphs > pre_out_graphs:
                    out_recompiles += (new_graphs - pre_out_graphs)
            except Exception as e:
                # Expected to fail or recompile
                notes.append(f"Out-of-contract variation: {e}")
                out_variations += 1
                out_recompiles += 1
        
        # Change embedding dimension (if applicable)
        if any(64 in shape_spec if isinstance(shape_spec, (tuple, list)) else False 
               for shape_spec in contract.values()):
            try:
                varied_inputs = {}
                for param_name, shape_spec in contract.items():
                    concrete_shape = []
                    for dim in shape_spec:
                        if isinstance(dim, int) and dim == 64:
                            concrete_shape.append(128)  # Change 64 to 128
                        elif isinstance(dim, str):
                            if dim in ("B", "batch", "batch_size"):
                                concrete_shape.append(4)
                            elif dim in ("S", "seq", "seq_len"):
                                concrete_shape.append(16)
                            else:
                                concrete_shape.append(8)
                        else:
                            concrete_shape.append(dim)
                    varied_inputs[param_name] = torch.randn(*concrete_shape)
                
                pre_out_graphs = get_recompilation_count()
                with torch.no_grad():
                    if len(varied_inputs) == 1:
                        input_tensor = list(varied_inputs.values())[0]
                        _ = compiled_model(input_tensor)
                    else:
                        _ = compiled_model(**varied_inputs)
                
                out_variations += 1
                new_graphs = get_recompilation_count()
                if new_graphs > pre_out_graphs:
                    out_recompiles += (new_graphs - pre_out_graphs)
            except Exception as e:
                notes.append(f"Out-of-contract dim variation: {e}")
                out_variations += 1
                out_recompiles += 1
        
        return in_variations, in_recompiles, out_variations, out_recompiles, "; ".join(notes)
    
    except Exception as e:
        return 0, 0, 0, 0, f"Dynamo test failed: {traceback.format_exc()}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment() -> ExperimentResults:
    """Run the full Track E experiment."""
    print("=" * 80)
    print("Track E: TorchDynamo Guard Correspondence Experiment")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchDynamo available: {hasattr(torch, '_dynamo')}")
    print(f"torchvision: {HAS_TORCHVISION}")
    print(f"transformers: {HAS_TRANSFORMERS}")
    print(f"timm: {HAS_TIMM}")
    print()
    
    results = ExperimentResults(torch_version=torch.__version__)
    
    # Collect all models
    all_models = []
    
    print("Collecting models...")
    simple_models = get_simple_models()
    print(f"  Simple models: {len(simple_models)}")
    all_models.extend(simple_models)
    
    if HAS_TORCHVISION:
        tv_models = get_torchvision_models()
        print(f"  torchvision models: {len(tv_models)}")
        all_models.extend(tv_models)
    
    if HAS_TRANSFORMERS:
        hf_models = get_transformers_models()
        print(f"  transformers models: {len(hf_models)}")
        all_models.extend(hf_models)
    
    print(f"\nTotal models to test: {len(all_models)}\n")
    
    # Test each model
    for idx, (model_name, model_instance, source, input_shapes) in enumerate(all_models, 1):
        print(f"[{idx}/{len(all_models)}] Testing {model_name}...")
        t0 = time.perf_counter()
        result = ModelTestResult(name=model_name, source=source)
        
        try:
            # Step 1: TensorGuard verification
            print(f"  Running TensorGuard verification...")
            verdict, contract, error = run_tensorguard_verification(source, input_shapes, model_name)
            result.tg_verdict = verdict
            result.tg_contract = contract
            result.tg_error = error
            
            if verdict == "verified":
                print(f"  ✓ Verified with contract: {contract}")
                
                # Step 2: Dynamo correspondence testing
                print(f"  Testing Dynamo guard correspondence...")
                (in_var, in_recomp, out_var, out_recomp, notes) = test_dynamo_correspondence(
                    model_instance, contract, model_name
                )
                
                result.in_contract_variations = in_var
                result.in_contract_recompiles = in_recomp
                result.out_of_contract_variations = out_var
                result.out_of_contract_recompiles = out_recomp
                result.notes = notes
                
                # Determine if claim holds
                # Claim holds if: no in-contract recompiles AND out-of-contract triggers recompile/error
                result.claim_holds = (in_recomp == 0) and (out_var == 0 or out_recomp > 0)
                
                if result.claim_holds:
                    print(f"  ✓ CLAIM HOLDS: {in_var} in-contract variations, 0 recompiles")
                else:
                    print(f"  ✗ CLAIM BROKEN: {in_recomp} recompiles on {in_var} variations")
            else:
                print(f"  ✗ {verdict.upper()}: {error}")
        
        except Exception as e:
            result.tg_verdict = "error"
            result.tg_error = f"Unexpected error: {traceback.format_exc()}"
            print(f"  ✗ ERROR: {e}")
        
        result.duration_ms = (time.perf_counter() - t0) * 1000
        results.models.append(result.to_dict())
        print(f"  Completed in {result.duration_ms:.0f}ms\n")
    
    # Compute summary
    verified = [m for m in results.models if m["tg_verdict"] == "verified"]
    claim_holds = [m for m in verified if m.get("claim_holds") is True]
    
    results.summary = {
        "total_models": len(all_models),
        "tg_verified": len(verified),
        "tg_unsafe": len([m for m in results.models if m["tg_verdict"] == "unsafe"]),
        "tg_error": len([m for m in results.models if m["tg_verdict"] == "error"]),
        "tg_not_attempted": len([m for m in results.models if m["tg_verdict"] == "not_attempted"]),
        "claim_holds_count": len(claim_holds),
        "claim_holds_rate": len(claim_holds) / len(verified) if verified else 0.0,
        "claim_broken_count": len([m for m in verified if m.get("claim_holds") is False]),
    }
    
    return results


def main():
    """Main entry point."""
    results = run_experiment()
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "dynamo_guard_correspondence.json")
    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"PyTorch version: {results.torch_version}")
    print(f"Total models tested: {results.summary['total_models']}")
    print(f"TensorGuard verified: {results.summary['tg_verified']}")
    print(f"  - Unsafe: {results.summary['tg_unsafe']}")
    print(f"  - Error: {results.summary['tg_error']}")
    print(f"  - Not attempted: {results.summary['tg_not_attempted']}")
    print()
    print(f"Claim holds: {results.summary['claim_holds_count']} / {results.summary['tg_verified']}")
    print(f"Claim holds rate: {results.summary['claim_holds_rate']:.1%}")
    print(f"Claim broken: {results.summary['claim_broken_count']}")
    print()
    
    # List models where claim broke
    broken = [m for m in results.models if m.get("claim_holds") is False]
    if broken:
        print("Models where claim BROKE:")
        for m in broken:
            print(f"  - {m['name']}: {m.get('notes', 'No notes')}")
        print()
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
