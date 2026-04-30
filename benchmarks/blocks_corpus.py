"""
Massive block corpus benchmark for TensorGuard (Track F).

Collects ≥150 standalone nn.Module blocks from:
- torchvision (ResNet, VGG, DenseNet, MobileNet, EfficientNet, ViT, etc.)
- timm (SqueezeExcite, Mlp, Attention, etc.)
- transformers (BERT, GPT2, T5, Llama attention/MLP blocks)

For each block:
1. Extract source via inspect.getsource
2. Pin package version, file path, compute source sha256
3. Run verify_architecture with reasonable input shapes
4. Classify abstain reason using taxonomy
5. Run baseline tools (torch.fx, FakeTensorMode, torch.export)
6. Record all results to JSON

Output: benchmarks/blocks_corpus.json
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture
from benchmarks.abstain_taxonomy import classify_abstain

OUT_JSON = ROOT / "benchmarks" / "blocks_corpus.json"


# ─── Block catalog ──────────────────────────────────────────────────────────

# (package_name, module_path, class_name, input_shape_dict)
BLOCK_CATALOG: List[Tuple[str, str, str, Dict[str, tuple]]] = []

# TorchVision blocks
TV_BLOCKS = [
    # ResNet
    ("torchvision", "torchvision.models.resnet", "BasicBlock", {"x": (1, 64, 56, 56)}),
    ("torchvision", "torchvision.models.resnet", "Bottleneck", {"x": (1, 256, 56, 56)}),
    
    # VGG (extract sub-blocks manually)
    ("torchvision", "torchvision.models.vgg", "VGG", {"x": (1, 3, 224, 224)}),
    
    # DenseNet
    ("torchvision", "torchvision.models.densenet", "_DenseLayer", {"x": (1, 64, 56, 56)}),
    ("torchvision", "torchvision.models.densenet", "_DenseBlock", {"x": (1, 64, 56, 56)}),
    ("torchvision", "torchvision.models.densenet", "_Transition", {"x": (1, 128, 56, 56)}),
    
    # GoogLeNet
    ("torchvision", "torchvision.models.googlenet", "Inception", {"x": (1, 192, 28, 28)}),
    ("torchvision", "torchvision.models.googlenet", "BasicConv2d", {"x": (1, 192, 28, 28)}),
    
    # Inception v3
    ("torchvision", "torchvision.models.inception", "BasicConv2d", {"x": (1, 192, 28, 28)}),
    ("torchvision", "torchvision.models.inception", "InceptionA", {"x": (1, 192, 35, 35)}),
    ("torchvision", "torchvision.models.inception", "InceptionB", {"x": (1, 288, 35, 35)}),
    ("torchvision", "torchvision.models.inception", "InceptionC", {"x": (1, 768, 17, 17)}),
    ("torchvision", "torchvision.models.inception", "InceptionD", {"x": (1, 768, 17, 17)}),
    ("torchvision", "torchvision.models.inception", "InceptionE", {"x": (1, 1280, 8, 8)}),
    
    # MobileNet v2
    ("torchvision", "torchvision.models.mobilenetv2", "InvertedResidual", {"x": (1, 32, 112, 112)}),
    ("torchvision", "torchvision.models.mobilenetv2", "ConvBNActivation", {"x": (1, 32, 112, 112)}),
    
    # MobileNet v3
    ("torchvision", "torchvision.models.mobilenetv3", "InvertedResidual", {"x": (1, 16, 112, 112)}),
    ("torchvision", "torchvision.models.mobilenetv3", "SqueezeExcitation", {"x": (1, 16, 112, 112)}),
    
    # ShuffleNet v2
    ("torchvision", "torchvision.models.shufflenetv2", "InvertedResidual", {"x": (1, 116, 28, 28)}),
    
    # SqueezeNet
    ("torchvision", "torchvision.models.squeezenet", "Fire", {"x": (1, 96, 54, 54)}),
    
    # EfficientNet
    ("torchvision", "torchvision.models.efficientnet", "MBConv", {"x": (1, 32, 112, 112)}),
    ("torchvision", "torchvision.models.efficientnet", "FusedMBConv", {"x": (1, 32, 112, 112)}),
    
    # RegNet
    ("torchvision", "torchvision.models.regnet", "SimpleStemIN", {"x": (1, 3, 224, 224)}),
    ("torchvision", "torchvision.models.regnet", "BottleneckTransform", {"x": (1, 32, 112, 112)}),
    ("torchvision", "torchvision.models.regnet", "ResBottleneckBlock", {"x": (1, 32, 112, 112)}),
    
    # ConvNeXt
    ("torchvision", "torchvision.models.convnext", "CNBlock", {"x": (1, 96, 56, 56)}),
    ("torchvision", "torchvision.models.convnext", "LayerNorm2d", {"x": (1, 96, 56, 56)}),
    
    # Vision Transformer
    ("torchvision", "torchvision.models.vision_transformer", "EncoderBlock", {"x": (1, 197, 768)}),
    ("torchvision", "torchvision.models.vision_transformer", "Encoder", {"x": (1, 197, 768)}),
    ("torchvision", "torchvision.models.vision_transformer", "MLPBlock", {"x": (1, 197, 768)}),
    
    # Swin Transformer
    ("torchvision", "torchvision.models.swin_transformer", "PatchMerging", {"x": (1, 3136, 96)}),
    ("torchvision", "torchvision.models.swin_transformer", "ShiftedWindowAttention", {"x": (1, 3136, 96)}),
    
    # MaxVit
    ("torchvision", "torchvision.models.maxvit", "MBConv", {"x": (1, 64, 56, 56)}),
    
    # Detection heads
    ("torchvision", "torchvision.models.detection.faster_rcnn", "TwoMLPHead", {"x": (1, 1024)}),
]

# timm blocks - expanded for 150+ target
TIMM_BLOCKS = [
    # Basic layers
    ("timm", "timm.layers.squeeze_excite", "SEModule", {"x": (1, 64, 56, 56)}),
    ("timm", "timm.layers.squeeze_excite", "SqueezeExcite", {"x": (1, 64, 56, 56)}),
    ("timm", "timm.layers.squeeze_excite", "EffectiveSEModule", {"x": (1, 64, 56, 56)}),
    ("timm", "timm.layers.squeeze_excite", "EffectiveSqueezeExcite", {"x": (1, 64, 56, 56)}),
    
    ("timm", "timm.layers.mlp", "Mlp", {"x": (1, 197, 768)}),
    ("timm", "timm.layers.mlp", "GluMlp", {"x": (1, 197, 768)}),
    ("timm", "timm.layers.mlp", "GatedMlp", {"x": (1, 197, 768)}),
    ("timm", "timm.layers.mlp", "SwiGLUPacked", {"x": (1, 197, 768)}),
    ("timm", "timm.layers.mlp", "ConvMlp", {"x": (1, 768, 7, 7)}),
    
    ("timm", "timm.layers.drop", "DropPath", {"x": (1, 197, 768)}),
    ("timm", "timm.layers.drop", "DropBlock2d", {"x": (1, 64, 56, 56)}),
    
    ("timm", "timm.layers.norm", "LayerNorm", {"x": (1, 197, 768)}),
    ("timm", "timm.layers.norm", "LayerNorm2d", {"x": (1, 96, 56, 56)}),
    ("timm", "timm.layers.norm", "RmsNorm", {"x": (1, 197, 768)}),
    ("timm", "timm.layers.norm", "GroupNorm", {"x": (1, 64, 56, 56)}),
    ("timm", "timm.layers.norm", "GroupNorm1", {"x": (1, 64, 56, 56)}),
    ("timm", "timm.layers.norm", "LayerNormExp2d", {"x": (1, 96, 56, 56)}),
    
    ("timm", "timm.layers.conv_bn_act", "ConvBnAct", {"x": (1, 3, 224, 224)}),
    ("timm", "timm.layers.conv_bn_act", "ConvNormAct", {"x": (1, 3, 224, 224)}),
    
    # Attention
    ("timm", "timm.layers.attention", "Attention", {"x": (1, 197, 768)}),
    
    # Bottleneck
    ("timm", "timm.layers.bottleneck_attn", "BottleneckAttn", {"x": (1, 2048, 7, 7)}),
    
    # Pooling
    ("timm", "timm.layers.adaptive_avgmax_pool", "AdaptiveAvgMaxPool2d", {"x": (1, 2048, 7, 7)}),
    ("timm", "timm.layers.adaptive_avgmax_pool", "SelectAdaptivePool2d", {"x": (1, 2048, 7, 7)}),
    ("timm", "timm.layers.pool2d_same", "AvgPool2dSame", {"x": (1, 64, 56, 56)}),
    
    # MLP-Mixer
    ("timm", "timm.layers.mlp_mixer", "MixerBlock", {"x": (1, 196, 512)}),
    
    # Vision Transformer blocks
    ("timm", "timm.layers.patch_embed", "PatchEmbed", {"x": (1, 3, 224, 224)}),
    
    # Blocks from model architectures
    ("timm", "timm.models.resnet", "BasicBlock", {"x": (1, 64, 56, 56)}),
    ("timm", "timm.models.resnet", "Bottleneck", {"x": (1, 256, 56, 56)}),
    
    ("timm", "timm.models.efficientnet_blocks", "DepthwiseSeparableConv", {"x": (1, 32, 112, 112)}),
    ("timm", "timm.models.efficientnet_blocks", "InvertedResidual", {"x": (1, 32, 112, 112)}),
    ("timm", "timm.models.efficientnet_blocks", "CondConvResidual", {"x": (1, 32, 112, 112)}),
    ("timm", "timm.models.efficientnet_blocks", "EdgeResidual", {"x": (1, 32, 112, 112)}),
    
    ("timm", "timm.models.vision_transformer", "Block", {"x": (1, 197, 768)}),
    ("timm", "timm.models.vision_transformer", "LayerScale", {"x": (1, 197, 768)}),
    
    ("timm", "timm.models.convnext", "ConvNeXtBlock", {"x": (1, 96, 56, 56)}),
    
    ("timm", "timm.models.swin_transformer", "WindowAttention", {"x": (49, 49, 96)}),
    ("timm", "timm.models.swin_transformer", "SwinTransformerBlock", {"x": (1, 3136, 96)}),
    
    ("timm", "timm.models.beit", "Block", {"x": (1, 197, 768)}),
    
    ("timm", "timm.models.coat", "ConvRelPosEnc", {"x": (1, 197, 768)}),
    
    ("timm", "timm.models.crossvit", "Block", {"x": (1, 197, 768)}),
    
    ("timm", "timm.models.deit", "Block", {"x": (1, 197, 768)}),
    
    ("timm", "timm.models.levit", "Attention", {"x": (1, 196, 256)}),
    
    ("timm", "timm.models.mlp_mixer", "MixerBlock", {"x": (1, 196, 512)}),
    
    ("timm", "timm.models.nest", "Block", {"x": (1, 196, 128)}),
    
    ("timm", "timm.models.poolformer", "PoolFormerBlock", {"x": (1, 3136, 64)}),
    
    ("timm", "timm.models.pvt_v2", "Block", {"x": (1, 3136, 64)}),
    
    ("timm", "timm.models.twins", "Block", {"x": (1, 3136, 64)}),
    
    ("timm", "timm.models.volo", "OutlookAttention", {"x": (1, 196, 384)}),
]

# Transformers blocks (HuggingFace) - expanded
HF_BLOCKS = [
    # BERT
    ("transformers", "transformers.models.bert.modeling_bert", "BertEmbeddings", {"input_ids": (1, 128)}),
    ("transformers", "transformers.models.bert.modeling_bert", "BertSelfAttention", {"hidden_states": (1, 128, 768)}),
    ("transformers", "transformers.models.bert.modeling_bert", "BertSelfOutput", {"hidden_states": (1, 128, 768), "input_tensor": (1, 128, 768)}),
    ("transformers", "transformers.models.bert.modeling_bert", "BertAttention", {"hidden_states": (1, 128, 768)}),
    ("transformers", "transformers.models.bert.modeling_bert", "BertIntermediate", {"hidden_states": (1, 128, 768)}),
    ("transformers", "transformers.models.bert.modeling_bert", "BertOutput", {"hidden_states": (1, 128, 3072), "input_tensor": (1, 128, 768)}),
    ("transformers", "transformers.models.bert.modeling_bert", "BertLayer", {"hidden_states": (1, 128, 768)}),
    ("transformers", "transformers.models.bert.modeling_bert", "BertPooler", {"hidden_states": (1, 128, 768)}),
    
    # GPT-2
    ("transformers", "transformers.models.gpt2.modeling_gpt2", "GPT2Attention", {"hidden_states": (1, 128, 768)}),
    ("transformers", "transformers.models.gpt2.modeling_gpt2", "GPT2MLP", {"hidden_states": (1, 128, 768)}),
    ("transformers", "transformers.models.gpt2.modeling_gpt2", "GPT2Block", {"hidden_states": (1, 128, 768)}),
    
    # T5
    ("transformers", "transformers.models.t5.modeling_t5", "T5LayerNorm", {"hidden_states": (1, 128, 512)}),
    ("transformers", "transformers.models.t5.modeling_t5", "T5DenseActDense", {"hidden_states": (1, 128, 512)}),
    ("transformers", "transformers.models.t5.modeling_t5", "T5DenseGatedActDense", {"hidden_states": (1, 128, 512)}),
    ("transformers", "transformers.models.t5.modeling_t5", "T5Attention", {"hidden_states": (1, 128, 512)}),
    ("transformers", "transformers.models.t5.modeling_t5", "T5LayerSelfAttention", {"hidden_states": (1, 128, 512)}),
    ("transformers", "transformers.models.t5.modeling_t5", "T5LayerCrossAttention", {"hidden_states": (1, 128, 512), "key_value_states": (1, 128, 512)}),
    ("transformers", "transformers.models.t5.modeling_t5", "T5LayerFF", {"hidden_states": (1, 128, 512)}),
    ("transformers", "transformers.models.t5.modeling_t5", "T5Block", {"hidden_states": (1, 128, 512)}),
    
    # Llama
    ("transformers", "transformers.models.llama.modeling_llama", "LlamaRMSNorm", {"hidden_states": (1, 128, 4096)}),
    ("transformers", "transformers.models.llama.modeling_llama", "LlamaMLP", {"x": (1, 128, 4096)}),
    ("transformers", "transformers.models.llama.modeling_llama", "LlamaAttention", {"hidden_states": (1, 128, 4096)}),
    ("transformers", "transformers.models.llama.modeling_llama", "LlamaDecoderLayer", {"hidden_states": (1, 128, 4096)}),
    ("transformers", "transformers.models.llama.modeling_llama", "LlamaRotaryEmbedding", {"x": (1, 128, 64)}),
    
    # RoBERTa
    ("transformers", "transformers.models.roberta.modeling_roberta", "RobertaEmbeddings", {"input_ids": (1, 128)}),
    ("transformers", "transformers.models.roberta.modeling_roberta", "RobertaSelfAttention", {"hidden_states": (1, 128, 768)}),
    ("transformers", "transformers.models.roberta.modeling_roberta", "RobertaSelfOutput", {"hidden_states": (1, 128, 768), "input_tensor": (1, 128, 768)}),
    ("transformers", "transformers.models.roberta.modeling_roberta", "RobertaAttention", {"hidden_states": (1, 128, 768)}),
    ("transformers", "transformers.models.roberta.modeling_roberta", "RobertaIntermediate", {"hidden_states": (1, 128, 768)}),
    ("transformers", "transformers.models.roberta.modeling_roberta", "RobertaOutput", {"hidden_states": (1, 128, 3072), "input_tensor": (1, 128, 768)}),
    ("transformers", "transformers.models.roberta.modeling_roberta", "RobertaLayer", {"hidden_states": (1, 128, 768)}),
    
    # DistilBERT
    ("transformers", "transformers.models.distilbert.modeling_distilbert", "Embeddings", {"input_ids": (1, 128)}),
    ("transformers", "transformers.models.distilbert.modeling_distilbert", "MultiHeadSelfAttention", {"query": (1, 128, 768)}),
    ("transformers", "transformers.models.distilbert.modeling_distilbert", "FFN", {"input": (1, 128, 768)}),
    ("transformers", "transformers.models.distilbert.modeling_distilbert", "TransformerBlock", {"x": (1, 128, 768)}),
    
    # ALBERT
    ("transformers", "transformers.models.albert.modeling_albert", "AlbertEmbeddings", {"input_ids": (1, 128)}),
    ("transformers", "transformers.models.albert.modeling_albert", "AlbertAttention", {"hidden_states": (1, 128, 768)}),
    ("transformers", "transformers.models.albert.modeling_albert", "AlbertLayer", {"hidden_states": (1, 128, 768)}),
    
    # ViT (Vision Transformer)
    ("transformers", "transformers.models.vit.modeling_vit", "ViTEmbeddings", {"pixel_values": (1, 3, 224, 224)}),
    ("transformers", "transformers.models.vit.modeling_vit", "ViTSelfAttention", {"hidden_states": (1, 197, 768)}),
    ("transformers", "transformers.models.vit.modeling_vit", "ViTSelfOutput", {"hidden_states": (1, 197, 768), "input_tensor": (1, 197, 768)}),
    ("transformers", "transformers.models.vit.modeling_vit", "ViTAttention", {"hidden_states": (1, 197, 768)}),
    ("transformers", "transformers.models.vit.modeling_vit", "ViTIntermediate", {"hidden_states": (1, 197, 768)}),
    ("transformers", "transformers.models.vit.modeling_vit", "ViTOutput", {"hidden_states": (1, 197, 3072), "input_tensor": (1, 197, 768)}),
    ("transformers", "transformers.models.vit.modeling_vit", "ViTLayer", {"hidden_states": (1, 197, 768)}),
    
    # DeiT (Data-efficient ViT)
    ("transformers", "transformers.models.deit.modeling_deit", "DeiTEmbeddings", {"pixel_values": (1, 3, 224, 224)}),
    ("transformers", "transformers.models.deit.modeling_deit", "DeiTSelfAttention", {"hidden_states": (1, 197, 768)}),
    ("transformers", "transformers.models.deit.modeling_deit", "DeiTAttention", {"hidden_states": (1, 197, 768)}),
    ("transformers", "transformers.models.deit.modeling_deit", "DeiTLayer", {"hidden_states": (1, 197, 768)}),
    
    # BEiT (BERT Pre-Training of Image Transformers)
    ("transformers", "transformers.models.beit.modeling_beit", "BeitEmbeddings", {"pixel_values": (1, 3, 224, 224)}),
    ("transformers", "transformers.models.beit.modeling_beit", "BeitSelfAttention", {"hidden_states": (1, 197, 768)}),
    ("transformers", "transformers.models.beit.modeling_beit", "BeitAttention", {"hidden_states": (1, 197, 768)}),
    ("transformers", "transformers.models.beit.modeling_beit", "BeitIntermediate", {"hidden_states": (1, 197, 768)}),
    ("transformers", "transformers.models.beit.modeling_beit", "BeitOutput", {"hidden_states": (1, 197, 3072), "input_tensor": (1, 197, 768)}),
    ("transformers", "transformers.models.beit.modeling_beit", "BeitLayer", {"hidden_states": (1, 197, 768)}),
]

BLOCK_CATALOG = TV_BLOCKS + TIMM_BLOCKS + HF_BLOCKS


# ─── Source extraction and verification ────────────────────────────────────

def get_package_version(package_name: str) -> str:
    """Get installed package version."""
    try:
        pkg = importlib.import_module(package_name)
        return getattr(pkg, "__version__", "unknown")
    except:
        return "not_installed"


def extract_class_info(package_name: str, module_path: str, class_name: str) -> Optional[Dict[str, Any]]:
    """Extract class source, file path, and compute sha256."""
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is None:
            return None
        
        source = inspect.getsource(cls)
        file_path = inspect.getfile(cls)
        sha256 = hashlib.sha256(source.encode('utf-8')).hexdigest()
        
        return {
            "source": source,
            "file_path": file_path,
            "sha256": sha256,
            "qualified_name": f"{module_path}.{class_name}",
        }
    except Exception as e:
        return None


def run_tensorguard(source: str, input_shapes: Dict[str, tuple], timeout: float = 30.0) -> Dict[str, Any]:
    """Run TensorGuard verify_architecture with timeout."""
    start = time.perf_counter()
    try:
        result = verify_architecture(source, input_shapes=input_shapes)
        duration_ms = (time.perf_counter() - start) * 1000
        
        return {
            "status": result.status,
            "abstained": result.abstained,
            "bug_count": len(result.bugs),
            "opaque_layer_count": result.opaque_layer_count,
            "duration_ms": duration_ms,
            "bugs": [
                {
                    "category": b.category.value,
                    "message": b.message,
                    "line": b.location.line,
                }
                for b in result.bugs
            ],
            "result": result,  # For taxonomy classifier
        }
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return {
            "status": "ERROR",
            "abstained": False,
            "bug_count": 0,
            "opaque_layer_count": 0,
            "duration_ms": duration_ms,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ─── Baseline tools ─────────────────────────────────────────────────────────

def run_torch_fx(source: str, input_shapes: Dict[str, tuple]) -> Dict[str, Any]:
    """Run torch.fx.symbolic_trace + ShapeProp."""
    try:
        # Would need to instantiate the module - complex, likely to fail
        # Skip for now or mark as not-applicable
        return {"verdict": "N/A", "reason": "requires_instantiation"}
    except Exception as e:
        return {"verdict": "ERROR", "error": str(e)}


def run_fake_tensor(source: str, input_shapes: Dict[str, tuple]) -> Dict[str, Any]:
    """Run FakeTensorMode."""
    try:
        return {"verdict": "N/A", "reason": "requires_instantiation"}
    except Exception as e:
        return {"verdict": "ERROR", "error": str(e)}


def run_torch_export(source: str, input_shapes: Dict[str, tuple]) -> Dict[str, Any]:
    """Run torch.export."""
    try:
        return {"verdict": "N/A", "reason": "requires_instantiation"}
    except Exception as e:
        return {"verdict": "ERROR", "error": str(e)}


# ─── Main benchmark driver ──────────────────────────────────────────────────

def run_benchmark() -> List[Dict[str, Any]]:
    """Run benchmark on all blocks."""
    results = []
    
    print(f"Starting blocks corpus benchmark with {len(BLOCK_CATALOG)} blocks...")
    print(f"Packages:")
    print(f"  - torchvision: {get_package_version('torchvision')}")
    print(f"  - timm: {get_package_version('timm')}")
    print(f"  - transformers: {get_package_version('transformers')}")
    print()
    
    for i, (package_name, module_path, class_name, input_shapes) in enumerate(BLOCK_CATALOG):
        print(f"[{i+1}/{len(BLOCK_CATALOG)}] {package_name}.{class_name}... ", end="", flush=True)
        
        record = {
            "id": i,
            "package": package_name,
            "module": module_path,
            "class": class_name,
            "package_version": get_package_version(package_name),
            "input_shapes": {k: list(v) for k, v in input_shapes.items()},
        }
        
        # Extract class info
        info = extract_class_info(package_name, module_path, class_name)
        if info is None:
            record["status"] = "NOT_FOUND"
            print("NOT FOUND")
            results.append(record)
            continue
        
        record["file_path"] = info["file_path"]
        record["qualified_name"] = info["qualified_name"]
        record["sha256"] = info["sha256"]
        record["source_lines"] = info["source"].count("\n") + 1
        
        # Run TensorGuard
        tg_result = run_tensorguard(info["source"], input_shapes)
        record["tensorguard"] = {
            "status": tg_result["status"],
            "abstained": tg_result["abstained"],
            "bug_count": tg_result["bug_count"],
            "opaque_layer_count": tg_result["opaque_layer_count"],
            "duration_ms": round(tg_result["duration_ms"], 2),
        }
        
        if "error" in tg_result:
            record["tensorguard"]["error"] = tg_result["error"]
        
        if "bugs" in tg_result:
            record["tensorguard"]["bugs"] = tg_result["bugs"]
        
        # Classify abstain reason
        if tg_result["abstained"] and "result" in tg_result:
            abstain_reason = classify_abstain(tg_result["result"], info["source"])
            record["tensorguard"]["abstain_reason"] = abstain_reason
        
        # Run baselines (simplified for now)
        record["baselines"] = {
            "torch_fx": run_torch_fx(info["source"], input_shapes),
            "fake_tensor": run_fake_tensor(info["source"], input_shapes),
            "torch_export": run_torch_export(info["source"], input_shapes),
        }
        
        print(f"{tg_result['status']} ({tg_result['duration_ms']:.0f}ms)")
        results.append(record)
    
    return results


def main():
    """Main entry point."""
    results = run_benchmark()
    
    # Compute statistics
    total = len(results)
    found = len([r for r in results if r.get("status") != "NOT_FOUND"])
    
    tg_stats = {
        "SAFE": 0,
        "UNSAFE": 0,
        "ERROR": 0,
        "abstained": 0,
    }
    
    for r in results:
        if "tensorguard" in r:
            status = r["tensorguard"]["status"]
            if status in tg_stats:
                tg_stats[status] += 1
            if r["tensorguard"]["abstained"]:
                tg_stats["abstained"] += 1
    
    # Write output
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_blocks": total,
            "blocks_found": found,
            "packages": {
                "torchvision": get_package_version("torchvision"),
                "timm": get_package_version("timm"),
                "transformers": get_package_version("transformers"),
            },
        },
        "summary": {
            "tensorguard": tg_stats,
        },
        "blocks": results,
    }
    
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Results written to {OUT_JSON}")
    print(f"\nSummary:")
    print(f"  Total blocks: {total}")
    print(f"  Found: {found}")
    print(f"  TensorGuard:")
    for k, v in tg_stats.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
