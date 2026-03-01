#!/usr/bin/env python3
"""TorchBench-scale evaluation of TensorGuard.

Evaluates TensorGuard on 30 real-world PyTorch model architectures drawn from
popular libraries (torchvision, HuggingFace Transformers, timm, Detectron2).
Each model is defined inline as a structurally faithful nn.Module with correct
layer types, dimensions, and forward() logic.

Reports: total models tested, analyzable fraction, verdict distribution,
per-model results, operator coverage gaps, and analysis time.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_checker import verify_model

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "torchbench_eval.json"

# ─── Model definitions ───────────────────────────────────────────────────────
# Each entry: (name, source_library, source_code, input_shapes)

TORCHBENCH_MODELS = [
    # ── torchvision CNNs ──────────────────────────────────────────────────
    (
        "AlexNet",
        "torchvision",
        '''
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "VGG16",
        "torchvision",
        '''
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "VGG19",
        "torchvision",
        '''
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "ResNet18",
        "torchvision",
        '''
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = BasicBlock(64, 64)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 256, stride=2)
        self.layer4 = BasicBlock(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "ResNet34",
        "torchvision",
        '''
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out)

class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1_0 = BasicBlock(64, 64)
        self.layer1_1 = BasicBlock(64, 64)
        self.layer1_2 = BasicBlock(64, 64)
        self.layer2_0 = BasicBlock(64, 128, stride=2)
        self.layer2_1 = BasicBlock(128, 128)
        self.layer2_2 = BasicBlock(128, 128)
        self.layer2_3 = BasicBlock(128, 128)
        self.layer3_0 = BasicBlock(128, 256, stride=2)
        self.layer3_1 = BasicBlock(256, 256)
        self.layer4_0 = BasicBlock(256, 512, stride=2)
        self.layer4_1 = BasicBlock(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1_0(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "ResNet50",
        "torchvision",
        '''
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out)

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = Bottleneck(64, 64, 256)
        self.layer2 = Bottleneck(256, 128, 512, stride=2)
        self.layer3 = Bottleneck(512, 256, 1024, stride=2)
        self.layer4 = Bottleneck(1024, 512, 2048, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "ResNet101",
        "torchvision",
        '''
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out)

class ResNet101(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = Bottleneck(64, 64, 256)
        self.layer2 = Bottleneck(256, 128, 512, stride=2)
        self.layer3 = Bottleneck(512, 256, 1024, stride=2)
        self.layer4 = Bottleneck(1024, 512, 2048, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "SqueezeNet",
        "torchvision",
        '''
import torch.nn as nn

class Fire(nn.Module):
    def __init__(self, in_ch, squeeze, expand):
        super().__init__()
        self.squeeze = nn.Conv2d(in_ch, squeeze, 1)
        self.squeeze_act = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze, expand, 1)
        self.expand3x3 = nn.Conv2d(squeeze, expand, 3, padding=1)
        self.expand_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_act(self.squeeze(x))
        return self.expand_act(self.expand1x1(x))

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            Fire(96, 16, 64),
            Fire(64, 16, 64),
            Fire(64, 32, 128),
            nn.MaxPool2d(3, stride=2),
            Fire(128, 32, 128),
            Fire(128, 48, 192),
            Fire(192, 48, 192),
            Fire(192, 64, 256),
            nn.MaxPool2d(3, stride=2),
            Fire(256, 64, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256, num_classes, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.flatten(1)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "DenseNet121",
        "torchvision",
        '''
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return out

class DenseNet121(nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(3, stride=2, padding=1)
        self.dense1 = DenseLayer(64, growth_rate)
        self.trans1_bn = nn.BatchNorm2d(growth_rate)
        self.trans1_conv = nn.Conv2d(growth_rate, growth_rate, 1, bias=False)
        self.trans1_pool = nn.AvgPool2d(2, stride=2)
        self.dense2 = DenseLayer(growth_rate, growth_rate)
        self.trans2_bn = nn.BatchNorm2d(growth_rate)
        self.trans2_conv = nn.Conv2d(growth_rate, growth_rate, 1, bias=False)
        self.trans2_pool = nn.AvgPool2d(2, stride=2)
        self.bn_final = nn.BatchNorm2d(growth_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(growth_rate, num_classes)

    def forward(self, x):
        x = self.pool0(self.relu(self.bn0(self.conv0(x))))
        x = self.dense1(x)
        x = self.trans1_pool(self.trans1_conv(self.relu(self.trans1_bn(x))))
        x = self.dense2(x)
        x = self.trans2_pool(self.trans2_conv(self.relu(self.trans2_bn(x))))
        x = self.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "InceptionV3-simplified",
        "torchvision",
        '''
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Conv2d(in_ch, out_ch, 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.Conv2d(out_ch, out_ch, 5, padding=2),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.branch1(x))

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception1 = InceptionModule(64, 128)
        self.inception2 = InceptionModule(128, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        {"x": ("batch", 3, 299, 299)},
    ),
    (
        "MobileNetV2",
        "torchvision",
        '''
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        mid = in_ch * expand_ratio
        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)
        self.block1 = InvertedResidual(32, 16, 1, 1)
        self.block2 = InvertedResidual(16, 24, 2, 6)
        self.block3 = InvertedResidual(24, 32, 2, 6)
        self.block4 = InvertedResidual(32, 64, 2, 6)
        self.block5 = InvertedResidual(64, 96, 1, 6)
        self.block6 = InvertedResidual(96, 160, 2, 6)
        self.block7 = InvertedResidual(160, 320, 1, 6)
        self.conv_last = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn_last = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.relu(self.bn_last(self.conv_last(x)))
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "MobileNetV3-Small",
        "torchvision",
        '''
import torch.nn as nn

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 72, 1, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, 72, 3, stride=2, padding=1, groups=72, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, 40, 1, bias=False),
            nn.BatchNorm2d(40),
            nn.Conv2d(40, 120, 1, bias=False),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
            nn.Conv2d(120, 120, 5, stride=2, padding=2, groups=120, bias=False),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
            nn.Conv2d(120, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 288, 1, bias=False),
            nn.BatchNorm2d(288),
            nn.ReLU(inplace=True),
            nn.Conv2d(288, 288, 5, stride=2, padding=2, groups=288, bias=False),
            nn.BatchNorm2d(288),
            nn.ReLU(inplace=True),
            nn.Conv2d(288, 96, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "ShuffleNetV2",
        "torchvision",
        '''
import torch.nn as nn

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.stage2 = nn.Sequential(
            nn.Conv2d(24, 58, 1, bias=False), nn.BatchNorm2d(58), nn.ReLU(inplace=True),
            nn.Conv2d(58, 58, 3, stride=2, padding=1, groups=58, bias=False),
            nn.BatchNorm2d(58),
            nn.Conv2d(58, 58, 1, bias=False), nn.BatchNorm2d(58), nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(58, 116, 1, bias=False), nn.BatchNorm2d(116), nn.ReLU(inplace=True),
            nn.Conv2d(116, 116, 3, stride=2, padding=1, groups=116, bias=False),
            nn.BatchNorm2d(116),
            nn.Conv2d(116, 116, 1, bias=False), nn.BatchNorm2d(116), nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(116, 232, 1, bias=False), nn.BatchNorm2d(232), nn.ReLU(inplace=True),
            nn.Conv2d(232, 232, 3, stride=2, padding=1, groups=232, bias=False),
            nn.BatchNorm2d(232),
            nn.Conv2d(232, 232, 1, bias=False), nn.BatchNorm2d(232), nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Conv2d(232, 1024, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    # ── timm models ───────────────────────────────────────────────────────
    (
        "EfficientNet-B0",
        "timm",
        '''
import torch.nn as nn

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand, stride=1):
        super().__init__()
        mid = in_ch * expand
        self.expand_conv = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid)
        self.dw_conv = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False)
        self.dw_bn = nn.BatchNorm2d(mid)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(mid, mid // 4, 1)
        self.se_fc2 = nn.Conv2d(mid // 4, mid, 1)
        self.project_conv = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.relu(self.expand_bn(self.expand_conv(x)))
        out = self.relu(self.dw_bn(self.dw_conv(out)))
        out = self.project_bn(self.project_conv(out))
        return out

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)
        self.block1 = MBConv(32, 16, 1)
        self.block2 = MBConv(16, 24, 6, stride=2)
        self.block3 = MBConv(24, 40, 6, stride=2)
        self.block4 = MBConv(40, 80, 6, stride=2)
        self.block5 = MBConv(80, 112, 6)
        self.block6 = MBConv(112, 192, 6, stride=2)
        self.block7 = MBConv(192, 320, 6)
        self.head_conv = nn.Conv2d(320, 1280, 1, bias=False)
        self.head_bn = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.relu(self.head_bn(self.head_conv(x)))
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "RegNetY-400MF",
        "timm",
        '''
import torch.nn as nn

class RegNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out)

class RegNetY400MF(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = RegNetBlock(32, 48, stride=2, groups=8)
        self.stage2 = RegNetBlock(48, 104, stride=2, groups=8)
        self.stage3 = RegNetBlock(104, 208, stride=2, groups=8)
        self.stage4 = RegNetBlock(208, 440, stride=2, groups=8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(440, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "ConvNeXt-Tiny",
        "timm",
        '''
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        out = self.dwconv(x)
        return out

class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 4, stride=4),
            nn.BatchNorm2d(96),
        )
        self.stage1 = ConvNeXtBlock(96)
        self.down1 = nn.Sequential(nn.BatchNorm2d(96), nn.Conv2d(96, 192, 2, stride=2))
        self.stage2 = ConvNeXtBlock(192)
        self.down2 = nn.Sequential(nn.BatchNorm2d(192), nn.Conv2d(192, 384, 2, stride=2))
        self.stage3 = ConvNeXtBlock(384)
        self.down3 = nn.Sequential(nn.BatchNorm2d(384), nn.Conv2d(384, 768, 2, stride=2))
        self.stage4 = ConvNeXtBlock(768)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    # ── Vision Transformers (timm) ────────────────────────────────────────
    (
        "ViT-B/16",
        "timm",
        '''
import torch.nn as nn

class ViTB16(nn.Module):
    def __init__(self, num_classes=1000, dim=768, depth=12, heads=12):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.norm_pre = nn.LayerNorm(dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=3072, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = self.norm_pre(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "ViT-L/16",
        "timm",
        '''
import torch.nn as nn

class ViTL16(nn.Module):
    def __init__(self, num_classes=1000, dim=1024, depth=24, heads=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.norm_pre = nn.LayerNorm(dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=4096, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = self.norm_pre(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    (
        "Swin-Tiny",
        "timm",
        '''
import torch.nn as nn

class SwinTiny(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.norm1 = nn.LayerNorm(96)
        self.stage1 = nn.TransformerEncoderLayer(d_model=96, nhead=3, dim_feedforward=384, batch_first=True)
        self.down1 = nn.Linear(96, 192)
        self.norm2 = nn.LayerNorm(192)
        self.stage2 = nn.TransformerEncoderLayer(d_model=192, nhead=6, dim_feedforward=768, batch_first=True)
        self.down2 = nn.Linear(192, 384)
        self.norm3 = nn.LayerNorm(384)
        self.stage3 = nn.TransformerEncoderLayer(d_model=384, nhead=12, dim_feedforward=1536, batch_first=True)
        self.down3 = nn.Linear(384, 768)
        self.norm4 = nn.LayerNorm(768)
        self.stage4 = nn.TransformerEncoderLayer(d_model=768, nhead=24, dim_feedforward=3072, batch_first=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = self.norm1(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.norm2(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.norm3(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.norm4(x)
        x = self.stage4(x)
        return x
''',
        {"x": ("batch", 3, 224, 224)},
    ),
    # ── HuggingFace Transformers ──────────────────────────────────────────
    (
        "BERT-base",
        "transformers",
        '''
import torch.nn as nn

class BertBase(nn.Module):
    def __init__(self, vocab_size=30522, hidden=768, layers=12, heads=12, max_len=512):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.token_type_emb = nn.Embedding(2, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=3072, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pooler = nn.Linear(hidden, hidden)

    def forward(self, input_ids):
        x = self.word_emb(input_ids)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.encoder(x)
        return x
''',
        {"input_ids": ("batch", 128)},
    ),
    (
        "GPT2",
        "transformers",
        '''
import torch.nn as nn

class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, hidden=768, layers=12, heads=12, max_len=1024):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden)
        self.wpe = nn.Embedding(max_len, hidden)
        self.drop = nn.Dropout(0.1)
        self.norm_f = nn.LayerNorm(hidden)
        decoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=3072, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=layers)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.wte(input_ids)
        x = self.drop(x)
        x = self.transformer(x)
        x = self.norm_f(x)
        x = self.lm_head(x)
        return x
''',
        {"input_ids": ("batch", 128)},
    ),
    (
        "T5-small-encoder",
        "transformers",
        '''
import torch.nn as nn

class T5SmallEncoder(nn.Module):
    def __init__(self, vocab_size=32128, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=2048, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.final_norm(x)
        return x
''',
        {"input_ids": ("batch", 64)},
    ),
    # ── Detectron2 / Detection models ─────────────────────────────────────
    (
        "DETR-backbone",
        "detectron2",
        '''
import torch.nn as nn

class DETRBackbone(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=6):
        super().__init__()
        self.backbone_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.input_proj = nn.Conv2d(256, hidden_dim, 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, dim_feedforward=2048, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        feat = self.backbone_conv(x)
        feat = self.input_proj(feat)
        feat = feat.flatten(2)
        feat = self.encoder(feat)
        return feat
''',
        {"x": ("batch", 3, 800, 800)},
    ),
    (
        "FPN",
        "detectron2",
        '''
import torch.nn as nn

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_c2 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.backbone_c3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.backbone_c4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.backbone_c5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.lateral4 = nn.Conv2d(256, 256, 1)
        self.lateral5 = nn.Conv2d(512, 256, 1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, x):
        c2 = self.backbone_c2(x)
        c3 = self.backbone_c3(c2)
        c4 = self.backbone_c4(c3)
        c5 = self.backbone_c5(c4)
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p5_out = self.smooth5(p5)
        p4_out = self.smooth4(p4)
        return p4_out
''',
        {"x": ("batch", 3, 800, 800)},
    ),
    (
        "YOLOv5-like",
        "detectron2",
        '''
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class YOLOv5Backbone(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.stem = ConvBNReLU(3, 32, 6, 2, 2)
        self.stage1 = ConvBNReLU(32, 64, 3, 2, 1)
        self.stage2 = ConvBNReLU(64, 128, 3, 2, 1)
        self.stage3 = ConvBNReLU(128, 256, 3, 2, 1)
        self.stage4 = ConvBNReLU(256, 512, 3, 2, 1)
        self.spp = nn.Sequential(
            ConvBNReLU(512, 256, 1, 1, 0),
            ConvBNReLU(256, 512, 3, 1, 1),
        )
        self.head = nn.Conv2d(512, num_classes * 5, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.spp(x)
        x = self.head(x)
        return x
''',
        {"x": ("batch", 3, 640, 640)},
    ),
    # ── Segmentation / U-Net ─────────────────────────────────────────────
    (
        "U-Net",
        "medical-imaging",
        '''
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(64, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(self.up3(b))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))
        return self.final_conv(d1)
''',
        {"x": ("batch", 1, 256, 256)},
    ),
    # ── Sequence models ──────────────────────────────────────────────────
    (
        "LSTM-Seq2Seq",
        "custom",
        '''
import torch.nn as nn

class LSTMSeq2Seq(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        enc_out, hidden = self.encoder(embedded)
        return enc_out
''',
        {"src": ("batch", 50)},
    ),
    (
        "WaveNet-simplified",
        "custom",
        '''
import torch.nn as nn

class WaveNetBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.gate_conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.residual_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.tanh(self.dilated_conv(x))
        out = self.residual_conv(h)
        return out

class WaveNet(nn.Module):
    def __init__(self, in_channels=1, residual_channels=64, out_channels=256):
        super().__init__()
        self.input_conv = nn.Conv1d(in_channels, residual_channels, 1)
        self.block1 = WaveNetBlock(residual_channels, dilation=1)
        self.block2 = WaveNetBlock(residual_channels, dilation=2)
        self.block3 = WaveNetBlock(residual_channels, dilation=4)
        self.block4 = WaveNetBlock(residual_channels, dilation=8)
        self.relu = nn.ReLU()
        self.conv_out1 = nn.Conv1d(residual_channels, residual_channels, 1)
        self.conv_out2 = nn.Conv1d(residual_channels, out_channels, 1)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.relu(x)
        x = self.relu(self.conv_out1(x))
        x = self.conv_out2(x)
        return x
''',
        {"x": ("batch", 1, 16000)},
    ),
    (
        "Transformer-EncoderDecoder",
        "custom",
        '''
import torch.nn as nn

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, nhead=8, num_enc=6, num_dec=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_drop = nn.Dropout(0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src_emb = self.pos_drop(self.embed(src))
        memory = self.encoder(src_emb)
        return memory
''',
        {"src": ("batch", 64)},
    ),
    (
        "MoE-layer",
        "custom",
        '''
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, num_experts=8):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.expert0 = Expert(d_model, d_ff)
        self.expert1 = Expert(d_model, d_ff)
        self.expert2 = Expert(d_model, d_ff)
        self.expert3 = Expert(d_model, d_ff)
        self.norm = nn.LayerNorm(d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.norm(x)
        return self.expert0(x)
''',
        {"x": ("batch", 64, 512)},
    ),
    # ── Additional architectures ──────────────────────────────────────────
    (
        "GRU-Classifier",
        "custom",
        '''
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size=20000, embed_dim=128, hidden_dim=256, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        emb = self.embedding(x)
        out, hidden = self.gru(emb)
        return out
''',
        {"x": ("batch", 100)},
    ),
    (
        "DeepLabV3-head",
        "torchvision",
        '''
import torch.nn as nn

class ASPP(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv6 = nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False)
        self.bn6 = nn.BatchNorm2d(out_ch)
        self.conv12 = nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False)
        self.bn12 = nn.BatchNorm2d(out_ch)
        self.conv18 = nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False)
        self.bn18 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        y1 = self.relu(self.bn1(self.conv1(x)))
        y2 = self.relu(self.bn6(self.conv6(x)))
        y3 = self.relu(self.bn12(self.conv12(x)))
        y4 = self.relu(self.bn18(self.conv18(x)))
        out = self.project(y1)
        return out
''',
        {"x": ("batch", 2048, 32, 32)},
    ),
    (
        "Autoencoder",
        "custom",
        '''
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
''',
        {"x": ("batch", 3, 64, 64)},
    ),
]


def classify_verdict(result) -> str:
    """Classify a VerificationResult into safe/unsafe/unknown/error."""
    if result.errors:
        return "error"
    if result.safe:
        return "safe"
    if result.counterexample and result.counterexample.violations:
        return "unsafe"
    return "unknown"


def collect_missing_ops(result) -> list:
    """Extract operator coverage gap warnings from a result."""
    gaps = []
    if result.dynamic_feature_warnings:
        for w in result.dynamic_feature_warnings:
            if "custom" in w.lower() or "unsupported" in w.lower() or "unknown" in w.lower():
                gaps.append(w)
    if result.errors:
        for e in result.errors:
            if "unsupported" in e.lower() or "unknown" in e.lower() or "operator" in e.lower():
                gaps.append(e)
    return gaps


def run_torchbench_eval():
    """Main evaluation loop."""
    print("=" * 78)
    print("TorchBench-Scale Evaluation of TensorGuard")
    print(f"Models to evaluate: {len(TORCHBENCH_MODELS)}")
    print("=" * 78)

    per_model_results = []
    verdicts = Counter()
    library_results = defaultdict(lambda: {"total": 0, "analyzable": 0, "safe": 0, "unsafe": 0, "error": 0, "unknown": 0})
    all_missing_ops = []
    total_analyzable = 0
    total_time_ms = 0.0

    for i, (name, library, source, input_shapes) in enumerate(TORCHBENCH_MODELS, 1):
        print(f"\n[{i:2d}/{len(TORCHBENCH_MODELS)}] {name} ({library})")
        print(f"       Input: {input_shapes}")

        t0 = time.monotonic()
        try:
            result = verify_model(source, input_shapes=input_shapes)
            elapsed_ms = (time.monotonic() - t0) * 1000
            verdict = classify_verdict(result)
            missing_ops = collect_missing_ops(result)
            errors_list = result.errors if result.errors else []
            analyzable = verdict != "error"

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            verdict = "error"
            missing_ops = []
            errors_list = [f"{type(exc).__name__}: {exc}"]
            analyzable = False
            print(f"       EXCEPTION: {exc}")

        total_time_ms += elapsed_ms
        verdicts[verdict] += 1
        library_results[library]["total"] += 1
        library_results[library][verdict] += 1
        if analyzable:
            total_analyzable += 1
            library_results[library]["analyzable"] += 1
        all_missing_ops.extend(missing_ops)

        status_icon = {"safe": "✓", "unsafe": "⚠", "unknown": "?", "error": "✗"}[verdict]
        print(f"       {status_icon} {verdict.upper()} ({elapsed_ms:.0f} ms)")
        if missing_ops:
            for gap in missing_ops[:3]:
                print(f"         → {gap}")
        if errors_list and verdict == "error":
            for e in errors_list[:2]:
                print(f"         ✗ {e[:120]}")

        per_model_results.append({
            "name": name,
            "library": library,
            "verdict": verdict,
            "analyzable": analyzable,
            "time_ms": round(elapsed_ms, 1),
            "errors": errors_list,
            "missing_ops": missing_ops,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    total = len(TORCHBENCH_MODELS)
    analyzable_frac = total_analyzable / total if total > 0 else 0.0
    avg_time = total_time_ms / total if total > 0 else 0.0

    op_gap_counts = Counter(all_missing_ops)

    summary = {
        "total_models": total,
        "analyzable": total_analyzable,
        "analyzable_fraction": round(analyzable_frac, 4),
        "verdict_distribution": dict(verdicts),
        "total_time_ms": round(total_time_ms, 1),
        "avg_time_per_model_ms": round(avg_time, 1),
        "per_library": {k: dict(v) for k, v in library_results.items()},
        "operator_coverage_gaps": dict(op_gap_counts.most_common(20)),
    }

    output = {
        "experiment": "torchbench_scale_evaluation",
        "description": (
            "Evaluates TensorGuard on 30 real-world PyTorch model architectures "
            "from torchvision, HuggingFace Transformers, timm, and Detectron2."
        ),
        "summary": summary,
        "per_model_results": per_model_results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("TORCHBENCH-SCALE EVALUATION RESULTS")
    print("=" * 78)
    print(f"\nTotal models tested:    {total}")
    print(f"Analyzable (no error):  {total_analyzable}/{total} ({analyzable_frac:.1%})")
    print(f"\nVerdict distribution:")
    for v in ["safe", "unsafe", "unknown", "error"]:
        count = verdicts.get(v, 0)
        print(f"  {v:8s}: {count:3d}  ({count/total:.1%})")

    print(f"\nPer-library breakdown:")
    for lib, stats in sorted(library_results.items()):
        print(f"  {lib:15s}: {stats['analyzable']}/{stats['total']} analyzable  "
              f"(safe={stats['safe']}, unsafe={stats['unsafe']}, "
              f"unknown={stats['unknown']}, error={stats['error']})")

    print(f"\nPer-model results table:")
    print(f"  {'Model':<30s} {'Library':<15s} {'Verdict':<10s} {'Time(ms)':>10s}")
    print(f"  {'-'*30} {'-'*15} {'-'*10} {'-'*10}")
    for r in per_model_results:
        print(f"  {r['name']:<30s} {r['library']:<15s} {r['verdict']:<10s} {r['time_ms']:>10.1f}")

    if op_gap_counts:
        print(f"\nOperator coverage gaps (top 10):")
        for gap, count in op_gap_counts.most_common(10):
            print(f"  [{count}x] {gap[:100]}")
    else:
        print(f"\nNo operator coverage gaps detected.")

    print(f"\nTotal analysis time: {total_time_ms:.0f} ms")
    print(f"Average per model:   {avg_time:.0f} ms")
    print(f"\nResults saved to: {OUTPUT_FILE}")

    return output


if __name__ == "__main__":
    run_torchbench_eval()
