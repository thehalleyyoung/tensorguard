"""
Modern PyTorch pattern benchmarks for TensorGuard evaluation.

10 benchmarks exercising torch.compile, mixed-precision, DDP, DataParallel,
data-dependent control flow, torch.export, torch.jit.script/trace, and
dynamic shapes — the patterns identified as gaps in the original evaluation.
"""

MODERN_PYTORCH_BENCHMARKS = {

    # =========================================================================
    # 1. torch.compile decorator on forward
    # =========================================================================
    "torch_compile_forward": {
        "source": """
import torch
import torch.nn as nn

class CompiledModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    @torch.compile
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 256)},
        "is_buggy": False,
        "category": "torch_compile",
        "expected_features": ["torch_compile_present", "torch_compile_forward"],
        "description": "Model with @torch.compile on forward — shapes are correct",
    },

    # =========================================================================
    # 2. torch.compile with dynamic=True and a shape bug
    # =========================================================================
    "torch_compile_dynamic_bug": {
        "source": """
import torch
import torch.nn as nn

compiled_model = None  # populated externally

class DynamicCompiledNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(128, 10)  # BUG: expects 128, fc1 outputs 256

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)

# compiled_model = torch.compile(DynamicCompiledNet(), dynamic=True)
""",
        "input_shapes": {"x": ("batch", 512)},
        "is_buggy": True,
        "category": "torch_compile",
        "expected_features": ["torch_compile_present", "torch_compile_dynamic_shapes"],
        "bug_description": "fc2 expects 128 features but fc1 outputs 256",
        "description": "torch.compile(dynamic=True) with Linear dimension mismatch",
    },

    # =========================================================================
    # 3. Mixed-precision autocast in forward
    # =========================================================================
    "autocast_forward": {
        "source": """
import torch
import torch.nn as nn

class AutocastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        with torch.amp.autocast('cuda'):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": False,
        "category": "mixed_precision",
        "expected_features": ["mixed_precision", "forward_uses_autocast"],
        "description": "Correct model using torch.amp.autocast in forward",
    },

    # =========================================================================
    # 4. torch.cuda.amp.autocast with GradScaler and shape bug
    # =========================================================================
    "autocast_gradscaler_bug": {
        "source": """
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # BUG: expects 64 in, conv1 outputs 32
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x = self.conv1(x)
            x = self.conv2(x)  # shape error: 32 channels vs 64 expected
            x = x.mean(dim=[2, 3])
            return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "mixed_precision",
        "expected_features": ["mixed_precision", "forward_uses_autocast", "grad_scaler"],
        "bug_description": "conv2 expects 64 input channels but conv1 outputs 32",
        "description": "Mixed-precision model with channel mismatch hidden by autocast",
    },

    # =========================================================================
    # 5. DistributedDataParallel wrapping
    # =========================================================================
    "ddp_wrapped_model": {
        "source": """
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

class DDPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# model = DistributedDataParallel(DDPModel().cuda(), device_ids=[0])
""",
        "input_shapes": {"x": ("batch", 784)},
        "is_buggy": False,
        "category": "distributed",
        "expected_features": ["distributed_ddp"],
        "description": "Correct model wrapped with DDP — shapes verified on single replica",
    },

    # =========================================================================
    # 6. DataParallel with shape bug
    # =========================================================================
    "data_parallel_bug": {
        "source": """
import torch
import torch.nn as nn

class DataParallelBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 10)  # BUG: should be 64

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# model = nn.DataParallel(DataParallelBug())
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "data_parallel",
        "expected_features": ["data_parallel"],
        "bug_description": "classifier expects 32 features but features block outputs 64",
        "description": "DataParallel model with Linear input dimension mismatch",
    },

    # =========================================================================
    # 7. Data-dependent control flow (value-dependent branching)
    # =========================================================================
    "data_dependent_control_flow": {
        "source": """
import torch
import torch.nn as nn
import torch.nn.functional as F

class DataDependentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc_high = nn.Linear(64, 10)
        self.fc_low = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if x.mean().item() > 0.5:
            return self.fc_high(x)
        else:
            return self.fc_low(x)
""",
        "input_shapes": {"x": ("batch", 128)},
        "is_buggy": False,
        "category": "data_dependent",
        "expected_features": ["data_dependent_branches"],
        "description": "Branching on tensor value — both paths have correct shapes",
    },

    # =========================================================================
    # 8. torch.jit.script model
    # =========================================================================
    "jit_script_model": {
        "source": """
import torch
import torch.nn as nn

class ScriptableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# scripted = torch.jit.script(ScriptableModel())
""",
        "input_shapes": {"x": ("batch", 100)},
        "is_buggy": False,
        "category": "jit",
        "expected_features": ["jit_script"],
        "description": "Model destined for torch.jit.script — correct shapes",
    },

    # =========================================================================
    # 9. torch.jit.trace with data-dependent flow (tracing pitfall)
    # =========================================================================
    "jit_trace_data_dependent": {
        "source": """
import torch
import torch.nn as nn

class TraceUnsafeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(64, 32)
        self.fc_b = nn.Linear(64, 16)  # different output size
        self.fc_out = nn.Linear(32, 10)  # BUG if fc_b path taken: 16 != 32

    def forward(self, x):
        if x.sum().item() > 0:
            x = torch.relu(self.fc_a(x))
        else:
            x = torch.relu(self.fc_b(x))  # output is 16, fc_out expects 32
        return self.fc_out(x)

# traced = torch.jit.trace(TraceUnsafeModel(), torch.randn(1, 64))
""",
        "input_shapes": {"x": ("batch", 64)},
        "is_buggy": True,
        "category": "jit",
        "expected_features": ["jit_trace", "data_dependent_branches"],
        "bug_description": "fc_b outputs 16 features but fc_out expects 32 — traced model only captures one path",
        "description": "torch.jit.trace on model with data-dependent branch — shape mismatch on alternative path",
    },

    # =========================================================================
    # 10. torch.export with shape-dependent reshape
    # =========================================================================
    "torch_export_reshape": {
        "source": """
import torch
import torch.nn as nn

class ExportableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# exported = torch.export.export(ExportableModel(), (torch.randn(1, 3, 32, 32),))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": False,
        "category": "export",
        "expected_features": ["torch_export", "shape_dependent_ops"],
        "description": "Correct model with torch.export and dynamic reshape",
    },
}
