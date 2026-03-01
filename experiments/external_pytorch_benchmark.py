"""
External Benchmark Suite for TensorGuard.

Real PyTorch nn.Module architectures from well-known papers/repos with
independently verifiable shape bugs.  Every model is a faithful
reproduction of the *architecture* described in the cited paper; buggy
variants inject a single, realistic dimension error that would cause a
RuntimeError at inference time.

Sources:
  - ResNet: He et al., "Deep Residual Learning" (CVPR 2015)
  - VGG: Simonyan & Zisserman, "Very Deep Convolutional Networks" (ICLR 2015)
  - BERT: Devlin et al., "BERT" (NAACL 2019)
  - GPT-2: Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
  - U-Net: Ronneberger et al., "U-Net" (MICCAI 2015)
  - DCGAN: Radford et al., "Unsupervised Representation Learning with DCGANs" (ICLR 2016)
  - WaveNet: van den Oord et al., "WaveNet" (2016)
  - MobileNetV2: Sandler et al., "MobileNetV2" (CVPR 2018)
"""

# Each entry:
#   name          – unique id
#   source        – nn.Module source code string
#   input_shapes  – dict passed to verify_model
#   is_buggy      – ground-truth label
#   category      – architecture family
#   description   – what the model is / what the bug is

EXTERNAL_PYTORCH_BENCHMARKS = {

    # =====================================================================
    #  1. ResNet-18 – correct
    # =====================================================================
    "resnet18_correct": {
        "source": """
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = BasicBlock(64, 64)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1000)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": False,
        "category": "ResNet",
        "description": "ResNet-18 (He et al. 2015) – two BasicBlocks, correct FC after adaptive pool",
    },

    # =====================================================================
    #  2. ResNet-18 – buggy (FC input mismatch after pool)
    # =====================================================================
    "resnet18_fc_bug": {
        "source": """
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out)

class ResNet18Bug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = BasicBlock(64, 64)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)  # BUG: should be 128 (layer2 out_ch)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": True,
        "category": "ResNet",
        "description": "ResNet-18 BUG: FC expects 256 features but AdaptiveAvgPool2d(1,1) on 128 channels yields 128",
    },

    # =====================================================================
    #  3. ResNet-50 bottleneck – correct
    # =====================================================================
    "resnet50_bottleneck_correct": {
        "source": """
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out)

class ResNet50Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.block1 = Bottleneck(64, 64, 256)
        self.block2 = Bottleneck(256, 128, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": False,
        "category": "ResNet",
        "description": "ResNet-50 bottleneck (He et al. 2015) – correct 1x1→3x3→1x1 pipeline",
    },

    # =====================================================================
    #  4. ResNet-50 bottleneck – buggy (1x1 output mismatch)
    # =====================================================================
    "resnet50_bottleneck_bug": {
        "source": """
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out)

class ResNet50StageBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.block1 = Bottleneck(64, 64, 256)
        self.block2 = Bottleneck(128, 128, 512, stride=2)  # BUG: in_ch=128 but block1 outputs 256
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": True,
        "category": "ResNet",
        "description": "ResNet-50 BUG: block2 expects 128 input channels but block1 outputs 256",
    },

    # =====================================================================
    #  5. VGG-16 – correct
    # =====================================================================
    "vgg16_correct": {
        "source": """
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": False,
        "category": "VGG",
        "description": "VGG-16 first 4 conv layers (Simonyan & Zisserman 2015) – correct flatten dim",
    },

    # =====================================================================
    #  6. VGG-16 – buggy (wrong flatten dim for different input size)
    # =====================================================================
    "vgg16_cifar_bug": {
        "source": """
import torch.nn as nn

class VGG16CIFARBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 4096),  # BUG: computed for 224x224 but input is 32x32
            nn.ReLU(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "VGG",
        "description": "VGG-16 BUG: classifier expects 128*56*56=401408 but 32x32 input → 128*8*8=8192 after pooling",
    },

    # =====================================================================
    #  7. BERT encoder block – correct
    # =====================================================================
    "bert_encoder_correct": {
        "source": """
import torch.nn as nn

class BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(30522, 768)
        self.layernorm1 = nn.LayerNorm(768)
        self.attention = nn.MultiheadAttention(768, 12, batch_first=True)
        self.layernorm2 = nn.LayerNorm(768)
        self.ffn_up = nn.Linear(768, 3072)
        self.ffn_down = nn.Linear(3072, 768)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layernorm1(self.embedding(x))
        attn_out, _ = self.attention(x, x, x)
        x = self.layernorm2(x + attn_out)
        ffn_out = self.ffn_down(self.relu(self.ffn_up(x)))
        return x + ffn_out
""",
        "input_shapes": {"x": ("batch", 128, 30522)},
        "is_buggy": False,
        "category": "BERT",
        "description": "BERT encoder block (Devlin et al. 2019) – correct attention + FFN with residuals",
    },

    # =====================================================================
    #  8. BERT encoder – buggy (FFN down projection input dim)
    # =====================================================================
    "bert_encoder_ffn_bug": {
        "source": """
import torch.nn as nn

class BERTEncoderFFNBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(30522, 768)
        self.layernorm1 = nn.LayerNorm(768)
        self.attention = nn.MultiheadAttention(768, 12, batch_first=True)
        self.layernorm2 = nn.LayerNorm(768)
        self.ffn_up = nn.Linear(768, 3072)
        self.ffn_down = nn.Linear(768, 768)  # BUG: input should be 3072 not 768
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layernorm1(self.embedding(x))
        attn_out, _ = self.attention(x, x, x)
        x = self.layernorm2(x + attn_out)
        ffn_out = self.ffn_down(self.relu(self.ffn_up(x)))
        return x + ffn_out
""",
        "input_shapes": {"x": ("batch", 128, 30522)},
        "is_buggy": True,
        "category": "BERT",
        "description": "BERT BUG: ffn_down expects 768 but ffn_up outputs 3072",
    },

    # =====================================================================
    #  9. BERT pooler – correct
    # =====================================================================
    "bert_pooler_correct": {
        "source": """
import torch.nn as nn

class BERTPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(512, 768)
        self.layernorm = nn.LayerNorm(768)
        self.attention = nn.MultiheadAttention(768, 12, batch_first=True)
        self.pooler_dense = nn.Linear(768, 768)
        self.pooler_act = nn.Tanh()

    def forward(self, x):
        x = self.layernorm(self.embedding(x))
        x, _ = self.attention(x, x, x)
        cls_token = x[:, 0, :]
        return self.pooler_act(self.pooler_dense(cls_token))
""",
        "input_shapes": {"x": ("batch", 128, 512)},
        "is_buggy": False,
        "category": "BERT",
        "description": "BERT pooler (Devlin et al. 2019) – correct CLS token extraction and dense",
    },

    # =====================================================================
    # 10. BERT pooler – buggy (pooler dense input dim)
    # =====================================================================
    "bert_pooler_bug": {
        "source": """
import torch.nn as nn

class BERTPoolerBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(512, 768)
        self.layernorm = nn.LayerNorm(768)
        self.attention = nn.MultiheadAttention(768, 12, batch_first=True)
        self.pooler_dense = nn.Linear(512, 768)  # BUG: should be 768 (attention output)
        self.pooler_act = nn.Tanh()

    def forward(self, x):
        x = self.layernorm(self.embedding(x))
        x, _ = self.attention(x, x, x)
        cls_token = x[:, 0, :]
        return self.pooler_act(self.pooler_dense(cls_token))
""",
        "input_shapes": {"x": ("batch", 128, 512)},
        "is_buggy": True,
        "category": "BERT",
        "description": "BERT pooler BUG: pooler_dense expects 512 but cls_token is 768-dim from attention",
    },

    # =====================================================================
    # 11. GPT-2 decoder block – correct
    # =====================================================================
    "gpt2_decoder_correct": {
        "source": """
import torch.nn as nn

class GPT2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(768, 12, batch_first=True)
        self.ln2 = nn.LayerNorm(768)
        self.mlp_up = nn.Linear(768, 3072)
        self.mlp_down = nn.Linear(3072, 768)
        self.gelu = nn.GELU()

    def forward(self, x):
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        normed2 = self.ln2(x)
        mlp_out = self.mlp_down(self.gelu(self.mlp_up(normed2)))
        return x + mlp_out
""",
        "input_shapes": {"x": ("batch", 1024, 768)},
        "is_buggy": False,
        "category": "GPT-2",
        "description": "GPT-2 decoder block (Radford et al. 2019) – correct pre-norm + MLP + residuals",
    },

    # =====================================================================
    # 12. GPT-2 decoder – buggy (MLP down projection mismatch)
    # =====================================================================
    "gpt2_decoder_mlp_bug": {
        "source": """
import torch.nn as nn

class GPT2BlockBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(768, 12, batch_first=True)
        self.ln2 = nn.LayerNorm(768)
        self.mlp_up = nn.Linear(768, 3072)
        self.mlp_down = nn.Linear(768, 768)  # BUG: input should be 3072
        self.gelu = nn.GELU()

    def forward(self, x):
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        normed2 = self.ln2(x)
        mlp_out = self.mlp_down(self.gelu(self.mlp_up(normed2)))
        return x + mlp_out
""",
        "input_shapes": {"x": ("batch", 1024, 768)},
        "is_buggy": True,
        "category": "GPT-2",
        "description": "GPT-2 BUG: mlp_down expects 768 features but mlp_up outputs 3072",
    },

    # =====================================================================
    # 13. GPT-2 LM head – correct
    # =====================================================================
    "gpt2_lm_head_correct": {
        "source": """
import torch.nn as nn

class GPT2LMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(768, 12, batch_first=True)
        self.mlp_up = nn.Linear(768, 3072)
        self.mlp_down = nn.Linear(3072, 768)
        self.ln_f = nn.LayerNorm(768)
        self.lm_head = nn.Linear(768, 50257)
        self.gelu = nn.GELU()

    def forward(self, x):
        normed = self.ln(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        mlp_out = self.mlp_down(self.gelu(self.mlp_up(x)))
        x = x + mlp_out
        return self.lm_head(self.ln_f(x))
""",
        "input_shapes": {"x": ("batch", 1024, 768)},
        "is_buggy": False,
        "category": "GPT-2",
        "description": "GPT-2 LM head (Radford et al. 2019) – correct final projection to vocab",
    },

    # =====================================================================
    # 14. GPT-2 LM head – buggy (lm_head input dim)
    # =====================================================================
    "gpt2_lm_head_bug": {
        "source": """
import torch.nn as nn

class GPT2LMHeadBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(768, 12, batch_first=True)
        self.mlp_up = nn.Linear(768, 3072)
        self.mlp_down = nn.Linear(3072, 768)
        self.ln_f = nn.LayerNorm(768)
        self.lm_head = nn.Linear(1024, 50257)  # BUG: input should be 768 not 1024
        self.gelu = nn.GELU()

    def forward(self, x):
        normed = self.ln(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        mlp_out = self.mlp_down(self.gelu(self.mlp_up(x)))
        x = x + mlp_out
        return self.lm_head(self.ln_f(x))
""",
        "input_shapes": {"x": ("batch", 1024, 768)},
        "is_buggy": True,
        "category": "GPT-2",
        "description": "GPT-2 LM head BUG: lm_head expects 1024 (seq_len) but receives 768 (hidden_dim)",
    },

    # =====================================================================
    # 15. U-Net encoder – correct
    # =====================================================================
    "unet_encoder_correct": {
        "source": """
import torch.nn as nn

class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1_conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bottleneck2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.relu(self.enc1_conv2(self.relu(self.enc1_conv1(x))))
        p1 = self.pool1(e1)
        e2 = self.relu(self.enc2_conv2(self.relu(self.enc2_conv1(p1))))
        p2 = self.pool2(e2)
        b = self.relu(self.bottleneck2(self.relu(self.bottleneck1(p2))))
        return b
""",
        "input_shapes": {"x": ("batch", 1, 256, 256)},
        "is_buggy": False,
        "category": "U-Net",
        "description": "U-Net encoder (Ronneberger et al. 2015) – correct double-conv + pool",
    },

    # =====================================================================
    # 16. U-Net encoder – buggy (conv channel mismatch)
    # =====================================================================
    "unet_encoder_bug": {
        "source": """
import torch.nn as nn

class UNetEncoderBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1_conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck1 = nn.Conv2d(64, 256, 3, padding=1)  # BUG: in_ch=64 but pool2 output is 128
        self.bottleneck2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.relu(self.enc1_conv2(self.relu(self.enc1_conv1(x))))
        p1 = self.pool1(e1)
        e2 = self.relu(self.enc2_conv2(self.relu(self.enc2_conv1(p1))))
        p2 = self.pool2(e2)
        b = self.relu(self.bottleneck2(self.relu(self.bottleneck1(p2))))
        return b
""",
        "input_shapes": {"x": ("batch", 1, 256, 256)},
        "is_buggy": True,
        "category": "U-Net",
        "description": "U-Net encoder BUG: bottleneck1 expects 64 input channels but pool2 outputs 128",
    },

    # =====================================================================
    # 17. U-Net decoder – correct
    # =====================================================================
    "unet_decoder_correct": {
        "source": """
import torch.nn as nn

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1_conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.dec1_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec2_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.final = nn.Conv2d(64, 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        u1 = self.up1(x)
        d1 = self.relu(self.dec1_conv2(self.relu(self.dec1_conv1(u1))))
        u2 = self.up2(d1)
        d2 = self.relu(self.dec2_conv2(self.relu(self.dec2_conv1(u2))))
        return self.final(d2)
""",
        "input_shapes": {"x": ("batch", 256, 64, 64)},
        "is_buggy": False,
        "category": "U-Net",
        "description": "U-Net decoder (Ronneberger et al. 2015) – correct transposed conv upsampling",
    },

    # =====================================================================
    # 18. U-Net decoder – buggy (transposed conv output channel mismatch)
    # =====================================================================
    "unet_decoder_bug": {
        "source": """
import torch.nn as nn

class UNetDecoderBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1_conv1 = nn.Conv2d(256, 128, 3, padding=1)  # BUG: in_ch=256 but up1 outputs 128
        self.dec1_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec2_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.final = nn.Conv2d(64, 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        u1 = self.up1(x)
        d1 = self.relu(self.dec1_conv2(self.relu(self.dec1_conv1(u1))))
        u2 = self.up2(d1)
        d2 = self.relu(self.dec2_conv2(self.relu(self.dec2_conv1(u2))))
        return self.final(d2)
""",
        "input_shapes": {"x": ("batch", 256, 64, 64)},
        "is_buggy": True,
        "category": "U-Net",
        "description": "U-Net decoder BUG: dec1_conv1 expects 256 ch but up1 outputs only 128",
    },

    # =====================================================================
    # 19. DCGAN generator – correct
    # =====================================================================
    "dcgan_generator_correct": {
        "source": """
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(100, 512 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.relu(self.project(z))
        x = x.view(x.size(0), 512, 4, 4)
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        return self.tanh(self.deconv3(x))
""",
        "input_shapes": {"z": ("batch", 100)},
        "is_buggy": False,
        "category": "DCGAN",
        "description": "DCGAN generator (Radford et al. 2016) – correct latent→image upsampling",
    },

    # =====================================================================
    # 20. DCGAN generator – buggy (project dim mismatch)
    # =====================================================================
    "dcgan_generator_bug": {
        "source": """
import torch.nn as nn

class DCGANGeneratorBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(100, 256 * 4 * 4)  # BUG: projects to 256 channels, not 512
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.relu(self.project(z))
        x = x.view(x.size(0), 512, 4, 4)
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        return self.tanh(self.deconv3(x))
""",
        "input_shapes": {"z": ("batch", 100)},
        "is_buggy": True,
        "category": "DCGAN",
        "description": "DCGAN generator BUG: project outputs 256*4*4=4096 but view reshapes to 512*4*4=8192",
    },

    # =====================================================================
    # 21. DCGAN discriminator – correct
    # =====================================================================
    "dcgan_discriminator_correct": {
        "source": """
import torch.nn as nn

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 4 * 4, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.sigmoid(self.fc(x))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": False,
        "category": "DCGAN",
        "description": "DCGAN discriminator (Radford et al. 2016) – correct strided conv chain",
    },

    # =====================================================================
    # 22. DCGAN discriminator – buggy (FC expects wrong spatial)
    # =====================================================================
    "dcgan_discriminator_bug": {
        "source": """
import torch.nn as nn

class DCGANDiscriminatorBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 8 * 8, 1)  # BUG: spatial is 4x4 not 8x8 after 3 stride-2 convs
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.sigmoid(self.fc(x))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "DCGAN",
        "description": "DCGAN discriminator BUG: FC expects 256*8*8 but conv chain 32→16→8→4 yields 256*4*4",
    },

    # =====================================================================
    # 23. WaveNet causal conv – correct
    # =====================================================================
    "wavenet_correct": {
        "source": """
import torch.nn as nn

class WaveNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.causal_conv = nn.Conv1d(256, 256, kernel_size=2, dilation=1, padding=1)
        self.gate_conv = nn.Conv1d(256, 256, kernel_size=2, dilation=1, padding=1)
        self.residual_conv = nn.Conv1d(256, 256, 1)
        self.skip_conv = nn.Conv1d(256, 512, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.causal_conv(x)
        g = self.gate_conv(x)
        z = self.tanh(h) * self.sigmoid(g)
        s = self.skip_conv(z)
        r = self.residual_conv(z) + x
        return r, s
""",
        "input_shapes": {"x": ("batch", 256, 16000)},
        "is_buggy": False,
        "category": "WaveNet",
        "description": "WaveNet dilated block (van den Oord et al. 2016) – correct gated activation",
    },

    # =====================================================================
    # 24. WaveNet – buggy (residual conv channel mismatch)
    # =====================================================================
    "wavenet_residual_bug": {
        "source": """
import torch.nn as nn

class WaveNetBlockBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.causal_conv = nn.Conv1d(256, 256, kernel_size=2, dilation=1, padding=1)
        self.gate_conv = nn.Conv1d(256, 256, kernel_size=2, dilation=1, padding=1)
        self.residual_conv = nn.Conv1d(128, 256, 1)  # BUG: input should be 256 not 128
        self.skip_conv = nn.Conv1d(256, 512, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.causal_conv(x)
        g = self.gate_conv(x)
        z = self.tanh(h) * self.sigmoid(g)
        s = self.skip_conv(z)
        r = self.residual_conv(z) + x
        return r, s
""",
        "input_shapes": {"x": ("batch", 256, 16000)},
        "is_buggy": True,
        "category": "WaveNet",
        "description": "WaveNet BUG: residual_conv expects 128 ch but gated output z has 256 ch",
    },

    # =====================================================================
    # 25. MobileNetV2 inverted residual – correct
    # =====================================================================
    "mobilenetv2_correct": {
        "source": """
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(32, 192, 1)
        self.bn1 = nn.BatchNorm2d(192)
        self.depthwise = nn.Conv2d(192, 192, 3, padding=1, groups=192)
        self.bn2 = nn.BatchNorm2d(192)
        self.project = nn.Conv2d(192, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        out = self.relu6(self.bn1(self.expand(x)))
        out = self.relu6(self.bn2(self.depthwise(out)))
        out = self.bn3(self.project(out))
        return out
""",
        "input_shapes": {"x": ("batch", 32, 56, 56)},
        "is_buggy": False,
        "category": "MobileNetV2",
        "description": "MobileNetV2 inverted residual (Sandler et al. 2018) – correct expand→DW→project",
    },

    # =====================================================================
    # 26. MobileNetV2 – buggy (depthwise groups mismatch)
    # =====================================================================
    "mobilenetv2_dw_bug": {
        "source": """
import torch.nn as nn

class InvertedResidualBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(32, 192, 1)
        self.bn1 = nn.BatchNorm2d(192)
        self.depthwise = nn.Conv2d(192, 192, 3, padding=1, groups=32)  # BUG: groups=32, should be 192
        self.bn2 = nn.BatchNorm2d(192)
        self.project = nn.Conv2d(192, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        out = self.relu6(self.bn1(self.expand(x)))
        out = self.relu6(self.bn2(self.depthwise(out)))
        out = self.bn3(self.project(out))
        return out
""",
        "input_shapes": {"x": ("batch", 32, 56, 56)},
        "is_buggy": True,
        "category": "MobileNetV2",
        "description": "MobileNetV2 BUG: depthwise groups=32 but channels=192 (groups should equal channels for DW conv)",
    },

    # =====================================================================
    # 27. Simple MLP correct
    # =====================================================================
    "mlp_mnist_correct": {
        "source": """
import torch.nn as nn

class MLPMNIST(nn.Module):
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
""",
        "input_shapes": {"x": ("batch", 784)},
        "is_buggy": False,
        "category": "MLP",
        "description": "Classic MNIST MLP – correct 784→256→128→10 chain",
    },

    # =====================================================================
    # 28. Simple MLP – buggy (chain break)
    # =====================================================================
    "mlp_mnist_bug": {
        "source": """
import torch.nn as nn

class MLPMNISTBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 64)  # BUG: input should be 256 not 128
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 784)},
        "is_buggy": True,
        "category": "MLP",
        "description": "MLP BUG: fc2 expects 128 features but fc1 outputs 256",
    },

    # =====================================================================
    # 29. AlexNet-style – correct
    # =====================================================================
    "alexnet_correct": {
        "source": """
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": False,
        "category": "AlexNet",
        "description": "AlexNet (Krizhevsky et al. 2012) – correct feature chain + classifier",
    },

    # =====================================================================
    # 30. AlexNet – buggy (classifier expects wrong flattened dim)
    # =====================================================================
    "alexnet_classifier_bug": {
        "source": """
import torch.nn as nn

class AlexNetBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),  # BUG: spatial is 6x6 not 7x7
            nn.ReLU(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": True,
        "category": "AlexNet",
        "description": "AlexNet BUG: classifier expects 256*7*7=12544 but features produce 256*6*6=9216",
    },

    # =====================================================================
    # 31. Transformer encoder – correct
    # =====================================================================
    "transformer_encoder_correct": {
        "source": """
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(512, 256)
        self.ln1 = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.ln2 = nn.LayerNorm(256)
        self.ffn1 = nn.Linear(256, 1024)
        self.ffn2 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.ln1(self.embedding(x))
        attn_out, _ = self.attn(x, x, x)
        x = self.ln2(x + attn_out)
        return x + self.ffn2(self.relu(self.ffn1(x)))
""",
        "input_shapes": {"x": ("batch", 64, 512)},
        "is_buggy": False,
        "category": "Transformer",
        "description": "Standard transformer encoder block – correct attention + FFN",
    },

    # =====================================================================
    # 32. Transformer encoder – buggy (attention embed_dim vs head count)
    # =====================================================================
    "transformer_encoder_head_bug": {
        "source": """
import torch.nn as nn

class TransformerEncoderHeadBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(512, 256)
        self.ln1 = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(256, 7, batch_first=True)  # BUG: 256 not divisible by 7
        self.ln2 = nn.LayerNorm(256)
        self.ffn1 = nn.Linear(256, 1024)
        self.ffn2 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.ln1(self.embedding(x))
        attn_out, _ = self.attn(x, x, x)
        x = self.ln2(x + attn_out)
        return x + self.ffn2(self.relu(self.ffn1(x)))
""",
        "input_shapes": {"x": ("batch", 64, 512)},
        "is_buggy": True,
        "category": "Transformer",
        "description": "Transformer BUG: embed_dim=256 not divisible by num_heads=7",
    },

    # =====================================================================
    # 33. Autoencoder – correct
    # =====================================================================
    "autoencoder_correct": {
        "source": """
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
""",
        "input_shapes": {"x": ("batch", 784)},
        "is_buggy": False,
        "category": "Autoencoder",
        "description": "Symmetric autoencoder – correct encoder/decoder dimension chain",
    },

    # =====================================================================
    # 34. Autoencoder – buggy (decoder first layer mismatch)
    # =====================================================================
    "autoencoder_bug": {
        "source": """
import torch.nn as nn

class AutoencoderBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 64),  # BUG: input should be 16 (encoder output) not 64
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
""",
        "input_shapes": {"x": ("batch", 784)},
        "is_buggy": True,
        "category": "Autoencoder",
        "description": "Autoencoder BUG: decoder first layer expects 64 but encoder outputs 16",
    },

    # =====================================================================
    # 35. LeNet-5 – correct
    # =====================================================================
    "lenet5_correct": {
        "source": """
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 1, 32, 32)},
        "is_buggy": False,
        "category": "LeNet",
        "description": "LeNet-5 (LeCun et al. 1998) – correct conv→pool→flatten→FC chain",
    },

    # =====================================================================
    # 36. LeNet-5 – buggy (FC1 input dim wrong for 28x28 input)
    # =====================================================================
    "lenet5_input_bug": {
        "source": """
import torch.nn as nn

class LeNet5Bug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # BUG: for 28x28 input, spatial is 4x4 not 5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 1, 28, 28)},
        "is_buggy": True,
        "category": "LeNet",
        "description": "LeNet-5 BUG: fc1 expects 16*5*5=400 but 28x28 input yields 16*4*4=256 after convs+pools",
    },

    # =====================================================================
    # 37. Conv classifier – correct
    # =====================================================================
    "conv_classifier_correct": {
        "source": """
import torch.nn as nn

class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": False,
        "category": "ConvNet",
        "description": "Simple conv classifier with global avg pool – correct",
    },

    # =====================================================================
    # 38. Conv classifier – buggy (FC after global pool wrong dim)
    # =====================================================================
    "conv_classifier_bug": {
        "source": """
import torch.nn as nn

class ConvClassifierBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)  # BUG: should be 128 not 64
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "ConvNet",
        "description": "Conv classifier BUG: FC expects 64 but conv3 outputs 128 channels",
    },

    # =====================================================================
    # 39. SqueezeNet fire module – correct
    # =====================================================================
    "squeezenet_fire_correct": {
        "source": """
import torch.nn as nn

class FireModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeeze = nn.Conv2d(256, 32, 1)
        self.expand1x1 = nn.Conv2d(32, 128, 1)
        self.expand3x3 = nn.Conv2d(32, 128, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        s = self.relu(self.squeeze(x))
        e1 = self.relu(self.expand1x1(s))
        return e1
""",
        "input_shapes": {"x": ("batch", 256, 14, 14)},
        "is_buggy": False,
        "category": "SqueezeNet",
        "description": "SqueezeNet fire module (Iandola et al. 2016) – correct squeeze→expand",
    },

    # =====================================================================
    # 40. SqueezeNet fire – buggy (expand input from wrong channel)
    # =====================================================================
    "squeezenet_fire_bug": {
        "source": """
import torch.nn as nn

class FireModuleBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeeze = nn.Conv2d(256, 32, 1)
        self.expand1x1 = nn.Conv2d(64, 128, 1)  # BUG: input should be 32 not 64
        self.expand3x3 = nn.Conv2d(32, 128, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        s = self.relu(self.squeeze(x))
        e1 = self.relu(self.expand1x1(s))
        return e1
""",
        "input_shapes": {"x": ("batch", 256, 14, 14)},
        "is_buggy": True,
        "category": "SqueezeNet",
        "description": "SqueezeNet fire BUG: expand1x1 expects 64 ch but squeeze outputs 32 ch",
    },

    # =====================================================================
    # 41. WaveNet deeper – correct
    # =====================================================================
    "wavenet_stack_correct": {
        "source": """
import torch.nn as nn

class WaveNetStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv1d(1, 64, 1)
        self.dilated1 = nn.Conv1d(64, 64, 2, dilation=1, padding=1)
        self.dilated2 = nn.Conv1d(64, 64, 2, dilation=2, padding=2)
        self.dilated3 = nn.Conv1d(64, 64, 2, dilation=4, padding=4)
        self.out_conv = nn.Conv1d(64, 256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_conv(x))
        x = self.relu(self.dilated1(x))
        x = self.relu(self.dilated2(x))
        x = self.relu(self.dilated3(x))
        return self.out_conv(x)
""",
        "input_shapes": {"x": ("batch", 1, 16000)},
        "is_buggy": False,
        "category": "WaveNet",
        "description": "WaveNet dilated stack (van den Oord et al. 2016) – correct increasing dilation",
    },

    # =====================================================================
    # 42. WaveNet deeper – buggy (dilated conv channel mismatch)
    # =====================================================================
    "wavenet_stack_bug": {
        "source": """
import torch.nn as nn

class WaveNetStackBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv1d(1, 64, 1)
        self.dilated1 = nn.Conv1d(64, 128, 2, dilation=1, padding=1)  # outputs 128
        self.dilated2 = nn.Conv1d(64, 64, 2, dilation=2, padding=2)   # BUG: expects 64 but dilated1 outputs 128
        self.dilated3 = nn.Conv1d(64, 64, 2, dilation=4, padding=4)
        self.out_conv = nn.Conv1d(64, 256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_conv(x))
        x = self.relu(self.dilated1(x))
        x = self.relu(self.dilated2(x))
        x = self.relu(self.dilated3(x))
        return self.out_conv(x)
""",
        "input_shapes": {"x": ("batch", 1, 16000)},
        "is_buggy": True,
        "category": "WaveNet",
        "description": "WaveNet stack BUG: dilated2 expects 64 ch but dilated1 outputs 128 ch",
    },

    # =====================================================================
    # 43. MobileNetV2 full block – correct
    # =====================================================================
    "mobilenetv2_block_correct": {
        "source": """
import torch.nn as nn

class MobileNetV2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(16, 96, 1)
        self.bn1 = nn.BatchNorm2d(96)
        self.depthwise = nn.Conv2d(96, 96, 3, stride=2, padding=1, groups=96)
        self.bn2 = nn.BatchNorm2d(96)
        self.project = nn.Conv2d(96, 24, 1)
        self.bn3 = nn.BatchNorm2d(24)
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        out = self.relu6(self.bn1(self.expand(x)))
        out = self.relu6(self.bn2(self.depthwise(out)))
        out = self.bn3(self.project(out))
        return out
""",
        "input_shapes": {"x": ("batch", 16, 112, 112)},
        "is_buggy": False,
        "category": "MobileNetV2",
        "description": "MobileNetV2 block with stride-2 DW conv (Sandler et al. 2018) – correct",
    },

    # =====================================================================
    # 44. MobileNetV2 – buggy (project input mismatch)
    # =====================================================================
    "mobilenetv2_project_bug": {
        "source": """
import torch.nn as nn

class MobileNetV2BlockBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(16, 96, 1)
        self.bn1 = nn.BatchNorm2d(96)
        self.depthwise = nn.Conv2d(96, 96, 3, stride=2, padding=1, groups=96)
        self.bn2 = nn.BatchNorm2d(96)
        self.project = nn.Conv2d(48, 24, 1)  # BUG: expects 48 but DW outputs 96
        self.bn3 = nn.BatchNorm2d(24)
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        out = self.relu6(self.bn1(self.expand(x)))
        out = self.relu6(self.bn2(self.depthwise(out)))
        out = self.bn3(self.project(out))
        return out
""",
        "input_shapes": {"x": ("batch", 16, 112, 112)},
        "is_buggy": True,
        "category": "MobileNetV2",
        "description": "MobileNetV2 BUG: project expects 48 channels but depthwise outputs 96",
    },

    # =====================================================================
    # 45. Seq2Seq encoder – correct
    # =====================================================================
    "seq2seq_encoder_correct": {
        "source": """
import torch.nn as nn

class Seq2SeqEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(10000, 256)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.embed(x))
        x = self.relu(self.fc1(x))
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 50, 10000)},
        "is_buggy": False,
        "category": "Seq2Seq",
        "description": "Simple seq2seq encoder – correct embedding→FC chain",
    },

    # =====================================================================
    # 46. Seq2Seq encoder – buggy (FC chain mismatch)
    # =====================================================================
    "seq2seq_encoder_bug": {
        "source": """
import torch.nn as nn

class Seq2SeqEncoderBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(10000, 256)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(128, 256)  # BUG: input should be 512 not 128
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.embed(x))
        x = self.relu(self.fc1(x))
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 50, 10000)},
        "is_buggy": True,
        "category": "Seq2Seq",
        "description": "Seq2Seq BUG: fc2 expects 128 but fc1 outputs 512",
    },

    # =====================================================================
    # 47. ResNeXt-style grouped conv – correct
    # =====================================================================
    "resnext_correct": {
        "source": """
import torch.nn as nn

class ResNeXtBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, groups=32)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + x)
""",
        "input_shapes": {"x": ("batch", 256, 14, 14)},
        "is_buggy": False,
        "category": "ResNeXt",
        "description": "ResNeXt block (Xie et al. 2017) – correct grouped 3x3 conv with residual",
    },

    # =====================================================================
    # 48. ResNeXt – buggy (grouped conv channel not divisible by groups)
    # =====================================================================
    "resnext_groups_bug": {
        "source": """
import torch.nn as nn

class ResNeXtBlockBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, groups=48)  # BUG: 128 not divisible by 48
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + x)
""",
        "input_shapes": {"x": ("batch", 256, 14, 14)},
        "is_buggy": True,
        "category": "ResNeXt",
        "description": "ResNeXt BUG: grouped conv has groups=48 but channels=128 (128%48!=0)",
    },

    # =====================================================================
    # 49. DenseNet transition – correct
    # =====================================================================
    "densenet_transition_correct": {
        "source": """
import torch.nn as nn

class DenseNetTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(256)
        self.conv = nn.Conv2d(256, 128, 1)
        self.pool = nn.AvgPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(x))
        x = self.conv(x)
        return self.pool(x)
""",
        "input_shapes": {"x": ("batch", 256, 28, 28)},
        "is_buggy": False,
        "category": "DenseNet",
        "description": "DenseNet transition layer (Huang et al. 2017) – correct BN→1x1conv→pool",
    },

    # =====================================================================
    # 50. DenseNet transition – buggy (BN channels mismatch)
    # =====================================================================
    "densenet_transition_bug": {
        "source": """
import torch.nn as nn

class DenseNetTransitionBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(128)  # BUG: should be 256 to match input
        self.conv = nn.Conv2d(256, 128, 1)
        self.pool = nn.AvgPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(x))
        x = self.conv(x)
        return self.pool(x)
""",
        "input_shapes": {"x": ("batch", 256, 28, 28)},
        "is_buggy": True,
        "category": "DenseNet",
        "description": "DenseNet transition BUG: BatchNorm expects 128 channels but input is 256",
    },
}


def get_benchmark_summary():
    """Return summary statistics about the benchmark suite."""
    total = len(EXTERNAL_PYTORCH_BENCHMARKS)
    buggy = sum(1 for b in EXTERNAL_PYTORCH_BENCHMARKS.values() if b["is_buggy"])
    correct = total - buggy
    categories = set(b["category"] for b in EXTERNAL_PYTORCH_BENCHMARKS.values())
    return {
        "total": total,
        "buggy": buggy,
        "correct": correct,
        "categories": sorted(categories),
        "num_categories": len(categories),
    }
