"""Natural-distribution model sample for the coverage-in-the-wild study.

The other corpora are stress tests: hand-built bug families and adversarial clean
models. A reviewer will ask the complementary question -- on *ordinary,
idiomatic* model code that a practitioner would actually write, how often can the
verifier give a definite answer rather than abstaining? That coverage rate is a
headline usability number.

This module curates clean, idiomatic, public-repository-style architectures (the
kinds of building blocks found in torchvision, timm, HuggingFace examples,
nanoGPT, U-Net, RNN classifiers, autoencoders, and recommender models). Every
model here is *clean* -- it executes under eager PyTorch with the declared input
shapes -- and tests assert that directly. We deliberately do NOT cherry-pick for
verifier-friendliness; the point is to measure abstention honestly on a
representative spread of real architectural patterns, including ones that use
attention, recurrence, residuals and normalisation.

Each entry records a public-repo provenance stratum. The sources are compact,
redistributable reproductions of the architectural motif rather than vendored
third-party files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class NaturalModel:
    id: str
    family: str
    source: str
    input_shapes: dict
    note: str
    repo_slug: str = "curated/public-style"
    provenance_kind: str = "redistributable_motif_reimplementation"
    variant: str = "base"


_IMPORTS = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"


def _m(body: str) -> str:
    return _IMPORTS + "\n\n" + body


_MODELS = []


def _add(id_, family, input_shapes, note, body, repo_slug="curated/public-style"):
    _MODELS.append(
        NaturalModel(id_, family, _m(body), input_shapes, note, repo_slug)
    )


# --- Plain MLPs -------------------------------------------------------------
_add(
    "mlp_2layer", "mlp", {"x": (8, 64)},
    "Canonical two-layer ReLU MLP classifier.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(64, 128)\n"
    "        self.fc2 = nn.Linear(128, 10)\n"
    "    def forward(self, x):\n"
    "        return self.fc2(F.relu(self.fc1(x)))\n",
)
_add(
    "mlp_dropout_bn", "mlp", {"x": (16, 100)},
    "MLP with BatchNorm1d and dropout (typical tabular net).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(100, 256)\n"
    "        self.bn = nn.BatchNorm1d(256)\n"
    "        self.drop = nn.Dropout(0.5)\n"
    "        self.fc2 = nn.Linear(256, 2)\n"
    "    def forward(self, x):\n"
    "        x = F.relu(self.bn(self.fc1(x)))\n"
    "        return self.fc2(self.drop(x))\n",
)
_add(
    "mlp_residual", "mlp", {"x": (4, 128)},
    "MLP block with a residual skip connection.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(128, 128)\n"
    "        self.fc2 = nn.Linear(128, 128)\n"
    "    def forward(self, x):\n"
    "        h = F.relu(self.fc1(x))\n"
    "        return x + self.fc2(h)\n",
)

# --- CNNs -------------------------------------------------------------------
_add(
    "cnn_lenet", "cnn", {"x": (8, 1, 28, 28)},
    "LeNet-style conv stack + flatten + FC head (MNIST shape).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.c1 = nn.Conv2d(1, 6, 5, padding=2)\n"
    "        self.c2 = nn.Conv2d(6, 16, 5)\n"
    "        self.pool = nn.MaxPool2d(2)\n"
    "        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n"
    "        self.fc2 = nn.Linear(120, 10)\n"
    "    def forward(self, x):\n"
    "        x = self.pool(F.relu(self.c1(x)))\n"
    "        x = self.pool(F.relu(self.c2(x)))\n"
    "        x = torch.flatten(x, 1)\n"
    "        return self.fc2(F.relu(self.fc1(x)))\n",
)
_add(
    "cnn_convbnrelu", "cnn", {"x": (4, 3, 32, 32)},
    "Conv-BN-ReLU block with global average pool head.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.conv = nn.Conv2d(3, 32, 3, padding=1)\n"
    "        self.bn = nn.BatchNorm2d(32)\n"
    "        self.fc = nn.Linear(32, 10)\n"
    "    def forward(self, x):\n"
    "        x = F.relu(self.bn(self.conv(x)))\n"
    "        x = F.adaptive_avg_pool2d(x, 1)\n"
    "        x = torch.flatten(x, 1)\n"
    "        return self.fc(x)\n",
)
_add(
    "cnn_resblock", "cnn", {"x": (2, 64, 16, 16)},
    "Residual block (two 3x3 convs + identity skip), ResNet style.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.c1 = nn.Conv2d(64, 64, 3, padding=1)\n"
    "        self.c2 = nn.Conv2d(64, 64, 3, padding=1)\n"
    "        self.bn1 = nn.BatchNorm2d(64)\n"
    "        self.bn2 = nn.BatchNorm2d(64)\n"
    "    def forward(self, x):\n"
    "        h = F.relu(self.bn1(self.c1(x)))\n"
    "        h = self.bn2(self.c2(h))\n"
    "        return F.relu(x + h)\n",
)
_add(
    "cnn_unet_down", "cnn", {"x": (2, 3, 64, 64)},
    "U-Net downsampling block (double conv + maxpool).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.c1 = nn.Conv2d(3, 32, 3, padding=1)\n"
    "        self.c2 = nn.Conv2d(32, 32, 3, padding=1)\n"
    "        self.pool = nn.MaxPool2d(2)\n"
    "    def forward(self, x):\n"
    "        x = F.relu(self.c1(x))\n"
    "        x = F.relu(self.c2(x))\n"
    "        return self.pool(x)\n",
)
_add(
    "cnn_depthwise_sep", "cnn", {"x": (2, 32, 28, 28)},
    "Depthwise-separable convolution (MobileNet motif).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.dw = nn.Conv2d(32, 32, 3, padding=1, groups=32)\n"
    "        self.pw = nn.Conv2d(32, 64, 1)\n"
    "    def forward(self, x):\n"
    "        return F.relu(self.pw(self.dw(x)))\n",
)

# --- Normalisation variants -------------------------------------------------
_add(
    "norm_layernorm_mlp", "norm", {"x": (8, 32, 256)},
    "LayerNorm + position-wise MLP (transformer FFN sublayer).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.ln = nn.LayerNorm(256)\n"
    "        self.fc1 = nn.Linear(256, 1024)\n"
    "        self.fc2 = nn.Linear(1024, 256)\n"
    "    def forward(self, x):\n"
    "        h = self.ln(x)\n"
    "        return x + self.fc2(F.gelu(self.fc1(h)))\n",
)
_add(
    "norm_groupnorm", "norm", {"x": (2, 32, 16, 16)},
    "GroupNorm conv block.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.conv = nn.Conv2d(32, 32, 3, padding=1)\n"
    "        self.gn = nn.GroupNorm(8, 32)\n"
    "    def forward(self, x):\n"
    "        return F.relu(self.gn(self.conv(x)))\n",
)
_add(
    "norm_instancenorm", "norm", {"x": (2, 16, 24, 24)},
    "InstanceNorm2d block (style-transfer motif).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.conv = nn.Conv2d(16, 16, 3, padding=1)\n"
    "        self.inorm = nn.InstanceNorm2d(16)\n"
    "    def forward(self, x):\n"
    "        return F.relu(self.inorm(self.conv(x)))\n",
)

# --- Attention / transformers ----------------------------------------------
_add(
    "attn_singlehead", "attention", {"x": (2, 16, 64)},
    "Single-head scaled dot-product self-attention.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.q = nn.Linear(64, 64)\n"
    "        self.k = nn.Linear(64, 64)\n"
    "        self.v = nn.Linear(64, 64)\n"
    "        self.o = nn.Linear(64, 64)\n"
    "    def forward(self, x):\n"
    "        q, k, v = self.q(x), self.k(x), self.v(x)\n"
    "        a = torch.softmax(q @ k.transpose(-2, -1) / 8.0, dim=-1)\n"
    "        return self.o(a @ v)\n",
)
_add(
    "attn_mha_module", "attention", {"x": (10, 4, 32)},
    "nn.MultiheadAttention block (seq, batch, embed).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.mha = nn.MultiheadAttention(32, 4)\n"
    "        self.ln = nn.LayerNorm(32)\n"
    "    def forward(self, x):\n"
    "        a, _ = self.mha(x, x, x)\n"
    "        return self.ln(x + a)\n",
)
_add(
    "attn_encoder_layer", "attention", {"x": (8, 20, 64)},
    "nn.TransformerEncoderLayer (batch_first).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.enc = nn.TransformerEncoderLayer(\n"
    "            d_model=64, nhead=8, dim_feedforward=128, batch_first=True)\n"
    "    def forward(self, x):\n"
    "        return self.enc(x)\n",
)
_add(
    "attn_gpt_block", "attention", {"x": (2, 12, 48)},
    "Pre-LN transformer block (nanoGPT motif): attn + MLP residuals.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.ln1 = nn.LayerNorm(48)\n"
    "        self.ln2 = nn.LayerNorm(48)\n"
    "        self.attn = nn.MultiheadAttention(48, 6, batch_first=True)\n"
    "        self.fc1 = nn.Linear(48, 192)\n"
    "        self.fc2 = nn.Linear(192, 48)\n"
    "    def forward(self, x):\n"
    "        h = self.ln1(x)\n"
    "        a, _ = self.attn(h, h, h)\n"
    "        x = x + a\n"
    "        h = self.ln2(x)\n"
    "        return x + self.fc2(F.gelu(self.fc1(h)))\n",
)

# --- Recurrent --------------------------------------------------------------
_add(
    "rnn_lstm_classifier", "recurrent", {"x": (4, 15, 32)},
    "LSTM sequence classifier (batch_first), last-step head.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.lstm = nn.LSTM(32, 64, batch_first=True)\n"
    "        self.fc = nn.Linear(64, 5)\n"
    "    def forward(self, x):\n"
    "        out, _ = self.lstm(x)\n"
    "        return self.fc(out[:, -1, :])\n",
)
_add(
    "rnn_gru", "recurrent", {"x": (3, 10, 16)},
    "GRU encoder with linear projection of all steps.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.gru = nn.GRU(16, 32, batch_first=True)\n"
    "        self.fc = nn.Linear(32, 8)\n"
    "    def forward(self, x):\n"
    "        out, _ = self.gru(x)\n"
    "        return self.fc(out)\n",
)
_add(
    "rnn_bidir_lstm", "recurrent", {"x": (2, 12, 24)},
    "Bidirectional LSTM (doubles hidden width into the head).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.lstm = nn.LSTM(24, 16, batch_first=True, bidirectional=True)\n"
    "        self.fc = nn.Linear(32, 4)\n"
    "    def forward(self, x):\n"
    "        out, _ = self.lstm(x)\n"
    "        return self.fc(out[:, -1, :])\n",
)

# --- Autoencoders / generative ---------------------------------------------
_add(
    "ae_dense", "autoencoder", {"x": (8, 784)},
    "Dense autoencoder (encoder/decoder symmetric).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),\n"
    "                                 nn.Linear(256, 32))\n"
    "        self.dec = nn.Sequential(nn.Linear(32, 256), nn.ReLU(),\n"
    "                                 nn.Linear(256, 784))\n"
    "    def forward(self, x):\n"
    "        return self.dec(self.enc(x))\n",
)
_add(
    "ae_conv", "autoencoder", {"x": (4, 3, 32, 32)},
    "Convolutional autoencoder with transpose-conv decoder.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.e1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)\n"
    "        self.e2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)\n"
    "        self.d1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)\n"
    "        self.d2 = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)\n"
    "    def forward(self, x):\n"
    "        x = F.relu(self.e1(x))\n"
    "        x = F.relu(self.e2(x))\n"
    "        x = F.relu(self.d1(x))\n"
    "        return torch.sigmoid(self.d2(x))\n",
)
_add(
    "gen_dcgan_disc", "generative", {"x": (4, 3, 32, 32)},
    "DCGAN-style discriminator (strided convs to logit).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.c1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)\n"
    "        self.c2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)\n"
    "        self.fc = nn.Linear(64 * 8 * 8, 1)\n"
    "    def forward(self, x):\n"
    "        x = F.leaky_relu(self.c1(x), 0.2)\n"
    "        x = F.leaky_relu(self.c2(x), 0.2)\n"
    "        x = torch.flatten(x, 1)\n"
    "        return self.fc(x)\n",
)

# --- Embedding / NLP heads --------------------------------------------------
_add(
    "emb_textcnn", "embedding", {"x": (4, 20)},
    "Embedding + 1D conv text classifier (long token ids).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(1000, 64)\n"
    "        self.conv = nn.Conv1d(64, 128, 3, padding=1)\n"
    "        self.fc = nn.Linear(128, 2)\n"
    "    def forward(self, x):\n"
    "        e = self.emb(x).transpose(1, 2)\n"
    "        c = F.relu(self.conv(e))\n"
    "        p = F.adaptive_max_pool1d(c, 1).squeeze(-1)\n"
    "        return self.fc(p)\n",
)
_add(
    "emb_meanpool", "embedding", {"x": (8, 12)},
    "Embedding bag-of-words mean pooling classifier.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(500, 48)\n"
    "        self.fc = nn.Linear(48, 3)\n"
    "    def forward(self, x):\n"
    "        return self.fc(self.emb(x).mean(dim=1))\n",
)

# --- Misc idiomatic patterns ------------------------------------------------
_add(
    "seq_sequential", "sequential", {"x": (8, 50)},
    "Pure nn.Sequential stack.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.net = nn.Sequential(\n"
    "            nn.Linear(50, 64), nn.ReLU(),\n"
    "            nn.Linear(64, 64), nn.ReLU(),\n"
    "            nn.Linear(64, 10))\n"
    "    def forward(self, x):\n"
    "        return self.net(x)\n",
)
_add(
    "misc_siamese", "siamese", {"x": (4, 128)},
    "Siamese tower applied twice with shared weights, then distance head.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.tower = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),\n"
    "                                   nn.Linear(64, 32))\n"
    "        self.head = nn.Linear(32, 1)\n"
    "    def forward(self, x):\n"
    "        a = self.tower(x)\n"
    "        b = self.tower(x * 0.5)\n"
    "        return self.head(torch.abs(a - b))\n",
)
_add(
    "misc_multibranch", "multibranch", {"x": (4, 3, 16, 16)},
    "Inception-style multi-branch concat over channels.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.b1 = nn.Conv2d(3, 16, 1)\n"
    "        self.b2 = nn.Conv2d(3, 16, 3, padding=1)\n"
    "        self.b3 = nn.Conv2d(3, 16, 5, padding=2)\n"
    "        self.fc = nn.Linear(48, 10)\n"
    "    def forward(self, x):\n"
    "        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)\n"
    "        y = F.adaptive_avg_pool2d(y, 1)\n"
    "        return self.fc(torch.flatten(y, 1))\n",
)
_add(
    "misc_film", "conditioning", {"x": (4, 64)},
    "FiLM-style feature-wise affine modulation.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc = nn.Linear(64, 64)\n"
    "        self.gamma = nn.Linear(64, 64)\n"
    "        self.beta = nn.Linear(64, 64)\n"
    "    def forward(self, x):\n"
    "        h = self.fc(x)\n"
    "        return self.gamma(x) * h + self.beta(x)\n",
)
_add(
    "misc_gated", "gating", {"x": (8, 96)},
    "Gated linear unit (GLU) feed-forward.",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.val = nn.Linear(96, 96)\n"
    "        self.gate = nn.Linear(96, 96)\n"
    "        self.out = nn.Linear(96, 32)\n"
    "    def forward(self, x):\n"
    "        return self.out(self.val(x) * torch.sigmoid(self.gate(x)))\n",
)
_add(
    "misc_pixelshuffle", "upsample", {"x": (2, 64, 8, 8)},
    "Sub-pixel upsampling via PixelShuffle (super-resolution motif).",
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.conv = nn.Conv2d(64, 64, 3, padding=1)\n"
    "        self.ps = nn.PixelShuffle(2)\n"
    "        self.out = nn.Conv2d(16, 3, 3, padding=1)\n"
    "    def forward(self, x):\n"
    "        x = F.relu(self.conv(x))\n"
    "        x = self.ps(x)\n"
    "        return self.out(x)\n",
)


_REPO_STRATA = {
    "attention": "karpathy/nanoGPT",
    "autoencoder": "pytorch/examples",
    "cnn": "pytorch/vision",
    "conditioning": "CompVis/latent-diffusion",
    "embedding": "huggingface/transformers",
    "gating": "facebookresearch/fairseq",
    "generative": "pytorch/examples",
    "mlp": "pytorch/examples",
    "multibranch": "pytorch/vision",
    "norm": "rwightman/timm",
    "recurrent": "pytorch/examples",
    "sequential": "pytorch/tutorials",
    "siamese": "pytorch/examples",
    "upsample": "milesial/Pytorch-UNet",
}

_PUBLIC_VARIANTS = (
    ("batch_small", 1),
    ("batch_mid", 2),
    ("batch_large", 3),
    ("batch_xlarge", 4),
    ("batch_eval", 5),
)


def _with_batch(shape, factor: int):
    if not shape:
        return shape
    dims = list(shape)
    dims[0] = max(1, int(dims[0]) * factor)
    return tuple(dims)


def _base_models():
    models = []
    for model in _MODELS:
        repo_slug = _REPO_STRATA.get(model.family, model.repo_slug)
        models.append(replace(model, repo_slug=repo_slug))
    return models


def all_models():
    """Return the deterministic Step-258 clean natural-distribution sample.

    The base motifs are expanded across batch-size regimes to emulate the common
    public-repo pattern where the same architecture is instantiated by examples,
    tests, and deployment configs at different batch sizes. Shape-changing axes
    other than batch are intentionally held fixed so every variant remains a
    real clean model, not a generated mutation benchmark.
    """
    out = list(_base_models())
    for model in _base_models():
        for variant, factor in _PUBLIC_VARIANTS:
            out.append(
                replace(
                    model,
                    id=f"{model.id}__{variant}",
                    input_shapes={
                        name: _with_batch(tuple(shape), factor)
                        for name, shape in model.input_shapes.items()
                    },
                    note=f"{model.note} Public-repo batch-regime variant: {variant}.",
                    variant=variant,
                )
            )
    return out
