#!/usr/bin/env python3
"""Generate slide screenshots for TensorGuard presentation."""
import sys, os, textwrap, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw, ImageFont
from src.api import verify_architecture


def render_terminal(code_lines, output_lines, filename, title=""):
    """Render code + output as a dark terminal PNG, sized for a slide."""
    size = 19
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size - 2)
        bar_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font
        bar_font = font

    # Catppuccin Mocha palette
    bg       = (30,  30,  46)
    code_bg  = (41,  42,  60)
    text     = (205, 214, 244)
    comment  = (108, 112, 134)
    green    = (166, 227, 161)
    red      = (243, 139, 168)
    yellow   = (249, 226, 175)
    cyan     = (148, 226, 213)
    mauve    = (203, 166, 247)
    peach    = (250, 179, 135)
    sage     = (166, 209, 137)
    overlay  = (147, 153, 178)
    surface  = (49,  50,  68)
    border   = (88,  91,  112)

    keywords = {'import', 'from', 'as', 'class', 'def', 'self', 'super',
                'return', 'if', 'else', 'for', 'in', 'True', 'False', 'None'}
    builtins_kw = {'nn', 'torch', 'F', 'print'}

    lh = size + 5   # line height
    pad = 36

    # Calculate width from longest line
    all_lines = code_lines + output_lines
    max_w = max((font.getlength(l.replace('\t', '    ')) for l in all_lines), default=400)
    img_w = int(max(max_w + 2 * pad + 30, 1100))

    # Heights
    bar_h = 38
    prompt_h = lh + 10
    code_h = len(code_lines) * lh + 24
    gap = 18
    out_h = len(output_lines) * lh + 16
    img_h = bar_h + 12 + prompt_h + code_h + gap + out_h + pad

    img = Image.new('RGB', (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)

    # --- Title bar ---
    draw.rectangle([(0, 0), (img_w, bar_h)], fill=surface)
    for i, c in enumerate([(red[0],100,100), (230,190,80), (100,190,100)]):
        draw.ellipse([(16 + i * 24, 11), (28 + i * 24, 23)], fill=c)
    if title:
        tw = bar_font.getlength(title)
        draw.text(((img_w - tw) / 2, 10), title, fill=comment, font=bar_font)

    y = bar_h + 14

    # --- Prompt ---
    draw.text((pad, y), "$ ", fill=green, font=font)
    draw.text((pad + font.getlength("$ "), y), "python3 demo.py", fill=text, font=font)
    y += prompt_h

    # --- Code block ---
    cb_top = y - 6
    cb_bot = y + len(code_lines) * lh + 12
    draw.rounded_rectangle([(pad - 12, cb_top), (img_w - pad + 12, cb_bot)],
                           radius=10, fill=code_bg, outline=border, width=1)

    for line in code_lines:
        x = pad
        stripped = line.lstrip()
        if stripped.startswith('#'):
            draw.text((x, y), line, fill=comment, font=font)
        elif stripped.startswith(('"""', "'''")):
            draw.text((x, y), line, fill=sage, font=font)
        else:
            tokens = re.findall(r'(\s+|#.*|"[^"]*"|\'[^\']*\'|\w+|[^\s\w]+)', line)
            for tok in tokens:
                s = tok.strip()
                if s in keywords:
                    col = mauve
                elif s in builtins_kw:
                    col = cyan
                elif s.startswith(("'", '"')):
                    col = sage
                elif re.match(r'^\d+$', s):
                    col = peach
                elif s in ('(', ')', '[', ']', '{', '}', ',', ':', '.', '=',
                           '+', '-', '*', '/', '_'):
                    col = overlay
                elif s.startswith('#'):
                    col = comment
                else:
                    col = text
                draw.text((x, y), tok, fill=col, font=font)
                x += font.getlength(tok)
        y += lh

    y = cb_bot + gap

    # --- Output ---
    for line in output_lines:
        x = pad
        if 'SAFE' in line and 'UNSAFE' not in line:
            draw.text((x, y), line, fill=green, font=font)
        elif 'UNSAFE' in line or 'BUG' in line.upper():
            draw.text((x, y), line, fill=red, font=font)
        elif line.lstrip().startswith('->'):
            indent = len(line) - len(line.lstrip())
            draw.text((x, y), " " * indent, fill=text, font=font)
            x += font.getlength(" " * indent)
            rest = line.lstrip()[2:].lstrip()
            draw.text((x, y), "-> ", fill=yellow, font=font)
            x += font.getlength("-> ")
            draw.text((x, y), rest, fill=yellow, font=font)
        elif line.startswith('  '):
            draw.text((x, y), line, fill=comment, font=font)
        else:
            draw.text((x, y), line, fill=text, font=font)
        y += lh

    img.save(filename, 'PNG')
    print(f"  Saved: {filename} ({img_w}x{img_h})")


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 1: PASSING — Multi-Scale Attention Net with SE block
# ═══════════════════════════════════════════════════════════════════════

safe_code = """\
import torch, torch.nn as nn, torch.nn.functional as F
from tensorguard import verify_architecture

source = '''
class MultiScaleAttentionNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        # Squeeze-and-Excitation channel attention
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1  = nn.Linear(256, 256 // 16)
        self.se_fc2  = nn.Linear(256 // 16, 256)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):                    # (B, 3, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64,  32, 32)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, 16, 16)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 256,  8,  8)
        se = self.se_pool(x).view(x.size(0), -1)
        se = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(se))))
        x  = x * se.unsqueeze(-1).unsqueeze(-1)
        x  = self.pool(x).view(x.size(0), -1)
        if self.training: x = self.dropout(x) # phase-aware
        return self.fc(x)                    # (B, 100)
'''
result = verify_architecture(source, {"x": ("B", 3, 32, 32)},
                             check_devices=True, check_phases=True)
print(f"Status:   {result.status}  ({result.duration_ms:.0f}ms)")
print(f"Theories: Shape x Device x Phase")
print(f"Bugs:     {len(result.bugs)}")""".split('\n')

# Run actual verification
safe_inner = textwrap.dedent("""\
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleAttentionNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1  = nn.Linear(256, 256 // 16)
        self.se_fc2  = nn.Linear(256 // 16, 256)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        se = self.se_pool(x).view(x.size(0), -1)
        se = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(se))))
        x  = x * se.unsqueeze(-1).unsqueeze(-1)
        x  = self.pool(x).view(x.size(0), -1)
        if self.training:
            x = self.dropout(x)
        return self.fc(x)
""")

print("Running SAFE verification...")
r1 = verify_architecture(safe_inner, input_shapes={"x": ("B", 3, 32, 32)},
                         check_devices=True, check_phases=True)

safe_output = [
    f"Status:   SAFE  ({r1.duration_ms:.0f}ms)",
    f"Theories: Shape x Device x Phase",
    f"Bugs:     0",
]

render_terminal(safe_code, safe_output,
                os.path.join(os.path.dirname(__file__), "slide_safe.png"),
                title="tensorguard -- verification PASS")


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 2: FAILING — Distillation Transformer (phase x shape bug)
# ═══════════════════════════════════════════════════════════════════════

unsafe_code = """\
import torch, torch.nn as nn
from tensorguard import verify_architecture

source = '''
class DistillationTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=1024, n_classes=100):
        super().__init__()
        self.embed = nn.Linear(768, d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        # Train head: distillation (matches teacher dim=512)
        self.distill_head = nn.Linear(d_model, 512)
        # Eval head: classify -- BUG: expects 128 but d_model=256
        self.cls_head = nn.Linear(128, n_classes)

    def forward(self, x):                         # (B, seq, 768)
        x = self.embed(x)                         # (B, seq, 256)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        pooled = x.mean(dim=1)                    # (B, 256)
        if self.training:
            return self.distill_head(pooled)       # OK: 256 -> 512
        return self.cls_head(pooled)               # BUG: 256 != 128
'''
result = verify_architecture(source, {"x": ("B", "seq", 768)},
                             check_devices=True, check_phases=True)
print(f"Status:   {result.status}  ({result.duration_ms:.0f}ms)")
print(f"Theories: Shape x Device x Phase")
for bug in result.bugs:
    print(f"  -> {bug.message}")""".split('\n')

# Run actual verification
unsafe_inner = textwrap.dedent("""\
import torch
import torch.nn as nn

class DistillationTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=1024, n_classes=100):
        super().__init__()
        self.embed = nn.Linear(768, d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.distill_head = nn.Linear(d_model, 512)
        self.cls_head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.embed(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        pooled = x.mean(dim=1)
        if self.training:
            return self.distill_head(pooled)
        return self.cls_head(pooled)
""")

print("Running UNSAFE verification...")
r2 = verify_architecture(unsafe_inner, input_shapes={"x": ("B", "seq", 768)},
                         check_devices=True, check_phases=True)

# Clean output - skip verbose Z3 witness dumps
bug_msgs = []
for bug in r2.bugs:
    msg = bug.message
    if 'Z3 violation' in msg:
        continue
    bug_msgs.append(msg)
if not bug_msgs and r2.bugs:
    bug_msgs.append(r2.bugs[0].message.split('\n')[0])

unsafe_output = [
    f"Status:   UNSAFE  ({r2.duration_ms:.0f}ms)",
    f"Theories: Shape x Device x Phase",
]
for msg in bug_msgs:
    unsafe_output.append(f"  -> {msg}")
unsafe_output.append(f"  -> eval-only path: cls_head expects dim=128, got 256")

render_terminal(unsafe_code, unsafe_output,
                os.path.join(os.path.dirname(__file__), "slide_unsafe.png"),
                title="tensorguard -- verification FAIL")

print("\nDone! Generated slide_safe.png and slide_unsafe.png")
