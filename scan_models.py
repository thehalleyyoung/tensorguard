"""
Scan real model files from GitHub for TG-detectable bugs.
"""
import sys, json, base64, urllib.request, urllib.error
sys.path.insert(0, '.')
from src.api import verify_architecture

TOKEN = open('.github_token', 'r').read().strip() if __import__('os').path.exists('.github_token') else None

import subprocess
result = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True)
TOKEN = result.stdout.strip() if result.returncode == 0 else None

def fetch_file(owner, repo, path, ref='HEAD'):
    """Fetch a file from GitHub and return its content."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    req = urllib.request.Request(url)
    if TOKEN:
        req.add_header('Authorization', f'token {TOKEN}')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            if 'content' in data:
                return base64.b64decode(data['content']).decode('utf-8', errors='replace')
    except Exception as e:
        return None
    return None

# Model files to test
MODEL_FILES = [
    # (owner, repo, path, ref, input_shapes)
    ('kyegomez', 'VisionMamba', 'vision_mamba/model.py', 'HEAD', {'x': (1, 3, 224, 224)}),
    ('karpathy', 'nanoGPT', 'model.py', 'HEAD', {'x': (1, 64)}),
    ('lucidrains', 'vit-pytorch', 'vit_pytorch/vit_pytorch.py', 'HEAD', {'img': (1, 3, 224, 224)}),
    ('lucidrains', 'vit-pytorch', 'vit_pytorch/max_vit.py', 'HEAD', {'img': (1, 3, 256, 256)}),
    ('lucidrains', 'vit-pytorch', 'vit_pytorch/nest.py', '1cc0f182a609', {'img': (1, 3, 224, 224)}),  # buggy commit
    ('facebookresearch', 'mae', 'models_mae.py', 'HEAD', {'imgs': (1, 3, 224, 224), 'mask_ratio': (1,)}),
]

for owner, repo, path, ref, input_shapes in MODEL_FILES:
    content = fetch_file(owner, repo, path, ref)
    if not content:
        print(f"SKIP {owner}/{repo} {path}: could not fetch")
        continue
    
    print(f"\nTesting {owner}/{repo} {path} @ {ref[:8]}")
    try:
        result = verify_architecture(content, input_shapes=input_shapes)
        if result.bugs:
            print(f"  BUGS FOUND: {len(result.bugs)}")
            for b in result.bugs[:3]:
                print(f"    [{b.confidence:.2f}] {b.message[:100]}")
        else:
            print(f"  No bugs detected")
    except Exception as e:
        print(f"  ERROR: {str(e)[:100]}")

