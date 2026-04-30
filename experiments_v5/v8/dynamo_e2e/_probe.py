import sys, inspect, time, json
sys.path.insert(0, '.')
import logging
for name in ('torch._dynamo','torch','transformers','timm'):
    logging.getLogger(name).setLevel(logging.ERROR)
from src.api import verify_architecture
import torchvision.models as tvm
import timm

candidates = [
    ('torchvision.resnet.BasicBlock', tvm.resnet.BasicBlock,
     'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch import Tensor\nfrom typing import Optional, Callable\n\ndef conv3x3(*a,**k): return nn.Conv2d(64,64,3,padding=1)\ndef conv1x1(*a,**k): return nn.Conv2d(64,64,1)\n',
     {'x': ('B', 64, 'H', 'W')}),
    ('torchvision.resnet.Bottleneck', tvm.resnet.Bottleneck,
     'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch import Tensor\nfrom typing import Optional, Callable\n\ndef conv3x3(*a,**k): return nn.Conv2d(64,64,3,padding=1)\ndef conv1x1(*a,**k): return nn.Conv2d(64,64,1)\n',
     {'x': ('B', 64, 'H', 'W')}),
    ('torchvision.mobilenetv2.InvertedResidual', tvm.mobilenetv2.InvertedResidual,
     'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch import Tensor\nfrom typing import Optional, Callable, List\nimport functools\n',
     {'x': ('B', 32, 'H', 'W')}),
    ('torchvision.squeezenet.Fire', tvm.squeezenet.Fire,
     'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch import Tensor\n',
     {'x': ('B', 64, 'H', 'W')}),
]
try:
    import timm.models.vision_transformer as vt
    candidates.append(('timm.vit.Mlp', vt.Mlp,
        'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom typing import Optional, Callable\n', {'x': ('B','S', 128)}))
    candidates.append(('timm.vit.Attention', vt.Attention,
        'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom typing import Optional, Callable\n', {'x': ('B','S', 128)}))
    candidates.append(('timm.vit.Block', vt.Block,
        'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom typing import Optional, Callable\n', {'x': ('B','S', 128)}))
except Exception as e:
    print('timm err', e)

results = []
for name, cls, prefix, shapes in candidates:
    try:
        body = inspect.getsource(cls)
    except Exception as e:
        results.append({'name': name, 'tg_verdict': 'error', 'tg_detail': f'getsource failed: {e}'})
        continue
    full_src = prefix + body
    t = time.time()
    try:
        r = verify_architecture(full_src, input_shapes=shapes)
        dur = time.time() - t
        verdict = 'verified' if getattr(r, 'status', '') == 'SAFE' else 'tg_failed'
        results.append({
            'name': name,
            'tg_verdict': verdict,
            'tg_status': getattr(r, 'status', None),
            'n_bugs': len(r.bugs),
            'bugs': [str(b)[:120] for b in r.bugs[:5]],
            'duration_s': round(dur, 3),
            'input_shapes': {k: list(v) for k, v in shapes.items()},
            'src_len_chars': len(full_src),
        })
    except Exception as e:
        results.append({'name': name, 'tg_verdict': 'error', 'tg_detail': f'{type(e).__name__}: {str(e)[:200]}', 'duration_s': round(time.time()-t, 3)})

print(json.dumps(results, indent=2))
