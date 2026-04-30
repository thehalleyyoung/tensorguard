#!/usr/bin/env python3
"""Quick test of unbind and .T implementations."""

from src.api import verify_architecture

# Test 1: unbind with tuple unpacking (from ChannelAttention)
test_unbind = """
import torch
import torch.nn as nn

class TestUnbind(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        k = k * 0.125
        attn = k.transpose(-1, -2) @ v
        attn = attn.softmax(dim=-1)
        x = (attn @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
"""

# Test 2: .T attribute (from FalconLinear)
test_t_attribute = """
import torch
import torch.nn as nn

class TestT(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # x @ weight.T where weight is (out_features, in_features)
        # so weight.T is (in_features, out_features)
        return x @ self.weight.T
"""

def run_test(name, code, input_shapes):
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    try:
        result = verify_architecture(code, input_shapes=input_shapes, max_cegar_iterations=3)
        print(f"Status: {result.status}")
        print(f"Abstained: {result.abstained}")
        print(f"Bug count: {result.bug_count}")
        if result.bugs:
            for bug in result.bugs[:3]:
                print(f"  - {bug.category.value}: {bug.message[:150]}")
        return result
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Testing unbind and .T implementations...")
    
    # Test unbind
    r1 = run_test(
        "unbind with tuple unpacking",
        test_unbind,
        {"x": (1, 196, 192)}
    )
    
    # Test .T
    r2 = run_test(
        ".T attribute transpose",
        test_t_attribute,
        {"x": (1, 128, 256)}
    )
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"unbind test: {'PASS' if r1 and not r1.abstained else 'FAIL'}")
    print(f".T test: {'PASS' if r2 and not r2.abstained else 'FAIL'}")
