"""
Test hook for Lean ↔ Python parity.

Loads experiments/lean_parity_results.json and asserts agreement rate >= 99%.
"""

import json
import os
import pytest

def test_lean_parity_results_exist():
    """Verify that parity results file exists."""
    results_path = "experiments/lean_parity_results.json"
    assert os.path.exists(results_path), f"Parity results not found at {results_path}"

def test_lean_parity_agreement_rates():
    """Assert that each operator has >= 99% agreement rate."""
    results_path = "experiments/lean_parity_results.json"
    
    with open(results_path) as f:
        results = json.load(f)
    
    assert "ops" in results, "Results missing 'ops' field"
    assert len(results["ops"]) >= 20, f"Expected ≥20 ops, got {len(results['ops'])}"
    
    for op in results["ops"]:
        agreement_rate = op["agreements"] / op["tests"]
        assert agreement_rate >= 0.99, (
            f"Operator {op['name']} has agreement rate {agreement_rate:.2%} < 99%"
        )

def test_lean_parity_overall_agreement():
    """Assert that overall agreement rate is >= 99%."""
    results_path = "experiments/lean_parity_results.json"
    
    with open(results_path) as f:
        results = json.load(f)
    
    overall_rate = results["summary"]["overall_agreement_rate"]
    assert overall_rate >= 0.99, (
        f"Overall agreement rate {overall_rate:.2%} < 99%"
    )

def test_lean_parity_num_tests():
    """Assert that we have ≥20,000 total test cases."""
    results_path = "experiments/lean_parity_results.json"
    
    with open(results_path) as f:
        results = json.load(f)
    
    total_tests = results["summary"]["total_tests"]
    assert total_tests >= 20000, (
        f"Expected ≥20,000 total tests, got {total_tests}"
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
