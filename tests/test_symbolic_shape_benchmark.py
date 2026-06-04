import json

from reproducibility import symbolic_shape_benchmark as bench


def test_symbolic_shape_benchmark_has_100_cases():
    items = bench.cases()
    assert len(items) == 100
    assert {case.family for case in items} == {
        "annotated_linear_bug",
        "conv2d_bug",
        "conv2d_safe",
        "docstring_linear_bug",
        "linear_abstain",
    }


def test_symbolic_shape_benchmark_expected_results():
    results = bench.run_cases(bench.cases())
    assert len(results) == 100
    assert all(result.passed for result in results)
    payload = json.loads(bench._render_json(results))
    assert payload["summary"]["passed"] == 100
    assert payload["summary"]["families"]["docstring_linear_bug"]["passed"] == 10
    assert payload["summary"]["families"]["linear_abstain"]["passed"] == 10
