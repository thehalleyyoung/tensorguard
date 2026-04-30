● Read round03_experiments.prompt.md
  │ .comet_neurips/round03_experiments.prompt.md
  └ 64 lines read

● List directory tests
  └ 109 files found

● Search (glob)
  │ "Makefile"
  └ No matches found

● Search (glob)
  │ "pyproject.toml"
  └ 2 files found

● Search (glob)
  │ "benchmarks/**/*.py"
  └ 151 files found

● Search (glob)
  │ "experiments/**/*.py"
  └ 209 files found

● Read pyproject.toml
  └ 44 lines read

● Check README and paper for headline claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -100 README.md 2>/dev/null || echo "No README"; head -50 neurips.tex
  │ 2>/dev/null | grep -A5 "abstract\|contributions\|achieve\|accuracy\|precisio
  │ n\|recall\|F1\|benchmark" | head -60
  └ 107 lines...

● Check experiments_v5 directory (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls experiments_v5/ | head -30
  └ 31 lines...

● Get paper abstract (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -A 30 "begin{abstract}" neurips.tex | head -50
  └ 32 lines...

● Check more experiments_v5 files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls experiments_v5/ | tail -30
  └ 31 lines...

● Check real_benchmarks and reproducibility dirs (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls real_benchmarks/ 2>/dev/null | head -20; ls reproducibility/
  │ 2>/dev/null | head -20
  └ 25 lines...

● Check if package imports (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python -c "import src; print('import ok')" 2>&1 | head -5
  └ 2 lines...

● Check existing benchmark results (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat experiments_v5/v5_benchmark_results.json | python -m json.tool
  │ 2>/dev/null | grep -E
  │ '"total|"detected|"precision|"recall|"f1|"rp_count|"bugs' | head -30
  └ 1 line...

● Check existing benchmark results JSON (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat experiments_v5/v5_benchmark_results.json | python3 -m json.tool
  │ 2>/dev/null | head -60
  └ 61 lines...

● Check if package imports with python3 (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "import src; print('import ok')" 2>&1 | head -5
  └ 2 lines...

● Find key numbers in benchmark results (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat experiments_v5/v5_benchmark_results.json | python3 -m json.tool
  │ 2>/dev/null | grep -E '"rp_|"total|"bugs_detected|"precision|"recall|"f1|unc
  │ onditional|53|60|32|34|25|15|26|488|356' | head -40
  └ 41 lines...

● Analyze bug corpus (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat experiments_v5/v5_bug_corpus.jsonl | python3 -c "
  │ import json,sys
  │ lines = [json.loads(l) for l in sys.stdin]
  │ print(f'Total entries: {len(lines)}')…
  └ 7 lines...

● Parse benchmark results JSON structure (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -m json.tool experiments_v5/v5_benchmark_results.json
  │ 2>/dev/null | python3 -c "
  │ import json,sys
  │ data = json.load(sys.stdin)…
  └ 59 lines...

● Check HCO results for 15/488 and 26/356 claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -m json.tool experiments_v5/v5_benchmark_results.json
  │ 2>/dev/null | python3 -c "
  │ import json,sys
  │ data = json.load(sys.stdin)…
  └ 54 lines...

● Check HCO numbers (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -m json.tool experiments_v5/v5_benchmark_results_hco.json
  │ 2>/dev/null | python3 -c "
  │ import json,sys
  │ data = json.load(sys.stdin)…
  └ 21 lines...

Now let me run the actual benchmarks to get fresh numbers:

● Run bug corpus tests (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && timeout 300 python3 -m pytest tests/test_bugs.py -v --tb=short 2>&1 |
  │ tail -30
  └ 31 lines...

● Run v5 benchmark (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && timeout 300 python3 experiments_v5/run_v5_benchmark.py 2>&1 | tail -40
  └ 41 lines...

● Get key benchmark numbers (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && timeout 300 python3 experiments_v5/run_v5_benchmark.py 2>&1 | grep -E
  │ "total|Refuted|Verified|Abstain|bug|block|elapsed|build_time" | head -30
  └ 31 lines...

● Run HCO benchmark for 15/488 and 26/356 claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && timeout 300 python3 experiments_v5/run_v5_benchmark_hco.py 2>&1 | grep -E
  │ "total|Refuted|Verified|Abstain|bug|block|rp_|unconditional|audited|elapsed|
  │ 356" | head -40
  └ 41 lines...

● Run Pytea baseline comparison for 32/34 vs 25/34 claim (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && timeout 300 python3 experiments_v5/run_baseline_comparison.py 2>&1 | tail
  │ -50
  └ 51 lines...

● List reproducibility files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls reproducibility/ | head -40
  └ 41 lines...

● Check Pytea baseline results (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat experiments_v5/pytea_baseline_results.json | python3 -c "
  │ import json,sys
  │ data = json.load(sys.stdin)
  │ print(json.dumps(data, indent=2)[:2000])…
  └ 29 lines...

● Check Pytea results keys (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data = json.load(open('experiments_v5/pytea_baseline_results.json'))
  │ keys = list(data.keys())…
  └ 4 lines...

● Find HuggingFace claim files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -r
  │ "9/9\|huggingface\|HuggingFace\|Llama\|Qwen\|Mistral\|Phi-3\|Gemma"
  │ experiments_v5/ --include="*.json" -l 2>/dev/null | head -5
  └ 6 lines...

● Check Pytea bug corpus comparison numbers (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data = json.load(open('experiments_v5/pytea_baseline_results.json'))
  │ bc = data.get('bug_corpus',{})…
  └ 61 lines...

● Check real bug corpus for HF numbers (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data = json.load(open('experiments_v5/v8/real_bug_corpus.json'))
  │ print(json.dumps(list(data.keys()), indent=2))…
  └ 13 lines...

● Find HF and Pytea-related reproducibility files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls reproducibility/ | grep -E "hf|huggingface|real_bug|pytea|fragment" |
  │ head -20
  └ 21 lines...

● Check real bug corpus list structure (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data = json.load(open('experiments_v5/v8/real_bug_corpus.json'))
  │ print(f'Type: {type(data)}')…
  └ 16 lines...

● Check all real bug corpus items (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data = json.load(open('experiments_v5/v8/real_bug_corpus.json'))
  │ for item in data:…
  └ 13 lines...

● Check Pytea symmetric comparison results (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data = json.load(open('reproducibility/pytea_2022_symmetric.json'))
  │ print(json.dumps(data, indent=2)[:2000])…
  └ 16 lines...

● Check audited footprint unconditional RP for 26/356 claim (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data =
  │ json.load(open('reproducibility/audited_footprint_unconditional_rp.json'))…
  └ 73 lines...

● Quick test of model checker (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && timeout 60 python3 -m pytest tests/test_model_checker.py -x --tb=short -q
  │ 2>&1 | tail -15
  └ 3 lines...

● Check block corpus 488 reconciliation (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data =
  │ json.load(open('reproducibility/block_corpus_488_reconciliation.json'))…
  └ 77 lines...

● Check per_block_user_visible_rp for 15/488 claim (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data = json.load(open('experiments_v5/v8/per_block_user_visible_rp.json'))
  │ print(json.dumps(data, indent=2)[:2000])…
  └ 70 lines...

● Check real bugs directory (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls experiments_v5/v8/real_bugs/ | head -15
  └ 11 lines...

● Run full test suite for pass/fail count (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && time timeout 300 python3 -m pytest tests/ -q --tb=no 2>&1 | tail -10
  └ 1 line...

● Read shell output Waiting up to 120 seconds for command output
  └ 15 lines...

● Run HF real bug repros to verify 9/9 claim (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && time timeout 120 python3 -c "
  │ import json, os, subprocess, time, sys
  │ sys.path.insert(0, '.')
  │ from src.api import verify_module_source…
  └ 34 lines...

● Check public API functions (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "^def \|^class " src/api.py | head -20
  └ 21 lines...

● Run HF real bug repros with analyze API (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && time timeout 120 python3 -c "
  │ import json, os, time, sys
  │ sys.path.insert(0, '.')
  │ from src.api import analyze…
  └ 41 lines...

● Check format of repro file (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -30 experiments_v5/v8/real_bugs/rb_001_xlstm_matq_view.py
  └ 31 lines...

● See full repro file (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat experiments_v5/v8/real_bugs/rb_001_xlstm_matq_view.py
  └ 34 lines...

● Test analyze on single repro file (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import sys; sys.path.insert(0,'.')
  │ from src.api import analyze
  │ src = open('experiments_v5/v8/real_bugs/rb_001_xlstm_matq_view.py').read()…
  └ 11 lines...

● Check AnalysisResult attributes (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import sys; sys.path.insert(0,'.')
  │ from src.api import analyze
  │ src = open('experiments_v5/v8/real_bugs/rb_001_xlstm_matq_view.py').read()…
  └ 13 lines...

● Test verify_architecture on view mismatch (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import sys; sys.path.insert(0,'.')
  │ from src.api import analyze
  │ src = open('experiments_v5/v8/real_bugs/rb_001_xlstm_matq_view.py').read()…
  └ 23 lines...

● Check how v5 benchmark runs (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -100 experiments_v5/run_v5_benchmark.py
  └ 101 lines...

● Check how bug repros are run (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -A 30 "def run_one_bug" experiments_v5/run_v5_benchmark.py | head
  │ -40
  └ 1 line...

● How bug repros are processed (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "bug_repro\|BUG_REPRO\|repro" experiments_v5/run_v5_benchmark.py
  │ | head -20
  └ 9 lines...

● Read run_v5_benchmark.py
  │ experiments_v5/run_v5_benchmark.py
  └ L208:270 (61 lines read)

● Find _run_one and _decide functions (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "def _run_one\|def _decide" experiments_v5/run_v5_benchmark.py
  └ 3 lines...

● Read run_v5_benchmark.py
  │ experiments_v5/run_v5_benchmark.py
  └ L115:160 (46 lines read)

● Run HF repros with input_shapes parameter (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import sys; sys.path.insert(0,'.')
  │ from src.api import verify_architecture
  │ import json, re, time…
  └ 41 lines...

● List all pytea reproducibility files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls reproducibility/ | grep pytea
  └ 15 lines...

● Check Pytea comparison details (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data = json.load(open('reproducibility/pytea_2022_symmetric.json'))
  │ print(json.dumps(data, indent=2))…
  └ 16 lines...

● Check pytea_fragment_fair for 32/34 vs 25/34 claim (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data = json.load(open('reproducibility/pytea_fragment_fair.json'))
  │ print(json.dumps(data, indent=2)[:1500])…
  └ 32 lines...

● Check bug corpus integrity for 53/60 claim (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data = json.load(open('experiments_v5/v5_bug_corpus_integrity.json'))
  │ print(json.dumps(data, indent=2)[:2000])…
  └ 60 lines...

● Check bugs within soundness scope for 53/60 claim (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data =
  │ json.load(open('reproducibility/bug_corpus_in_soundness_scope.json'))…
  └ 75 lines...

● Check soundness scope summary (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "
  │ import json
  │ data =
  │ json.load(open('reproducibility/bug_corpus_in_soundness_scope.json'))…
  └ 11 lines...

● Find 53/60 claim and Refuted-Proof references in paper (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "53\|88.3\|88\.3\|Refuted-Proof\|rp_count" neurips.tex | head -20
  └ 4 lines...

● Understand Refuted-Proof vs Refuted distinction (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "Refuted-Proof\|RP\|high_confidence\|proof_backed\|z3_proof"
  │ src/api.py | head -20
  └ 12 lines...

● Find HIGH confidence mapping (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "HIGH\|MEDIUM\|severity\|confidence" src/api.py | grep -i
  │ "confidence\|HIGH" | head -15
  └ 16 lines...

● Get details on the 2 failing tests (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && timeout 30 python3 -m pytest tests/test_real_model_analysis.py::TestKnown
  │ ShapeBugs::test_missing_unsqueeze tests/test_refinement_track_c.py::TestSymb
  │ olicConfig::test_make_expression_symbolic_multiply -v --tb=short 2>&1 | tail
  │ -30
  └ 31 lines...

Now I have all the data. Let me write the report:

● Ensure output directory exists (shell)
  │ mkdir -p /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorg
  │ uard/.comet_neurips
  └ 1 line...

● Create round03_experiments.md +99
  └ .comet_neurips/round03_experiments.md

Report written to `.comet_neurips/round03_experiments.md`. Key findings:

**Agrees with paper:**
- Pytea head-to-head: **32/34 vs. 25/34**, McNemar p=0.0156 ✅
- Unconditional RP on audited footprint: **26 total, 5 inside Lean-audited handlers** ✅

**Disagrees with paper:**
- Bug corpus: code now finds **56/60 (93.3%)**, not claimed 53/60 (88.3%) — 3 extra catches
- HF real-bug claim: corpus is **10/10** (not 9/9), and the 10 bugs are from xLSTM/GPT-NeoX/ConvBERT/etc., not from Llama/Qwen2/Mistral/Phi-3/Gemma 2 as stated in the abstract

**Also noted:** 2 pre-existing test failures — `ast.Num` removed in Python 3.12+ (compatibility break) and one missed-bug regression in `test_missing_unsqueeze`.


Changes   +99 -0
Requests  1 Premium (11m 58s)
Tokens    ↑ 3.0m • ↓ 20.1k • 2.9m (cached)
