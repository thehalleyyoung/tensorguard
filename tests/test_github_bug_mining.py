"""Offline regression tests for the frozen GitHub-mined bug dataset (Step 11).

No network: operates only on the committed dataset + manifest. Guarantees the
dataset cannot silently drift, contains >= 500 real labeled bugs with valid
GitHub provenance, and that every label is consistent with its matched PyTorch
error signature.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MINE_DIR = os.path.join(REPO, "experiments_v5", "github_bug_mining")
if MINE_DIR not in sys.path:
    sys.path.insert(0, MINE_DIR)

import load as mined  # noqa: E402


def test_dataset_integrity_and_hash():
    # Raises MinedCorpusError on hash drift or label inconsistency.
    records = mined.verify_integrity()
    assert len(records) >= 500


def test_meets_500_target():
    m = mined.load_manifest()["meta"]
    assert m["total"] >= 500, f"only {m['total']} mined bugs (< 500 target)"
    assert m["total"] == len(mined.load_records())


def test_every_record_has_github_provenance():
    for r in mined.load_records():
        assert r["source_url"].startswith("https://github.com/"), r["source_url"]
        assert "/" in r["repository"]
        assert r["domain"] in ("shape", "device")
        assert r["category"]
        assert r["matched_signature"]


def test_labels_match_signatures():
    sigmap = mined.signature_label_map()
    for r in mined.load_records():
        assert r["matched_signature"] in sigmap
        assert (r["domain"], r["category"]) == sigmap[r["matched_signature"]]


def test_covers_both_domains_and_multiple_categories():
    recs = mined.load_records()
    domains = {r["domain"] for r in recs}
    cats = {r["category"] for r in recs}
    assert {"shape", "device"} <= domains
    assert len(cats) >= 6


def test_no_duplicate_urls():
    urls = [r["source_url"] for r in mined.load_records()]
    assert len(urls) == len(set(urls))


def test_manifest_breakdowns_match_dataset():
    recs = mined.load_records()
    m = mined.load_manifest()["meta"]
    by_domain, by_cat = {}, {}
    for r in recs:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    assert by_domain == m["by_domain"]
    assert by_cat == m["by_category"]
