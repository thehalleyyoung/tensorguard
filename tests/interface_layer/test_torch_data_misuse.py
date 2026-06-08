from src.interface_layer.torch_data_misuse import (
    GuaranteeTier,
    TorchDataMisuseKind,
    analyze_torch_data_source,
    render_torch_data_report_json,
    render_torch_data_report_text,
)


WORKER_RNG_BUG = '''
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AugDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        x = self.items[i]
        noise = np.random.randn(*x.shape)   # unsafe: shared across workers
        return x + noise

ds = AugDataset(data)
loader = DataLoader(ds, batch_size=32, num_workers=4, shuffle=True)
'''


def test_worker_rng_duplication_flagged():
    report = analyze_torch_data_source(WORKER_RNG_BUG, path="train.py")
    kinds = {f.kind for f in report.findings}
    assert TorchDataMisuseKind.WORKER_RNG_DUPLICATION in kinds
    f = next(f for f in report.findings if f.kind is TorchDataMisuseKind.WORKER_RNG_DUPLICATION)
    assert f.guarantee is GuaranteeTier.HEURISTIC
    assert "AugDataset" in f.message


def test_worker_rng_safe_with_worker_init_fn():
    src = WORKER_RNG_BUG.replace(
        "num_workers=4, shuffle=True)",
        "num_workers=4, shuffle=True, worker_init_fn=seed_worker)",
    )
    report = analyze_torch_data_source(src)
    assert not any(
        f.kind is TorchDataMisuseKind.WORKER_RNG_DUPLICATION for f in report.findings
    )


def test_worker_rng_safe_with_torch_rng():
    src = WORKER_RNG_BUG.replace(
        "noise = np.random.randn(*x.shape)   # unsafe: shared across workers",
        "noise = torch.randn(x.shape)",
    )
    report = analyze_torch_data_source(src)
    assert not any(
        f.kind is TorchDataMisuseKind.WORKER_RNG_DUPLICATION for f in report.findings
    )


def test_worker_rng_safe_when_single_worker():
    src = WORKER_RNG_BUG.replace("num_workers=4", "num_workers=0")
    report = analyze_torch_data_source(src)
    assert not any(
        f.kind is TorchDataMisuseKind.WORKER_RNG_DUPLICATION for f in report.findings
    )


DROP_LAST_EVAL_BUG = '''
from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=True)
'''


def test_drop_last_on_eval_flagged_only_for_val():
    report = analyze_torch_data_source(DROP_LAST_EVAL_BUG, path="loaders.py")
    findings = [f for f in report.findings if f.kind is TorchDataMisuseKind.DROP_LAST_ON_EVAL]
    assert len(findings) == 1
    assert "val" in findings[0].message.lower()


FIT_BEFORE_SPLIT_BUG = '''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
'''


def test_fit_before_split_leakage_flagged():
    report = analyze_torch_data_source(FIT_BEFORE_SPLIT_BUG, path="prep.py")
    findings = [f for f in report.findings if f.kind is TorchDataMisuseKind.FIT_BEFORE_SPLIT_LEAKAGE]
    assert len(findings) == 1
    assert "leak" in findings[0].message.lower()


def test_fit_after_split_is_clean():
    src = '''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
'''
    report = analyze_torch_data_source(src)
    assert not any(
        f.kind is TorchDataMisuseKind.FIT_BEFORE_SPLIT_LEAKAGE for f in report.findings
    )


def test_clean_source_has_no_findings():
    src = '''
from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
'''
    report = analyze_torch_data_source(src)
    assert report.ok


def test_unparseable_source_abstains():
    report = analyze_torch_data_source("def broken(:\n  pass")
    assert report.abstained
    assert report.ok


def test_serialization_roundtrip():
    report = analyze_torch_data_source(WORKER_RNG_BUG, path="t.py")
    js = render_torch_data_report_json(report)
    assert '"torch-data-misuse"' in js
    txt = render_torch_data_report_text(report)
    assert "worker-rng-duplication" in txt
