# Third-party conformance certification

TensorGuard exposes two extension paths for library authors:

- declarative community stubs, validated by `src.stub_governance`;
- trusted executable operator plugins, validated by `src.operator_plugin_abi`.

`src.third_party_conformance` is the reusable pytest-facing certification layer
on top of both. It does not merely call a transfer function. It installs the
stub/plugin in an isolated registry snapshot, runs real `verify_architecture`
scenarios, and checks the expected `SAFE` / `UNSAFE` / `UNKNOWN` verdict in
`sound`, `balanced`, and `heuristic` modes. The registry is restored after the
run, so a failed certification cannot leak a bad stub into later tests.

## Plugin example

```python
from src.third_party_conformance import (
    ThirdPartyConformanceScenario,
    assert_conformance_passed,
    certify_plugin_contracts,
)
from my_library.tensorguard_plugin import contracts

GOOD = """
import torch.nn as nn
from my_library import FancyBlock
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = FancyBlock(8, 16)
        self.head = nn.Linear(16, 2)
    def forward(self, x):
        return self.head(self.block(x))
"""

BAD = GOOD.replace("nn.Linear(16, 2)", "nn.Linear(99, 2)")

def test_tensorguard_certification():
    report = certify_plugin_contracts(
        contracts(),
        [
            ThirdPartyConformanceScenario(
                name="clean-path",
                source=GOOD,
                input_shapes={"x": ("batch", 8)},
                expected_verdicts={"*": "SAFE"},
            ),
            ThirdPartyConformanceScenario(
                name="downstream-mismatch",
                source=BAD,
                input_shapes={"x": ("batch", 8)},
                expected_verdicts={"*": "UNSAFE"},
                expected_bug_substrings=("Linear expects",),
            ),
        ],
        extension_name="my-library",
    )
    assert_conformance_passed(report)
```

## Stub-manifest example

```python
from src.third_party_conformance import (
    ThirdPartyConformanceScenario,
    certify_stub_manifests,
)

report = certify_stub_manifests(
    [manifest_dict],
    [
        ThirdPartyConformanceScenario(
            name="sound-mode-clean",
            source=GOOD,
            input_shapes={"x": ("batch", 8)},
            expected_verdicts={"sound": "SAFE", "balanced": "SAFE"},
        )
    ],
    modes=("sound", "balanced"),
)
```

Reports can be persisted with `report.to_json()` or `report.to_markdown()` for a
release artifact or model-card certification note.
