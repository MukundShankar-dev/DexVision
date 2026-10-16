"""Phase windows are causal source derivations; consumed parents are refused."""

import pytest
import yaml
from dexvision.logging.visual_stream import digest_file


def test_consumed_phase_parent_rejected_before_payload_access(tmp_path, monkeypatch):
    import dexvision.logging.perception_late_windows as module

    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({"parents": [["protocol", "absent"]]}))
    p.with_suffix(".sha256").write_text(f"{digest_file(p)}  plan.yaml\n")
    monkeypatch.setattr(
        module, "load_protocol", lambda _: {"dataset_role": "confirmation"}
    )
    monkeypatch.setattr(
        module,
        "verify_dataset",
        lambda *_: pytest.fail("Consumed payload must stay closed"),
    )
    with pytest.raises(ValueError, match="Consumed/test"):
        module.freeze_phase_windows(p, tmp_path / "out")
    assert not (tmp_path / "out").exists()
