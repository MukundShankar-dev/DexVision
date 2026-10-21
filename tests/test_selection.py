import copy
from dataclasses import replace

import numpy as np
import pytest

from dexvision.data.demos import generate, load_data_config
from dexvision.eval.evaluator import evaluate
from dexvision.eval.ledger import read_ledger
from dexvision.eval.selection import evaluate_checkpoints, select
from dexvision.learning.dataset import DemoWindows
from dexvision.learning.policy import load_learned_policy
from dexvision.learning.train import load_method_config, train
from dexvision.policies import make_policy
from dexvision.sim.workcell import Workcell
from dexvision.tasks import make_task


def test_selection_prefers_successes_then_the_later_step():
    rows = [{"step": 10, "successes": 5}, {"step": 20, "successes": 7},
            {"step": 30, "successes": 7}, {"step": 40, "successes": 6}]
    assert select(rows)["step"] == 30
    with pytest.raises(ValueError):
        select([])


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """Two tiny MLP checkpoints trained on two real expert episodes."""
    root = tmp_path_factory.mktemp("run")
    workcell, task = Workcell(), make_task("pick")
    generate(workcell, task, make_policy("expert", workcell.action_space, "pick"),
             dict(load_data_config("pick"), episodes=2), root / "data")
    config = copy.deepcopy(load_method_config())
    config["mlp"]["hidden_dims"] = [32]
    config["training"].update(batch_size=16, checkpoint_every=3)
    h = config["horizons"]
    windows = DemoWindows.from_directory(root / "data", task.spec.observation,
                                         h["observation"], h["prediction"])
    train("mlp", windows, config, steps=6, record={"skill": "pick"},
          checkpoint=lambda t: t.save(root / f"step_{t.step:06d}.pt"))
    return root, task


def test_checkpoint_must_match_the_task(checkpoints, tmp_path):
    root, task = checkpoints
    policy, info = load_learned_policy(root / "step_000006.pt", task.spec)
    assert info["step"] == 6 and len(info["checkpoint_sha256"]) == 64
    assert info["training"]["skill"] == "pick"
    with pytest.raises(ValueError):
        load_learned_policy(root / "step_000006.pt", replace(task.spec, skill="push"))
    with pytest.raises(ValueError):
        load_learned_policy(root / "step_000006.pt",
                            replace(task.spec, observation=task.spec.observation[:-1]))


def test_checkpoint_test_runs_are_ledgered_with_their_hash(checkpoints, tmp_path):
    root, task = checkpoints
    policy, info = load_learned_policy(root / "step_000006.pt", task.spec)
    task.spec = replace(task.spec, horizon_steps=10)
    ledger = tmp_path / "ledger.jsonl"
    report = evaluate(Workcell(), task, policy, "test", 1, ledger_path=ledger, policy_info=info)
    entry = read_ledger(ledger)[0]
    assert entry["policy"] == "mlp" and report.policy_info["step"] == 6
    assert entry["policy_info"]["checkpoint_sha256"] == info["checkpoint_sha256"]


def test_parallel_checkpoint_evaluation(checkpoints):
    root, _ = checkpoints
    table = evaluate_checkpoints(sorted(root.glob("step_*.pt")), "pick", 1, root, workers=2)
    assert [row["step"] for row in table] == [3, 6]
    assert all(row["n"] == 1 and (root / row["report"]).is_file() for row in table)
    assert np.isin(select(table)["step"], [3, 6])
