from dataclasses import replace
from functools import partial

import pytest
import yaml

from dexvision.eval.evaluator import evaluate, run_episode
from dexvision.eval.ledger import BudgetExceeded, check_test_budget, read_ledger
from dexvision.policies import make_policy
from dexvision.sim.workcell import Workcell
from dexvision.tasks import make_task
from dexvision.tasks.spec import TaskSpecError, load_task_spec


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


def short_task(steps=5):
    task = make_task("pick")
    task.spec = replace(task.spec, horizon_steps=steps)
    return task


def test_random_policy_episodes_are_reproducible(workcell):
    task, policy = short_task(15), make_policy("random", workcell.action_space, "pick")
    first = run_episode(workcell, task, policy, 100003)
    second = run_episode(workcell, task, policy, 100003)
    assert first == second


def test_report_counts_unique_seeds_with_intervals(workcell, tmp_path):
    zero = make_policy("zero", workcell.action_space, "pick")
    report = evaluate(workcell, short_task(), zero, "val", 4, ledger_path=tmp_path / "ledger.jsonl")
    assert report.overall.n == 4 and report.overall.successes == 0
    assert report.overall.ci95[0] == 0.0 and 0 < report.overall.ci95[1] < 1
    assert sorted(r.seed for r in report.episodes) == [100000, 100001, 100002, 100003]
    assert sum(rate.n for rate in report.by_object.values()) == 4
    assert not (tmp_path / "ledger.jsonl").exists()  # val runs are not ledgered


def test_progress_callback_sees_every_episode(workcell, tmp_path):
    zero = make_policy("zero", workcell.action_space, "pick")
    calls = []
    report = evaluate(workcell, short_task(), zero, "val", 3, ledger_path=tmp_path / "ledger.jsonl",
                      on_episode=lambda episodes, total: calls.append((list(episodes), total)))
    assert [(len(episodes), total) for episodes, total in calls] == [(1, 3), (2, 3), (3, 3)]
    assert calls[-1][0] == report.episodes


def _random_policy(workcell):
    return make_policy("random", workcell.action_space, "pick")


def test_parallel_evaluation_matches_serial(workcell, tmp_path):
    task, ledger = short_task(15), tmp_path / "ledger.jsonl"
    serial = evaluate(workcell, task, _random_policy(workcell), "val", 12, ledger_path=ledger)
    calls = []
    parallel = evaluate(workcell, task, _random_policy(workcell), "val", 12, ledger_path=ledger,
                        workers=2, policy_factory=partial(_random_policy),
                        on_episode=lambda episodes, total: calls.append(len(episodes)))
    assert parallel.episodes == serial.episodes
    assert calls == list(range(1, 13))


def test_test_runs_are_ledgered_and_budgeted(workcell, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    task, policy = short_task(), make_policy("zero", workcell.action_space, "pick")
    for _ in range(task.spec.test_budget):
        evaluate(workcell, task, policy, "test", 1, ledger_path=ledger)
    entries = read_ledger(ledger)
    assert len(entries) == task.spec.test_budget
    assert entries[0]["skill"] == "pick" and entries[0]["seeds"] == [200000, 200001]
    assert "commit" in entries[0]["code"]
    with pytest.raises(BudgetExceeded):
        evaluate(workcell, task, policy, "test", 1, ledger_path=ledger)
    test_range = task.spec.splits["test"]
    assert check_test_budget("pick", "random", task.spec.test_budget, test_range, ledger) == 3
    # a new test range (a changed spec) starts a new budget; the old runs stay in the ledger
    assert check_test_budget("pick", "zero", task.spec.test_budget, (300000, 310000), ledger) == 3


def test_split_seed_ranges_are_validated(tmp_path):
    spec = load_task_spec("pick")
    with pytest.raises(TaskSpecError):
        spec.seeds("val", 10**6)
    raw = dict(spec.raw, splits={"train": [0, 10], "val": [5, 20], "test": [30, 40]})
    path = tmp_path / "overlap.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(TaskSpecError):
        load_task_spec(path)
