import json
import threading
import urllib.error
import urllib.request

import pytest

from dexvision.dashboard import collect, make_server, outcome


@pytest.fixture
def runs(tmp_path):
    run = tmp_path / "pick" / "mlp_seed0"
    run.mkdir(parents=True)
    rows = ["step,loss,seconds"] + [f"{100 * i},{1 / i:.6f},{10.0 * i:.1f}" for i in range(1, 31)]
    (run / "train_log.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for step, successes in ((1000, 3), (2000, 7)):
        (run / f"step_{step:06d}.pt").touch()
        report = {"overall": {"successes": successes, "n": 10, "ci95": [0.1, 0.9]}}
        (run / f"val_step_{step:06d}.json").write_text(json.dumps(report), encoding="utf-8")
    (run / "step_003000.pt").touch()                                 # saved, not yet evaluated
    (run / "val_step_003000.json").write_text("{", encoding="utf-8")  # being written
    (run / "selection.json").write_text(json.dumps({"selected": {"step": 2000}}), "utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run.log").write_text("first\n\nprogress 1\rprogress 2\nlast\n", "utf-8")
    (tmp_path / "videos").mkdir()                                    # not a run: no train log
    return tmp_path


def test_collect_reads_loss_checkpoints_val_and_log(runs):
    state = collect(runs)
    (run,) = state["tasks"]["pick"]
    assert (run["method"], run["seed"]) == ("mlp", "0")
    assert run["loss"][0] == [100, 1.0, 10.0] and len(run["loss"]) == 30
    assert run["steps_per_s"] == pytest.approx(10.0)
    assert run["checkpoints"] == [1000, 2000, 3000]
    assert [(v["step"], v["successes"]) for v in run["val"]] == [(1000, 3), (2000, 7)]
    assert run["selected"] == 2000 and run["video"] is None
    assert state["logs"] == [{"name": "run.log",
                              "lines": ["first", "progress 1", "progress 2", "last"]}]


def test_video_names_give_the_outcome():
    assert outcome("keypoints_pick_vision_varied_100000_no_contact") == "no contact"
    assert outcome("pick_mlp_val_100000_success") == "success"


def test_server_serves_the_page_and_state_and_nothing_else(runs):
    server = make_server(runs, "127.0.0.1", 0, videos=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/", timeout=10) as reply:
            assert reply.status == 200 and b"DexVision training" in reply.read()
        with urllib.request.urlopen(base + "/api/state", timeout=10) as reply:
            assert list(json.load(reply)["tasks"]) == ["pick"]
        for path in ("/video/pick/mlp_seed0/step_001000/x.mp4", "/../AGENTS.md"):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path, timeout=10)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
