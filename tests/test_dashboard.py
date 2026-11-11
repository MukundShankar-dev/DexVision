import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

from dexvision.dashboard import EpisodeVideos, collect, current_runs, make_server, outcome


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
    stale = time.time() - 86400
    os.utime(run / "train_log.csv", (stale, stale))                  # trained yesterday
    (tmp_path / "videos").mkdir()                                    # not a run: no train log
    return tmp_path


def test_collect_reads_loss_checkpoints_val_and_log(runs):
    state = collect(runs)
    assert state["skills"]["pick"]["task"] == "pick"
    (run,) = state["skills"]["pick"]["runs"]
    assert (run["method"], run["seed"]) == ("mlp", "0")
    assert run["loss"][0] == [100, 1.0, 10.0] and len(run["loss"]) == 30
    assert run["steps_per_s"] == pytest.approx(10.0)
    assert run["checkpoints"] == [1000, 2000, 3000]
    assert [(v["step"], v["successes"]) for v in run["val"]] == [(1000, 3), (2000, 7)]
    assert run["selected"] == 2000 and run["video"] is None
    (log,) = state["logs"]
    assert (log["name"], log["lines"]) == ("run.log", ["first", "progress 1", "progress 2", "last"])
    old = runs / "logs" / "finished.log"
    old.write_text("Traceback from an earlier job\n", encoding="utf-8")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    assert [x["name"] for x in collect(runs)["logs"]] == ["run.log"]   # only logs being written


def test_episode_status_says_what_is_shown_and_what_is_pending(runs):
    videos = EpisodeVideos(runs)
    run = runs / "pick" / "mlp_seed0"
    assert videos.status(runs / "pick" / "nothing_seed0")["state"] == "none"
    assert videos.status(run) == {"state": "queued", "shown": None, "newest_step": 3000,
                                  "error": None}
    assert videos._targets() == []                      # an old run nobody is looking at
    videos.want("pick")
    assert videos._targets() == [run / "step_003000.pt"]
    earlier = runs / "dashboard" / "pick" / "mlp_seed0" / "step_002000"
    earlier.mkdir(parents=True)
    (earlier / "pick_mlp_val_100000_success.mp4").touch()
    videos.busy = run / "step_003000.pt"
    status = videos.status(run)                         # the earlier episode stays up meanwhile
    assert status["state"] == "recording" and status["shown"] == {
        "step": 2000, "outcome": "success",
        "url": "/video/pick/mlp_seed0/step_002000/pick_mlp_val_100000_success.mp4"}
    newest = runs / "dashboard" / "pick" / "mlp_seed0" / "step_003000"
    newest.mkdir()
    (newest / "pick_mlp_val_100000_dropped.mp4").touch()
    assert videos.status(run)["state"] == "ready" and videos._targets() == []


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
            assert list(json.load(reply)["skills"]) == ["pick"]
        for path in ("/video/pick/mlp_seed0/step_001000/x.mp4", "/../AGENTS.md"):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path, timeout=10)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_only_the_current_version_of_each_skill_is_shown(runs):
    def run(task, name, age_s):
        directory = runs / task / name
        directory.mkdir(parents=True)
        (directory / "train_log.csv").write_text("step,loss,seconds\n", encoding="utf-8")
        os.utime(directory / "train_log.csv", (time.time() - age_s,) * 2)
        return directory
    newer = [run("pick_vision_varied", "diffusion_seed0", 5),
             run("pick_vision_varied", "mlp_seed1", 9)]
    run("pick_vision_varied", "mlp_seed0", 500)        # an earlier seed of the same method
    press = run("press_depth_limit", "mlp_seed0", 50)  # a variant whose spec is long gone
    run("logs_like", "mlp_seed0", 1)                   # not a skill
    current = current_runs(runs)
    assert current == {"pick": ("pick_vision_varied", newer),
                       "press": ("press_depth_limit", [press])}
    state = collect(runs)
    shown = [r["name"] for r in state["skills"]["pick"]["runs"]]
    assert shown == ["diffusion_seed0", "mlp_seed1"]
