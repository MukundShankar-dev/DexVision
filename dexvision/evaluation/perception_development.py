"""Validation-selected repair and a separately frozen one-use confirmation."""
from __future__ import annotations

import json
from pathlib import Path

from dexvision.evaluation.perception_audit import accepted_inventory, feature_parity_diagnostics
from dexvision.evaluation.perception_confirmation import TypedBaseline, score_confirmation
from dexvision.learning.perception_development import development_rows, load_repair_recipe
from dexvision.logging.perception_development import freeze_protocol, load_protocol, verify_dataset
from dexvision.logging.visual_stream import digest_file, write_json
from dexvision.perception.perception_baseline import PerceptionBaseline, load_model, model_digest, save_artifact_index
from dexvision.perception.world_estimate import world_estimate_from_dict


def selected_method(config: str | Path):
    recipe = load_repair_recipe(config)
    model = load_model(recipe)
    return {**recipe, "tracker": model["tracker_config"]}, model


def review_validation(config: str | Path, output_dir: str | Path) -> dict:
    recipe, model = selected_method(config)
    rows = [r for r in development_rows(recipe) if r["split"] == "validation"]
    return evaluate_rows(recipe, model, rows, Path(output_dir), scope="validation")


def freeze_confirmation(config: str | Path, output_dir: str | Path) -> dict:
    import yaml
    spec = yaml.safe_load(Path(config).read_text())
    recipe, model = selected_method(spec["selected_recipe"])
    report = json.loads((Path(recipe["model_dir"])/"validation.json").read_text())
    if not report["all_numerical_gates_pass"] or not report["all_primary_gates_pass"]:
        raise ValueError("Both original and supplemental validation gates must pass before confirmation.")
    if spec["role"] != "confirmation" or spec["selected_model_sha256"] != model_digest(model):
        raise ValueError("Confirmation must bind the validation-selected model.")
    required = {spec["selected_recipe"], str(Path(recipe["model_dir"])/"selected.json"),
                str(Path(recipe["model_dir"])/"selection.json")}
    if not required.issubset(spec["implementation_files"]):
        raise ValueError("Confirmation must pin selected method artifacts before generating pixels.")
    return freeze_protocol(config, output_dir)


def evaluate_rows(recipe: dict, model: dict, rows: list[dict], output: Path, *, scope: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    wrapper = TypedBaseline(PerceptionBaseline(recipe, model))
    report = score_confirmation({**recipe, "review_scope": scope}, rows, wrapper, output)
    # Roundtrip real RGB estimates and verify all unavailable consumers reject.
    for frame in wrapper.frames:
        world_estimate_from_dict(frame["world"])
        if not frame["accepted_control_state_blocked"]:
            raise ValueError("Partial RGB evidence unexpectedly admitted a full control state.")
    write_json(output/"typed_inferred_frames.json", wrapper.frames)
    report.update(split=scope, model_sha256=model_digest(model), recipe_sha256=recipe["_recipe_sha256"],
                  automated_baseline_pass=report["all_numerical_gates_pass"] and report["all_primary_gates_pass"],
                  manual_acceptance="pending" if scope == "fresh_confirmation" else "development_only",
                  accepted_bundles=accepted_inventory(recipe), feature_diagnostics=feature_parity_diagnostics(recipe),
                  full_inferred_control_states=0, checkpoint_complete=False, future_checkpoint_started=False)
    write_json(output/"report.json", report)
    save_artifact_index(output)
    return report


def evaluate_confirmation(protocol_path: str | Path, dataset_dir: str | Path, output_dir: str | Path) -> dict:
    path, root, output = Path(protocol_path), Path(dataset_dir), Path(output_dir)
    protocol = load_protocol(path)
    if (protocol["version"] != "level5/perception-confirmation-v3" or
            protocol["dataset_role"] != "confirmation" or set(protocol["splits"]) != {"test"}):
        raise ValueError("Confirmation requires a separate fresh test-only protocol.")
    # One reservation per protocol, before any new label/image decoding or inference.
    attempt = path.with_name("confirmation_attempt.json")
    if attempt.exists():
        raise ValueError("Confirmation already consumed; another output path cannot retry it.")
    if output.exists():
        raise ValueError("Evaluation output exists.")
    verify_dataset(root, path)
    recipe, model = selected_method(protocol["selected_recipe"])
    if model_digest(model) != protocol["selected_model_sha256"]:
        raise ValueError("Selected confirmation method changed.")
    with attempt.open("x") as handle:
        json.dump({"protocol_sha256": digest_file(path), "model_sha256": model_digest(model),
                   "output_dir": output.as_posix(), "consumed": True}, handle, indent=2)
    owners = json.loads((root/"owners.json").read_text())
    rows = [json.loads(line) for line in (root/"frames.jsonl").read_text().splitlines()]
    cases = {c["episode_id"] for c in protocol["cases"]}
    if any(r["split"] != "test" or r["source_episode_id"] not in cases for r in rows):
        raise ValueError("Confirmation split/case ownership differs from frozen protocol.")
    for row in rows:
        row["evaluation_only_source_owner"] = owners[row["source_episode_id"]]
    recipe = {**recipe, "visual_root": root.as_posix()}
    return evaluate_rows(recipe, model, rows, output, scope="fresh_confirmation")


def review_artifacts(directory: str | Path) -> dict:
    """Replay saved media only; completion is not owner acceptance."""
    import cv2
    root = Path(directory)
    for line in (root/"SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        file = root/relative
        if not file.resolve().is_relative_to(root.resolve()) or digest_file(file) != expected:
            raise ValueError("Review artifact checksum/path mismatch.")
    report = json.loads((root/"report.json").read_text())
    review = json.loads((root/"manual_review.json").read_text())
    if not report["automated_baseline_pass"] or review["scope"] != "fresh_confirmation":
        raise ValueError("Owner qualification review requires passing fresh confirmation.")
    if any(p["status"] != "present" for p in review["panels"]):
        raise ValueError("Missing required class/condition review panel.")
    window = "DexVision 5.6 review: SPACE next/play; Q or ESC incomplete"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 960)

    def check_key(delay):
        key = cv2.waitKey(delay) & 0xff
        if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            raise ValueError("Manual review incomplete; no acceptance recorded.")
        return key

    try:
        sheet = cv2.imread(str(root/review["contact_sheet"]))
        if sheet is None:
            raise ValueError("Cannot decode contact sheet.")
        cv2.imshow(window, sheet)
        while check_key(100) != 32:
            pass
        for index, item in enumerate(review["videos"]):
            print(f"Review {index+1}/{len(review['videos'])}: {item['episode']} {item['condition']}", flush=True)
            capture = cv2.VideoCapture(str(root/item["path"]))
            frames = 0
            try:
                delay = max(1, round(1000/max(1., capture.get(cv2.CAP_PROP_FPS))))
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames += 1
                    cv2.imshow(window, frame)
                    if check_key(delay) == 32:
                        while check_key(100) != 32:
                            pass
                if frames == 0:
                    raise ValueError(f"Missing/empty review video: {item['path']}")
                print("Clip complete; SPACE for next clip.", flush=True)
                while check_key(100) != 32:
                    pass
            finally:
                capture.release()
    finally:
        cv2.destroyAllWindows()
    return {"reviewed_sequences": len(review["videos"]), "owner_acceptance": "must_be_confirmed_by_owner"}
