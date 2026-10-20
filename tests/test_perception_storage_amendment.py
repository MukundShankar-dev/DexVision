"""A storage ceiling amendment must not change frozen controller/entry behavior."""

import ast
from pathlib import Path

import pytest
import yaml

from dexvision.evaluation.instrumented_rollouts import storage_parent_digest
from dexvision.learning.training_manifest import digest_file


def parent_recipe(tmp_path, *, tamper=False):
    import dexvision.evaluation.instrumented_rollouts as module

    tree = ast.parse(Path(module.__file__).read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "run_instrumented_evaluation"
        ):
            node.body = [
                stmt
                for stmt in node.body
                if not (
                    isinstance(stmt, ast.Assign)
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "entry_protocol"
                )
            ]

            class Normalize(ast.NodeTransformer):
                def visit_Name(self, item):
                    return (
                        ast.parse('protocol["protocol_sha256"]', mode="eval").body
                        if item.id == "entry_protocol"
                        else item
                    )

            Normalize().visit(node)
        if (
            tamper
            and isinstance(node, ast.FunctionDef)
            and node.name == "execute_attempt"
        ):
            node.body.append(ast.Expr(value=ast.Constant(value="changed control")))
    own = "dexvision/evaluation/instrumented_rollouts.py"
    snapshot = tmp_path / "source_snapshot"
    source = snapshot / own
    source.parent.mkdir(parents=True)
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    parent = snapshot / "configs/level5/evaluation_perception.yaml"
    parent.parent.mkdir(parents=True)
    old = {
        "maximum_saved_bytes": 320 * 1024**2,
        "camera_stride": 3,
        "implementation_sha256": {own: digest_file(source), "observer": "fixed-digest"},
    }
    parent.write_text(yaml.safe_dump(old))
    current = {
        **old,
        "maximum_saved_bytes": 640 * 1024**2,
        "implementation_sha256": {
            **old["implementation_sha256"],
            own: digest_file(Path(module.__file__)),
        },
        "storage_amendment": {
            "parent_recipe": str(parent),
            "parent_sha256": digest_file(parent),
        },
    }
    return current, parent


def test_storage_only_amendment_retains_parent_and_rejects_method_changes(tmp_path):
    current, parent = parent_recipe(tmp_path)
    assert storage_parent_digest(current) == digest_file(parent)
    with pytest.raises(ValueError, match="Only the storage"):
        storage_parent_digest({**current, "camera_stride": 2})
    with pytest.raises(ValueError, match="observation/policy"):
        storage_parent_digest(
            {
                **current,
                "implementation_sha256": {
                    **current["implementation_sha256"],
                    "observer": "changed",
                },
            }
        )


def test_storage_amendment_cannot_hide_control_code_change(tmp_path):
    current, _ = parent_recipe(tmp_path, tamper=True)
    with pytest.raises(
        ValueError, match="changed control or evaluation: execute_attempt"
    ):
        storage_parent_digest(current)
