"""Select or freeze the checkpoint5.6 identity baseline; no optimizer updates."""

from __future__ import annotations
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("select", "freeze"))
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    a = p.parse_args()
    print("DexVision5.6 current-RGB localization:", a.command, flush=True)
    try:
        if a.command == "select":
            from dexvision.learning.perception_guarded_body import select_identity

            return 0 if select_identity(a.config)["strict_validation_pass"] else 2
        if not a.output_dir:
            raise ValueError("--output-dir required to freeze confirmation.")
        from dexvision.evaluation.perception_guarded_body import freeze_context_confirmation

        freeze_context_confirmation(a.config, a.output_dir)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as e:
        print("Current-RGB baseline failed:", e, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
