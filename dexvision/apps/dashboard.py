"""Serve a live view of the training runs in runs/ to a browser.

    python -m dexvision.apps.dashboard                 # then open http://localhost:8765

On the desktop itself, open that address in a browser. From another machine, forward the port
over ssh and open the same address there:

    ssh -N -L 8765:localhost:8765 desktop

The page shows each run's loss, checkpoints, val successes per checkpoint and one recorded val
episode of its newest checkpoint (dexvision/dashboard.py). It only listens on this machine
unless --host says otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dexvision import REPO_ROOT
from dexvision.dashboard import make_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs",
                        help="the runs directory to show (default: this checkout's)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-videos", action="store_true",
                        help="do not record an episode of each new checkpoint (one CPU core, "
                             "under a minute per checkpoint)")
    args = parser.parse_args()
    server = make_server(args.runs.resolve(), args.host, args.port, not args.no_videos)
    print(f"Dashboard for {args.runs} at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
