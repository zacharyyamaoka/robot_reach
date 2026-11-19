"""Command line.

    python -m robot_reach examples            # compute (or reuse) all five examples, view them
    python -m robot_reach run ur5e rr         # compute + save + PNG figures
    python -m robot_reach view out/ur5e       # view saved results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "out"


def _run(names, out: Path, fresh: bool):
    from . import analyze, load
    from .examples import EXAMPLES

    results = {}
    for name in names:
        folder = out / name
        if not fresh and (folder / "result.json").exists():
            try:
                results[name] = load(folder)
                print(f"{name}: reused {folder}", flush=True)
                continue
            except Exception as exc:  # stale schema -> recompute
                print(f"{name}: recomputing ({exc})", flush=True)
        factory, kwargs = EXAMPLES[name]
        t = time.perf_counter()
        res = analyze(factory(), **kwargs)
        res.save(folder)
        try:
            from .plots import save_figures

            save_figures(res, folder / "figures")
        except ImportError:
            pass
        print(f"{name}: computed in {time.perf_counter() - t:.1f} s -> {folder}", flush=True)
        print(json.dumps(res.summary(), indent=None)[:600], flush=True)
        results[name] = res
    return results


def main(argv=None) -> int:
    from .examples import EXAMPLES

    ap = argparse.ArgumentParser(prog="robot_reach")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "examples"):
        p = sub.add_parser(cmd)
        if cmd == "run":
            p.add_argument("arms", nargs="+", choices=[*EXAMPLES, "all"])
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--fresh", action="store_true", help="recompute even if a saved result exists")
        p.add_argument("--port", type=int, default=0)
        p.add_argument("--no-view", action="store_true")
    v = sub.add_parser("view")
    v.add_argument("folders", nargs="+", type=Path)
    v.add_argument("--port", type=int, default=0)
    for p in (sub.choices["run"], sub.choices["examples"], v):
        p.add_argument("--arm", help="which arm to open first")
        p.add_argument("--select", help="x,y,z: preselect the nearest workspace point (a scripted click)")
        p.add_argument("--solution", type=int, default=1, help="which of that point's solutions to show")
    a = ap.parse_args(argv)

    from .viewer import ReachViewer

    if a.cmd == "view":
        from . import load

        results = {f.name: load(f) for f in a.folders}
        _open(ReachViewer, results, a)
        return 0
    names = list(EXAMPLES) if a.cmd == "examples" or "all" in a.arms else a.arms
    results = _run(names, a.out, a.fresh)
    if not a.no_view:
        _open(ReachViewer, results, a)
    return 0


def _open(ReachViewer, results, a):
    if a.arm:
        results = {a.arm: results[a.arm], **{k: v for k, v in results.items() if k != a.arm}}
    viewer = ReachViewer(results, port=a.port)
    if a.select:
        viewer.select_nearest([float(v) for v in a.select.split(",")], a.solution)
    viewer.run()


if __name__ == "__main__":
    sys.exit(main())
