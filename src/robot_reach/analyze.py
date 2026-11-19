"""analyze(): compute everything once; ReachResult: the result log you view later.

WHY compute-once, view-later (Zach, 2026-11-19: "it does the computation once and then you can like
view the results in viser. Reachmap.display()"; kept from v1, "Reach computation remains independent
of visualization"): the result is plain arrays on disk, so a viewer, a report, a notebook or
another package reads the same numbers, and nothing recomputes to look.
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .arm import Arm
from .codes import Code
from .joint_space import JointMap, JointRanges, joint_map, joint_ranges
from .orientations import OrientationSet
from .reach import ReachMap, reach_map
from .sampling import Box, fk_cloud, sample_joints

RESULT_SCHEMA = "robot_reach.result.v1"


@dataclass(eq=False)
class ReachResult:
    arm_info: dict
    cloud_q: np.ndarray  # (N, dof) joint samples
    cloud_p: np.ndarray  # (N, 3) their FK positions (NaN where FK failed)
    cloud_code: np.ndarray  # (N,) joint-space code of each sample
    reach: ReachMap | None
    joint_maps: list[JointMap]
    ranges: JointRanges
    config: dict
    timings: dict
    arm: Arm | None = None  # live functions; None when loaded and the arm cannot be rebuilt
    path: Path | None = field(default=None)

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        out = {
            "arm": self.arm_info["name"],
            "dof": len(self.arm_info["joint_names"]),
            "task": self.arm_info["task"],
            "fk_samples": int(len(self.cloud_q)),
            "fk_sample_codes": _counts(self.cloud_code),
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
        }
        if self.reach is not None:
            lv = self.reach.levels()
            out["reach"] = {
                "positions": self.reach.n_positions,
                "orientations": self.reach.n_orientations,
                "ik_branches": self.reach.n_branches,
                "targets": self.reach.n_positions * self.reach.n_orientations,
                "ok_solutions": int(len(self.reach.sol_index)),
                "branch_codes": self.reach.code_counts(),
                "points_by_level": {str(k): int((lv == k).sum()) for k in (3, 4, 5, 6)},
                "max_solutions_at_one_pose": int(self.reach.n_solutions.max()) if self.reach.n_positions else 0,
            }
        out["joint_maps"] = [
            {"axes": list(m.names), "mode": m.mode, "ok_fraction": round(float(m.ok_fraction.mean()), 4), "codes": m.code_counts()}
            for m in self.joint_maps
        ]
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.reach is not None:
            self.reach.save(folder / "reach")
        np.savez_compressed(folder / "cloud.npz", q=self.cloud_q.astype(np.float32), p=self.cloud_p.astype(np.float32), code=self.cloud_code)
        jm = {}
        for i, m in enumerate(self.joint_maps):
            jm[f"m{i}_code"] = m.code
            jm[f"m{i}_ok"] = m.ok_fraction
            jm[f"m{i}_q0"] = m.q0
            for a, g in enumerate(m.grids):
                jm[f"m{i}_grid{a}"] = g
            if m.torque is not None:
                jm[f"m{i}_torque"] = m.torque
            if m.manipulability is not None:
                jm[f"m{i}_manip"] = m.manipulability
        r = self.ranges
        jm["ranges_limits"] = r.limits
        if r.used is not None:
            jm["ranges_used"], jm["ranges_edges"], jm["ranges_hist"] = r.used, r.hist_edges, r.hist
        np.savez_compressed(folder / "joint_maps.npz", **jm)
        doc = {
            "schema": RESULT_SCHEMA,
            "arm": self.arm_info,
            "config": self.config,
            "joint_maps": [{"axes": list(m.axes), "names": list(m.names), "mode": m.mode} for m in self.joint_maps],
            "summary": self.summary(),
        }
        (folder / "result.json").write_text(json.dumps(doc, indent=2, default=_json_default))
        self.path = folder
        return folder

    def display(self, **kwargs):
        """Open the viser viewer on this result (blocks until Ctrl-C unless block=False)."""
        from .viewer import ReachViewer

        return ReachViewer(self, **kwargs).run()


def load(folder: str | Path, rebuild_arm: bool = True) -> ReachResult:
    folder = Path(folder)
    doc = json.loads((folder / "result.json").read_text())
    if doc.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"{folder}: schema {doc.get('schema')!r}, expected {RESULT_SCHEMA!r}")
    reach_path = folder / "reach.reach.npz"
    reach = ReachMap.load(reach_path) if reach_path.exists() else None
    with np.load(folder / "cloud.npz", allow_pickle=False) as z:
        cloud_q, cloud_p, cloud_code = z["q"].astype(float), z["p"].astype(float), z["code"]
    maps = []
    with np.load(folder / "joint_maps.npz", allow_pickle=False) as z:
        for i, spec in enumerate(doc["joint_maps"]):
            grids = tuple(z[f"m{i}_grid{a}"] for a in range(len(spec["axes"])))
            maps.append(
                JointMap(
                    arm_name=doc["arm"]["name"],
                    axes=tuple(spec["axes"]),
                    names=tuple(spec["names"]),
                    grids=grids,
                    code=z[f"m{i}_code"],
                    ok_fraction=z[f"m{i}_ok"],
                    mode=spec["mode"],
                    q0=z[f"m{i}_q0"],
                    torque=z[f"m{i}_torque"] if f"m{i}_torque" in z else None,
                    manipulability=z[f"m{i}_manip"] if f"m{i}_manip" in z else None,
                )
            )
        ranges = JointRanges(
            tuple(doc["arm"]["joint_names"]),
            z["ranges_limits"],
            z["ranges_used"] if "ranges_used" in z else None,
            z["ranges_edges"] if "ranges_edges" in z else None,
            z["ranges_hist"] if "ranges_hist" in z else None,
        )
    arm = rebuild(doc["arm"]) if rebuild_arm else None
    return ReachResult(
        arm_info=doc["arm"],
        cloud_q=cloud_q,
        cloud_p=cloud_p,
        cloud_code=cloud_code,
        reach=reach,
        joint_maps=maps,
        ranges=ranges,
        config=doc["config"],
        timings=doc["summary"].get("timings_s", {}),
        arm=arm,
        path=folder,
    )


def rebuild(arm_info: dict) -> Arm | None:
    """Re-create an arm from its recorded factory, so an offline view can still pose it."""
    factory = arm_info.get("factory")
    if not factory:
        return None
    module, name, kwargs = factory
    try:
        return getattr(importlib.import_module(module), name)(**kwargs)
    except Exception:  # a renamed factory must not stop the data from being viewed
        return None


def analyze(
    arm: Arm,
    *,
    step: float | None = None,
    box: Box | None = None,
    orientations: OrientationSet | None = None,
    n_fk: int = 20_000,
    joint_axes: list[tuple[int, ...]] | None = None,
    floor_z: float | None = None,
    singular_tol: float = 0.01,
    seed: int = 0,
    progress=None,
) -> ReachResult:
    """Everything worth seeing about one arm, computed once.

    1. FK cloud: joint samples -> positions, each classified (OK / torque / collision / singular).
    2. Reach map (if the arm has IK and its workspace is not a curve): every IK branch on a grid
       that the FK cloud bounds, crossed with one orientation set.
    3. Joint maps: the 2R picture for chosen joint pairs (slice and sweep), or a line for 1 DOF.
    4. Joint ranges: limits vs the range the OK solutions actually use."""
    timings = {}
    t = time.perf_counter()
    cloud = fk_cloud(arm, n_fk, seed=seed)
    w_ref = float(np.nanpercentile(arm.manipulability(sample_joints(arm, 2000, seed)), 95)) if singular_tol > 0 else None
    cloud_code = arm.check(cloud.q, floor_z=floor_z, singular_tol=singular_tol, w_ref=w_ref)
    cloud_p = np.where(cloud.ok[:, None], cloud.T[:, :3, 3], np.nan)
    timings["fk_cloud"] = time.perf_counter() - t

    pts = cloud.points
    if box is None:
        box = Box.around(pts)
    extent = float(np.max(box.size))
    flat_axes = int(np.sum(np.asarray(box.size) == 0))
    grid_dims = 3 - flat_axes
    if step is None:
        step = extent / (40 if grid_dims <= 2 else 18)
    positions = box.grid(step)

    reach = None
    if arm.ik is not None and grid_dims <= arm.dof:
        if orientations is None and arm.task != "position":
            orientations = OrientationSet.sample()
        t = time.perf_counter()
        reach = reach_map(arm, positions, orientations, floor_z=floor_z, singular_tol=singular_tol, w_ref=w_ref, progress=progress)
        timings["reach_map"] = time.perf_counter() - t

    if joint_axes is None:
        joint_axes = default_joint_axes(arm)
    t = time.perf_counter()
    maps = []
    for axes in joint_axes:
        maps.append(joint_map(arm, axes, mode="slice", floor_z=floor_z, singular_tol=max(singular_tol, 0.02), w_ref=w_ref))
        if arm.dof > len(axes):
            maps.append(joint_map(arm, axes, mode="sweep", n=91, floor_z=floor_z, singular_tol=max(singular_tol, 0.02), w_ref=w_ref))
    timings["joint_maps"] = time.perf_counter() - t
    ranges = joint_ranges(arm, reach)

    arm_info = {
        "name": arm.name,
        "joint_names": list(arm.joint_names),
        "lower": list(arm.lower),
        "upper": list(arm.upper),
        "task": arm.task,
        "description": arm.description,
        "torque_limits": None if arm.torque_limits is None else list(arm.torque_limits),
        "factory": arm.meta.get("factory"),
    }
    config = {
        "step": step,
        "box": {"center": list(box.center), "size": list(box.size)},
        "n_fk": n_fk,
        "floor_z": floor_z,
        "singular_tol": singular_tol,
        "w_ref": w_ref,
        "seed": seed,
        "orientations": None if reach is None else int(reach.n_orientations),
    }
    return ReachResult(
        arm_info=arm_info,
        cloud_q=cloud.q,
        cloud_p=cloud_p,
        cloud_code=cloud_code,
        reach=reach,
        joint_maps=maps,
        ranges=ranges,
        config=config,
        timings=timings,
        arm=arm,
    )


def default_joint_axes(arm: Arm) -> list[tuple[int, ...]]:
    """1 DOF: the line. 2 DOF: the whole picture. More: shoulder pitch x elbow (joints 1, 2)."""
    if arm.dof == 1:
        return [(0,)]
    if arm.dof == 2:
        return [(0, 1)]
    return [(1, 2)]


def _counts(codes: np.ndarray) -> dict[str, int]:
    vals, counts = np.unique(codes, return_counts=True)
    return {Code(int(v)).name: int(c) for v, c in zip(vals, counts)}


def _json_default(o):
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
