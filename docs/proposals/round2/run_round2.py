"""Round 2 of the downstream API: five shapes, the same three happy paths + a two-arm cell, run for real.

    taskset -c 2-5,16,17 uv run --project /home/bam/robot_reach --extra viz python /home/bam/robot_reach/docs/proposals/round2/run_round2.py

Loads the saved UR5e result read-only (out/ur5e), tags every IK solution with its IK configuration,
then executes each proposal's snippets exactly as the report prints them (exec of the same string)
and FK-checks every pose any snippet drew. `run(result)` is what build_proposals.py calls in-process.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import core  # noqa: E402
from ik_configurations import UR5E_CONFIGURATIONS  # noqa: E402

TASKS = [
    ("full / reachable / dexterous, then a dexterous sample",
     "Zach: “full workspace, the reachable … .dexterous/.hemisphere.sample()”"),
    ("over the belt, hemisphere, elbow up only: 3 A → B pairs",
     "trajectory generation: A → B pairs inside a task zone, one IK configuration per pair"),
    ("over the belt, straight down, wrist J5+ only: 3 tours of 5 waypoints",
     "trajectory generation: 2–10 waypoints inside a zone"),
    ("two arms 0.85 m apart: grid points both reach with every hemisphere orientation",
     "multi-chain later: the same shape, one level up"),
]

RECOMMENDED_EXTRAS = [  # P1 + the borrowed pieces, run in P1's namespace after its four jobs
    """cone = ws.reachable.inside(belt).within(30).only(elbow="up")     # P2's custom cone, no sixth attribute
print(cone)
print(json.dumps(cone.spec()))                                     # P2's query, as data a campaign logs
for line in cone.sample_pairs(2, rng).lines(): print(line)""",
    """s = over_belt.to_sampler()                                        # P4's draw-only hand-off
print(PoseSampler.load(s.save(tmp / "over_belt.npz")))
m = over_belt.as_reach_map_like()                                  # what robot_trajectories duck-types today
print(m.joint_solutions.shape, m.valid_mask.shape, m.reach_scores.shape, int(m.valid_mask.sum()), "valid (position, orientation) pairs")""",
]

PROPOSALS = [
    {
        "id": "P1", "name": "Attribute namespace", "module": "p1_namespace",
        "thesis": "The kinds are attributes; each is a lazy region you narrow (inside, only) and sample. Reads like your sentence.",
        "msgs": """@dataclass(frozen=True, eq=False)
class ReachableWorkspace(Msg):          # robot_msgs/manipulation/reachable_workspace.py
    chain: str; frame: str; joint_names: tuple[str, ...]
    positions: Array          # (P, 3) every grid point analysed, m
    voxel_size: float
    orientations: Array       # (O, 3, 3) shared orientation set
    kind_orientations: dict   # "top_down" | "hemisphere" | "dexterous" -> (O,) bool
    solution_pose: Array      # (K,) p * O + o, sorted
    solution_q: Array         # (K, dof) rad, checked
    solution_configuration: Array   # (K,) index into configurations
    configurations: IKConfigurations
    allowed: Array | None = None    # IK-configuration filter carried by views

    full / reachable / top_down / hemisphere / dexterous -> WorkspaceRegion   # properties
    def only(self, *labels, **answers) -> ReachableWorkspace

class WorkspaceRegion:                  # a view, never stored
    def inside(self, zone) -> WorkspaceRegion
    def only(self, *labels, **answers) -> WorkspaceRegion
    def contains(self, points) -> Array          # (N,) bool
    def sample(self, n, rng) -> PoseSamples                       # (n,)
    def sample_pairs(self, n, rng, *, same_configuration=True)    # (n, 2)
    def sample_waypoints(self, n, k, rng, *, same_configuration=True)  # (n, k)
    def __and__(self, other) -> SharedRegion       # several chains

class Cell(Msg):  chains: dict[str, ReachableWorkspace]    # cell.left.hemisphere""",
        "config": "<code>.only(elbow=\"up\")</code>, <code>.only(wrist=(\"J5+\", \"J5-\"))</code> or <code>.only(\"front/up/J5+\")</code> on the workspace or any region; the two commute, and membership is re-judged with the filter (hemisphere-with-elbow-up, not hemisphere-then-drop-elbow-down).",
        "multi": "One more attribute level: <code>cell.left.hemisphere &amp; cell.right.hemisphere</code> is a <code>SharedRegion</code>. A single-chain <code>ReachableWorkspace</code> never changes; a cell is a dict of them in one frame.",
        "weak": "The vocabulary is fixed by the attributes. A custom approach cone (“within 30° of down”) is not one of the five, so it needs a sixth attribute or a <code>ws[\"…\"]</code> escape hatch with a parameter — and then you have a query builder under another name.",
        "snippets": [
            """ws = reachable_workspace(reach)
print(ws)
for line in ws.dexterous.sample(3, rng).lines(): print(line)""",
            """over_belt = ws.hemisphere.inside(belt).only(elbow="up")
print(over_belt)
for line in over_belt.sample_pairs(3, rng).lines(): print(line)""",
            """region = ws.top_down.inside(belt).only(wrist="J5+")
print(region.count_by_configuration())
tours = region.sample_waypoints(3, 5, rng)
for line in tours.lines(): print(line)""",
            """cell = Cell({"left": reachable_workspace(left), "right": reachable_workspace(right)})
handover = cell.left.hemisphere & cell.right.hemisphere
print(len(handover), "grid points")
for name, s in handover.sample(2, rng).items(): print(name, s.lines()[0], "|", s.lines()[1])""",
        ],
    },
    {
        "id": "P2", "name": "Query builder", "module": "p2_query",
        "thesis": "The question is its own frozen, JSON-able value. A campaign config stores it; any arm's workspace answers it.",
        "msgs": """@dataclass(frozen=True)
class WorkspaceQuery(Msg):              # robot_msgs/manipulation/workspace_query.py
    kind: str = "reachable"
    zones: tuple[Zone, ...] = ()
    configurations: tuple[tuple[str, tuple[str, ...]], ...] = ()   # (("elbow", ("up",)),)
    approach_within_deg: float | None = None
    chain: str | None = None
    same_configuration: bool = True
    # builder: Q.hemisphere, .of(kind), .inside(zone), .configuration(**answers),
    #          .approach_within(deg), .on(chain), .any_configuration(), to_json / from_json
Q = WorkspaceQuery()

@dataclass(frozen=True, eq=False)
class ReachableWorkspace(Msg):          # the same arrays as P1, no views
    ...
    def count(self, q) -> dict          # voxels, IK solutions
    def contains(self, q, points) -> Array
    def sample(self, q, n, rng, *, k=1) -> PoseSamples   # k=2 pairs, k>2 waypoints

class Cell(Msg):  chains: dict[str, ReachableWorkspace]
    def sample(self, q_with_chain, n, rng, *, k=1)
    def sample_shared(self, queries, n, rng)""",
        "config": "<code>.configuration(elbow=\"up\")</code> — stored in the query as <code>((\"elbow\", (\"up\",)),)</code>, so the filter is in the JSON a campaign logs. Repeating an axis ORs its options.",
        "multi": "A query names its chain (<code>.on(\"left\")</code>); a cell runs one query per chain or a list of them coupled at one grid point (<code>sample_shared</code>). The single-chain query is unchanged — <code>chain=None</code> means “the only one”.",
        "weak": "Two objects for every call (<code>ws.sample(q, …)</code>), and a kind or axis name is a string checked only when the query runs. The simplest ask — a dexterous sample — reads <code>ws.sample(Q.dexterous, 3, rng)</code>, not <code>ws.dexterous.sample(3)</code>.",
        "snippets": [
            """ws = reachable_workspace(reach)
for kind in ("full", "reachable", "dexterous"): print(kind, ws.count(Q.of(kind)))
for line in ws.sample(Q.dexterous, 3, rng).lines(): print(line)""",
            """q = Q.hemisphere.inside(belt_box).configuration(elbow="up")
print(q.to_json())
print(ws.count(q))
for line in ws.sample(q, 3, rng, k=2).lines(): print(line)""",
            """q = Q.top_down.inside(belt_box).configuration(wrist="J5+")
for line in ws.sample(q, 3, rng, k=5).lines(): print(line)
cone = Q.reachable.inside(belt_box).approach_within(30)   # a custom approach cone: no new API
print("within 30 deg of down:", ws.count(cone))""",
            """cell = Cell({"left": reachable_workspace(left), "right": reachable_workspace(right)})
s = cell.sample_shared([Q.hemisphere.on("left"), Q.hemisphere.on("right")], 2, rng)
for name, x in s.items(): print(name, x.lines()[0], "|", x.lines()[1])""",
        ],
    },
    {
        "id": "P3", "name": "Numpy masks", "module": "p3_masks",
        "thesis": "The workspace is one boolean cube ok[position, orientation, IK configuration]. Every question is numpy broadcasting.",
        "msgs": """@dataclass(frozen=True, eq=False)
class ReachableWorkspace(Msg):          # robot_msgs/manipulation/reachable_workspace.py
    ...                                 # P1's arrays, plus
    ok: Array                           # (P, O, C) bool, dense

    def configuration(self, *labels, **answers) -> Array   # (C,)   broadcasts on ok
    def orientations(self, kind, *, within_deg=None) -> Array  # (O, 1)
    def inside(self, zone) -> Array                         # (P,)
    def voxels(self, kind, ok=None) -> Array                # (P,) judged on ok
    full / reachable / top_down / hemisphere / dexterous    # (P,) voxel masks
    def poses(self, voxels, kind, ok=None) -> Array         # (P, O, C)
    def sample(self, mask, n, rng, *, k=1, same_configuration=True) -> PoseSamples

class CellMasks:  chains: dict[str, ReachableWorkspace]   # masks gain a chain axis (Ch, G)""",
        "config": "<code>ws.ok &amp; ws.configuration(elbow=\"up\")</code>: a (C,) mask that broadcasts onto the cube's last axis. Any boolean over the eight configurations is a valid filter — no special syntax for “these two only”.",
        "multi": "One shared cell grid and a leading chain axis: <code>cell.voxels(\"hemisphere\")</code> is (Ch, G); “both arms” is <code>.all(0)</code>, “either” is <code>.any(0)</code>. Needs the chains' grids aligned (placed a whole number of voxels apart, or re-gridded).",
        "weak": "Dense: P × O × C bits — 7.2 MB here, but a 25 mm grid over the same box with 132 orientations is ~470 MB, and you must remember which axis is which (<code>[:, None, None]</code>). The happy paths are the longest of the five.",
        "snippets": [
            """ws = reachable_workspace(reach)
print(ws.ok.shape, f"{ws.ok.nbytes / 1e6:.1f} MB", {k: int(getattr(ws, k).sum()) for k in ("full", "reachable", "dexterous")})
for line in ws.sample(ws.poses(ws.dexterous, "dexterous"), 3, rng).lines(): print(line)""",
            """up = ws.ok & ws.configuration(elbow="up")
belt_hemi = ws.voxels("hemisphere", up) & ws.inside(belt)
print(int(belt_hemi.sum()), "voxels")
for line in ws.sample(ws.poses(belt_hemi, "hemisphere", up), 3, rng, k=2).lines(): print(line)""",
            """j5 = ws.ok & ws.configuration(wrist="J5+")
m = ws.poses(ws.voxels("top_down", j5) & ws.inside(belt), "top_down", j5)
print(int(m.sum()), "poses ·", "per configuration", m.sum((0, 1)).tolist())
for line in ws.sample(m, 3, rng, k=5).lines(): print(line)""",
            """cell = CellMasks({"left": reachable_workspace(left), "right": reachable_workspace(right)})
v = cell.voxels("hemisphere")
print(v.shape, int(v.all(0).sum()), "grid points")
s = cell.sample_shared(v.all(0), "hemisphere", 2, rng)
for name, x in s.items(): print(name, x.lines()[0], "|", x.lines()[1])""",
        ],
    },
    {
        "id": "P4", "name": "Sampler first", "module": "p4_sampler",
        "thesis": "Downstream never holds a workspace — only a small, saved, ready-to-draw set of checked poses for one question.",
        "msgs": """@dataclass(frozen=True, eq=False)
class PoseSampler(Msg):                 # robot_msgs/manipulation/pose_sampler.py
    chain: str; frame: str; joint_names: tuple[str, ...]
    description: str                    # "ur5e hemisphere ∩ belt · elbow=up"
    position: Array                     # (M, 3)  eligible rows only
    rotation: Array                     # (M, 3, 3)
    q: Array                            # (M, dof) checked
    configuration: Array                # (M,)
    configuration_labels: tuple[str, ...]
    pose_key: Array                     # (M,) same pose, several configurations
    voxel_size: float
    same_configuration: bool = True     # the drawing policy travels with it
    def poses(self, n, rng); pairs(self, n, rng); waypoints(self, n, k, rng)
    def stream(self, rng, k=1) -> Iterator[PoseSamples]
    def save(self, path); load(cls, path)

def sampler(reach, kind, *, inside=None, configurations=None, within_deg=None) -> PoseSampler   # robot_reach

class CoupledSampler(Msg):  samplers: dict[str, PoseSampler]; couple: "same_position" | "independent\"""",
        "config": "Resolved once at the producer: <code>sampler(reach, \"hemisphere\", configurations=dict(elbow=\"up\"))</code>. The consumer cannot re-filter — it gets exactly the poses that were asked for.",
        "multi": "<code>CoupledSampler({\"left\": …, \"right\": …}, couple=\"same_position\")</code> draws one pose per chain per draw; <code>\"independent\"</code> draws each on its own. One-chain samplers are untouched.",
        "weak": "It answers one question. A new zone, kind or filter is a trip back to robot_reach and the full map, and it cannot say what is NOT reachable (no <code>full</code>, no <code>contains</code>, no why-not) — so it cannot be the robot_msgs workspace type, only a hand-off beside one.",
        "snippets": [
            """print(sampler(reach, "reachable"))
dex = sampler(reach, "dexterous")
print(dex)
for line in dex.poses(3, rng).lines(): print(line)""",
            """s = sampler(reach, "hemisphere", inside=belt, configurations=dict(elbow="up"))
s = PoseSampler.load(s.save(tmp / "belt.poses.npz"))     # the hand-off is a file
print(s)
for line in s.pairs(3, rng).lines(): print(line)""",
            """tours = sampler(reach, "top_down", inside=belt, configurations=dict(wrist="J5+"))
print(tours)
for line in tours.waypoints(3, 5, rng).lines(): print(line)
stream = tours.stream(rng, k=5)
print("next(stream):", next(stream).pose.shape)""",
            """both = CoupledSampler({"left": sampler(left, "hemisphere"), "right": sampler(right, "hemisphere")})
s = both.draw(2, rng)
for name, x in s.items(): print(name, x.lines()[0], "|", x.lines()[1])""",
        ],
    },
    {
        "id": "P5", "name": "Table of IK solutions", "module": "p5_table",
        "thesis": "One row per checked IK solution; columns are the vocabulary (x y z, tilt, configuration, elbow, q, in_hemisphere). Filter it like a dataframe.",
        "msgs": """@dataclass(frozen=True, eq=False)
class SolutionTable(Msg):               # robot_msgs/manipulation/solution_table.py
    columns: dict[str, Array]           # K rows each:
    #   chain, position, orientation, x, y, z, key (cell grid), pose_key, tilt_deg,
    #   configuration ("front/up/J5+"), shoulder, elbow, wrist (one per axis), q (K, dof),
    #   o_top_down, o_hemisphere (row's orientation is in that set),
    #   in_top_down, in_hemisphere, in_dexterous (voxel membership, all configurations)
    header: dict                        # chain, frame, joint_names, voxel_size, orientations, kinds
    t.col / t["col"]; t[bool_mask] -> SolutionTable
    def complete(self, kind) -> Array   # membership re-judged on THESE rows
    def sample(self, n, rng) -> SolutionTable
    def sample_groups(self, by, *, n, k, rng) -> SolutionTable   # + "group" column
    def counts(self, by) -> dict;  to_samples(k);  to_pandas()   # pandas optional
    concat(tables) -> SolutionTable;  shared_by("chain") -> Array""",
        "config": "Columns: <code>t.elbow == \"up\"</code>, <code>t.configuration == \"front/up/J5+\"</code>, <code>np.isin(t.wrist, [\"J5+\"])</code>. Grouping by <code>configuration</code> gives same-configuration pairs and tours.",
        "multi": "A <code>chain</code> column: <code>SolutionTable.concat([left, right])</code>; “both arms reach it” is <code>t.shared_by(\"chain\")</code> on the cell grid key. A one-chain table is a table with one chain value.",
        "weak": "The <code>in_*</code> flags were computed with every configuration and go stale after a filter (elbow-up only can drop a voxel from the hemisphere), so the belt path needs <code>b[b.complete(\"hemisphere\")]</code> — easy to forget, silently overclaims. And it has no rows for unreachable points, so <code>full</code> needs the header.",
        "snippets": [
            """t = solution_table(reach)
print(f"{len(t):,} rows", "· voxels: reachable", len(np.unique(t.position)), "dexterous", len(np.unique(t.position[t.in_dexterous])))
for line in t[t.in_dexterous].sample(3, rng).to_samples().lines(): print(line)""",
            """b = t[(t.elbow == "up") & belt.contains(t.xyz) & t.o_hemisphere]
b = b[b.complete("hemisphere")]          # re-judge membership with elbow-up rows only
print(f"{len(b):,} rows", b.counts("configuration"))
ab = b.sample_groups("configuration", n=3, k=2, rng=rng)
for line in ab.to_samples(k=2).lines(): print(line)""",
            """c = t[(t.wrist == "J5+") & belt.contains(t.xyz) & t.o_top_down]
c = c[c.complete("top_down")]
tours = c.sample_groups("configuration", n=3, k=5, rng=rng)
for line in tours.to_samples(k=5).lines(): print(line)
print(tours.head(3, ("group", "x", "y", "z", "tilt_deg", "configuration")))""",
            """cell = SolutionTable.concat([solution_table(left), solution_table(right)])
h = cell[cell.in_hemisphere & cell.o_hemisphere]
both = h[h.shared_by("chain")]
print(len(np.unique(both.key)), "grid points", both.counts("chain"))""",
        ],
    },
]


def fk_check(arm, drawn) -> dict:
    """Re-run the arm's FK on every IK solution a snippet drew; compare with the pose handed out."""
    if not drawn:
        return {"n": 0}
    T = np.concatenate([d.pose.reshape(-1, 4, 4) - _shift(d) for d in drawn])
    q = np.concatenate([d.q.reshape(-1, d.q.shape[-1]) for d in drawn])
    F = arm.fk(q)
    F = F[0] if isinstance(F, tuple) else F
    pos = np.linalg.norm(F[:, :3, 3] - T[:, :3, 3], axis=1) * 1000
    cos = (np.einsum("nij,nij->n", F[:, :3, :3], T[:, :3, :3]) - 1) / 2
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return {"n": int(len(q)), "pos_mm": float(pos.max()), "rot_deg": float(ang.max())}


def _shift(s) -> np.ndarray:
    out = np.zeros((1, 4, 4))
    out[0, :3, 3] = s.base_xyz
    return out


def run(result, solution_configuration: np.ndarray | None = None) -> dict:
    from robot_msgs.shape import SolidPrimitive

    import p1_namespace
    import p2_query
    import p3_masks
    import p4_sampler
    import p5_table
    from ik_branches import tag_all

    arm = result.arm
    if solution_configuration is None:
        solution_configuration = UR5E_CONFIGURATIONS.index_of(tag_all(result.reach.sol_q.astype(float), arm.meta["frames"]))
    reach = core.from_result(result, UR5E_CONFIGURATIONS, solution_configuration)
    off = 4 * reach.voxel_size  # WHY a whole number of voxels: both arms then share one cell grid
    left = core.placed(reach, "left", (0.0, +off, 0.0))
    right = core.placed(reach, "right", (0.0, -off, 0.0))
    # the belt: 0.30-0.65 m in front, +-0.40 m across, 0.05-0.40 m up (round 1's zone)
    belt = core.Zone(SolidPrimitive.box(0.35, 0.80, 0.35), (0.475, 0.0, 0.225), "belt")
    belt_box = p2_query.BoxZone((0.30, -0.40, 0.05), (0.65, 0.40, 0.40), "belt")
    mods = {"p1_namespace": p1_namespace, "p2_query": p2_query, "p3_masks": p3_masks, "p4_sampler": p4_sampler, "p5_table": p5_table}
    tmp = Path(tempfile.mkdtemp(prefix="round2-"))
    out = {"tasks": TASKS, "proposals": [], "facts": {
        "P": reach.P, "O": reach.O, "K": int(len(reach.solution_q)), "C": len(UR5E_CONFIGURATIONS), "voxel_mm": reach.voxel_size * 1000,
        "labels": list(UR5E_CONFIGURATIONS.labels), "per_configuration": np.bincount(solution_configuration, minlength=8).tolist(),
        "offset_m": off}}
    for prop in PROPOSALS:
        mod = mods[prop["module"]]
        ns = {k: getattr(mod, k) for k in dir(mod) if not k.startswith("_")}
        ns.update(np=np, reach=reach, left=left, right=right, belt=belt, belt_box=belt_box, tmp=tmp, rng=np.random.default_rng(7))
        runs = []
        for code in prop["snippets"]:
            core.DRAWN.clear()
            buf = io.StringIO()
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(buf):
                exec(compile(code, f"<{prop['id']}>", "exec"), ns)
            dt = time.perf_counter() - t0
            drawn = [d for d in core.DRAWN if d.q.size <= 60_000]  # the stream's 256-batch and per-draw views, not whole tables
            runs.append({"code": code, "output": buf.getvalue().rstrip(), "seconds": round(dt, 3), "fk": fk_check(arm, drawn)})
        extras = []
        if prop["id"] == "P1":
            ns.update(json=json, PoseSampler=p4_sampler.PoseSampler)
            for code in RECOMMENDED_EXTRAS:
                core.DRAWN.clear()
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, "<P1+>", "exec"), ns)
                extras.append({"code": code, "output": buf.getvalue().rstrip(), "fk": fk_check(arm, list(core.DRAWN))})
        out["proposals"].append({k: v for k, v in prop.items() if k != "snippets"} | {"runs": runs, "extras": extras})
        print(f"== {prop['id']} {prop['name']}", file=sys.stderr)
        for r in runs:
            print(r["code"], "\n---\n", r["output"], "\n", r["fk"], f"{r['seconds']} s\n", file=sys.stderr)
    import p3_masks as _p3
    import p5_table as _p5

    sparse = sum(a.nbytes for a in (reach.solution_pose, reach.solution_q, reach.solution_configuration, reach.positions, reach.orientations))
    table = _p5.solution_table(reach)
    out["facts"]["bytes"] = {"sparse (P1, P2)": sparse, "dense cube ok (P3, extra)": _p3.ReachableWorkspace.of(reach).ok.nbytes,
                             "table (P5)": int(sum(v.nbytes for v in table.columns.values()))}
    del table
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)  # the saved .npz hand-offs were only there to prove the round trip
    rml = core.reach_map_like(reach, UR5E_CONFIGURATIONS.allowed(elbow="up"))
    out["reach_map_like"] = {"joint_names": list(rml.joint_names), "joint_solutions": list(rml.joint_solutions.shape),
                             "valid_mask": list(rml.valid_mask.shape), "reach_scores": list(rml.reach_scores.shape),
                             "valid": int(rml.valid_mask.sum())}
    return out


if __name__ == "__main__":
    import robot_reach as rr

    res = rr.load("/home/bam/robot_reach/out/ur5e")
    data = run(res)
    print(json.dumps({"reach_map_like": data["reach_map_like"], "bytes": data["facts"]["bytes"]}))
