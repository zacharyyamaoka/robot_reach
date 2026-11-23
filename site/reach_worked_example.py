# %% [markdown]
# # robot_reach, worked end to end
#
# Kinematics in, workspace out. An arm is its **FK** and **IK**, passed in as pure batched numpy
# functions (or built from a URDF / RobotDescription). `rr.analyze()` computes, once, where the arm
# can go (Cartesian space) and which postures work (joint space), with one status code for every
# "why not". The result is plain arrays on disk; viser shows it, and anything downstream reads it.
#
# Run it as a script (every cell prints what it computes):
#
# ```bash
# uv run --project /home/bam/robot_reach --extra viz python /home/bam/robot_reach/notebooks/reach_worked_example.py
# ```
#
# or open `reach_worked_example.ipynb` beside it (the same cells, executed, outputs saved).
#
# Sections: 1 an arm from two functions · 2 analyze it · 3 any DOF · 4 a saved UR5e map, queried
# the way a downstream consumer would · 5 over a conveyor belt · 6 from a URDF or RobotDescription
# · 7 out to robot_msgs · 8 into someone else's viser scene · 9 the viewer.

# %%
import time
from pathlib import Path

import numpy as np

import robot_reach as rr
from robot_reach import Code

np.set_printoptions(precision=3, suppress=True)
REPO = Path("/home/bam/robot_reach")
print("robot_reach exports:", ", ".join(rr.__all__))

# %% [markdown]
# ## 1. An arm is two batched functions
#
# A planar 2R arm (shoulder + elbow, link lengths 0.30 m and 0.25 m). FK maps `(N, dof)` joint
# angles to `(N, 4, 4)` poses. IK maps `(N, 4, 4)` targets to **every** branch, `(N, S, dof)`, with
# a `Code` per branch. Here S = 2: elbow up and elbow down. `task="position"` says the arm is only
# asked to match the position, never a false orientation promise.

# %%
L1, L2 = 0.30, 0.25


def fk_2r(q):
    """(N, 2) rad -> (N, 4, 4) base->tool, the arm in the x-z plane."""
    a, b = q[:, 0], q[:, 0] + q[:, 1]
    T = np.tile(np.eye(4), (len(q), 1, 1))
    T[:, 0, 3] = L1 * np.cos(a) + L2 * np.cos(b)
    T[:, 2, 3] = L1 * np.sin(a) + L2 * np.sin(b)
    return T


def ik_2r(T):
    """(N, 4, 4) -> (q (N, 2, 2), code (N, 2)): both elbow branches, a Code each."""
    x, z = T[:, 0, 3], T[:, 2, 3]
    c2 = (x**2 + z**2 - L1**2 - L2**2) / (2 * L1 * L2)
    reachable = np.abs(c2) <= 1.0
    q = np.zeros((len(T), 2, 2))
    code = np.where(reachable, int(Code.OK), int(Code.NO_SOLUTION))[:, None].repeat(2, axis=1)
    for branch, sign in enumerate((+1.0, -1.0)):
        q2 = sign * np.arccos(np.clip(c2, -1.0, 1.0))
        q1 = np.arctan2(z, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        q[:, branch] = np.stack([q1, q2], axis=1)
    return q, code.astype(np.int8)


arm = rr.Arm(
    name="my_2r",
    joint_names=("shoulder", "elbow"),
    lower=np.deg2rad([-10.0, -150.0]).tolist(),
    upper=np.deg2rad([190.0, 150.0]).tolist(),
    fk=fk_2r,
    ik=ik_2r,
    task="position",
    plane=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # a planar arm: grids collapse onto its plane
)

q = np.deg2rad([[30.0, 45.0], [90.0, -60.0]])
T, ok = arm.forward(q)
print("FK tool positions (m):\n", T[:, :3, 3], "ok:", ok)
q_all, codes = arm.inverse(T)
print("IK, every branch (deg):\n", np.rad2deg(q_all))
print("codes:", [[Code(int(c)).name for c in row] for row in codes])
print("manipulability at q:", arm.manipulability(q))

# %% [markdown]
# The library never trusts a solver's OK. `arm.check(q)` applies the checks it can do itself
# (joint limits, the FK round trip, collision, torque, singularity), and the reach map runs the
# same checks on every IK branch. Below, a posture outside the shoulder limit and one at the
# fully stretched singularity.

# %%
probe = np.deg2rad([[30.0, 45.0], [-60.0, 20.0], [60.0, 0.0]])
print([Code(int(c)).name for c in arm.check(probe, singular_tol=0.05)])

# %% [markdown]
# ## 2. Analyze it: compute once
#
# `rr.analyze()` runs four things and returns one `ReachResult`:
# the **FK cloud** (sample joints, run FK; bounds the grid so nobody guesses an extent),
# the **reach map** (IK at every grid point x orientation, every branch kept),
# the **joint maps** (the 2R picture for joint pairs) and the **joint ranges** (limits vs what the
# working solutions actually use).

# %%
t = time.perf_counter()
result = rr.analyze(arm, floor_z=-0.05)
print(f"analyze: {time.perf_counter() - t:.2f} s")
s = result.summary()
print({k: s[k] for k in ("arm", "dof", "task", "fk_samples")})
print("reach:", s["reach"])
print("joint maps:", [(m["axes"], m["mode"], m["ok_fraction"]) for m in s["joint_maps"]])

# %% [markdown]
# The result log is plain files: `result.json` plus compressed npz, no pickle. A viewer, a report,
# this notebook or another package reads the same numbers; nothing recomputes to look.

# %%
folder = result.save(REPO / "out" / "notebook_my_2r")
print(sorted(p.name for p in folder.iterdir()))
again = rr.load(folder, rebuild_arm=False)
print("reloaded positions:", again.reach.n_positions, "| arm rebuilt:", again.arm is not None,
      "(a hand-written arm has no factory; the built-ins do)")

# %% [markdown]
# ## 3. Any DOF: the built-in arms
#
# 1-DOF pendulum, 2-DOF RR, 2-DOF five-bar (a closed chain whose FK can fail), 3-DOF shoulder and
# 6-DOF UR5e. The same call on each. The UR5e (about 20 s) is loaded from its saved result in the
# next section instead of recomputed here.

# %%
from robot_reach.robots import five_bar, pendulum, rr as rr_arm, shoulder3  # noqa: E402

for factory in (pendulum, rr_arm, five_bar, shoulder3):
    a = factory()
    t = time.perf_counter()
    res = rr.analyze(a)
    dt = time.perf_counter() - t
    r = res.reach
    if r is None:
        print(f"{a.name:9s} dof={a.dof} no grid (its workspace is a curve: see the FK cloud) {dt:5.2f} s")
        continue
    lv = r.levels()
    print(f"{a.name:9s} dof={a.dof} branches={r.n_branches} grid={r.n_positions:5d} "
          f"reachable={(lv >= 3).sum():5d} OK branches={len(r.sol_index):6d} {dt:5.2f} s")

# %% [markdown]
# ## 4. A downstream consumer: query a saved UR5e map
#
# `rr.load()` reads the result log and rebuilds the arm from its recorded factory, so the live FK
# is back too. The `ReachMap` is the thing a downstream package reads:
#
# | Call | Shape | Meaning |
# |---|---|---|
# | `positions` | (P, 3) | grid points, arm base frame, m |
# | `orientations.R` | (O, 3, 3) | one orientation set shared by every point |
# | `code` | (P, O, S) | final `Code` per IK branch |
# | `levels()` | (P,) | pose DOF controlled at each point: 0, 3, 4, 5, 6 |
# | `workspace(kind)` | (P,) bool | `reachable`, `four_dof`, `hemisphere`, `dexterous` |
# | `solutions(p)` | list | every OK `(orientation, branch, q)` at point p |
# | `target(p, o)` | (4, 4) | the pose a solution reaches |
# | `joint_solutions` | (P, O, dof) | first OK branch per pose (bam_reach v1's field) |

# %%
ur = rr.load(REPO / "out" / "ur5e")
m = ur.reach
print(f"{m.arm_name}: {m.n_positions} points x {m.n_orientations} orientations x {m.n_branches} branches")
lv = m.levels()
for level, name in rr.LEVEL_NAMES.items():
    print(f"  level {level} {name:12s} {(lv == level).sum():5d} points")
print("branch codes:", m.code_counts())

# %% [markdown]
# Sampling a workspace: the points of the hemisphere workspace (every approach within 90 deg of
# straight down, every roll), five of them at random. Today this is three lines of numpy over the
# map; a `.sample()` helper is one of the proposals for the next API, not something that exists yet.

# %%
rng = np.random.default_rng(0)
hemi = np.flatnonzero(m.workspace("hemisphere"))
picks = rng.choice(hemi, 5, replace=False)
print(f"hemisphere workspace: {len(hemi)} points; five picks (m):\n", m.positions[picks])

# %% [markdown]
# Every IK branch at one point, and the round-trip check done by hand: FK of each stored solution
# lands back on its target.

# %%
p = int(picks[0])
sols = m.solutions(p)
print(f"point {p} at {m.positions[p]}: {len(sols)} OK (orientation, branch) solutions")
o, b, q_sol = sols[0]
print(f"first: orientation {o}, branch {b}, q (deg) = {np.rad2deg(q_sol)}")
T_fk, _ = ur.arm.forward(np.array([s[2] for s in sols]))
T_target = np.array([m.target(p, s[0]) for s in sols])
print(f"max FK(IK) position error over all {len(sols)}: {np.abs(T_fk[:, :3, 3] - T_target[:, :3, 3]).max():.2e} m")
n_here = m.n_solutions[p][m.n_solutions[p] > 0]
print("orientations here by number of OK IK branches:", {int(k): int((n_here == k).sum()) for k in np.unique(n_here)})

# %% [markdown]
# ## 5. Over a conveyor belt
#
# The use case that motivated the downstream question: which points over the belt can be picked
# straight down at any yaw (4-DOF), and which with any approach in the hemisphere? A `Box` marks
# the belt's pick volume in the arm base frame.

# %%
belt = rr.Box(center=(0.60, 0.0, 0.15), size=(0.50, 0.80, 0.20))  # x 0.35-0.85 m, z 0.05-0.25 m
inside = belt.contains(m.positions)
for kind in ("reachable", "four_dof", "hemisphere", "dexterous"):
    w = m.workspace(kind) & inside
    print(f"  {kind:10s} {w.sum():3d} of {inside.sum()} grid points over the belt")

# %% [markdown]
# The grid step here is the saved map's (about 0.11 m). For a finer look at just that volume, map
# the box itself: same arm, same orientation set, a 5 cm grid.

# %%
t = time.perf_counter()
fine = rr.reach_map(ur.arm, belt.grid(0.05), rr.OrientationSet.sample(), floor_z=0.0)
print(f"belt box at 5 cm: {fine.n_positions} points in {time.perf_counter() - t:.1f} s")
for kind in ("four_dof", "hemisphere", "dexterous"):
    print(f"  {kind:10s} {fine.workspace(kind).mean():6.1%} of the belt volume")

# %% [markdown]
# ## 6. From a URDF or a RobotDescription
#
# FK comes from the URDF (stdlib XML + numpy: revolute, continuous, prismatic, mimic). With no
# analytic IK, a batched damped-least-squares IK runs from several seeds and keeps the distinct
# branches. A bam `RobotDescription` goes in through `arm_from_description`, which only calls its
# `to_urdf_string()`, so robot_reach never imports the description package.

# %%
from robot_reach.urdf import analyze_chains, arm_from_description, arm_from_urdf, chains  # noqa: E402

URDF = """<robot name="yaw_rr">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="tool"/>
  <joint name="J1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.10"/><axis xyz="0 0 1"/><limit lower="-2.9" upper="2.9" effort="5" velocity="3"/></joint>
  <joint name="J2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0"/><axis xyz="0 -1 0"/><limit lower="-0.3" upper="3.4" effort="5" velocity="3"/></joint>
  <joint name="J3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0.30 0 0"/><axis xyz="0 -1 0"/><limit lower="-2.6" upper="2.6" effort="5" velocity="3"/></joint>
  <joint name="tool_fixed" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="0.25 0 0"/></joint>
</robot>"""

print("chains:", chains(URDF))
urdf_arm = arm_from_urdf(URDF, "tool")
print(urdf_arm.name, urdf_arm.joint_names, "task:", urdf_arm.task)
T, _ = urdf_arm.forward(np.deg2rad([[20.0, 60.0, -60.0]]))
print("FK at (20, 60, -60) deg:", T[0, :3, 3])
q_ik, c_ik = urdf_arm.inverse(T)
good = c_ik[0] == Code.OK
print(f"numeric IK: {good.sum()} distinct OK branches (deg):\n", np.rad2deg(q_ik[0, good]))


class TinyDescription:
    """Stands in for a bam RobotDescription: robot_reach only calls to_urdf_string()."""

    def to_urdf_string(self, provenance: bool = True) -> str:
        return URDF


t = time.perf_counter()
per_chain = analyze_chains(TinyDescription(), floor_z=0.0)
for label, res in per_chain.items():
    lv = res.reach.levels()
    print(f"{label}: {res.reach.n_positions} grid points, {(lv >= 3).sum()} reachable, "
          f"{time.perf_counter() - t:.1f} s")
print("same arm via arm_from_description:", arm_from_description(TinyDescription(), "tool").joint_names)

# %% [markdown]
# ## 7. Out to robot_msgs
#
# What crosses a package boundary today. robot_reach's own outputs are numpy arrays and its result
# log; it hands robot_msgs types at two seams:
#
# - **`NamedPose`** (`robot_msgs.manipulation`): the rest-pose solver's pick becomes one
#   (`goals.to_named_pose(record)`), and `rest.named_poses()` reads a robot_msgs named-poses JSON.
# - **rotation helpers** (`robot_msgs.geometry.rotations.quat_from_matrix`, `xyzw_to_wxyz`): the
#   viewers pose meshes with them.
#
# A consumer that wants messages for a reach solution builds them from the arrays: a 4x4 target is
# a `Pose`, a joint vector is a `JointState`, a chosen solution is a `NamedPose`. These three lines
# are consumer code, not a robot_reach function (a typed downstream API is in the proposals).

# %%
from robot_msgs.geometry import Pose  # noqa: E402
from robot_msgs.manipulation import NamedPose, named_poses_from_json, named_poses_to_json  # noqa: E402
from robot_msgs.sensor import JointState  # noqa: E402

pose = Pose.from_matrix(m.target(p, o))
state = JointState(name=m.joint_names, position=q_sol)
named = NamedPose(name="belt_hover", group="arm", joint_names=m.joint_names, positions=q_sol,
                  description="A hemisphere-workspace point picked from the UR5e reach map.",
                  why=f"reach map point {p}, orientation {o}, IK branch {b}")
print(pose)
print(state)
text = named_poses_to_json([named], robot="ur5e", source="robot_reach worked example")
print("round trip:", named_poses_from_json(text)[0].name, "digest", named.digest())

# %% [markdown]
# The other direction: a named-poses file goes into the rest-pose tools, which check each pose
# against the arm (joint names, limits) before measuring it.

# %%
from robot_reach.rest import named_poses  # noqa: E402

poses_file = REPO / "out" / "notebook_named_poses.json"
poses_file.write_text(text)
choices = named_poses(poses_file, ur.arm)
print([(c.label, np.round(np.rad2deg(c.q), 1).tolist()) for c in choices])

# %% [markdown]
# ## 8. Into someone else's viser scene
#
# `add_reach_map` draws a reach map under any frame of a host's viser scene (a RobotDescription
# viewer, a facility layout), so it moves when the robot is placed. It imports nothing from viser
# and works on viser 1.0.16 and 1.1.1.

# %%
import viser  # noqa: E402

from robot_reach.overlay import add_reach_map  # noqa: E402

server = viser.ViserServer(host="127.0.0.1", port=8120, verbose=False)
try:
    server.scene.add_frame("/cell/robot_1", position=(1.0, 0.5, 0.0), show_axes=False)
    overlay = add_reach_map(server, m, parent_frame="/cell/robot_1")
    for kind in ("reachable", "four_dof", "hemisphere", "dexterous"):
        overlay.kind = kind
        print(f"overlay kind={overlay.kind:10s} visible={overlay.visible}")
    overlay.visible = False
    print("hidden:", not overlay.visible)
finally:
    server.stop()

# %% [markdown]
# ## 9. The viewer
#
# Not run here (it serves until Ctrl-C). Each line is a whole command:
#
# ```bash
# uv run --project /home/bam/robot_reach --extra viz python -m robot_reach examples
# uv run --project /home/bam/robot_reach --extra viz python -m robot_reach urdf /abs/path/robot.urdf --floor 0.0
# uv run --project /home/bam/robot_reach --extra viz python -m robot_reach rest shoulder3 --solve
# ```
#
# In Python: `rr.load("out/ur5e").display()`. Click a voxel to put the arm there; **Next ▶** walks
# every (orientation, IK branch) that works at that point; **Cut**, **Blob**, **Jog** and **Ask
# here** (Compute box, Drop POI) compute live in the Python backend.
