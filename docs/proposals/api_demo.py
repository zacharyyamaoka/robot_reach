"""Proposal G, run for real on the saved UR5e result: the consumer's view, then the simple alternative.

    uv run --project /home/bam/robot_reach --extra viz python /home/bam/robot_reach/docs/proposals/api_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import robot_reach as rr  # noqa: E402
from ik_branches import UR5E_BRANCHES, tag_all  # noqa: E402
from workspace_api import AxisBox, workspace_from_result  # noqa: E402

result = rr.load("/home/bam/robot_reach/out/ur5e")
arm = result.arm
types = UR5E_BRANCHES.type_of(tag_all(result.reach.sol_q.astype(float), arm.meta["frames"]))

# --- the consumer: a belt in front of the arm, top-down picks, elbow up only -------------------
belt = AxisBox(lo=(0.30, -0.40, 0.05), hi=(0.65, 0.40, 0.40))
ws = workspace_from_result(result, "hemisphere", types=UR5E_BRANCHES.allowed(elbow="up"), sol_type=types)
print(ws)
over_belt = ws.region(belt)
print(over_belt)
rng = np.random.default_rng(7)
a, b = over_belt.sample_pairs(5, rng, same_type=True)
for i in range(5):
    print(f"A {np.round(a.T[i, :3, 3], 3)} -> B {np.round(b.T[i, :3, 3], 3)}  type: {UR5E_BRANCHES.type_name(int(a.branch_type[i]))}")

# every sampled q really reaches its pose (FK of the stored joints vs the pose)
s = over_belt.sample(2000, rng)
T_fk = arm.fk(s.q)
err_mm = np.linalg.norm(T_fk[:, :3, 3] - s.T[:, :3, 3], axis=1) * 1000
print(f"FK check on 2,000 samples: worst position error {err_mm.max():.2e} mm")
print("contains([0.5, 0, 0.2]):", bool(over_belt.contains([0.5, 0.0, 0.2])[0]),
      "· contains([0.5, 0, 1.5]):", bool(over_belt.contains([0.5, 0.0, 1.5])[0]))
want = np.eye(4); want[:3, 3] = (0.62, 0.05, 0.47); want[:3, :3] = np.diag([1.0, -1.0, -1.0])
T_ok, q_ok, dist, ang = over_belt.nearest_ok_pose(want)
print(f"nearest_ok_pose to (0.62, 0.05, 0.47) pointing down: {dist * 1000:.0f} mm away, {ang:.0f} deg off, "
      f"q = {np.round(np.degrees(q_ok), 1).tolist()} deg")
mask, origin, step = over_belt.voxels()
print(f"voxels(): mask {mask.shape}, {int(mask.sum())} filled, step {step[0] * 1000:.0f} mm")

# --- branch-aware vs not: how far the joints travel A -> B -------------------------------------
def travel(same_type: bool) -> np.ndarray:
    a, b = over_belt.sample_pairs(3000, np.random.default_rng(1), same_type=same_type)
    d = np.abs(np.degrees(a.q - b.q))
    d = np.minimum(d, 360 - d)  # the shorter way round a revolute joint
    return d.max(axis=1)

for flag in (True, False):
    t = travel(flag)
    print(f"A->B largest joint move, same_type={flag}: median {np.median(t):.0f} deg, 90th pct {np.percentile(t, 90):.0f} deg")

# --- Zach's simple alternative: sample joints, FK, keep what lands over the belt pointing down ---
q = arm.limits[:, 0] + np.random.default_rng(2).random((200_000, 6)) * (arm.limits[:, 1] - arm.limits[:, 0])
T = arm.fk(q)
code = arm.check(q, floor_z=0.0)
in_belt = belt.contains(T[:, :3, 3])
down = T[:, 2, 2] < -np.cos(np.radians(30))  # tool z within 30 deg of straight down
keep = in_belt & down & (code == 0)
print(f"FK sampling: {in_belt.mean() * 100:.1f} % land over the belt, {keep.mean() * 100:.2f} % also point down "
      f"(within 30 deg) and clear the floor -> {int(keep.sum())} of 200,000")
keep90 = in_belt & (T[:, 2, 2] < 0) & (code == 0)  # the hemisphere's own rule: within 90 deg of down
print(f"FK sampling, hemisphere rule (within 90 deg of down): {keep90.mean() * 100:.2f} % -> {int(keep90.sum())} of 200,000")
qa, qb = q[keep][: len(q[keep]) // 2], q[keep][len(q[keep]) // 2 : 2 * (len(q[keep]) // 2)]
d = np.abs(np.degrees(qa - qb)); d = np.minimum(d % 360, 360 - d % 360)
print(f"FK-sampled A->B largest joint move: median {np.median(d.max(1)):.0f} deg, 90th pct {np.percentile(d.max(1), 90):.0f} deg")
