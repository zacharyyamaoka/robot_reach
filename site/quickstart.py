"""The landing page's quickstart: load a robot -> its workspace -> sample a target. build_site.py shows this file
verbatim and, with --run, re-executes it and shows the printed output beside it (site/quickstart.out)."""

import numpy as np

import robot_reach as rr
from robot_reach.robots import ur5e

# 1. Load a robot: its kinematics are FK + IK (or rr.urdf.arm_from_description(description, "tool"))
result = rr.analyze(ur5e())
reach = result.reach
print(f"{reach.arm_name}: {reach.n_positions} grid points x {reach.n_orientations} orientations")

# 2. Its workspace: one mask per kind, every IK answer already re-checked by FK
inside = reach.workspace("hemisphere")
print(f"hemisphere workspace: {inside.sum()} of {reach.n_positions} points")

# 3. Sample: a target the arm reaches, with the joint angles that reach it
p = np.random.default_rng(0).choice(np.flatnonzero(inside))
orientation, configuration, q = reach.solutions(p)[0]
print("target (m):", reach.positions[p].round(3), "| q (deg):", np.rad2deg(q).round(1))
print(f"{len(reach.solutions(p))} working (orientation, IK configuration) pairs at this point")

result.save("out/quickstart_ur5e")  # plain npz + json; rr.load("out/quickstart_ur5e").display() opens it live in viser
