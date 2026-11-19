# robot_reach — BAM Reach v2

Reachability of any arm, in **Cartesian space** and **joint space**, from FK and IK passed in as
pure functions. Compute once, save the result, look at it in viser any time later.

```bash
uv run --project /home/bam/robot_reach python -m robot_reach examples
```

Opens one viewer with five arms behind an **Arm** dropdown: a 1-DOF pendulum, a 2-DOF RR, a 2-DOF
five-bar, a 3-DOF shoulder (yaw + RR) and a 6-DOF UR5e. Click a voxel to put the arm there, then
**Next ▶** walks every (orientation, IK branch) that works at that point.

## The API

```python
import robot_reach as rr
from robot_reach.robots import ur5e

result = rr.analyze(ur5e(), floor_z=0.0)   # FK cloud + reach map + joint maps + joint ranges
result.save("out/ur5e")                     # the result log: npz + result.json, no pickle
rr.load("out/ur5e").display()               # viser; the arm is rebuilt from its recorded factory
```

Your own arm is two batched functions:

```python
arm = rr.Arm(
    name="my_arm",
    joint_names=("q1", "q2", "q3"),
    lower=(-3.0, -1.0, -2.5),
    upper=(3.0, 2.0, 2.5),
    fk=my_fk,      # (N, dof) -> (N, 4, 4)            [or ((N,4,4), ok) when FK can fail]
    ik=my_ik,      # (N, 4, 4) -> (q (N, S, dof), code (N, S))   every branch, a Code each
    task="position",                       # "pose" (6-DOF), "axis" (5-DOF), "position"
)
```

One-pose-at-a-time functions work too: `rr.Arm.from_scalar(name, joints, lower, upper, fk, ik)`.
Optional hooks: `skeleton` (points to draw), `mass_points` + `masses` + `torque_limits` (static
gravity torque), `collision`, `singular` (a mechanism's own singularity test), `meta["home"]`.

The library re-checks everything an IK claims: joint limits (trying the ±2π twin of revolute
joints), the FK round trip, duplicate branches, collision, torque and singularity. A lazy solver
cannot overclaim.

## What you see

**Cartesian.** Grid points (bounded by the FK cloud, so nobody guesses the extent) × one shared
orientation set (22 approach directions × 6 rolls = 132 by default). Every point gets a level —
how many pose DOF the arm controls there:

| Level | Name | Meaning |
|---|---|---|
| 3 | reachable | at least one orientation works |
| 4 | 4-DOF | straight-down approach at every yaw (top-down picking) |
| 5 | hemisphere | every approach within 90° of the reference, every roll |
| 6 | dexterous | every sampled orientation |

All four are masks over one map. The approach direction is yours (`OrientationSet.sample(approach=...)`,
or the **Approach** dropdown in the viewer's Ask here panel).

**Joint space.** The 2R picture for any pair of joints: **slice** (others held at home) or **sweep**
(others sampled; colour = fraction of postures that work). A 1-DOF arm gets a line with its gravity
torque against the motor limit. Plus one bar per joint: limits vs the range the working IK solutions
actually use.

**Codes** (one vocabulary for every view): OK, no IK solution, outside joint limits, singular,
collision, over torque limit, FK(IK) mismatch, linkage cannot close. Empty is white; only problems
are saturated.

**Viewer extras.** Cut plane (look inside the shell), blob (voxel boundary surface: keeps holes and
separate bodies, unlike a convex hull), FK cloud, Jog sliders with the live code and torque, and
**Ask here**: drag a box and **Compute box** samples it finer; **Drop POI** checks every orientation
at one point and draws a green/grey spike per approach.

## Any URDF or RobotDescription, every kinematic chain

```python
from robot_reach.urdf import analyze_chains, arm_from_description, arm_from_urdf, chains

chains(desc)                                   # [(base, tip, dof), ...] — desc, URDF path or XML
arm = arm_from_description(desc, "ee_link")    # duck-typed: only calls desc.to_urdf_string()
results = analyze_chains(desc, floor_z=0.0)    # {"ee_link (6 DOF)": ReachResult, ...}
```

```bash
uv run --project /home/bam/robot_reach --extra viz python -m robot_reach urdf /abs/path/robot.urdf --floor 0.0
```

FK comes from the URDF with stdlib XML + numpy (revolute, continuous, prismatic, mimic). Without an
analytic IK, a batched damped-least-squares IK runs from 8 seeds per target and merges duplicates,
so you still get several distinct branches (UR5e: 100 % solved, 4.5 branches on average; every one
matches a closed-form branch). It is slower than closed form — UR5e quick map (48 orientations,
131k targets) ~75 s vs ~19 s for the 132-orientation closed-form map — so pass `ik=` when you have
one. `quick=True` (default) samples 48 orientations at a grid step of extent/12.

## Drawing into someone else's scene

```python
from robot_reach.overlay import add_reach_map

overlay = add_reach_map(server, result.reach, parent_frame="/facility/robot_1")
overlay.kind = "four_dof"      # reachable | four_dof | hemisphere | dexterous
overlay.visible = False
```

Imports nothing from viser and uses only calls present in viser 1.0.16 and 1.1.1.

## Examples (measured 2026-11-19, one process)

| Arm | DOF | IK branches | Grid | Time |
|---|---|---|---|---|
| pendulum | 1 | 1 | — (workspace is a circle: FK cloud) | 0.1 s |
| rr | 2 | 2 | 1,476 points | 0.3 s |
| five_bar | 2 | 4 | 1,517 points | 0.4 s |
| shoulder3 | 3 | 4 | 5,776 points | 1.2 s |
| ur5e | 6 | 8 | 6,859 points × 132 orientations | ~20 s (1.0 M valid solutions) |

## What v2 kept from bam_reach v1 (bam_ws, July 2026)

| Element | v1 `bam_reach` | v2 `robot_reach` | Must survive |
|---|---|---|---|
| FK/IK in as functions | `Kinematics` protocol, `FunctionalKinematics` | `Arm(fk=, ik=)`, batched | any solver plugs in |
| Validity rule | IK ∧ FK ∧ round trip ∧ no checked collision | same, plus limits, torque, singular | unchecked ≠ free |
| Workspaces | four masks over one map | same, as levels 3-6 | one IK run |
| Lower-DOF arms | position-only targets | `task="position"` | no false 6-DOF claim |
| Grid | centred, expand never shrink, FK-bounded | same | authored box never clipped |
| Storage | versioned npz, no pickle | same | readable without the code |
| v1 field names | `joint_names`, `joint_solutions`, `valid_mask`, `reach_scores` | same names | robot_trajectories duck-types them |
| IK branches | first one only | **all of them** | click through solutions |
| Speed | one Python call per pose | vectorised, chunked | — |
| Joint space | none | slice / sweep / line / range bars | — |

v1 still owns the workcell studio, the dog-leg campaign and the trajectory dataset; they move here
only when needed.

## Tests

```bash
uv run --project /home/bam/robot_reach --extra test pytest -q
```
