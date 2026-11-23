# robot_reach — BAM Reach v2

Reachability of any arm, in **Cartesian space** and **joint space**, from FK and IK passed in as
pure functions. Compute once, save the result, look at it in viser any time later.

```bash
uv run --project /home/bam/robot_reach python -m robot_reach examples
```

Opens one viewer with seven arms behind an **Arm** dropdown: a 1-DOF pendulum, a 2-DOF RR, a 2-DOF
five-bar, a 3-DOF shoulder (yaw + RR), a 6-DOF UR5e (closed form, skeleton) and two arms loaded from
stock URDFs with their meshes: **ur5e (URDF)** and **ur5e + 2F-85 (URDF)** (the Robotiq 2F-85 on
`tool0`; the reach is of its TCP, 0.150 m along the gripper, mid finger pad). Click a voxel to put the
arm there, then **Next ▶** walks every (orientation, IK branch) that works at that point; Esc or a
click on empty space deselects.

The URDF arms read robotdescription.com's gallery builds in place (`examples.UR5E_URDF`,
`ROBOTIQ_2F85_URDF`; env `ROBOT_REACH_UR5E_URDF` / `ROBOT_REACH_ROBOTIQ_2F85_URDF`). Their first run
takes ~1.5 min each (numeric IK), then is reused from `out/`. UR's visual meshes are `.dae`, which
trimesh reads only with pycollada, so run with `--with pycollada` for them; without it the viewer
draws the URDF collision meshes and says so:

```bash
uv run --project /home/bam/robot_reach --extra viz --with pycollada python -m robot_reach examples --arm "ur5e (URDF)"
```

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
gravity torque), `collision`, `singular` (a mechanism's own singularity test), `meta["home"]`,
`velocity_limits` + `acceleration_limits` + `friction` (per joint: how fast a pose is reached, what holds a
joint with its torque off), and
`meshes` + `mesh_poses` (rigid `ArmMesh` parts posed per q: `arm.surface(q)` gives dense points for
clearance and footprint, where the skeleton has no thickness).

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
(each cell = 48 postures of the other joints, uniform over their full limits or over ranges you pick
in the viewer's **Sweep: other joints' ranges**). A sweep cell is, worst first: *always singular*,
*always collision* (every posture fails that way: the two joints alone decide it, like a collision
matrix), *never OK* (pale: its worst failure, singular › collision › other), *OK for some* (shade =
share of postures), *OK for all*. A 1-DOF arm gets a line with its gravity torque against the motor
limit. Plus one bar per joint: limits vs the range the working IK solutions actually use. Every joint
name carries its chain index: `shoulder_lift_joint (J2)`.

**Codes** (one vocabulary for every view): OK, no IK solution, outside joint limits, singular,
collision, over torque limit, FK(IK) mismatch, linkage cannot close. Empty is white; only problems
are saturated.

**Workspace panel.** *Colour by* pose DOF, IK solutions, fraction of orientations or **consistency**
(did FK of every IK answer land back on its target: matched / some mismatched / all mismatched / no
answer to check). *Show* follows the colouring and always offers **All**: every sampled grid point,
the empty ones light grey. The FK cloud explains itself next to its checkbox: random joint postures
through FK, whose bounding box is the grid the voxels are computed on; its colours are each posture's
own joint-space check. *Selected point* and *Ask here* state the angular resolution behind "N / M
orientations" (e.g. `approach -z: tilt 0…180° in 45° rings (22 directions) × roll 0…300° every 60°
(6 rolls) = 132 orientations`).

**Robot model.** For an arm with meshes: *Show mesh* visual / collision / both / none and *Opacity*.
A colliding pose (Jog) marks the skeleton points under the floor and the parts in contact red, and
shows their collision mesh whatever Show says. The panel states what the check really uses: the floor
tested on the skeleton's joint points, not the meshes (no self-collision model yet).

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

FK comes from the URDF with stdlib XML + numpy (revolute, continuous, prismatic, mimic).
`arm_from_urdf(..., meshes=True)` (or `analyze_chains(..., arm_kw={"meshes": True})`) also reads every
link's visual and collision geometry (trimesh, the viz extra) into `Arm.meshes`, posed per q; links off
the chain (a gripper's fingers) ride along at their joint zero. Without an
analytic IK, a batched damped-least-squares IK runs from 8 seeds per target and merges duplicates,
so you still get several distinct branches (UR5e: 100 % solved, 4.5 branches on average; every one
matches a closed-form branch). It is slower than closed form — UR5e quick map (48 orientations,
131k targets) ~75 s vs ~19 s for the 132-orientation closed-form map — so pass `ik=` when you have
one. `quick=True` (default) samples 48 orientations at a grid step of extent/12.

## Find a good ready/rest pose for any arm

Where should an arm wait? `robot_reach.rest` measures one pose over a scene and the explorer shows
it: the arm in 3D, and beside it the three numbers that decide a rest pose.

```bash
uv run --project /home/bam/robot_reach --extra viz python -m robot_reach rest shoulder3
```

- **Footprint**: how far the hand reaches from a reference line (the belt's centre line, the
  shoulder's vertical) across and along, the hull area of the whole arm on the plane, and how far
  anything sticks past the arm's lane toward a neighbouring arm.
- **Clearance**: the lowest arm point over the region (the belt between its edges), measured along
  the plane's normal, and the clearance over each keep-out (a rail). Red only under the gate.
- **Holding power**: static gravity torque per joint by virtual work (tau = dU/dq through the arm's
  own FK, so a five-bar's closure is handled by its FK), into copper watts per motor
  (P = 3/2·R·i², i = |tau| / (kt·ratio)), or your measured `torque(q)` / `power(q, tau)`.

The template for your arm, in order (the shoulder3 example is `robot_reach.examples.rest_example`):

1. **The arm**: an `Arm` with masses (`mass_points` + `masses`) and, for real clearance, geometry:
   `meshes` + `mesh_poses` (CAD), or `rest.with_link_boxes(arm, width)` to fatten a skeleton.
2. **The scene** (arm frame, z up): `Scene(origin=<a point on the reference line>, across=, along=,
   plane_point=, plane_normal=, region_across=(lo, hi), keepouts=(KeepOut("rail", top=, across=),),
   neighbours=("along",), lane_along=(lo, hi), min_clearance=)`. Coordinates are measured from
   `origin`: across, along, up.
3. **The motors**: `Motor(kt, resistance, ratio)` per joint (`Motor.from_kv` for a Kv), or a
   measured `power(q, tau)`.
4. **Criteria before options**: `Gate`s (pass/fail) and anchored `Criterion`s (score 1 at `best`,
   0 at `worst`), fixed before you look at the candidates.
5. **The poses**: named poses from a robot_msgs named-poses JSON (`--poses`; name, description and
   digest shown) and candidates from any plain JSON (`--candidates`), then
   `rank(reports_for(study), study.gates, study.criteria)`.

```python
import numpy as np

from robot_reach.rest import Criterion, Gate, KeepOut, Motor, RestStudy, Scene, with_link_boxes
from robot_reach.robots import shoulder3

arm = with_link_boxes(shoulder3(), width=0.05, names=("base", "upper", "forearm"))
scene = Scene(origin=(0.0, 0.0, 0.0), plane_point=(0.35, 0.0, -0.05), region_across=(0.15, 0.55),
              keepouts=(KeepOut("far rail", top=0.0, across=(0.55, 0.58)),), lane_along=(-0.15, 0.15),
              min_clearance=0.10)
study = RestStudy(arm, scene, motors=(Motor(kt=0.10, resistance=0.25, ratio=15.0),) * arm.dof, hand=("forearm",),
                  gates=(Gate(">= 100 mm over the belt", "clearance_m", min=0.10),),
                  criteria=(Criterion("low holding power", "power_w", 2.0, best=0.0, worst=3.0),))
print(study.measure(np.deg2rad([0.0, 45.0, -90.0])).headline())
```

```text
{'footprint': '407 mm from the line · hull 0.022 m²', 'clearance': '168 mm', 'holding': '2.66 W'}
```

A candidates file is a list of rows (or `{"candidates": [...]}`, or `{name: row}`), each naming its
joints in radians (`q_rad`, `joint_rad`) or degrees (`q_deg`, `joint_deg`); a study's `aliases` map
other joint names (`J1`) to the arm's:

```json
{"candidates": [{"name": "fist up", "description": "claw up beside the forearm",
                 "joint_deg": {"J1": -2, "J2": 40, "C1": 70, "W1": 120, "W2": 90, "W3": 0}}]}
```

Viewer options (after the spec): `--pose NAME`, `--set JOINT=DEG` (a scripted jog), `--export DIR`
(one PNG + metrics JSON per pose and `index.json` with the ranking), `--out DIR` (where the
**Save PNG + card** button writes), `--no-view`, `--port`. The spec is an example name,
`/abs/file.py:func` or `module:func` returning a `RestStudy`.

### Solve for the best pose of a use case

Zach's four cases (*"you have a robot arm ... define a workspace where you want to do picking and then
like zones that you need to stay clear of ... and then within that zone you can pick the ideal
position"*) are each a `PoseGoal` with a few settings, solved on any `Arm`:

```bash
uv run --project /home/bam/robot_reach --extra viz python -m robot_reach rest shoulder3 --solve
```

| Setting | What it says | Example (a belt) |
|---|---|---|
| `clearances` | least distance per scene zone (`"plane"` or its label, a keep-out's name, `"lane"`); `0` = touching allowed, `None` = ignored | 0.120 m while waste flows, ~0.020 m over an empty belt, 0 with the machine off |
| `torque` | `"held"` (energy = copper W) or `"limp"` (torque off: a passive rest that settles within `settle_deg` and survives a `kick_deg` nudge; energy 0) | `"limp"` when the machine is off |
| `workspace` | the task poses to be ready for (tool poses solved by the arm's IK or a numeric IK, or your planner's joint configurations); **readiness** = time to each (trapezoid per joint under `Arm.velocity_limits` / `acceleration_limits`, the slowest joint), mean and worst | hover points over the pick zone |
| `weights` | over `energy` (W), `readiness` (s), `compactness` (the hand's reach from the reference line), `occlusion` (fraction of the workspace a `Camera` loses behind the arm) | ready: readiness 3, energy 1 |

Gates first (clearances, limits less `stop_margin_deg`, closure, self-collision, torque limits when
held, the passive rest when limp, your own `Gate`s), then a weighted score over objectives normalised
over the poses that pass (or your `anchors`). `solve` samples the joint space (a closed chain through
its FK: a pose that does not close fails), refines the best few by a pattern search, and returns the
Pareto set, the pick and a NamedPose-ready `record` (a one-line description with the headline numbers,
a why listing the settings and the result; `goals.to_named_pose(record)`).

**Limp, what is modelled**: gradient flow on the potential of the arm's point masses through its own
FK (no inertia, so no overshoot); the joint limits are the hard stops; a joint holds while its gravity
torque is under `Arm.friction` (the pick also reports where it would end with no friction at all); a
zone the goal lets it touch holds the arm where it first touches — sliding about the contact is not
modelled.

In the explorer the **Use case** folder (its **Goal** dropdown) shows a goal's settings, the pick and its Pareto runners-up (the
**Solution** dropdown), the arm at the one chosen and a fourth headline number, readiness.
`--solve --solutions FILE` saves the solutions; `--solutions FILE` alone reloads them (a goal whose
settings changed is dropped and must be re-solved).

**bam_fb_v2** (the instance this was built for) lives in bam_ws, not here (robot_reach never imports
bam_ws): `docs/named_poses/rest_explorer/` — the full arm's CAD meshes and 15c masses, the measured
belt, the fitted J2/C1 comp, the description's named poses — launched with its `rest-explorer`
script. Its numbers match the 2026-11-20 W1-plumb study on all 11 of its poses within 2e-6 W and 1e-12 mm (test tolerance 0.01 W, 0.1 mm: `rest-explorer test`).

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
| ur5e (URDF), numeric IK (2026-11-23) | 6 | up to 8 | 2,744 points × 48 orientations | 86 s |
| ur5e + 2F-85 (URDF), TCP | 6 | up to 8 | 2,366 points × 48 orientations | 84 s |

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
