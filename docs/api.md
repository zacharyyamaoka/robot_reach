# robot_reach API — what exists today

Kinematics in, workspace out. An arm is its **forward kinematics (FK)** and **inverse kinematics (IK)**,
passed in as pure batched numpy functions, or built from a URDF / `RobotDescription`. `rr.analyze()`
computes once where the arm can go and which postures work, with one status code for every "why not".
The result is plain arrays on disk; the viser viewer shows it and recomputes on demand in its Python
backend; anything downstream reads the same arrays.

This page describes the API **as it is on `main`** (2026-11-23). The worked example runs every call
below: [`notebooks/reach_worked_example.py`](../notebooks/reach_worked_example.py) (executed copy:
`reach_worked_example.ipynb`). What a typed downstream API could look like next is in
`docs/proposals/` and section G of `reports/reach-proposals-2026-11-23.html`; those are proposals, not code.

```text
 RobotDescription ─to_urdf_string()─┐
 URDF file / XML ───────────────────┼─> arm_from_urdf ─┐
 your fk(q), ik(T) ─────────────────┴──────────────────┴─> Arm ──> analyze() ──> ReachResult ──save()──> folder
                                                                                    │                     │
                                                         display() (viser) <────────┤        load() <─────┘
                                                         add_reach_map(server) <────┤
                                                         downstream: ReachMap arrays┘ (levels, workspace, solutions)
```

## 1. In: an `Arm`

```python
import robot_reach as rr

arm = rr.Arm(
    name="my_arm",
    joint_names=("q1", "q2", "q3"),
    lower=(-3.0, -1.0, -2.5),          # rad (m for prismatic)
    upper=(3.0, 2.0, 2.5),
    fk=my_fk,                          # (N, dof) -> (N, 4, 4), or ((N, 4, 4), ok (N,)) when FK can fail
    ik=my_ik,                          # (N, 4, 4) -> (q (N, S, dof), code (N, S)): EVERY branch, a Code each
    task="position",                   # "pose" (6-DOF) | "axis" (5-DOF, roll free) | "position"
)
```

| Field | Shape / type | Meaning |
|---|---|---|
| `fk` | `(N, dof) -> (N, 4, 4)` | base->tool pose; the tool's +Z is the approach axis |
| `ik` | `(N, 4, 4) -> (q (N, S, dof), code (N, S))` | all S branches (2 for an elbow, 8 for a UR); unused slot = `Code.PAD` |
| `task` | `"pose" \| "axis" \| "position"` | what part of a target the arm is asked to match; no false 6-DOF claim |
| `skeleton` | `(N, dof) -> (N, K, 3)` | points to draw the arm |
| `mass_points`, `masses`, `torque_limits` | `(N, dof) -> (N, M, 3)`, kg, N·m | static gravity torque vs motor limit |
| `collision` | `(N, dof) -> (N,)` bool | None = not checked (never assumed free) |
| `singular` | `(N, dof) -> (N,)` bool | a mechanism's own singularity test (a five-bar's) |
| `plane` | point, normal | a planar arm; grids collapse onto it |
| `meshes`, `mesh_poses` | `ArmMesh` parts, `(N, dof) -> (N, M, 4, 4)` | posed geometry for clearance and footprint |
| `velocity_limits`, `acceleration_limits`, `friction` | per joint | readiness (time to a pose) and the limp rest |
| `meta["home"]`, `meta["factory"]` | | home posture; the recipe `load()` uses to rebuild the arm |

Convenience: `rr.Arm.from_scalar(name, joints, lower, upper, fk, ik)` wraps one-pose-at-a-time
functions (`fk(q) -> 4x4`, `ik(T) -> [q, ...]`).

Methods every consumer can call on an `Arm`: `forward(q) -> (T, ok)`, `inverse(T) -> (q, code)`
(both chunked), `check(q, floor_z=, singular_tol=) -> code`, `jacobian(q)` (N, 6, dof),
`manipulability(q)`, `gravity_torque(q)`, `surface(q)`, `within_limits(q)`, `dof`, `limits`.

**The library re-checks every IK claim**: joint limits (trying the ±2π twin of revolute joints), the
FK round trip, duplicate branches, collision, torque and singularity. A lazy solver cannot overclaim.

### Built-in arms (`robot_reach.robots`)

`pendulum()` 1-DOF · `rr()` 2-DOF · `five_bar()` 2-DOF closed chain · `shoulder3()` 3-DOF ·
`ur5e()` 6-DOF closed-form IK, 8 branches.

### From a URDF or a RobotDescription (`robot_reach.urdf`)

```python
from robot_reach.urdf import analyze_chains, arm_from_description, arm_from_urdf, chains

chains(desc)                                  # [(base, tip, dof), ...]; desc = path, URDF XML, or a description
arm = arm_from_urdf("/abs/robot.urdf", "tool0")
arm = arm_from_description(desc, "tool0")     # duck-typed: only calls desc.to_urdf_string()
results = analyze_chains(desc, floor_z=0.0)   # {"tool0 (6 DOF)": ReachResult, ...}
```

FK from stdlib XML + numpy (revolute, continuous, prismatic, mimic). With no analytic IK, a batched
damped-least-squares IK runs from 8 seeds per target and merges duplicates, so several distinct
branches still come back (the worked example's 3-DOF URDF: 4 branches at one target). Pass `ik=` when
you have a closed form; it is several times faster.

## 2. Compute: `analyze()`

```python
result = rr.analyze(arm, floor_z=0.0)         # step=, box=, orientations=, n_fk=, joint_axes=, singular_tol=, seed=
```

1. **FK cloud** — sample joints, run FK, classify each sample. Works for any DOF (a pendulum's workspace
   is a circle no grid would hit) and bounds the grid, so nobody guesses an extent.
2. **Reach map** — a grid inside that box × one shared `OrientationSet` (22 approach directions × 6
   rolls = 132), IK at every target, every branch kept with its final code.
3. **Joint maps** — the 2R picture for joint pairs: `slice` (others at home) and `sweep` (others
   sampled; value = fraction that works). 1-DOF: a line with its gravity torque.
4. **Joint ranges** — limits vs the range the working solutions actually use.

Lower-level pieces are public too: `rr.fk_cloud`, `rr.reach_map(arm, positions, orientations)`,
`rr.joint_map`, `rr.joint_ranges`, `rr.Box`, `rr.OrientationSet.sample(approach=...)`.

## 3. Store and reload: the result log

```python
folder = result.save("out/ur5e")              # result.json + reach.reach.npz + cloud.npz + joint_maps.npz
result = rr.load("out/ur5e")                  # the arm is rebuilt from meta["factory"] when it has one
result.summary()                              # plain dict: counts, codes, levels, timings
```

Pickle-free and versioned (`schema` strings `robot_reach.result.vN` / `robot_reach.reach_map.vN`, bumped when the
layout changes; the examples CLI recomputes a result saved under an older one): readable without the code.

## 4. Out: what a downstream consumer reads

`ReachResult` holds `reach` (a `ReachMap`, or None for a curve workspace), `joint_maps`, `ranges`,
`cloud_q` / `cloud_p` / `cloud_code`, `config`, `timings` and the live `arm`.

**`ReachMap`** — the Cartesian answer:

| Member | Shape | Meaning |
|---|---|---|
| `positions` | (P, 3) | grid points, arm base frame, m |
| `orientations.R` | (O, 3, 3) | one orientation set shared by every point |
| `code` | (P, O, S) int8 | final `Code` per IK branch |
| `sol_index`, `sol_q` | (K,), (K, dof) | every OK branch: flat index into (P, O, S), and its joints |
| `n_solutions` | (P, O) | count of OK branches |
| `valid_mask`, `reach_scores` | (P, O), (P,) | bam_reach v1 field names, kept |
| `joint_solutions` | (P, O, dof) | first OK branch per pose, NaN if none (v1's shape; robot_trajectories duck-types it) |
| `levels(threshold=1.0)` | (P,) | pose DOF the arm controls there: 0, 3, 4, 5, 6 |
| `workspace(kind, threshold=1.0)` | (P,) bool | `reachable` · `four_dof` · `hemisphere` · `dexterous` |
| `score(kind)` | (P,) | fraction of that kind's orientations reachable |
| `solutions(p)` | list | every OK `(orientation index, branch index, q)` at point p |
| `target(p, o)` | (4, 4) | the pose a solution reaches |
| `code_counts()` | dict | branches per code |

Levels: **3** reachable (some orientation works) · **4** straight-down approach at every yaw (top-down
picking) · **5** hemisphere (every approach within 90° of the reference, every roll) · **6** dexterous
(every sampled orientation). All four are masks over one map.

**`Code`** (`rr.Code`, `rr.CODE_LABELS`, `rr.CODE_COLORS`) — one vocabulary for every view:
`OK`, `NO_SOLUTION`, `JOINT_LIMIT`, `SINGULAR`, `COLLISION`, `TORQUE_LIMIT`, `FK_MISMATCH`, `FK_FAIL`
(linkage cannot close), `PAD` (unused slot).

Typical downstream queries today are a few lines of numpy over these arrays (worked example §4-5):

```python
m = rr.load("out/ur5e").reach
hemi = np.flatnonzero(m.workspace("hemisphere"))             # sample a workspace
p = np.random.default_rng(0).choice(hemi)
for o, branch, q in m.solutions(p): ...                      # every IK branch there
belt = rr.Box(center=(0.6, 0, 0.15), size=(0.5, 0.8, 0.2))
over_belt = belt.contains(m.positions) & m.workspace("four_dof")  # top-down pickable over the belt
fine = rr.reach_map(arm, belt.grid(0.05), rr.OrientationSet.sample())   # finer map of one volume
```

There is no `ReachMap.sample()` or typed query object yet; that is what the proposals are about.

## 5. Look: viser

```python
result.display()                              # blocks; the viewer recomputes live in Python
from robot_reach.overlay import add_reach_map
overlay = add_reach_map(server, result.reach, parent_frame="/cell/robot_1")   # into anyone's viser scene
overlay.kind = "four_dof"; overlay.visible = False
```

`overlay.py` imports nothing from viser and works on viser 1.0.16 and 1.1.1. CLI:
`python -m robot_reach examples | urdf <path> | rest <spec> [--solve]`.

In the viewer: click a voxel to put the arm there, **Next ▶** walks every (orientation, IK branch);
**Cut**, **Blob** (voxel surface that keeps holes), **FK cloud**, **Jog** sliders with live code and
torque, **Ask here** → **Compute box** (finer map of a dragged box) and **Drop POI** (every
orientation at one point).

## 6. Rest and ready poses (`robot_reach.rest`, `robot_reach.goals`)

`RestStudy(arm, Scene, motors, gates, criteria).measure(q)` → footprint, clearance over a belt and
holding power; `goals.solve(study, PoseGoal(...))` → gates, Pareto set, pick, and a NamedPose-ready
`record`. See the README's "Find a good ready/rest pose" section.

## robot_msgs: types used and produced

robot_reach's core (FK, IK, the reach map, the result log) speaks **numpy arrays**: 4×4 homogeneous
poses and joint vectors in radians, SI. It is the shared vocabulary only at these seams today:

| robot_msgs name | Direction | Where | What for |
|---|---|---|---|
| `manipulation.NamedPose` | **produced** | `goals.to_named_pose(record)` | a solved rest/ready pose as a saved, digest-stamped joint pose |
| `manipulation.named_poses_from_json` (→ `NamedPose`) | **consumed** | `rest.named_poses(file, arm)`, `rest --poses` | read a description's named-poses file and check each pose against the arm |
| `geometry.rotations.quat_from_matrix`, `xyzw_to_wxyz` | used | `viewer.py`, `rest_viewer.py` | rotation matrix → viser's wxyz quaternion for posing meshes |

Not robot_msgs, but the kinematics input: **`RobotDescription`** (bam_robot_descriptions, in bam_ws) is
duck-typed through `to_urdf_string()`; robot_reach never imports it.

A consumer that wants messages for a reach solution builds them from the arrays (worked example §7):
`robot_msgs.geometry.Pose.from_matrix(m.target(p, o))`, `robot_msgs.sensor.JointState(name=m.joint_names,
position=q)`, `robot_msgs.manipulation.NamedPose(...)`. robot_reach does not return `Pose`, `JointState` or
a workspace message itself today; a typed workspace hand-off is in the proposals.

## Run it

```bash
uv run --project /home/bam/robot_reach --extra viz python /home/bam/robot_reach/notebooks/reach_worked_example.py
```
