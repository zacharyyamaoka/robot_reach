# robot_reach API — what exists today

Kinematics in, workspace out. An arm is its **forward kinematics (FK)** and **inverse kinematics (IK)**,
passed in as pure batched numpy functions, or built from a URDF / `RobotDescription`. `rr.analyze()`
computes once where the arm can go and which postures work, with one status code for every "why not".
The result is plain arrays on disk; the viser viewer shows it and recomputes on demand in its Python
backend; anything downstream reads the same arrays.

This page describes the API **as it is on `main`** (2026-11-24). The worked example runs every call
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
    fk=my_fk,                          # (N, dof) -> (N, 4, 4), or ((N, 4, 4), ok (N,)) / ((N, 4, 4), code (N,))
    ik=my_ik,                          # (N, 4, 4) -> (q (N, S, dof), code (N, S)): EVERY IK solution, a Code each
    task="position",                   # "pose" (6-DOF) | "axis" (5-DOF, roll free) | "position"
)
```

| Field | Shape / type | Meaning |
|---|---|---|
| `fk` | `(N, dof) -> (N, 4, 4)` | base->tool pose; the tool's +Z is the approach axis; may also return a `Code` per posture (`SELF_COLLISION`, `FK_FAIL`, ...) |
| `ik` | `(N, 4, 4) -> (q (N, S, dof), code (N, S))` | all S IK solutions (2 for an elbow, 8 for a UR); unused slot = `Code.PAD`; a collision is a code here |
| `task` | `"pose" \| "axis" \| "position"` | what part of a target the arm is asked to match; no false 6-DOF claim |
| `skeleton` | `(N, dof) -> (N, K, 3)` | points to draw the arm |
| `mass_points`, `masses`, `torque_limits` | `(N, dof) -> (N, M, 3)`, kg, N·m | static gravity torque vs motor limit |
| `collision` | `(N, dof) -> (N,)` bool or `Code` | optional thin adapter for a separate checker, folded into the codes; None = not checked here |
| `configurations` | `IKConfigurations` | the questions that name the IK solutions (UR: shoulder × elbow × wrist) and `tag(q)` answering them from a posture |
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

**The library re-checks what it can from the maths alone**: joint limits (trying the ±2π twin of revolute
joints), the FK round trip, duplicate solutions, torque and singularity. A lazy solver cannot overclaim.

**Collision is reported by your FK/IK**, never computed here: return `Code.SELF_COLLISION`,
`Code.ENVIRONMENT_COLLISION` (or `Code.COLLISION`) per IK solution or per FK posture and every view
enumerates and colours it. `analyze(floor_z=)` is only a convenience floor for toy arms.

**The tool frame is whatever your FK/IK return**: reach is measured for it (add a gripper's TCP to the
chain and the map is the fingers'). One serial chain per `Arm` today; multiple chains are a planned
extension (each its own `Arm` until then).

### Built-in arms (`robot_reach.robots`)

`pendulum()` 1-DOF · `rr()` 2-DOF · `five_bar()` 2-DOF closed chain · `shoulder3()` 3-DOF ·
`ur5e()` 6-DOF closed-form IK, 8 IK solutions named by `configurations`. No floor: the UR arms are in free space.

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
result = rr.analyze(arm)                      # step=, steps_across=, grid_shape=, box=, orientations=, n_fk=, joint_axes=, singular_tol=, seed=, floor_z=
```

1. **FK cloud** — sample joints, run FK, classify each sample. Works for any DOF (a pendulum's workspace
   is a circle no grid would hit) and bounds the grid, so nobody guesses an extent.
2. **Reach map** — the box's lattice kept inside the tightest of box / sphere / cylinder / convex hull
   (`grid_shape="auto"`, padded one step; `rr.sampling.fit_grid`) × one shared `OrientationSet` (22 approach
   directions × 6 rolls = 132), IK at every target, every IK solution kept with its final code and its IK
   configuration. `config["settings"]` records the step, grid shape, lattice and orientation steps.
3. **Joint maps** — the 2R picture for joint pairs: `slice` (others at home) and `sweep` (others
   sampled; value = fraction that works). 1-DOF: a line with its gravity torque.
4. **Joint ranges** — per joint, a 10° histogram of the working solutions' angles, against its limits.

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
| `sol_index`, `sol_q` | (K,), (K, dof) | every OK IK solution: flat index into (P, O, S), and its joints |
| `sol_config`, `configurations` | (K,) uint8, `IKConfigurations` | each solution's IK configuration (from its posture), and the questions |
| `filtered(allowed)` | `ReachMap` | only the allowed IK configurations: `m.filtered(m.configurations.allowed(elbow="up"))` |
| `n_solutions` | (P, O) | count of OK IK solutions |
| `valid_mask`, `reach_scores` | (P, O), (P,) | bam_reach v1 field names, kept |
| `joint_solutions` | (P, O, dof) | first OK solution per pose, NaN if none (v1's shape; robot_trajectories duck-types it) |
| `levels(threshold=1.0)` | (P,) | workspace level as an int key of `LEVEL_NAMES`: 0 unreachable, 3 reachable, 4 top-down, 5 hemisphere, 6 dexterous |
| `workspace(kind, threshold=1.0)` | (P,) bool | `reachable` · `four_dof` (top-down) · `hemisphere` · `dexterous` |
| `score(kind)` | (P,) | fraction of that kind's orientations reachable |
| `solutions(p)` | list | every OK `(orientation index, solver slot, q)` at point p (`solution_configs(p)`: their configurations) |
| `target(p, o)` | (4, 4) | the pose a solution reaches |
| `code_counts()` | dict | branches per code |

Workspaces, by name (`rr.LEVEL_NAMES`): **reachable** (at least one orientation works) · **top-down**
(straight down along the approach axis at every yaw) · **hemisphere** (every approach in the hemisphere,
every roll) · **dexterous** (every sampled orientation). Nested: each includes the next. All four are
masks over one map.

**`Code`** (`rr.Code`, `rr.CODE_LABELS`, `rr.CODE_COLORS`) — one vocabulary for every view:
`OK`, `NO_SOLUTION`, `JOINT_LIMIT`, `SINGULAR`, `COLLISION`, `SELF_COLLISION`, `ENVIRONMENT_COLLISION`
(the three collision codes are returned by your IK/FK; `rr.codes.is_collision`), `TORQUE_LIMIT`,
`FK_MISMATCH`, `FK_FAIL` (linkage cannot close), `EXCLUDED` (a filtered-out IK configuration, in memory
only), `PAD` (unused slot).

Typical downstream queries today are a few lines of numpy over these arrays (worked example §4-5):

```python
m = rr.load("out/ur5e").reach
hemi = np.flatnonzero(m.workspace("hemisphere"))             # sample a workspace
p = np.random.default_rng(0).choice(hemi)
for o, slot, q in m.solutions(p): ...                        # every IK solution there
belt = rr.Box(center=(0.6, 0, 0.15), size=(0.5, 0.8, 0.2))
over_belt = belt.contains(m.positions) & m.workspace("four_dof")  # top-down pickable over the belt
fine = rr.reach_map(arm, belt.grid(0.05), rr.OrientationSet.sample())   # finer map of one volume
```

### 4b. The workspace API: regions you sample from (`robot_reach.workspace`)

Zach's pick P1 (proposals round 2, G′), made general underneath (2026-11-24).

```python
ws = result.workspace()                       # ReachableWorkspace(chain, frame, positions, rotations, every checked IK solution)
ws.dexterous.sample(n, rng)                   # PoseSamples (n,)
ws.hemisphere.inside(belt).only(elbow="up").sample_pairs(n, rng)   # (n, 2)
ws.regions["dexterous"] == ws.region("dexterous") == ws.dexterous  # attributes are aliases
ws.regions.register("belt_pick", ws.top_down.inside(belt))         # then ws.belt_pick, ws.top_down.belt_pick
```

| Call | Returns | Notes |
|---|---|---|
| `ReachResult.workspace(*, frame="base_link", base_xyz=(0,0,0))` | `ReachableWorkspace` | also `ReachableWorkspace.from_reach_map(m, voxel_size=)` |
| `ws.regions` | `Regions` (Mapping) | `[name]`, iteration, `.register(name, region, *, parent=None)`, `.parent(name)`, `.children(name)` |
| `ws.region(name)`, `ws.full/.reachable/.top_down/.hemisphere/.dexterous`, `ws.<registered>` | `WorkspaceRegion` | built-ins nest in that order |
| `ws.only(allowed=None, **choices)` | `ReachableWorkspace` | same as filtering every region |
| `ws.from_spec(spec)` | `WorkspaceRegion` | rebuilds a `spec()` (box / cylinder zones) |
| `region.inside(zone)` | region | zone = anything with `contains(points)`: `rr.Box`, `rr.Cylinder` |
| `region.only(allowed=None, **choices)` | region | `elbow="up"`, `wrist=("J5+","J5−")`; membership re-judged; unknown question → ValueError |
| `region.within(deg)` | region | only orientations within `deg` of straight down |
| `region.regions`, `region.region(name)`, `region.<sub>` | regions | registered sub-regions, carrying this region's filters |
| `region.sample(n, rng)` / `.sample_pairs(n, rng, *, same_configuration=True)` / `.sample_waypoints(n, k, rng, ...)` | `PoseSamples` | uniform over poses; k distinct poses per draw |
| `region.contains(points)` | (N,) bool | the grid voxel the point falls in |
| `region.nearest_ok_pose(T, *, m_per_rad=0.1)` | `NearestPose(pose, q, distance_m, angle_deg, configuration)` | T is 4x4 or a position |
| `region.spec()` | dict (JSON) | schema `robot_reach.region.v1` |
| `region.to_sampler(*, same_configuration=True)` | `PoseSampler` | draw-only; `.save(path)` / `PoseSampler.load(path)` |
| `region.as_reach_map_like()` | namespace | robot_trajectories' duck type: `joint_names, joint_solutions, valid_mask, reach_scores` |
| `len(region)`, `.positions`, `.voxel_mask`, `.n_solutions`, `.count_by_configuration()` | | |

`PoseSamples`: `pose (...,4,4)`, `q (...,dof)`, `configuration`, `configuration_name(*i)`, and at the edge
`poses()` → `robot_msgs.geometry.Pose` list, `joint_states()` → `robot_msgs.sensor.JointState` list.
One chain is implemented; a workspace records its chain, frame and base placement so a cell of several
chains can be added without changing the region type.


## 5. Look: viser

```python
result.display()                              # blocks; the viewer recomputes live in Python
from robot_reach.overlay import add_reach_map
overlay = add_reach_map(server, result.reach, parent_frame="/cell/robot_1")   # into anyone's viser scene
overlay.kind = "four_dof"; overlay.update(colormap="magma", reverse=True); overlay.visible = False
```

`overlay.py` imports nothing from viser and works on viser 1.0.16 and 1.1.1. CLI:
`python -m robot_reach examples | urdf <path> | rest <spec> [--solve]`.

In the viewer: a settings card (what the map was computed with; edit → **Compute workspace**), colour by
workspace / IK solutions / fraction / consistency through a colormap (viridis default) or a traffic light,
a legend list (click a bin to hide it), **IK configurations** toggles that filter everything, click a
voxel to put the arm there, **Next ▶** walks every (orientation, IK solution) with its configuration's
name; **Cut**, **Blob** (voxel surface that keeps holes), **FK cloud**, **Jog** sliders with live code and
torque, **Ask here** → a box you move and tilt in 3-D (its −Z is the approach) → **Compute box** and
**Drop POI**; joint ranges as one histogram per joint; show/hide the skeleton.

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
position=q)`, `robot_msgs.manipulation.NamedPose(...)`. `PoseSamples.poses()` / `.joint_states()` return `Pose` / `JointState` (§4b). Not in robot_msgs yet: a
`manipulation.ReachableWorkspace` message holding §4b's arrays, with `WorkspaceRegion.spec()` as its query data.

## Run it

```bash
uv run --project /home/bam/robot_reach --extra viz python /home/bam/robot_reach/notebooks/reach_worked_example.py
```
