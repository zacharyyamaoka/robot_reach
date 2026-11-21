"""The example gallery: a 1-DOF pendulum, 2-DOF RR, 2-DOF five-bar, 3-DOF shoulder and 6-DOF UR5e.

WHY these five (Zach, 2026-11-19: "Then please show example for 1 DOF arm (pendulum) 2 DOF arm (RR)
3 DOF arm (just shoulder) and 6 DOF arm"); the five-bar is added because it is BAM's own shoulder
and the one 2-DOF arm whose joint-space picture has a "cannot close" hole.
"""

from __future__ import annotations

from .robots import five_bar, pendulum, rr, shoulder3, ur5e

# name -> (arm factory, analyze() keyword arguments)
EXAMPLES: dict[str, tuple] = {
    "pendulum": (pendulum, {}),
    "rr": (rr, {"floor_z": -0.15}),
    "five_bar": (five_bar, {}),
    "shoulder3": (shoulder3, {"floor_z": 0.0}),
    "ur5e": (ur5e, {"floor_z": 0.0}),
}


# --------------------------------------------------------------------------- rest-pose template
def rest_example():
    """The rest-pose template (README "Find a good ready/rest pose for any arm"): `shoulder3` on a stand
    beside a fake belt, five candidate poses, gates and anchored criteria fixed up front.

    The belt: its top 5 cm under the arm's base, its edges 0.15 and 0.55 m in front; the reference line
    is the arm's own vertical (x 0) along the belt; a rail beyond the far edge; the next arm 0.30 m
    along the belt (the lane, +-0.15 m). The skeleton gets 5 cm boxes (`with_link_boxes`) so clearance
    and footprint have thickness; the forearm is the hand."""
    import numpy as np

    from .rest import Criterion, Gate, KeepOut, Motor, PoseChoice, RestStudy, Scene, with_link_boxes

    arm = with_link_boxes(shoulder3(), width=0.05, names=("base", "upper", "forearm"))
    scene = Scene(
        origin=(0.0, 0.0, 0.0), across=(1.0, 0.0, 0.0), along=(0.0, 1.0, 0.0),
        plane_point=(0.35, 0.0, -0.05), region_across=(0.15, 0.55),
        keepouts=(KeepOut("far rail", top=0.0, across=(0.55, 0.58)),),
        neighbours=("along",), lane_along=(-0.15, 0.15), min_clearance=0.10,
        name="shoulder3 beside a fake belt", labels={"plane": "belt", "line": "the arm's vertical"},
    )
    D = np.deg2rad
    poses = (
        PoseChoice("upright", D([0.0, 90.0, 0.0]), "Straight up: no gravity torque, tallest", kind="candidate"),
        PoseChoice("home", D([0.0, 45.0, -90.0]), "The arm's own home: elbow up, hand forward", kind="candidate"),
        PoseChoice("over the belt", D([0.0, 20.0, -50.0]), "Hand out over the near half of the belt", kind="candidate"),
        PoseChoice("folded back", D([0.0, 110.0, -150.0]), "Forearm folded back down beside the upper arm", kind="candidate"),
        PoseChoice("turned along", D([90.0, 45.0, -90.0]), "Yawed 90 deg: the hand along the belt, toward the next arm", kind="candidate"),
    )
    motor = Motor(kt=0.10, resistance=0.25, ratio=15.0, name="illustrative BLDC, 15:1")
    return RestStudy(
        arm=arm, scene=scene, name="shoulder3 rest poses (template)", motors=(motor,) * 3, poses=poses, hand=("forearm",),
        gates=(Gate("checks OK", "code_ok"), Gate(">= 100 mm over the belt", "clearance_m", min=0.10),
               Gate("inside its lane", "past_shoulder_m", max=0.0)),
        criteria=(Criterion("low holding power", "power_w", 2.0, best=0.0, worst=3.0),
                  Criterion("compact", "reach_m", 1.0, best=0.10, worst=0.40),
                  Criterion("room over items", "clearance_m", 1.0, best=0.40, worst=0.10)),
    )


REST_EXAMPLES = {"shoulder3": rest_example}  # replaced below by the goals example (a superset)


def goals_example():
    """Zach's four use cases (robot_reach.goals) on the rest template: shoulder3 beside the fake belt,
    plus the stand it is bolted to (a 'table' the arm may lie on when it is off), drive limits,
    static friction and an overhead camera. Numbers are illustrative.

    - hover_empty_belt: hover just above an empty belt (20 mm), holding power first, then readiness;
    - sleep_limp: the machine is off: torque off, it may rest on the belt or the table;
    - standby: another arm is picking: 120 mm over the belt, inside its lane, low power, out of the camera's view;
    - ready_to_pick: reach every hover point over the belt fast, power second."""
    import dataclasses

    import numpy as np

    from .goals import Camera, PoseGoal, Workspace
    from .rest import KeepOut

    study = rest_example()
    arm = dataclasses.replace(study.arm, velocity_limits=(2.0, 2.0, 3.0), acceleration_limits=(6.0, 6.0, 10.0),
                              friction=(0.05, 0.10, 0.05))
    scene = dataclasses.replace(study.scene, keepouts=(*study.scene.keepouts, KeepOut("table", top=0.0, across=(-0.40, 0.15))))
    # the workspace: 3 x 3 hover points 6 cm over the belt, across 0.25..0.45 m, along -0.10..0.10 m
    grid = [(x, y) for x in (0.25, 0.35, 0.45) for y in (-0.10, 0.0, 0.10)]
    T = np.tile(np.eye(4), (len(grid), 1, 1))
    T[:, :3, 3] = [(x, y, -0.05 + 0.06) for x, y in grid]
    region = np.array([(x, y, -0.05) for x in np.linspace(0.15, 0.55, 9) for y in np.linspace(-0.15, 0.15, 7)])
    ws = Workspace("pick zone", targets=T, task="position", region=region, stat="mean")
    cam = Camera((0.35, 0.0, 0.80), name="overhead camera")
    goals = (
        PoseGoal("hover_empty_belt", {"belt": 0.020, "table": 0.0}, "held", workspace=ws, weights={"energy": 3.0, "readiness": 1.0},
                 description="Hover just above an empty belt"),
        PoseGoal("sleep_limp", {"belt": 0.0, "table": 0.0}, "limp", workspace=ws, weights={"compactness": 2.0, "readiness": 1.0},
                 settle_deg=3.0, description="Machine off: torque off, resting where gravity leaves it"),
        PoseGoal("standby", {"belt": 0.120, "table": 0.0, "lane": 0.0}, "held", workspace=ws, camera=cam,
                 weights={"energy": 2.0, "compactness": 1.0, "occlusion": 1.0}, description="Another arm is picking: clear and cheap"),
        PoseGoal("ready_to_pick", {"belt": 0.100, "table": 0.0, "lane": 0.0}, "held", workspace=ws,
                 weights={"readiness": 3.0, "energy": 1.0}, description="Ready to pick anywhere in the zone quickly"),
    )
    return dataclasses.replace(study, arm=arm, scene=scene, goals=goals, name="shoulder3 rest goals (template)")


REST_EXAMPLES = {"shoulder3": goals_example}
