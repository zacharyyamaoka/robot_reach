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


REST_EXAMPLES = {"shoulder3": rest_example}
