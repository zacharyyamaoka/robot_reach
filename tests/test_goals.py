"""robot_reach.goals against hand-derived answers: trapezoid timing, a pendulum's passive rests (minimum,
stop, friction, a support it may touch, an unstable balance), camera occlusion, and the 4-goal template."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from test_rest import H, L, M, S, pendulum_arm

from robot_reach.goals import (
    Camera,
    PoseGoal,
    load_solutions,
    occlusion,
    readiness,
    save_solutions,
    settle,
    solve,
    to_named_pose,
    trapezoid_time,
)
from robot_reach.rest import RestStudy, Scene

G = 9.81


def test_trapezoid_and_triangle_times():
    v, a = np.array([1.0]), np.array([2.0])
    # long move: d/v + v/a ; short move (d <= v^2/a = 0.5): 2 sqrt(d/a)
    assert trapezoid_time(np.array([2.0]), v, a)[0] == pytest.approx(2.0 + 0.5)
    assert trapezoid_time(np.array([-0.18]), v, a)[0] == pytest.approx(2 * math.sqrt(0.09))
    # the slowest joint sets a move's time; the fastest branch sets a target's
    targets = np.array([[[1.0, 0.0], [0.1, 0.1]], [[3.0, 3.0], [9.0, 9.0]]])
    valid = np.array([[True, True], [True, False]])
    t, mean, worst = readiness(np.zeros(2), targets, valid, np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert t[0] == pytest.approx(2 * math.sqrt(0.05)) and t[1] == pytest.approx(3.5)
    assert mean == pytest.approx((t[0] + t[1]) / 2) and worst == pytest.approx(3.5)


def _study(arm, **scene_kw):
    base = {"origin": (0.0, 0.0, 0.0), "region_across": (-0.5, 0.5)}
    base.update(scene_kw)
    return RestStudy(arm, Scene(**base))


def test_settle_frictionless_falls_to_the_minimum_and_stops_hold():
    st = _study(pendulum_arm())
    s = settle(st, [math.radians(30.0)], friction=False)
    assert abs(s.q_end[0]) < math.radians(0.5) and s.moved_deg == pytest.approx(30.0, abs=0.5)
    assert "potential minimum" in s.how
    # a stop at +10 deg: gravity presses the pendulum into it from 30 deg
    stopped = _study(pendulum_arm(lower=math.radians(10.0), upper=math.pi))
    s = settle(stopped, [math.radians(30.0)], friction=False)
    assert s.q_end[0] == pytest.approx(math.radians(10.0)) and "swing on its stop" in s.how


def test_settle_friction_holds_under_its_torque():
    tau = M * G * L * math.sin(math.radians(30.0))
    st = _study(dataclasses.replace(pendulum_arm(), friction=(tau * 1.05,)))
    s = settle(st, [math.radians(30.0)])
    assert s.moved_deg == pytest.approx(0.0) and "held by friction" in s.how
    weak = _study(dataclasses.replace(pendulum_arm(), friction=(tau * 0.5,)))
    s = settle(weak, [math.radians(30.0)])
    # it slides until gravity's torque drops under the friction: sin(q) = 0.5 sin(30)
    assert s.q_end[0] == pytest.approx(math.asin(0.5 * math.sin(math.radians(30.0))), abs=math.radians(0.6))


def test_settle_comes_to_rest_on_a_support_it_may_touch():
    # the plane 0.05 m under the hanging box's bottom; from 60 deg the swing meets it before the bottom
    floor = H - L - S / 2 - 0.05
    st = _study(pendulum_arm(), plane_point=(0.0, 0.0, floor + 0.07))  # a plane the hanging hand would cut through
    s = settle(st, [math.radians(60.0)], friction=False, supports=("plane",))
    assert s.on_support == "plane" and "rests on the plane" in s.how
    assert 0.0 < s.q_end[0] < math.radians(60.0)
    s_free = settle(st, [math.radians(60.0)], friction=False)
    assert abs(s_free.q_end[0]) < math.radians(0.5)  # without the support it swings to the bottom


def test_limp_goal_refuses_an_unstable_balance():
    # standing straight up on friction that holds it there, but not after a 3 deg kick
    up = math.pi - 1e-3
    tau3 = M * G * L * math.sin(math.radians(3.0))
    st = _study(dataclasses.replace(pendulum_arm(lower=-math.pi, upper=math.pi), friction=(tau3 * 0.5,)))
    goal = PoseGoal("limp", {"plane": None}, "limp", weights={"compactness": 1.0}, settle_deg=3.0, kick_deg=3.0)
    from robot_reach.goals import _Evaluator

    ev = _Evaluator(st, goal, None, None, np.ones(1), np.ones(1), None)
    c = ev(np.array([up]))
    assert not c.passed and any("kick" in f for f in c.failed)
    hanging = ev(np.array([0.0]))
    assert hanging.passed and hanging.objectives["energy"] == 0.0


def test_occlusion_of_a_plate_between_camera_and_region():
    region = np.array([(x, y, 0.0) for x in np.linspace(-0.1, 0.1, 11) for y in np.linspace(-0.1, 0.1, 11)])
    cam = Camera((0.0, 0.0, 1.0), resolution_deg=0.2, margin_bins=0)
    plate = np.array([(x, y, 0.5) for x in np.linspace(-0.2, 0.0, 120) for y in np.linspace(-0.2, 0.2, 240)])
    # the plate covers x <= 0 of the region as seen from above: half the points (x = 0 sits on its edge)
    assert occlusion(cam, region, plate) == pytest.approx(66 / 121, abs=0.05)
    assert occlusion(cam, region, np.zeros((0, 3))) == 0.0
    below = plate - np.array([0.0, 0.0, 0.6])  # under the region: it hides nothing
    assert occlusion(cam, region, below) == 0.0


def test_goal_validation():
    with pytest.raises(ValueError):
        PoseGoal("x", torque="floppy")
    with pytest.raises(ValueError):
        PoseGoal("x", weights={"readiness": 1.0})  # readiness needs a workspace
    with pytest.raises(ValueError):
        PoseGoal("x", weights={"beauty": 1.0})


def test_four_goals_on_the_template(tmp_path):
    from robot_reach.examples import goals_example

    st = goals_example()
    seeds = [p.q for p in st.poses]
    sols = {g.name: solve(st, g, samples=300 if g.torque == "held" else 60, seeds=seeds) for g in st.goals}
    for name, sol in sols.items():
        assert sol.pick is not None and sol.pick.passed, (name, sol.gate_failures)
        rep = sol.pick.report
        need = sol.goal.clearances["belt"]
        assert rep.clearance.plane is None or rep.clearance.plane >= need - 1e-9
        assert sol.record["description"].startswith(sol.goal.description) and "Settings" in sol.record["why"]
    sleep = sols["sleep_limp"].pick
    assert sleep.objectives["energy"] == 0.0 and sleep.settle.moved_deg <= 3.0
    ready, hover = sols["ready_to_pick"].pick, sols["hover_empty_belt"].pick
    assert ready.readiness_mean <= hover.readiness_mean  # readiness is what ready_to_pick is weighted on
    assert sols["standby"].pick.report.footprint.past_shoulder <= 0.0  # the lane gate
    # the Pareto set: nobody in it is beaten on every weighted objective by another member
    for sol in sols.values():
        keys = [k for k, w in sol.goal.weights.items() if w]
        for a in sol.pareto:
            assert not any(all(b.objectives[k] <= a.objectives[k] for k in keys) and any(b.objectives[k] < a.objectives[k] for k in keys)
                           for b in sol.pareto if b is not a)
    # NamedPose-ready, and the saved solutions round-trip
    pose = to_named_pose(sols["ready_to_pick"].record)
    assert pose.name == "ready_to_pick" and len(pose.digest()) == 16 and pose.description.startswith("Ready to pick")
    path = tmp_path / "solutions.json"
    save_solutions(st, sols, path, {k: (v.workspace_q, v.workspace_valid) for k, v in sols.items() if v.workspace_q is not None})
    back = load_solutions(st, path)
    assert set(back) == set(sols) and back["ready_to_pick"]["pick"]["q_rad"] == sols["ready_to_pick"].pick.to_dict(st.arm)["q_rad"]
    changed = dataclasses.replace(st, goals=(dataclasses.replace(st.goals[0], clearances={"belt": 0.05}), *st.goals[1:]))
    assert st.goals[0].name not in load_solutions(changed, path)  # a changed setting means re-solve
    json.loads(path.read_text())


def test_viewer_goal_picker(tmp_path):
    pytest.importorskip("viser")
    from robot_reach.examples import goals_example
    from robot_reach.goals import solve_all
    from robot_reach.rest_viewer import RestViewer

    st = goals_example()
    st = dataclasses.replace(st, goals=st.goals[3:])  # ready_to_pick only: quick
    sols = solve_all(st, samples={"ready_to_pick": 200}, path=tmp_path / "s.json")
    v = RestViewer(st, solutions=load_solutions(st, tmp_path / "s.json"))
    try:
        v.wait_idle()
        v.select_goal("ready_to_pick")
        v.wait_idle()
        assert np.allclose(v.report.q, sols["ready_to_pick"].pick.q)
        assert "Readiness" in v.g_head.content and "pick" in v.g_goal_pick.value
        assert "Clearances" in v.g_goal_md.content
        if len(v.g_goal_pick.options) > 1:
            v.select_goal("ready_to_pick", 1)
            v.wait_idle()
            assert np.allclose(v.report.q, sols["ready_to_pick"].pareto[1].q)
    finally:
        v.stop()
