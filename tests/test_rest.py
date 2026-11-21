"""robot_reach.rest against hand-derived answers: a pendulum with a box for a hand over a (tilted) belt."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from robot_reach import Arm, ArmMesh
from robot_reach.rest import (
    Criterion,
    Gate,
    KeepOut,
    Motor,
    RestStudy,
    Scene,
    _hull_points,
    _surface_labels,
    candidate_poses,
    convex_hull,
    named_poses,
    polygon_area,
    rank,
)

H, L, S, M = 0.60, 0.30, 0.04, 0.5  # pivot height, link length, box side, tip mass
G = 9.81


def _box(size) -> np.ndarray:
    hx, hy, hz = np.asarray(size, float) / 2
    return np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)])


def pendulum_arm(lower=-math.pi, upper=math.pi) -> Arm:
    """One joint about +y at (0, 0, H); q = 0 hangs straight down. A shoulder box at the pivot, a hand
    box (twice as long along y as the shoulder) at the tip, a point mass at the tip."""

    def tip(q):
        a = q[:, 0]
        return np.stack([L * np.sin(a), np.zeros_like(a), H - L * np.cos(a)], -1)

    def fk(q):
        T = np.tile(np.eye(4), (len(q), 1, 1))
        a = q[:, 0]
        c, s = np.cos(a), np.sin(a)
        T[:, 0, 0], T[:, 0, 2], T[:, 2, 0], T[:, 2, 2] = c, s, -s, c
        T[:, :3, 3] = tip(q)
        return T

    def mesh_poses(q):
        T = fk(q)
        P = np.tile(np.eye(4), (len(q), 2, 1, 1))
        P[:, 0, :3, 3] = (0.0, 0.0, H)
        P[:, 1] = T
        return P

    return Arm(
        name="pendulum_box", joint_names=("swing",), lower=(lower,), upper=(upper,), fk=fk, task="position",
        skeleton=lambda q: np.stack([np.tile((0.0, 0.0, H), (len(q), 1)), tip(q)], 1),
        mass_points=lambda q: tip(q)[:, None, :], masses=(M,), torque_limits=(2.0,),
        meshes=(ArmMesh("shoulder", _box((S, S, S))), ArmMesh("hand", _box((S, 2 * S, S)))), mesh_poses=mesh_poses,
    )


def belt(**kw) -> Scene:
    base = {"origin": (0.0, 0.0, 0.0), "region_across": (-0.5, 0.5), "min_clearance": 0.1}
    base.update(kw)
    return Scene(**base)


def test_footprint_and_clearance_hanging_straight_down():
    study = RestStudy(pendulum_arm(), belt(), hand=("hand",), shoulder=("shoulder",))
    r = study.measure([0.0])
    assert r.ok and r.code_label == "OK"
    f, c = r.footprint, r.clearance
    assert f.hand_across == pytest.approx((-S / 2, S / 2)) and f.hand_along == pytest.approx((-S, S))
    assert f.reach == pytest.approx(S)  # the hand box is S long either way along the line
    # the shoulder box is S/2 either side along: the hand sticks S/2 past it toward the neighbour
    assert f.lane["along"] == pytest.approx((-S / 2, S / 2))
    assert f.past_shoulder == pytest.approx(S / 2) and f.past_axis == "along"
    # both boxes stack straight down: the projected hull is the hand's S x 2S rectangle
    assert f.hull_area == pytest.approx(S * 2 * S)
    assert c.plane == pytest.approx(H - L - S / 2)
    assert c.ok


def test_clearance_is_measured_along_a_tilted_plane_normal():
    theta = math.radians(5.0)
    scene = belt(plane_point=(0.0, 0.0, 0.0), plane_normal=(math.sin(theta), 0.0, math.cos(theta)))
    r = RestStudy(pendulum_arm(), scene).measure([0.0])
    # the hand's bottom corners at (+-S/2, +-S, H-L-S/2): the lowest along the normal is the -x one
    assert r.clearance.plane == pytest.approx((H - L - S / 2) * math.cos(theta) - (S / 2) * math.sin(theta))


def test_region_and_keepouts():
    q = [math.pi / 2]  # swung out along +x: the hand at (L, 0, H)
    scene = belt(region_across=(0.5, 0.9), keepouts=(KeepOut("rail", top=H - 0.1, across=(L - S, L + S)), KeepOut("far", top=0.0, across=(2, 3))))
    r = RestStudy(pendulum_arm(), scene).measure(q)
    assert r.clearance.plane is None  # nothing of the arm is over the region
    assert r.clearance.keepouts["rail"] == pytest.approx(0.1 - S / 2)
    assert r.clearance.keepouts["far"] is None
    m = r.metrics()
    assert m["keepout_rail_m"] == pytest.approx(0.1 - S / 2) and m["clearance_m"] == math.inf  # nothing over the region: nothing to hit


def test_holding_power_virtual_work_and_copper():
    motor = Motor(kt=0.05, resistance=0.4, ratio=10.0)
    study = RestStudy(pendulum_arm(), belt(), motors=(motor,))
    q = math.radians(30.0)
    r = study.measure([q])
    tau = M * G * L * math.sin(q)  # dU/dq, U = m g (H - L cos q)
    assert r.holding.tau[0] == pytest.approx(tau, rel=1e-4)
    i = tau / (0.05 * 10.0)
    assert r.holding.current[0] == pytest.approx(i, rel=1e-4)
    assert r.holding.total == pytest.approx(1.5 * 0.4 * i * i, rel=1e-4)
    assert r.holding.limit_frac[0] == pytest.approx(tau / 2.0, rel=1e-4)
    # a measured model wins: torque(q) and power(q, tau)
    measured = RestStudy(pendulum_arm(), belt(), motors=(motor,), torque=lambda q: np.array([1.0]), power=lambda q, t: 7.0 * t)
    h = measured.measure([q]).holding
    assert h.tau[0] == 1.0 and h.total == pytest.approx(7.0)
    assert Motor.from_kv(100.0, 0.1).kt == pytest.approx(1.5 / math.sqrt(3) * 60 / (2 * math.pi) / 100.0)


def test_a_pose_that_cannot_close_reports_no_geometry():
    arm = pendulum_arm()
    import dataclasses

    broken = dataclasses.replace(arm, fk=lambda q: (arm.fk(q), np.zeros(len(q), bool)))
    r = RestStudy(broken, belt()).measure([0.0])
    assert not r.ok and r.footprint is None and r.holding.total is None and r.code_label == "linkage cannot close"


def test_rank_gates_then_weighted_anchored_score():
    study = RestStudy(pendulum_arm(), belt(), motors=(Motor(kt=0.05, resistance=0.4, ratio=10.0),), hand=("hand",))
    reports = {name: study.measure([math.radians(d)]) for name, d in (("down", 0.0), ("tilted", 40.0), ("out", 90.0))}
    gates = (Gate("clears the belt", "clearance_m", min=0.30),)
    crit = (Criterion("power", "power_w", 2.0, best=0.0, worst=0.1), Criterion("reach", "reach_m", 1.0))
    ranked = rank(reports, gates, crit)
    assert [r.name for r in ranked][-1] == "down"  # hanging: H - L - S/2 = 0.28 m over the belt, under the 0.30 gate
    assert not ranked[-1].passed and ranked[-1].failed == ("clears the belt",)
    tilted = next(r for r in ranked if r.name == "tilted")
    p = reports["tilted"].holding.total
    assert tilted.scores["power"] == pytest.approx(np.clip((p - 0.1) / (0.0 - 0.1), 0, 1))
    # reach without anchors: the best (smallest) of the three scores 1, the worst 0
    reach = {k: v.footprint.reach for k, v in reports.items()}
    assert next(r for r in ranked if r.name == min(reach, key=reach.get)).scores["reach"] == 1.0


def test_named_poses_and_candidates(tmp_path):
    from robot_msgs.manipulation import NamedPose, named_poses_to_json

    arm = pendulum_arm(lower=-0.5, upper=5.0)
    poses = [NamedPose(name="ready", group="arm", joint_names=("swing",), positions=[-2.0], description="waits here"),
             NamedPose(name="other", group="arm", joint_names=("elbow",), positions=[0.1])]
    path = tmp_path / "arm.named_poses.json"
    path.write_text(named_poses_to_json(poses, "pendulum_box"))
    got = named_poses(path, arm)
    assert [p.name for p in got] == ["ready"]  # "other" names a joint the arm lacks
    assert got[0].q[0] == pytest.approx(-2.0 + 2 * math.pi)  # wrapped into the limits
    assert got[0].description == "waits here" and got[0].digest == poses[0].digest()
    rows = {"v1": {"title": "hang", "joint_deg": {"J": 10.0}}, "v2": {"q_rad": {"swing": 0.5}}, "skip": {"joint_deg": {"X": 1}}}
    (tmp_path / "results.json").write_text(json.dumps(rows))
    cands = candidate_poses(tmp_path / "results.json", arm, aliases={"J": "swing"})
    assert [(c.name, round(float(c.q[0]), 6)) for c in cands] == [("v1", round(math.radians(10.0), 6)), ("v2", 0.5)]
    assert cands[0].label == "candidate · v1" and cands[0].description == "hang"


def test_hull_prefilter_is_exact_and_labels_match_the_surface():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(20_000, 2)) * (1.0, 0.3)
    assert polygon_area(convex_hull(_hull_points(pts))) == pytest.approx(polygon_area(convex_hull(pts)), rel=1e-12)
    arm = pendulum_arm()
    for cap in (None, 3):
        assert len(_surface_labels(arm, cap)) == arm.surface(np.zeros((1, 1)), max_per_mesh=cap).shape[1]


# --------------------------------------------------------------------------- the template, the viewer and the cards
def test_template_ranks_and_exports(tmp_path):
    pytest.importorskip("matplotlib")
    from robot_reach.examples import rest_example
    from robot_reach.rest import reports_for
    from robot_reach.rest_plots import export_cards

    study = rest_example()
    ranked = rank(reports_for(study), study.gates, study.criteria)
    by = {r.name.split(" · ")[-1]: r for r in ranked}
    assert by["turned along"].failed == ("inside its lane",)  # yawed 90 deg: the hand sits in the next arm's lane
    assert by["upright"].passed and by["upright"].report.holding.total == pytest.approx(0.0, abs=1e-6)
    assert by["upright"].report.metrics()["clearance_m"] == math.inf  # nothing over the belt: nothing to hit
    paths = export_cards(study, tmp_path)
    assert len(paths) == 5 and all(p.stat().st_size > 20_000 for p in paths)
    index = json.loads((tmp_path / "index.json").read_text())
    assert [r["pose"] for r in index["ranking"]] == [r.name for r in ranked]
    card = json.loads((tmp_path / "home.json").read_text())
    assert card["metrics"]["power_w"] == pytest.approx(by["home"].report.holding.total)


def test_viewer_pose_dropdown_and_jog():
    pytest.importorskip("viser")
    from robot_reach.examples import rest_example
    from robot_reach.rest_viewer import RestViewer

    v = RestViewer(rest_example())
    try:
        v.wait_idle()
        v.select("folded back")
        v.wait_idle()
        assert v.pose_name == "candidate · folded back"
        assert np.allclose(np.rad2deg(v.report.q), (0.0, 110.0, -150.0), atol=0.05)
        w0 = v.report.holding.total
        v.set_joint("shoulder", 60.0)
        v.wait_idle()
        assert v.report.q[1] == pytest.approx(math.radians(60.0), abs=1e-3)
        assert v.report.holding.total != pytest.approx(w0)
        assert "Holding" in v.g_head.content and "jogged from" in v.g_jog_md.content
    finally:
        v.stop()
