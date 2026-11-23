"""Planar example arms in the vertical x-z plane: a pendulum, an RR elbow, and a five-bar.

Convention: joint angles are measured from +x toward +z (q = 0 is horizontal, positive raises),
so gravity (-z) loads them the way a real shoulder is loaded. The tool frame's +X runs along the
last link; +Z is the out-of-plane rotation axis turned into the approach direction is not used —
these arms are position-only.
"""

from __future__ import annotations

import numpy as np

from ..arm import Arm
from ..codes import Code
from ..configurations import elbow_configurations

D = np.deg2rad
PLANE_XZ = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def _planar_T(x: np.ndarray, z: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Pose in the x-z plane: +X along `theta` (from +x toward +z), +Y out of plane = world +Y."""
    n = np.shape(x)[0]
    c, s = np.cos(theta), np.sin(theta)
    T = np.zeros((n, 4, 4))
    T[:, 0, 0], T[:, 0, 2] = c, -s
    T[:, 1, 1] = 1.0
    T[:, 2, 0], T[:, 2, 2] = s, c
    T[:, 0, 3], T[:, 2, 3] = x, z
    T[:, 3, 3] = 1.0
    return T


def _xz(points_xz: np.ndarray) -> np.ndarray:
    """(..., 2) x-z -> (..., 3) xyz with y = 0."""
    out = np.zeros(points_xz.shape[:-1] + (3,))
    out[..., 0], out[..., 2] = points_xz[..., 0], points_xz[..., 1]
    return out


def _off_plane(T: np.ndarray, tol: float) -> np.ndarray:
    return np.abs(T[:, 1, 3]) > tol


# --------------------------------------------------------------------------- 1 DOF
def pendulum(length: float = 0.30, tip_mass: float = 0.5, torque_limit: float = 1.0) -> Arm:
    """A 1-DOF pendulum: one motor, one link, a payload at the tip.

    Its workspace is a circle, so the interesting picture is the joint line: which angles the motor
    can hold against gravity (m g L cos q <= limit)."""

    def fk(q):
        th = q[:, 0]
        return _planar_T(length * np.cos(th), length * np.sin(th), th)

    def ik(T, tol=1e-4):
        x, z = T[:, 0, 3], T[:, 2, 3]
        q = np.arctan2(z, x)[:, None, None]
        code = np.where((np.abs(np.hypot(x, z) - length) > tol) | _off_plane(T, tol), Code.NO_SOLUTION, Code.OK)
        return q, code[:, None].astype(np.int8)

    def skeleton(q):
        th = q[:, 0]
        tip = np.stack([length * np.cos(th), length * np.sin(th)], -1)
        return _xz(np.stack([np.zeros_like(tip), tip], axis=1))

    def mass_points(q):
        return skeleton(q)[:, 1:2]

    return Arm(
        name="pendulum",
        joint_names=("q1",),
        lower=(-np.pi,),
        upper=(np.pi,),
        fk=fk,
        ik=ik,
        task="position",
        skeleton=skeleton,
        mass_points=mass_points,
        masses=(tip_mass,),
        torque_limits=(torque_limit,),
        plane=PLANE_XZ,
        description=f"1-DOF pendulum, L={length} m, {tip_mass} kg at the tip, motor limit {torque_limit} N·m",
        meta={"factory": ["robot_reach.robots", "pendulum", {"length": length, "tip_mass": tip_mass, "torque_limit": torque_limit}]},
    )


# --------------------------------------------------------------------------- 2 DOF serial
def _rr_ik(x, z, L1, L2):
    """Both elbow branches of a planar 2R. Returns q (N, 2, 2), reachable (N,)."""
    r2 = x**2 + z**2
    c2 = (r2 - L1**2 - L2**2) / (2 * L1 * L2)
    ok = np.abs(c2) <= 1 + 1e-12
    c2 = np.clip(c2, -1, 1)
    sols = []
    for sign in (+1, -1):  # elbow down (+), elbow up (-) for a forward-reaching arm
        q2 = sign * np.arccos(c2)
        q1 = np.arctan2(z, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
        sols.append(np.stack([q1, q2], -1))
    return np.stack(sols, axis=1), ok


def rr(
    l1: float = 0.30,
    l2: float = 0.25,
    lower_deg=(-30.0, -150.0),
    upper_deg=(210.0, 150.0),
    masses=(0.4, 0.3, 0.3, 0.5),
    torque_limits=(3.0, 1.5),
) -> Arm:
    """A planar 2R elbow arm (shoulder + elbow) in the vertical plane.

    Point masses: upper-link centre, elbow motor, forearm centre, payload at the tip."""

    def points(q):
        q1, q2 = q[:, 0], q[:, 1]
        e = np.stack([l1 * np.cos(q1), l1 * np.sin(q1)], -1)
        t = e + np.stack([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)], -1)
        return e, t

    def fk(q):
        _, t = points(q)
        return _planar_T(t[:, 0], t[:, 1], q[:, 0] + q[:, 1])

    def ik(T, tol=1e-6):
        q, ok = _rr_ik(T[:, 0, 3], T[:, 2, 3], l1, l2)
        ok &= ~_off_plane(T, tol)
        code = np.where(ok[:, None], Code.OK, Code.NO_SOLUTION).repeat(2, axis=1).astype(np.int8)
        code[~ok, 1] = Code.PAD
        return q, code

    def skeleton(q):
        e, t = points(q)
        return _xz(np.stack([np.zeros_like(e), e, t], axis=1))

    def mass_points(q):
        e, t = points(q)
        return _xz(np.stack([e / 2, e, (e + t) / 2, t], axis=1))

    return Arm(
        name="rr",
        joint_names=("shoulder", "elbow"),
        lower=tuple(D(lower_deg)),
        upper=tuple(D(upper_deg)),
        fk=fk,
        ik=ik,
        task="position",
        skeleton=skeleton,
        mass_points=mass_points,
        masses=tuple(masses),
        torque_limits=tuple(torque_limits),
        plane=PLANE_XZ,
        configurations=elbow_configurations(("shoulder", "elbow"), skeleton, shoulder=0, elbow=1, tip=2),
        description=f"2-DOF RR, L1={l1} m, L2={l2} m, vertical plane, motor limits {torque_limits} N·m",
        meta={"factory": ["robot_reach.robots", "rr", {"l1": l1, "l2": l2, "lower_deg": list(lower_deg), "upper_deg": list(upper_deg), "masses": list(masses), "torque_limits": list(torque_limits)}]},
    )


# --------------------------------------------------------------------------- 2 DOF parallel
def five_bar(
    spacing: float = 0.12,
    proximal: float = 0.20,
    distal: float = 0.24,
    lower_deg=(-60.0, -60.0),
    upper_deg=(240.0, 240.0),
    min_transmission_deg: float = 15.0,
    masses=(0.3, 0.3, 0.5),
    torque_limits=(3.0, 3.0),
) -> Arm:
    """A symmetric planar five-bar: two motors `spacing` apart, two proximal + two distal links,
    the tool where the distal links meet ("working mode up": tool above the elbow line).

    Unlike the RR, its FK can FAIL — where the two distal links cannot meet — which is exactly the
    picture worth seeing in joint space. Both singularity types are flagged with a margin:
    a leg folding straight (proximal and distal collinear) and the distal links aligning
    (transmission angle under `min_transmission_deg`). Illustrative dimensions, not BAM fb_v2."""
    m1 = np.array([-spacing / 2, 0.0])
    m2 = np.array([+spacing / 2, 0.0])
    margin = D(min_transmission_deg)

    def elbows(q):
        e1 = m1 + proximal * np.stack([np.cos(q[:, 0]), np.sin(q[:, 0])], -1)
        e2 = m2 + proximal * np.stack([np.cos(q[:, 1]), np.sin(q[:, 1])], -1)
        return e1, e2

    def tool(q):
        e1, e2 = elbows(q)
        d = e2 - e1
        dist = np.linalg.norm(d, axis=1)
        ok = (dist < 2 * distal) & (dist > 1e-9)
        h = np.sqrt(np.clip(distal**2 - (dist / 2) ** 2, 0, None))
        n = np.stack([-d[:, 1], d[:, 0]], -1) / np.where(dist > 1e-12, dist, 1.0)[:, None]
        p = (e1 + e2) / 2 + h[:, None] * n
        return p, e1, e2, ok

    def fk(q):
        p, e1, e2, ok = tool(q)
        theta = np.arctan2(p[:, 1] - e1[:, 1], p[:, 0] - e1[:, 0])
        T = _planar_T(p[:, 0], p[:, 1], theta)
        T[~ok, :3, 3] = np.nan
        return T, ok

    def ik(T, tol=1e-6):
        p = np.stack([T[:, 0, 3], T[:, 2, 3]], -1)
        n = len(p)
        q = np.zeros((n, 4, 2))
        code = np.full((n, 4), int(Code.NO_SOLUTION), np.int8)
        angs = []
        reach = []
        for m in (m1, m2):
            v = p - m
            r = np.linalg.norm(v, axis=1)
            c = (proximal**2 + r**2 - distal**2) / (2 * proximal * np.where(r > 1e-12, r, 1.0))
            reach.append(np.abs(c) <= 1)
            phi = np.arctan2(v[:, 1], v[:, 0])
            a = np.arccos(np.clip(c, -1, 1))
            angs.append((phi + a, phi - a))
        k = 0
        for i in range(2):
            for j in range(2):
                q[:, k, 0] = angs[0][i]
                q[:, k, 1] = angs[1][j]
                e1, e2 = elbows(q[:, k])
                cross = (e2[:, 0] - e1[:, 0]) * (p[:, 1] - e1[:, 1]) - (e2[:, 1] - e1[:, 1]) * (p[:, 0] - e1[:, 0])
                # WHY not OK when the tool sits on the other side of the elbow line: that branch
                # needs the other assembly mode, which this arm's FK never produces.
                good = reach[0] & reach[1] & (cross > 0) & ~_off_plane(T, tol)
                code[good, k] = Code.OK
                k += 1
        q = (q + np.pi) % (2 * np.pi) - np.pi
        return q, code

    def singular(q):
        p, e1, e2, ok = tool(q)
        out = np.zeros(len(q), bool)
        for m, e in ((m1, e1), (m2, e2)):
            a = e - m
            b = p - e
            ang = np.arctan2(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b, axis=1))
            out |= (ang < margin) | (ang > np.pi - margin)  # a leg near straight / folded flat
        a, b = p - e1, p - e2
        ang = np.arctan2(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b, axis=1))
        out |= ang < margin  # the distal links nearly aligned: transmission lost
        out |= ang > np.pi - margin
        return out & ok

    def skeleton(q):
        p, e1, e2, ok = tool(q)
        p = np.where(ok[:, None], p, np.nan)
        pts = np.stack([np.broadcast_to(m1, p.shape), e1, p, e2, np.broadcast_to(m2, p.shape)], axis=1)
        return _xz(pts)

    def mass_points(q):
        p, e1, e2, _ = tool(q)
        return _xz(np.stack([e1, e2, p], axis=1))

    return Arm(
        name="five_bar",
        joint_names=("motor_left", "motor_right"),
        lower=tuple(D(lower_deg)),
        upper=tuple(D(upper_deg)),
        fk=fk,
        ik=ik,
        task="position",
        skeleton=skeleton,
        mass_points=mass_points,
        masses=tuple(masses),
        torque_limits=tuple(torque_limits),
        plane=PLANE_XZ,
        description=f"2-DOF five-bar, motors {spacing} m apart, proximal {proximal} m, distal {distal} m",
        singular=singular,
        meta={
            "factory": ["robot_reach.robots", "five_bar", {"spacing": spacing, "proximal": proximal, "distal": distal, "lower_deg": list(lower_deg), "upper_deg": list(upper_deg), "min_transmission_deg": min_transmission_deg, "masses": list(masses), "torque_limits": list(torque_limits)}],
        },
    )
