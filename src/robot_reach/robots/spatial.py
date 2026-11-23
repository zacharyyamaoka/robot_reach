"""Spatial example arms: a 3-DOF positioning shoulder and a 6-DOF UR5e."""

from __future__ import annotations

import numpy as np

from ..arm import Arm
from ..codes import Code
from ..configurations import elbow_configurations, ur_configurations
from .planar import _rr_ik

D = np.deg2rad


# --------------------------------------------------------------------------- 3 DOF
def shoulder3(
    height: float = 0.10,
    l1: float = 0.30,
    l2: float = 0.25,
    lower_deg=(-170.0, -20.0, -150.0),
    upper_deg=(170.0, 200.0, 150.0),
    masses=(0.4, 0.3, 0.3, 0.5),
    torque_limits=(5.0, 4.0, 1.5),
) -> Arm:
    """Yaw + shoulder pitch + elbow: the positioning part of an arm ("just the shoulder").

    Up to four IK branches: reach forward or over the back (yaw, yaw+180), times elbow up/down.
    Pitch is measured from horizontal (raising positive), so gravity loads the pitch joints."""

    def points(q):
        y, a, b = q[:, 0], q[:, 1], q[:, 2]
        r_e, z_e = l1 * np.cos(a), height + l1 * np.sin(a)
        r_t, z_t = r_e + l2 * np.cos(a + b), z_e + l2 * np.sin(a + b)
        cy, sy = np.cos(y), np.sin(y)
        n = len(q)
        base = np.zeros((n, 3))
        sh = np.stack([np.zeros(n), np.zeros(n), np.full(n, height)], -1)
        el = np.stack([r_e * cy, r_e * sy, z_e], -1)
        tp = np.stack([r_t * cy, r_t * sy, z_t], -1)
        return base, sh, el, tp

    def fk(q):
        _, _, _, tp = points(q)
        y, pitch = q[:, 0], q[:, 1] + q[:, 2]
        cy, sy, cp, sp = np.cos(y), np.sin(y), np.cos(pitch), np.sin(pitch)
        T = np.zeros((len(q), 4, 4))
        # tool +X along the forearm, +Y horizontal, +Z completes the frame (not scored: position task)
        T[:, :3, 0] = np.stack([cp * cy, cp * sy, sp], -1)
        T[:, :3, 1] = np.stack([-sy, cy, np.zeros_like(y)], -1)
        T[:, :3, 2] = np.cross(T[:, :3, 0], T[:, :3, 1])
        T[:, :3, 3] = tp
        T[:, 3, 3] = 1.0
        return T

    def ik(T):
        x, y, z = T[:, 0, 3], T[:, 1, 3], T[:, 2, 3]
        n = len(T)
        rho = np.hypot(x, y)
        yaw = np.arctan2(y, x)
        back = (yaw + 2 * np.pi) % (2 * np.pi) - np.pi  # yaw + 180 deg, wrapped
        q = np.zeros((n, 4, 3))
        code = np.full((n, 4), int(Code.NO_SOLUTION), np.int8)
        # forward reach (yaw, +rho) and over-the-back reach (yaw + 180, -rho): elbow up/down each.
        # A pitch past 180 deg comes back wrapped negative; the library's +-2pi wrap restores it.
        for k0, (yaw_k, r_k) in enumerate(((yaw, rho), (back, -rho))):
            sols, ok = _rr_ik(r_k, z - height, l1, l2)
            for b in range(2):
                k = 2 * k0 + b
                q[:, k, 0] = yaw_k
                q[:, k, 1:] = sols[:, b]
                code[ok, k] = Code.OK
        return q, code

    def skeleton(q):
        return np.stack(points(q), axis=1)

    def mass_points(q):
        _, sh, el, tp = points(q)
        return np.stack([(sh + el) / 2, el, (el + tp) / 2, tp], axis=1)

    names = ("yaw", "shoulder", "elbow")
    return Arm(
        name="shoulder3",
        joint_names=names,
        lower=tuple(D(lower_deg)),
        upper=tuple(D(upper_deg)),
        fk=fk,
        ik=ik,
        task="position",
        skeleton=skeleton,
        mass_points=mass_points,
        masses=tuple(masses),
        torque_limits=tuple(torque_limits),
        configurations=elbow_configurations(names, skeleton, shoulder=1, elbow=2, tip=3, yaw=0),
        description=f"3-DOF shoulder: yaw + pitch + elbow, L1={l1} m, L2={l2} m",
        meta={"home": [0.0, np.pi / 4, -np.pi / 2], "factory": ["robot_reach.robots", "shoulder3", {"height": height, "l1": l1, "l2": l2, "lower_deg": list(lower_deg), "upper_deg": list(upper_deg), "masses": list(masses), "torque_limits": list(torque_limits)}]},
    )


# --------------------------------------------------------------------------- 6 DOF
UR5E_DH = {"d1": 0.1625, "a2": -0.425, "a3": -0.3922, "d4": 0.1333, "d5": 0.0997, "d6": 0.0996}
_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])


def _dh(theta, d, a, alpha):
    """Standard DH, vectorised over theta: (N,) -> (N, 4, 4)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.zeros(np.shape(theta) + (4, 4))
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 2], T[..., 0, 3] = ct, -st * ca, st * sa, a * ct
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2], T[..., 1, 3] = st, ct * ca, -ct * sa, a * st
    T[..., 2, 1], T[..., 2, 2], T[..., 2, 3] = sa, ca, d
    T[..., 3, 3] = 1.0
    return T


def _inv(T):
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = R
    out[..., :3, 3] = -(R @ T[..., :3, 3, None])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def ur5e(dh: dict | None = None, elbow_limit_deg: float = 180.0) -> Arm:
    """UR5e with its closed-form IK: all eight IK solutions (shoulder front/back x elbow up/down x wrist
    flip), named by `configurations` from each posture.

    Limits: +-360 deg on every joint but the elbow (+-180 deg), as the real arm. The +-2pi twins
    of a solution are the same physical posture and are not enumerated. No collision model and no
    floor: the arm is in free space (Zach, 2026-11-23: "you never know how it will actually be mounted")."""
    p = dict(UR5E_DH if dh is None else dh)
    d = np.array([p["d1"], 0.0, 0.0, p["d4"], p["d5"], p["d6"]])
    a = np.array([0.0, p["a2"], p["a3"], 0.0, 0.0, 0.0])

    def frames(q):
        T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        out = [T]
        for i in range(6):
            T = T @ _dh(q[:, i], d[i], a[i], _ALPHA[i])
            out.append(T)
        return np.stack(out, axis=1)  # (N, 7, 4, 4)

    def fk(q):
        return frames(q)[:, -1]

    def ik(T06):
        n = len(T06)
        a2, a3, d4, d6 = p["a2"], p["a3"], p["d4"], p["d6"]
        q = np.zeros((n, 8, 6))
        ok = np.ones((n, 8), bool)
        P05 = (T06 @ np.array([0.0, 0.0, -d6, 1.0]))[:, :3]
        rxy = np.hypot(P05[:, 0], P05[:, 1])
        arg1 = d4 / np.where(rxy > 1e-12, rxy, 1e-12)
        ok &= (np.abs(arg1) <= 1)[:, None]
        phi = np.arccos(np.clip(arg1, -1, 1))
        psi = np.arctan2(P05[:, 1], P05[:, 0])
        for i1, s1 in enumerate((+1, -1)):
            t1 = psi + s1 * phi + np.pi / 2
            c1, sn1 = np.cos(t1), np.sin(t1)
            arg5 = (T06[:, 0, 3] * sn1 - T06[:, 1, 3] * c1 - d4) / d6
            ok5 = np.abs(arg5) <= 1 + 1e-12
            for i5, s5 in enumerate((+1, -1)):
                t5 = s5 * np.arccos(np.clip(arg5, -1, 1))
                st5 = np.sin(t5)
                safe = np.where(np.abs(st5) > 1e-9, st5, 1.0)
                # WHY X60/Y60 = rows of R06: Hawkins' formula uses frame 0's axes seen from frame 6.
                t6 = np.arctan2(
                    (-T06[:, 0, 1] * sn1 + T06[:, 1, 1] * c1) / safe,
                    (T06[:, 0, 0] * sn1 - T06[:, 1, 0] * c1) / safe,
                )
                t6 = np.where(np.abs(st5) > 1e-9, t6, 0.0)  # wrist singular: any q6 works, pick 0
                T01 = _dh(t1, d[0], a[0], _ALPHA[0])
                T45 = _dh(t5, d[4], a[4], _ALPHA[4])
                T56 = _dh(t6, d[5], a[5], _ALPHA[5])
                T14 = _inv(T01) @ T06 @ _inv(T45 @ T56)
                P14 = T14[:, :3, 3]
                # WHY x-y: with standard DH, joints 2-4 turn about z1, so the shoulder-elbow
                # 2R lies in frame 1's x-y plane (Hawkins writes "xz" for his own frame choice).
                rxy14 = np.hypot(P14[:, 0], P14[:, 1])
                arg3 = (rxy14**2 - a2**2 - a3**2) / (2 * a2 * a3)
                ok3 = np.abs(arg3) <= 1 + 1e-12
                for i3, s3 in enumerate((+1, -1)):
                    k = i1 * 4 + i5 * 2 + i3
                    t3 = s3 * np.arccos(np.clip(arg3, -1, 1))
                    t2 = np.arctan2(P14[:, 1], P14[:, 0]) - np.arctan2(a3 * np.sin(t3), a2 + a3 * np.cos(t3))
                    T12 = _dh(t2, d[1], a[1], _ALPHA[1])
                    T23 = _dh(t3, d[2], a[2], _ALPHA[2])
                    T34 = _inv(T12 @ T23) @ T14
                    t4 = np.arctan2(T34[:, 1, 0], T34[:, 0, 0])
                    q[:, k] = np.stack([t1, t2, t3, t4, t5, t6], -1)
                    ok[:, k] &= ok5 & ok3
        q = (q + np.pi) % (2 * np.pi) - np.pi
        code = np.where(ok, Code.OK, Code.NO_SOLUTION).astype(np.int8)
        return q, code

    def skeleton(q):
        F = frames(q)
        pts = F[:, :, :3, 3]
        # WHY the extra elbow points: DH origins skip the UR's lateral shoulder/elbow offsets, so a
        # skeleton through origins alone looks folded wrong; walk each link's z-offset explicitly.
        sh_off = pts[:, 1] + F[:, 1, :3, 2] * 0.138
        el_off = pts[:, 2] + F[:, 2, :3, 2] * 0.131
        return np.stack([pts[:, 0], pts[:, 1], sh_off, el_off, pts[:, 2], pts[:, 3], pts[:, 4], pts[:, 5], pts[:, 6]], axis=1)

    def joint_axes(q):
        F = frames(np.atleast_2d(q))[:, :6]  # joint k turns about frame k's z, through its origin
        return F[..., :3, 2], F[..., :3, 3]

    lim = np.deg2rad(360.0)
    el = np.deg2rad(elbow_limit_deg)
    names = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
    return Arm(
        name="ur5e",
        joint_names=names,
        lower=(-lim, -lim, -el, -lim, -lim, -lim),
        upper=(lim, lim, el, lim, lim, lim),
        fk=fk,
        ik=ik,
        task="pose",
        skeleton=skeleton,
        configurations=ur_configurations(names, joint_axes),
        description="UR5e, closed-form IK (8 IK solutions), DH from the UR datasheet",
        meta={
            "factory": ["robot_reach.robots", "ur5e", {"dh": p, "elbow_limit_deg": elbow_limit_deg}],
            "frames": frames,
            "joint_axes": joint_axes,
            "home": [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0],
        },
    )
