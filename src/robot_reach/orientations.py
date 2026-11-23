"""One shared orientation set, from which every workspace kind is a mask.

WHY one set crossed with every position (kept from bam_reach v1, FR "One orientation set is shared
across every position" and "One dexterous reach map derives workspace masks"): every point gets the
same orientation challenge, so scores compare across points, and the four workspace kinds come from
one IK run instead of four.

Directions are rings of constant angle from the approach axis (0 deg = straight along it), each
crossed with evenly spaced rolls about the tool axis. The ring at 0 deg is exactly the approach, so
"4-DOF" (approach fixed, yaw free) is an exact subset, never a nearest-neighbour guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _frame_from_z(z: np.ndarray) -> np.ndarray:
    """(N,3) unit z -> (N,3,3) right-handed frames with that +Z (x chosen deterministically)."""
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    helper = np.where(np.abs(z[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    x = helper - np.sum(helper * z, axis=-1, keepdims=True) * z
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def _rot_z(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    R = np.zeros(np.shape(angle) + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1.0
    return R


@dataclass(frozen=True, eq=False)
class OrientationSet:
    R: np.ndarray  # (O, 3, 3) base->tool rotations; column 2 is the approach axis
    tilt: np.ndarray  # (O,) angle between the tool +Z and the approach axis, rad
    roll: np.ndarray  # (O,) roll about the tool axis, rad
    approach: np.ndarray  # (3,) the reference approach direction (default straight down)

    def __len__(self) -> int:
        return len(self.R)

    @property
    def four_dof(self) -> np.ndarray:
        """Approach fixed, every roll: the top-down picking set."""
        return self.tilt < 1e-6

    @property
    def hemisphere(self) -> np.ndarray:
        """Every approach within 90 deg of the reference, every roll."""
        return self.tilt <= np.pi / 2 + 1e-6

    @property
    def dexterous(self) -> np.ndarray:
        return np.ones(len(self.R), bool)

    @classmethod
    def sample(
        cls,
        approach=(0.0, 0.0, -1.0),
        tilt_step_deg: float = 45.0,
        n_rolls: int = 6,
        max_tilt_deg: float = 180.0,
    ) -> OrientationSet:
        """Rings every `tilt_step_deg` from the approach out to `max_tilt_deg`, `n_rolls` rolls each.

        Defaults give 22 directions x 6 rolls = 132 orientations (4-DOF: 6, hemisphere: 90)."""
        a = np.asarray(approach, float)
        a = a / np.linalg.norm(a)
        base = _frame_from_z(a[None])[0]  # tool frame whose +Z is the approach
        tilts = np.deg2rad(np.arange(0.0, max_tilt_deg + 1e-9, tilt_step_deg))
        dirs, tilt_of = [], []
        for t in tilts:
            # WHY ~ sin(t) azimuths: roughly even spacing over the sphere; poles get exactly one.
            n_az = 1 if np.sin(t) < 1e-9 else max(3, int(round(2 * np.pi * np.sin(t) / np.deg2rad(tilt_step_deg))))
            for k in range(n_az):
                az = 2 * np.pi * k / n_az
                local = np.array([np.sin(t) * np.cos(az), np.sin(t) * np.sin(az), np.cos(t)])
                dirs.append(base @ local)
                tilt_of.append(t)
        dirs = np.asarray(dirs)
        frames = _frame_from_z(dirs)
        # The 0-tilt frame must be exactly `base`, so the 4-DOF rolls are rolls of one known frame.
        frames[0] = base
        rolls = 2 * np.pi * np.arange(n_rolls) / n_rolls
        R = (frames[:, None] @ _rot_z(rolls)[None]).reshape(-1, 3, 3)
        return cls(
            R=R,
            tilt=np.repeat(np.asarray(tilt_of), n_rolls),
            roll=np.tile(rolls, len(frames)),
            approach=a,
        )

    def resolution(self) -> dict:
        """The angular resolution of this set, read back from the set itself (so it is true for a
        loaded map too): tilt rings, roll steps and counts, in degrees.

        WHY (Zach, 2026-11-23: "its important though to basically set what you angular resolution is
        for this ... rotation range, and tilt range ... and then you set the step angle"): every
        "N / M orientations" in the viewer needs its denominator explained."""
        tilts = np.unique(np.round(np.rad2deg(self.tilt), 6))
        rolls = np.unique(np.round(np.rad2deg(self.roll), 6))
        n_rolls = len(rolls)
        return {
            "approach": tuple(float(v) for v in self.approach),
            "tilt_max_deg": float(tilts.max()) if len(tilts) else 0.0,
            "tilt_step_deg": float(np.min(np.diff(tilts))) if len(tilts) > 1 else 0.0,
            "n_tilt_rings": int(len(tilts)),
            "n_directions": int(len(self) // max(n_rolls, 1)),
            "roll_step_deg": float(360.0 / n_rolls) if n_rolls > 1 else 0.0,
            "roll_max_deg": float(rolls.max()) if len(rolls) else 0.0,
            "n_rolls": int(n_rolls),
            "n_orientations": int(len(self)),
        }

    def describe(self) -> str:
        """One line: ``approach -z: tilt 0…180° in 45° rings (22 directions) × roll 0…300° every 60°
        (6 rolls) = 132 orientations``."""
        r = self.resolution()
        a = np.asarray(r["approach"])
        axis = next((f"{'+-'[int(a[i] < 0)]}{'xyz'[i]}" for i in range(3) if abs(abs(a[i]) - 1) < 1e-9), f"({a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f})")
        tilt = (f"tilt 0…{r['tilt_max_deg']:.0f}° from it in {r['tilt_step_deg']:.0f}° rings" if r["n_tilt_rings"] > 1 else "tilt 0° only")
        roll = (f"roll 0…{r['roll_max_deg']:.0f}° every {r['roll_step_deg']:.0f}°" if r["n_rolls"] > 1 else "one roll")
        return (f"approach {axis}: {tilt} ({r['n_directions']} directions) × {roll} ({r['n_rolls']} rolls) "
                f"= {r['n_orientations']} orientations")

    @classmethod
    def identity(cls) -> OrientationSet:
        """Position-only arms: one placeholder orientation that is never scored."""
        return cls(R=np.eye(3)[None], tilt=np.zeros(1), roll=np.zeros(1), approach=np.array([0.0, 0.0, -1.0]))
