"""Proposal A (prototype, not src/): a robot_msgs-style type that NAMES an arm's IK branches.

Zach, 2026-11-23: "support kinda like a standard Robot MSG that allows you to name the different IK
solutions, and then also provide like the context so you can automatically generate like these
check box ui."

Two pieces:

- `IKBranches` (the message): which yes/no questions split this arm's solutions ("elbow: up or
  down"), each with the joint that flips it, shown as "(J3)". A viewer builds its checkboxes from
  this alone (`ui_spec()`), for any arm.
- a per-solution tag (`(K,) uint8` beside `ReachMap.sol_q`): the branch TYPE of every solution,
  computed from the solution's joints by the arm (`tag_ur5e` below), never from the solver's slot.

WHY tags from the posture, not the solver's slot number (measured on the saved UR5e map, 20,000
random solutions): the closed-form slot's elbow sign means "elbow up" when the shoulder is on one
side and "elbow down" on the other (99.5 % / 0.7 %), so a checkbox wired to the slot would flip
meaning half the time. A numeric IK has no stable slots at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BranchAxis:
    """One question that splits the solutions in two (or more): "elbow: up or down"."""

    name: str  # "elbow"
    options: tuple[str, ...]  # ("up", "down"); option i is tag value i
    joint: str = ""  # the joint whose sign or side decides it, shown as "(J3)"
    rule: str = ""  # how a solution is tagged, in plain words (shown as a tooltip)


@dataclass(frozen=True)
class IKBranches:
    """The named branch types of one arm's IK. Frozen, JSON-able, no functions inside."""

    arm: str
    joint_names: tuple[str, ...]
    axes: tuple[BranchAxis, ...]

    @property
    def n_types(self) -> int:
        return int(np.prod([len(a.options) for a in self.axes]))

    def joint_label(self, joint: str) -> str:
        """"elbow (J3)": Zach's ask, always the chain index beside the URDF name."""
        return f"{joint} (J{self.joint_names.index(joint) + 1})" if joint in self.joint_names else joint

    def type_of(self, tags: np.ndarray) -> np.ndarray:
        """(N, n_axes) option indices -> (N,) type index (mixed radix, first axis most significant)."""
        tags = np.atleast_2d(tags)
        out = np.zeros(len(tags), np.int64)
        for a, axis in enumerate(self.axes):
            out = out * len(axis.options) + tags[:, a]
        return out

    def tags_of(self, type_index: int) -> tuple[int, ...]:
        out = []
        for axis in reversed(self.axes):
            type_index, r = divmod(type_index, len(axis.options))
            out.append(r)
        return tuple(reversed(out))

    def type_name(self, type_index: int) -> str:
        return " · ".join(f"{a.name} {a.options[t]}" for a, t in zip(self.axes, self.tags_of(type_index)))

    def allowed(self, **choices: str | tuple[str, ...]) -> np.ndarray:
        """(n_types,) bool. allowed(elbow="up") or allowed(elbow="up", wrist=("J5+", "J5-"))."""
        ok = np.ones(self.n_types, bool)
        for i in range(self.n_types):
            for axis, t in zip(self.axes, self.tags_of(i)):
                want = choices.get(axis.name)
                if want is not None and axis.options[t] not in ((want,) if isinstance(want, str) else want):
                    ok[i] = False
        return ok

    def ui_spec(self) -> dict:
        """Everything a viewer needs to draw the checkbox UI: no arm-specific code in the viewer."""
        return {
            "arm": self.arm,
            "axes": [
                {"name": a.name, "label": self.joint_label(a.joint).replace(a.joint, a.name) if a.joint else a.name,
                 "options": list(a.options), "rule": a.rule}
                for a in self.axes
            ],
            "types": [{"index": i, "name": self.type_name(i), "tags": list(self.tags_of(i))} for i in range(self.n_types)],
        }


# ------------------------------------------------------------------------------------------ UR5e
UR5E_BRANCHES = IKBranches(
    arm="ur5e",
    joint_names=("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"),
    axes=(
        BranchAxis("shoulder", ("front", "back"), "shoulder_pan",
                   "front: the wrist is in front of the shoulder pan axis; back: the arm reaches over its back"),
        BranchAxis("elbow", ("up", "down"), "elbow", "up: the elbow is above the shoulder-to-wrist line"),
        BranchAxis("wrist", ("J5+", "J5−"), "wrist_2", "the sign of wrist_2 (J5): the wrist flip"),
    ),
)


def tag_ur5e(q: np.ndarray, frames) -> np.ndarray:
    """(N, 6) joints -> (N, 3) tags (shoulder, elbow, wrist), from the posture itself.

    `frames` is the arm's DH frame function (`arm.meta["frames"]`): (N, 6) -> (N, 7, 4, 4)."""
    F = frames(q)
    pts = F[:, :, :3, 3]
    shoulder, elbow, wrist = pts[:, 1], pts[:, 2], pts[:, 3]
    x1 = F[:, 1, :3, 0]
    front = np.sum(wrist[:, :2] * x1[:, :2], axis=1) > 0
    d = wrist - shoulder
    t = np.sum((elbow - shoulder) * d, axis=1) / np.maximum(np.sum(d * d, axis=1), 1e-12)
    up = (elbow - (shoulder + t[:, None] * d))[:, 2] > 0
    j5_pos = q[:, 4] > 0
    return np.stack([np.where(front, 0, 1), np.where(up, 0, 1), np.where(j5_pos, 0, 1)], axis=1).astype(np.uint8)


def tag_all(q: np.ndarray, frames, chunk: int = 100_000) -> np.ndarray:
    return np.concatenate([tag_ur5e(q[i : i + chunk], frames) for i in range(0, len(q), chunk)])


if __name__ == "__main__":
    import json

    print(json.dumps(UR5E_BRANCHES.ui_spec()["axes"], indent=1, ensure_ascii=False))
    print(UR5E_BRANCHES.type_name(0), "|", UR5E_BRANCHES.type_name(7))
    print("allowed(elbow='up'):", np.flatnonzero(UR5E_BRANCHES.allowed(elbow="up")).tolist())
