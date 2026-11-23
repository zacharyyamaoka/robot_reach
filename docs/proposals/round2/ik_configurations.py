"""The IK configuration vocabulary every round-2 proposal shares (prototype, not src/).

Zach, 2026-11-23: "for me the vocabulary is like around.. Ik Solutions ... The branch makes me think of
like multiple branches on a robot like 1 for the arm 1 for the leg etc. Ik Configuration... mabye?"

So: an **IK solution** is one joint vector that reaches a pose. Its **IK configuration** is which
posture family it belongs to ("front · elbow up · J5+"), named by answers to a few yes/no questions
(`ConfigurationAxis`). Round 1's `IKBranches` / `BranchAxis` (../ik_branches.py), renamed; the tags
still come from the posture, never the solver's slot (measured in round 1: the slot's elbow sign
flips meaning with the shoulder side). Target home: robot_msgs/manipulation/ik_configurations.py.
src/ is getting its own IK-configuration type in parallel; this one should converge with it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfigurationAxis:
    """One question that splits the IK solutions: "elbow: up or down"."""

    name: str  # "elbow"
    options: tuple[str, ...]  # ("up", "down"): option i is tag value i
    joint: str = ""  # the joint whose side decides it, shown as "elbow (J3)"
    rule: str = ""  # how a solution is tagged, in plain words


@dataclass(frozen=True)
class IKConfigurations:
    """The named IK configurations of one chain. Frozen, JSON-able, no functions inside."""

    chain: str
    joint_names: tuple[str, ...]
    axes: tuple[ConfigurationAxis, ...]

    def __len__(self) -> int:
        return int(np.prod([len(a.options) for a in self.axes]))

    def index_of(self, tags: np.ndarray) -> np.ndarray:
        """(N, n_axes) answers -> (N,) configuration index (first axis most significant)."""
        tags = np.atleast_2d(tags)
        out = np.zeros(len(tags), np.int64)
        for a, axis in enumerate(self.axes):
            out = out * len(axis.options) + tags[:, a]
        return out

    def answers(self, i: int) -> dict[str, str]:
        out = []
        for axis in reversed(self.axes):
            i, r = divmod(int(i), len(axis.options))
            out.append((axis.name, axis.options[r]))
        return dict(reversed(out))

    def label(self, i: int) -> str:
        """'front/up/J5+': short enough for a table cell, one option per axis in axis order."""
        return "/".join(self.answers(i).values())

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.label(i) for i in range(len(self)))

    def allowed(self, *labels: str, **answers: str | Iterable[str]) -> np.ndarray:
        """(n,) bool. allowed(elbow="up"), allowed(wrist=("J5+", "J5-")), allowed("front/up/J5+", "back/down/J5-").
        Positional labels are OR-ed; keyword answers are AND-ed across axes, OR-ed within one."""
        names = {a.name for a in self.axes}
        unknown = set(answers) - names
        if unknown:
            raise ValueError(f"no IK configuration axis {sorted(unknown)}; this chain has {sorted(names)}")
        ok = np.ones(len(self), bool)
        for i in range(len(self)):
            got = self.answers(i)
            for axis, want in answers.items():
                want = (want,) if isinstance(want, str) else tuple(want)
                if got[axis] not in want:
                    ok[i] = False
        if labels:
            bad = [lb for lb in labels if lb not in self.labels]
            if bad:
                raise ValueError(f"no IK configuration {bad}; this chain has {list(self.labels)}")
            ok &= np.isin(np.array(self.labels), labels)
        return ok


UR5E_CONFIGURATIONS = IKConfigurations(
    chain="ur5e",
    joint_names=("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"),
    axes=(
        ConfigurationAxis("shoulder", ("front", "back"), "shoulder_pan", "front: the wrist is in front of the pan axis"),
        ConfigurationAxis("elbow", ("up", "down"), "elbow", "up: the elbow is above the shoulder-to-wrist line"),
        ConfigurationAxis("wrist", ("J5+", "J5-"), "wrist_2", "the sign of wrist_2 (J5): the wrist flip"),
    ),
)
