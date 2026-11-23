"""IK configurations: name the different IK solutions of an arm, and tag every solution from its posture.

    arm.configurations.questions   # ("shoulder: front | back", "elbow: up | down", "wrist: J5+ | J5−")
    arm.configurations.tag(q)      # (N, dof) -> (N, n_questions) option index per question
    arm.configurations.allowed(elbow="up")   # (n_configurations,) bool

WHY "IK configuration", not "branch" (Zach, 2026-11-23: "The branch makes me think of like multiple
branches on a robot like 1 for the arm 1 for the leg etc. ... Ik Configuration ... mabye?").

WHY tags come from the posture, never from the solver's slot (measured on the saved UR5e map in
docs/proposals/ik_branches.py, 20,000 random solutions): the closed-form slot's elbow sign means "elbow
up" when the shoulder is on one side and "elbow down" on the other (99.5 % / 0.7 %), so a filter wired to
the slot would flip meaning half the time; a numeric IK has no stable slots at all.

Plain data plus one function (`tag`); `to_json` / `from_json` carry the questions (not the function) in a
saved result, so a loaded result filters by its stored per-solution tags without the arm.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ConfigurationQuestion:
    """One question that tells IK solutions apart: "elbow: up or down". Option i is tag value i."""

    name: str  # "elbow"
    options: tuple[str, ...]  # ("up", "down")
    joint: str = ""  # the joint that decides it, shown with its chain index: "elbow (J3)"
    rule: str = ""  # how a solution is tagged, in plain words (a tooltip)


@dataclass(frozen=True, eq=False)
class IKConfigurations:
    """The named IK configurations of one arm. ``tag`` answers every question from a posture."""

    joint_names: tuple[str, ...]
    questions: tuple[ConfigurationQuestion, ...]
    tag: Callable[[np.ndarray], np.ndarray] | None = field(default=None, repr=False)

    @property
    def n_configurations(self) -> int:
        return int(np.prod([len(a.options) for a in self.questions])) if self.questions else 1

    def label(self, question: ConfigurationQuestion) -> str:
        """``"elbow (J3)"``: the question with its joint's chain index."""
        if question.joint in self.joint_names:
            return f"{question.name} (J{self.joint_names.index(question.joint) + 1})"
        return question.name

    def index_of(self, tags: np.ndarray) -> np.ndarray:
        """(N, n_questions) option indices -> (N,) configuration index (first question most significant)."""
        tags = np.atleast_2d(np.asarray(tags, np.int64))
        out = np.zeros(len(tags), np.int64)
        for a, question in enumerate(self.questions):
            out = out * len(question.options) + tags[:, a]
        return out

    def tags_of(self, index: int) -> tuple[int, ...]:
        out = []
        for question in reversed(self.questions):
            index, r = divmod(int(index), len(question.options))
            out.append(r)
        return tuple(reversed(out))

    def name(self, index: int) -> str:
        """``"shoulder front · elbow up · wrist J5+"``."""
        return " · ".join(f"{q.name} {q.options[t]}" for q, t in zip(self.questions, self.tags_of(index)))

    def configuration(self, q: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        """(N, dof) postures -> (N,) uint8 configuration index. Needs ``tag``."""
        if self.tag is None:
            raise ValueError("these configurations carry no tag function (loaded from a saved result)")
        q = np.atleast_2d(np.asarray(q, float))
        parts = [self.index_of(self.tag(q[i : i + chunk])) for i in range(0, len(q), chunk)]
        return (np.concatenate(parts) if parts else np.zeros(0, np.int64)).astype(np.uint8)

    def allowed(self, **choices) -> np.ndarray:
        """(n_configurations,) bool: ``allowed(elbow="up")``, ``allowed(wrist=("J5+", "J5−"))``; a
        question not named allows every option."""
        selected = []
        for question in self.questions:
            want = choices.get(question.name)
            want = question.options if want is None else ((want,) if isinstance(want, str) else tuple(want))
            unknown = set(want) - set(question.options)
            if unknown:
                raise ValueError(f"{question.name}: no option {sorted(unknown)}; options {question.options}")
            selected.append(tuple(i for i, o in enumerate(question.options) if o in want))
        return self.allowed_from(selected)

    def allowed_from(self, selected) -> np.ndarray:
        """(n_configurations,) bool from the selected option indices per question (the viewer's toggles)."""
        ok = np.ones(self.n_configurations, bool)
        for i in range(self.n_configurations):
            ok[i] = all(t in sel for t, sel in zip(self.tags_of(i), selected))
        return ok

    def to_json(self) -> dict:
        return {"joint_names": list(self.joint_names),
                "questions": [{"name": q.name, "options": list(q.options), "joint": q.joint, "rule": q.rule} for q in self.questions]}

    @classmethod
    def from_json(cls, doc: dict) -> IKConfigurations:
        return cls(tuple(doc["joint_names"]), tuple(ConfigurationQuestion(q["name"], tuple(q["options"]), q.get("joint", ""), q.get("rule", ""))
                                                    for q in doc["questions"]))


def wrap(q: np.ndarray) -> np.ndarray:
    """Angles into (-pi, pi]: a joint's sign is read off the posture, not off its +-2pi twin."""
    return np.pi - np.mod(np.pi - np.asarray(q, float), 2 * np.pi)


# --------------------------------------------------------------------------- UR-style 6R arms
def ur_configurations(joint_names, joint_axes: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]) -> IKConfigurations:
    """A UR-style 6R (pan, lift, elbow, three wrist joints): shoulder front/back × elbow up/down × wrist
    J5+/J5− = the eight closed-form solutions.

    `joint_axes(q)` -> (axes (N, 6, 3), points (N, 6, 3)): every joint's world axis and a point on it
    (a URDF chain's `_walk`, or the DH frames' z axes and origins). Only the joints' places and the lift
    axis are used, so a DH model and a URDF of the same arm get the same tags whatever their axis signs:

    - shoulder **front** when the wrist (where the J5 and J6 axes cross) is on the side of the pan axis that the arm's
      sideways offset (along the lift axis, from J2 to the wrist) points 90° clockwise from, seen from above;
      **back** when it reaches over its back. Equal to the DH rule "wrist ahead along x1".
    - elbow **up** when, in the plane the lift and elbow turn in, the elbow sits above the shoulder-to-J4 line.
    - wrist **J5+ / J5−**: the sign of wrist_2 (J5), wrapped into ±180°: the wrist flip."""
    names = tuple(joint_names)

    def tag(q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(np.asarray(q, float))
        axes, points = joint_axes(q)
        a = axes[:, 1] / np.linalg.norm(axes[:, 1], axis=1, keepdims=True)  # the lift (J2) axis
        shoulder, elbow = points[:, 1], points[:, 2]
        # WHY the J5/J6 axes' crossing: it is the one wrist point every model of the arm agrees on (a DH
        # model and the URDF put their joint origins at different places along the same axes).
        wrist = closest_point(points[:, 4], axes[:, 4], points[:, 5], axes[:, 5])
        lateral = np.sum((wrist - shoulder) * a, axis=1, keepdims=True) * a  # the sideways offset, sign-free
        reach = wrist.copy()
        reach[:, 2] = 0.0
        front = np.cross(lateral, reach)[:, 2] > 0

        def flat(p):  # into the plane the lift and elbow turn in
            return p - np.sum(p * a, axis=1, keepdims=True) * a

        # WHY J4's place for the elbow, not the wrist point: the lift and elbow are a 2R that ends at J4
        # (flattened, a DH and a URDF J4 coincide); the two elbow solutions are mirror images across the
        # shoulder-to-J4 line, so "above that line" always splits them (via the wrist point it did not
        # when the elbow folds back, measured 2026-11-23: 215 of 2,264 UR5e targets).
        s, e, w = flat(shoulder), flat(elbow), flat(points[:, 3])
        d = w - s
        t = np.sum((e - s) * d, axis=1) / np.maximum(np.sum(d * d, axis=1), 1e-12)
        up = (e - (s + t[:, None] * d))[:, 2] > 0
        j5 = wrap(q[:, 4]) >= 0
        return np.stack([np.where(front, 0, 1), np.where(up, 0, 1), np.where(j5, 0, 1)], axis=1).astype(np.uint8)

    return IKConfigurations(
        joint_names=names,
        questions=(
            ConfigurationQuestion("shoulder", ("front", "back"), names[0],
                                  "front: the wrist is ahead of the pan axis; back: the arm reaches over its back"),
            ConfigurationQuestion("elbow", ("up", "down"), names[2], "up: the elbow is above the shoulder-to-wrist line"),
            ConfigurationQuestion("wrist", ("J5+", "J5−"), names[4], "the sign of wrist_2 (J5), ±180°: the wrist flip"),
        ),
        tag=tag,
    )


def closest_point(p1, d1, p2, d2) -> np.ndarray:
    """(N, 3) the midpoint of the shortest segment between lines p1 + s d1 and p2 + t d2 (their crossing
    when they cross); p1 where the lines are parallel."""
    w = p1 - p2
    a, b, c = np.sum(d1 * d1, 1), np.sum(d1 * d2, 1), np.sum(d2 * d2, 1)
    d, e = np.sum(d1 * w, 1), np.sum(d2 * w, 1)
    den = a * c - b * b
    ok = den > 1e-12
    s = np.where(ok, (b * e - c * d) / np.where(ok, den, 1.0), 0.0)
    t = np.where(ok, (a * e - b * d) / np.where(ok, den, 1.0), 0.0)
    return 0.5 * ((p1 + s[:, None] * d1) + (p2 + t[:, None] * d2))


def looks_like_ur(joint_names) -> bool:
    """A UR description's six joint names (``shoulder_pan(_joint)`` ... ``wrist_3(_joint)``)."""
    stems = [n.removesuffix("_joint") for n in joint_names]
    return stems == ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


# --------------------------------------------------------------------------- elbow arms
def elbow_configurations(joint_names, points: Callable[[np.ndarray], np.ndarray], shoulder: int, elbow: int, tip: int,
                         yaw: int | None = None) -> IKConfigurations:
    """An arm whose IK splits by elbow up/down (an RR, a yaw + RR shoulder), plus front/back when a yaw
    joint can turn the arm round (`yaw`). `points(q)` -> (N, K, 3) skeleton; `shoulder`, `elbow`, `tip`
    index it."""
    names = tuple(joint_names)
    questions = []
    if yaw is not None:
        questions.append(ConfigurationQuestion("shoulder", ("front", "back"), names[yaw],
                                               "front: the yaw joint faces the tip; back: it is turned 180° and the arm reaches over its top"))
    questions.append(ConfigurationQuestion("elbow", ("up", "down"), names[min(elbow, len(names) - 1)],
                                           "up: the elbow is above the shoulder-to-tip line"))

    def tag(q: np.ndarray) -> np.ndarray:
        pts = np.asarray(points(np.atleast_2d(np.asarray(q, float))), float)
        s, e, w = pts[:, shoulder], pts[:, elbow], pts[:, tip]
        d = w - s
        t = np.sum((e - s) * d, axis=1) / np.maximum(np.sum(d * d, axis=1), 1e-12)
        up = (e - (s + t[:, None] * d))[:, 2] > 0
        cols = []
        if yaw is not None:
            # front: the yaw joint faces the tip (back: turned 180° away, the arm folded over its top);
            # both put the joints at the same points, so only the yaw itself tells them apart.
            yaw_q = np.atleast_2d(np.asarray(q, float))[:, yaw]
            front = (np.cos(yaw_q) * w[:, 0] + np.sin(yaw_q) * w[:, 1]) >= 0
            cols.append(np.where(front, 0, 1))
        cols.append(np.where(up, 0, 1))
        return np.stack(cols, axis=1).astype(np.uint8)

    return IKConfigurations(names, tuple(questions), tag)
