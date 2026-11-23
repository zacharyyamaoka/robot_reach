"""URDF adapter: an `Arm` for one kinematic chain of any robot description, no hand-written FK/IK.

    from robot_reach.urdf import arm_from_urdf, chains

    chains("ur5e.urdf")                       # [("base_link", "tool0", 6)]
    arm = arm_from_urdf("ur5e.urdf", "tool0")
    robot_reach.analyze(arm)

- **FK** is read straight from the URDF XML: revolute, continuous, prismatic and fixed joints, and
  ``<mimic>`` joints driven from their master. Batched, ``(N, dof) -> (N, 4, 4)``, plus the
  geometric Jacobian in the same pass.
- **IK**, when no analytic solver is passed, is a generic multi-start damped least squares
  (`numeric_ik`) that returns every distinct branch it finds, in the `Arm` contract's shape.
- A bam ``RobotDescription`` goes in through `arm_from_description`, which only calls its
  ``to_urdf_string()``.

WHY no URDF library (urdfpy, yourdfpy, Pinocchio): robot_reach's runtime dependencies are numpy +
robot_msgs (Zach's dependency-free runtime cores), and a joint tree is a page of `xml.etree`.
Inertials are not read; xacro must be expanded first.

**Meshes** (``arm_from_urdf(..., meshes=True)``): every link's ``<visual>`` and ``<collision>``
geometry becomes an `ArmMesh`, posed by the arm's ``mesh_poses`` — links off the chain (a gripper's
fingers) ride along at their joints' zero (clipped into the limits). Visuals are drawn, collisions
measured (`Arm.surface`). Mesh files are read with trimesh (the ``viz`` extra), imported only here;
``package://pkg/...`` resolves under ``package_dirs``, the URDF's folder, its ``assets/`` and its
parents. A visual that cannot be read (a ``.dae`` needs pycollada) is replaced by its link's collision
mesh, and ``arm.meta["mesh_note"]`` says so.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arm import Arm, Task
from .codes import Code

MOVABLE = ("revolute", "continuous", "prismatic")
TWO_PI = 2 * np.pi


# --------------------------------------------------------------------------- parsing
@dataclass(frozen=True, eq=False)
class UrdfJoint:
    name: str
    kind: str  # revolute | continuous | prismatic | fixed | floating | planar
    parent: str
    child: str
    origin: np.ndarray  # (4, 4) parent link -> joint frame at zero joint value
    axis: np.ndarray  # (3,) unit axis in the joint frame
    lower: float
    upper: float
    mimic: tuple[str, float, float] | None = None  # (master joint, multiplier, offset)

    @property
    def movable(self) -> bool:
        return self.kind in MOVABLE


def _floats(text: str | None, n: int, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(n, default)
    values = [float(v) for v in text.split()]
    if len(values) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return np.array(values)


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy: fixed-axis roll about X, then pitch about Y, then yaw about Z = Rz Ry Rx."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _origin(element: ET.Element) -> np.ndarray:
    T = np.eye(4)
    origin = element.find("origin")
    if origin is not None:
        T[:3, :3] = rpy_matrix(*_floats(origin.get("rpy"), 3))
        T[:3, 3] = _floats(origin.get("xyz"), 3)
    return T


def _joint(element: ET.Element) -> UrdfJoint:
    name, kind = element.get("name"), element.get("type")
    parent, child = element.find("parent"), element.find("child")
    if name is None or kind is None or parent is None or child is None:
        raise ValueError(f"joint {name!r}: needs a name, a type, a <parent> and a <child>")
    axis_element = element.find("axis")
    axis = _floats(None if axis_element is None else axis_element.get("xyz"), 3)
    if axis_element is None:
        axis = np.array([1.0, 0.0, 0.0])  # the URDF default
    norm = float(np.linalg.norm(axis))
    if kind in MOVABLE and norm < 1e-12:
        raise ValueError(f"joint {name!r}: zero-length axis")
    axis = axis / norm if norm > 0 else axis  # URDF does not require a unit axis
    if kind == "continuous":
        # WHY +-pi: a continuous joint's postures repeat every 2pi, so one turn holds each exactly
        # once; IK answers anywhere else are wrapped back (here, and by the library's +-2pi wrap).
        lower, upper = -np.pi, np.pi
    elif kind in ("revolute", "prismatic"):
        limit = element.find("limit")
        # The URDF spec's defaults: a missing lower/upper is 0, said truthfully rather than guessed.
        lower = float(limit.get("lower", 0.0)) if limit is not None else 0.0
        upper = float(limit.get("upper", 0.0)) if limit is not None else 0.0
    else:
        lower = upper = 0.0
    mimic_element = element.find("mimic")
    mimic = None
    if mimic_element is not None:
        mimic = (
            mimic_element.get("joint"),
            float(mimic_element.get("multiplier", 1.0)),
            float(mimic_element.get("offset", 0.0)),
        )
    return UrdfJoint(name, kind, parent.get("link"), child.get("link"), _origin(element), axis, lower, upper, mimic)


@dataclass(frozen=True, eq=False)
class UrdfGeometry:
    """One ``<visual>`` or ``<collision>`` of a link: a mesh file or a box / cylinder / sphere."""

    link: str
    role: str  # "visual" | "collision"
    origin: np.ndarray  # (4, 4) link frame -> geometry frame
    shape: str  # "mesh" | "box" | "cylinder" | "sphere"
    filename: str | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: tuple[float, ...] = ()  # box (x, y, z); cylinder (radius, length); sphere (radius,)
    color: tuple[int, int, int] | None = None


def _geometries(root: ET.Element) -> tuple[UrdfGeometry, ...]:
    named = {}
    for material in root.findall("material"):
        color = material.find("color")
        if material.get("name") and color is not None and color.get("rgba"):
            named[material.get("name")] = tuple(int(round(255 * float(v))) for v in color.get("rgba").split()[:3])
    out = []
    for link in root.findall("link"):
        for role in ("visual", "collision"):
            for element in link.findall(role):
                geometry = element.find("geometry")
                if geometry is None or len(geometry) == 0:
                    continue
                g = geometry[0]
                color = None
                material = element.find("material")
                if material is not None:
                    rgba = material.find("color")
                    if rgba is not None and rgba.get("rgba"):
                        color = tuple(int(round(255 * float(v))) for v in rgba.get("rgba").split()[:3])
                    else:
                        color = named.get(material.get("name"))
                if g.tag == "mesh":
                    scale = tuple(_floats(g.get("scale"), 3, 1.0)) if g.get("scale") else (1.0, 1.0, 1.0)
                    out.append(UrdfGeometry(link.get("name"), role, _origin(element), "mesh", g.get("filename"), scale, color=color))
                elif g.tag == "box":
                    out.append(UrdfGeometry(link.get("name"), role, _origin(element), "box", size=tuple(_floats(g.get("size"), 3)), color=color))
                elif g.tag == "cylinder":
                    size = (float(g.get("radius", 0.0)), float(g.get("length", 0.0)))
                    out.append(UrdfGeometry(link.get("name"), role, _origin(element), "cylinder", size=size, color=color))
                elif g.tag == "sphere":
                    out.append(UrdfGeometry(link.get("name"), role, _origin(element), "sphere", size=(float(g.get("radius", 0.0)),), color=color))
    return tuple(out)


class UrdfModel:
    """The joint tree of one URDF: links, joints, the root, the leaves, and chains through it."""

    def __init__(self, name: str, links: tuple[str, ...], joints: tuple[UrdfJoint, ...], geometries: tuple[UrdfGeometry, ...] = ()):
        self.name, self.links, self.joints, self.geometries = name, links, joints, geometries
        self._parent_joint: dict[str, UrdfJoint] = {}
        for joint in joints:
            if joint.child in self._parent_joint:
                raise ValueError(f"{name}: link {joint.child!r} has two parent joints; a URDF is a tree")
            self._parent_joint[joint.child] = joint
        roots = [link for link in links if link not in self._parent_joint]
        if len(roots) != 1:
            raise ValueError(f"{name}: expected exactly one root link, found {roots}")
        self.root = roots[0]

    def leaf_links(self) -> list[str]:
        """Links no joint hangs from: the candidate tips."""
        parents = {joint.parent for joint in self.joints}
        return [link for link in self.links if link not in parents]

    def chain(self, tip_link: str, base_link: str | None = None, tool_offset: np.ndarray | None = None) -> Chain:
        base = self.root if base_link is None else base_link
        for role, link in (("tip", tip_link), ("base", base)):
            if link not in self.links:
                raise ValueError(f"{self.name}: no {role} link {link!r}; candidate tips: {self.leaf_links()}")
        path, link = [], tip_link
        while link != base:
            joint = self._parent_joint.get(link)
            if joint is None:
                raise ValueError(f"{self.name}: {base!r} is not an ancestor of {tip_link!r}")
            path.append(joint)
            link = joint.parent
        return Chain(self.name, tuple(reversed(path)), base, tip_link, tool_offset)

    def link_poser(self, chain: Chain, links) -> tuple[list[str], Callable[[np.ndarray], np.ndarray]]:
        """(links it can place, ``poses(q) -> (N, L, 4, 4)``) for any `links` of the tree below the
        chain's base: a chain link moves with q; a link off the chain hangs from its nearest chain
        ancestor through joints held at zero (clipped into their limits; a mimic joint at its offset)."""
        index = {link: i for i, link in enumerate(chain.links)}
        placed, anchors, constants = [], [], []
        for link in links:
            T, cursor = np.eye(4), link
            while cursor not in index:
                joint = self._parent_joint.get(cursor)
                if joint is None:
                    break
                value = joint.mimic[2] if joint.mimic else float(np.clip(0.0, joint.lower, joint.upper))
                motion = np.eye(4)
                if joint.kind == "prismatic":
                    motion[:3, 3] = joint.axis * value
                elif joint.movable:
                    motion[:3, :3] = _rotations(joint.axis, np.array([value]))[0]
                T = joint.origin @ motion @ T
                cursor = joint.parent
            if cursor in index:
                placed.append(link)
                anchors.append(index[cursor])
                constants.append(T)
        anchors_arr, constants_arr = np.array(anchors, int), np.array(constants, float).reshape(-1, 4, 4)

        def poses(q: np.ndarray) -> np.ndarray:
            frames = chain.link_frames(q)  # (N, K, 4, 4)
            return frames[:, anchors_arr] @ constants_arr[None]

        return placed, poses


def parse_urdf(xml: str) -> UrdfModel:
    """URDF XML text -> `UrdfModel`."""
    root = ET.fromstring(xml)
    if root.tag != "robot":
        raise ValueError(f"not a URDF: root element <{root.tag}>, expected <robot> (expand xacro first)")
    joints = tuple(_joint(element) for element in root.findall("joint"))
    links = [element.get("name") for element in root.findall("link")]
    for joint in joints:  # tolerate a link that only a joint names
        for link in (joint.parent, joint.child):
            if link not in links:
                links.append(link)
    return UrdfModel(root.get("name", "robot"), tuple(links), joints, _geometries(root))


def _read(source) -> tuple[UrdfModel, Path | None]:
    """A path, a URDF XML string, or anything with ``to_urdf_string()`` -> (model, file path or None)."""
    if hasattr(source, "to_urdf_string"):
        return parse_urdf(source.to_urdf_string()), None
    if isinstance(source, Path) or not str(source).lstrip().startswith("<"):
        path = Path(source).expanduser().resolve()
        return parse_urdf(path.read_text()), path
    return parse_urdf(source), None


def load_urdf(source) -> UrdfModel:
    """`UrdfModel` from a URDF path, a URDF XML string, or an object with ``to_urdf_string()``."""
    return _read(source)[0]


# --------------------------------------------------------------------------- one chain, batched
@dataclass(frozen=True)
class _Segment:
    """One moving joint with every fixed transform since the previous one folded in front of it."""

    pre: np.ndarray  # (4, 4) constant transform from the previous moving joint's frame
    axis: np.ndarray  # (3,) in the joint frame
    prismatic: bool
    index: int  # which DOF drives it
    multiplier: float  # joint value = multiplier * q[index] + offset (1, 0 unless a mimic joint)
    offset: float


def _skew(axis: np.ndarray) -> np.ndarray:
    x, y, z = axis
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotations(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rodrigues, batched over theta: (N,) -> (N, 3, 3)."""
    K = _skew(axis)
    s, c = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


class Chain:
    """One serial path base -> tip through a URDF tree, compiled for batched FK and Jacobians.

    The DOF are the chain's moving joints, except a mimic joint whose master is also on the chain
    (it follows the master). A mimic joint whose master is off the chain is a DOF of its own here:
    something else moves it, and this chain cannot know what."""

    def __init__(self, robot: str, joints: tuple[UrdfJoint, ...], base_link: str, tip_link: str, tool_offset=None):
        for joint in joints:
            if joint.kind not in (*MOVABLE, "fixed"):
                raise ValueError(
                    f"joint {joint.name!r} is {joint.kind}: "
                    "a serial arm has revolute, continuous, prismatic and fixed joints"
                )
        self.robot, self.joints, self.base_link, self.tip_link = robot, joints, base_link, tip_link
        self.tool_offset = None if tool_offset is None else np.asarray(tool_offset, float).reshape(4, 4)
        self.links = (base_link, *(joint.child for joint in joints))
        on_chain = {joint.name: joint for joint in joints if joint.movable}
        actuated = [j for j in on_chain.values() if not (j.mimic and j.mimic[0] in on_chain)]
        self.actuated = tuple(actuated)
        index = {joint.name: i for i, joint in enumerate(actuated)}

        def drive(joint: UrdfJoint, depth: int = 0) -> tuple[int, float, float]:
            if joint.name in index:
                return index[joint.name], 1.0, 0.0
            if depth > len(joints):
                raise ValueError(f"mimic joints form a cycle at {joint.name!r}")
            k, multiplier, offset = drive(on_chain[joint.mimic[0]], depth + 1)
            return k, joint.mimic[1] * multiplier, joint.mimic[1] * offset + joint.mimic[2]

        self._drive = {name: drive(joint) for name, joint in on_chain.items()}
        segments, pre = [], np.eye(4)
        for joint in joints:
            pre = pre @ joint.origin
            if joint.movable:
                segments.append(_Segment(pre, joint.axis, joint.kind == "prismatic", *self._drive[joint.name]))
                pre = np.eye(4)
        self._segments = tuple(segments)
        self._post = pre if self.tool_offset is None else pre @ self.tool_offset

    # ------------------------------------------------------------------ description
    @property
    def dof(self) -> int:
        return len(self.actuated)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.actuated)

    @property
    def lower(self) -> np.ndarray:
        return np.array([joint.lower for joint in self.actuated], float)

    @property
    def upper(self) -> np.ndarray:
        return np.array([joint.upper for joint in self.actuated], float)

    @property
    def revolute(self) -> np.ndarray:
        """(dof,) True for revolute/continuous joints (wrap by 2pi), False for prismatic."""
        return np.array([joint.kind != "prismatic" for joint in self.actuated], bool)

    @property
    def length(self) -> float:
        """The arm's size in metres: the joint offsets, the tool offset and the prismatic strokes.
        IK divides positions by it so a metre of error and a radian of error weigh alike."""
        size = sum(float(np.linalg.norm(joint.origin[:3, 3])) for joint in self.joints)
        size += sum(joint.upper - joint.lower for joint in self.actuated if joint.kind == "prismatic")
        if self.tool_offset is not None:
            size += float(np.linalg.norm(self.tool_offset[:3, 3]))
        return max(size, 1e-3)

    # ------------------------------------------------------------------ kinematics
    def _walk(self, q: np.ndarray):
        """Tip pose plus every moving joint's world axis and origin, (N, 4, 4), (N, J, 3), (N, J, 3)."""
        q = np.atleast_2d(np.asarray(q, float))
        n = len(q)
        R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        p = np.zeros((n, 3))
        axes = np.zeros((n, len(self._segments), 3))
        points = np.zeros((n, len(self._segments), 3))
        for s, seg in enumerate(self._segments):
            p = p + R @ seg.pre[:3, 3]
            R = R @ seg.pre[:3, :3]
            theta = seg.multiplier * q[:, seg.index] + seg.offset
            axes[:, s] = R @ seg.axis
            points[:, s] = p
            if seg.prismatic:
                p = p + axes[:, s] * theta[:, None]
            else:
                R = R @ _rotations(seg.axis, theta)
        p = p + R @ self._post[:3, 3]
        R = R @ self._post[:3, :3]
        T = np.zeros((n, 4, 4))
        T[:, :3, :3], T[:, :3, 3], T[:, 3, 3] = R, p, 1.0
        return T, axes, points

    def fk(self, q: np.ndarray) -> np.ndarray:
        """(N, dof) -> (N, 4, 4) base -> tip (· tool_offset)."""
        return self._walk(q)[0]

    def fk_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, dof) -> (T (N, 4, 4), J (N, 6, dof)): the geometric Jacobian in the base frame, rows
        0-2 linear velocity of the tip origin, rows 3-5 angular velocity (as `Arm.jacobian`)."""
        T, axes, points = self._walk(q)
        J = np.zeros((len(T), 6, self.dof))
        tip = T[:, :3, 3]
        for s, seg in enumerate(self._segments):
            if seg.prismatic:
                J[:, :3, seg.index] += seg.multiplier * axes[:, s]
            else:
                J[:, :3, seg.index] += seg.multiplier * np.cross(axes[:, s], tip - points[:, s])
                J[:, 3:, seg.index] += seg.multiplier * axes[:, s]
        return T, J

    def link_frames(self, q: np.ndarray) -> np.ndarray:
        """(N, dof) -> (N, K, 4, 4): every link frame along the chain, `links` order (base ... tip),
        plus the tool frame last when there is a tool offset."""
        q = np.atleast_2d(np.asarray(q, float))
        T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        frames = [T]
        for joint in self.joints:
            T = T @ joint.origin
            if joint.movable:
                k, multiplier, offset = self._drive[joint.name]
                theta = multiplier * q[:, k] + offset
                motion = np.broadcast_to(np.eye(4), T.shape).copy()
                if joint.kind == "prismatic":
                    motion[:, :3, 3] = joint.axis * theta[:, None]
                else:
                    motion[:, :3, :3] = _rotations(joint.axis, theta)
                T = T @ motion
            frames.append(T)
        if self.tool_offset is not None:
            frames.append(T @ self.tool_offset)
        return np.stack(frames, axis=1)

    def skeleton(self, q: np.ndarray) -> np.ndarray:
        """(N, dof) -> (N, K, 3): the origin of every link frame, base ... tip (... tool)."""
        return self.link_frames(q)[..., :3, 3]

    def plane(self, tol: float = 1e-9) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """(point, normal) when the chain moves in one plane, else None.

        WHY detect it: a planar arm's position Jacobian has rank 2, so without its plane
        `Arm.manipulability` would call every posture of a planar 3R singular. Planar means every
        revolute axis is parallel to one normal, every prismatic axis lies across it, and every link
        origin sits in one plane across it; motion keeps all three true, so one posture decides."""
        home = np.clip(np.zeros(self.dof), self.lower, self.upper)[None]
        _, axes, _ = self._walk(home)
        revolute = np.array([not seg.prismatic for seg in self._segments], bool)
        if not revolute.any():
            return None
        axes = axes[0]
        normal = axes[np.argmax(revolute)]
        if np.any(np.linalg.norm(np.cross(axes[revolute], normal), axis=1) > tol):
            return None
        if np.any(np.abs(axes[~revolute] @ normal) > tol):
            return None
        heights = self.skeleton(home)[0] @ normal
        if np.ptp(heights) > tol:
            return None
        normal = normal * np.sign(normal[np.argmax(np.abs(normal))])  # read +y, not -y
        point = normal * float(self.skeleton(home)[0, -1] @ normal)
        return tuple(float(v) for v in point), tuple(float(v) for v in normal)


# --------------------------------------------------------------------------- numeric IK
def default_task(dof: int) -> Task:
    """6+ DOF -> pose, 4-5 -> axis, 1-3 -> position.

    WHY 4 DOF gets "axis": a 4-DOF arm (a SCARA, a palletiser) controls position plus some approach
    directions; asking for the axis lets the map show which, where "position" would cap it at 3."""
    return "pose" if dof >= 6 else "axis" if dof >= 4 else "position"


def into_limits(q: np.ndarray, lower: np.ndarray, upper: np.ndarray, revolute: np.ndarray) -> np.ndarray:
    """Wrap revolute joints to [-pi, pi), shift by 2pi where that lands inside the limits, then clamp.

    WHY wrap before clamping: clamping a full-turn joint (a UR's +-360 deg) at its limit would trap
    the solver against a wall that is not there, and the 2pi twins of one posture become one value."""
    q = np.where(revolute, (q + np.pi) % TWO_PI - np.pi, q)
    for shift in (TWO_PI, -TWO_PI):
        moved = q + shift
        fix = revolute & ((q < lower) | (q > upper)) & (moved >= lower) & (moved <= upper)
        q = np.where(fix, moved, q)
    return np.clip(q, lower, upper)


def _rotvec(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 3) rotation vectors (log map), exact up to and including pi."""
    cos = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    skew = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)  # 2 sin(a) axis
    sin = np.sin(angle)
    out = skew * np.where(angle < 1e-8, 0.5, angle / (2.0 * np.where(sin > 1e-300, sin, 1.0)))[:, None]
    # WHY the symmetric part past 90 deg: the skew part vanishes as the angle nears pi, while
    # (R + R^T)/2 - cos I = (1 - cos) a a^T stays well conditioned there.
    wide = angle > np.pi / 2
    if wide.any():
        B = (R[wide] + np.swapaxes(R[wide], 1, 2)) / 2.0 - cos[wide, None, None] * np.eye(3)
        column = np.argmax(np.diagonal(B, axis1=1, axis2=2), axis=1)
        axis = B[np.arange(len(B)), :, column]
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        sign = np.where(np.sum(axis * skew[wide], axis=1) < 0, -1.0, 1.0)
        out[wide] = axis * (sign * angle[wide])[:, None]
    return out


def _residual(T: np.ndarray, J: np.ndarray, target: np.ndarray, task: str, length: float):
    """Task error and its Jacobian, positions divided by `length`; plus position / angle errors."""
    dp = target[:, :3, 3] - T[:, :3, 3]
    pos_err = np.linalg.norm(dp, axis=1)
    e_pos, J_pos = dp / length, J[:, :3] / length
    if task == "position":
        return e_pos, J_pos, pos_err, np.zeros(len(T))
    R, R_target, J_rot = T[:, :3, :3], target[:, :3, :3], J[:, 3:]
    if task == "pose":
        w = _rotvec(R_target @ np.swapaxes(R, 1, 2))  # world-frame rotation taking R onto the target
        return np.concatenate([e_pos, w], 1), np.concatenate([J_pos, J_rot], 1), pos_err, np.linalg.norm(w, axis=1)
    # axis: turn the tool +Z onto the target's +Z; roll about +Z is free, so keep only the two
    # components across it (tool x and y), which makes the system exactly 5 rows.
    z, z_target = R[:, :, 2], R_target[:, :, 2]
    cos = np.clip(np.sum(z * z_target, axis=1), -1.0, 1.0)
    angle = np.arccos(cos)
    cross = np.cross(z, z_target)
    sin = np.linalg.norm(cross, axis=1)
    w = cross * (angle / np.maximum(sin, 1e-300))[:, None]
    flipped = (sin < 1e-12) & (cos < 0)  # exactly opposite: any axis across z works; take tool x
    w[flipped] = R[flipped, :, 0] * angle[flipped, None]
    x, y = R[:, :, 0], R[:, :, 1]
    e_rot = np.stack([np.sum(x * w, axis=1), np.sum(y * w, axis=1)], -1)
    J_axis = np.stack([np.einsum("ni,nij->nj", x, J_rot), np.einsum("ni,nij->nj", y, J_rot)], 1)
    return np.concatenate([e_pos, e_rot], 1), np.concatenate([J_pos, J_axis], 1), pos_err, angle


def numeric_ik(
    fk_jacobian: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    lower,
    upper,
    *,
    revolute=None,
    task: Task = "pose",
    n_seeds: int = 8,
    home=None,
    length: float = 1.0,
    max_iter: int = 100,
    pos_tol: float = 1e-9,
    rot_tol: float = 1e-9,
    merge_tol: float = 1e-4,
    damping: float = 1e-10,
    max_step: float = 0.5,
    seed: int = 0,
    block: int = 4096,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """A batched multi-branch IK for any FK that also returns its geometric Jacobian.

    Every target starts from `n_seeds` postures: `home` (default zero clipped into the limits) and
    `n_seeds - 1` uniform samples inside the limits, the same seeded set for every target, so a
    run is deterministic. Each start runs Levenberg-Marquardt with Sugihara's damping
    (``lambda = |e|^2 / 2 + damping``: a big step near the answer, a careful one far from it),
    wrapping revolute joints and clamping to the limits every step. A start that converges
    (``pos_tol`` m, ``rot_tol`` rad) is a branch; converged starts that land on the same joint
    vector (``merge_tol``, after the 2pi wrap) count once.

    Returns ``ik(T (N, 4, 4)) -> (q (N, n_seeds, dof), code (N, n_seeds))``: OK branches first, in
    seed order (branch 0 is the one the home start reached, when it converged), then PAD; a target
    nothing converged on gets NO_SOLUTION in slot 0, with the closest attempt as its q. Branch k is
    not the same physical branch from one target to the next, as it is for a closed-form solver.
    A redundant arm (more DOF than the task) returns up to `n_seeds` points of its self-motion.

    WHY damping-only, no line search: Sugihara's damping converges globally on serial arms without
    an accept/reject step, which keeps every start in lock-step and the whole batch vectorised.
    WHY the constant part is only 1e-10 (it just keeps the system solvable at a singularity):
    measured on 2000 random UR5e poses, 1e-4 solved 97.5 %, 1e-6 99.85 %, 1e-10 100 % - a bigger
    constant slows convergence near singular postures until the stall rule drops the start."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    dof = len(lower)
    revolute = np.ones(dof, bool) if revolute is None else np.asarray(revolute, bool)
    home = np.zeros(dof) if home is None else np.asarray(home, float)
    starts = np.vstack([home, lower + np.random.default_rng(seed).random((max(n_seeds, 1) - 1, dof)) * (upper - lower)])
    starts = into_limits(starts, lower, upper, revolute)
    n_starts = len(starts)
    # WHY prismatic joints in units of the arm's size: with positions divided by `length` too, one
    # damping value and one step cap fit a 10 cm gripper and a 3 m gantry alike.
    joint_scale = np.where(revolute, 1.0, length)
    eye = np.eye(dof)

    def solve_block(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(targets)
        q = np.tile(starts, (n, 1))  # (n * n_starts, dof), target-major
        goal = np.repeat(targets, n_starts, axis=0)
        error = np.full(len(q), np.inf)  # last energy seen, to pick the closest failed attempt
        energy_10_ago = np.full(len(q), np.inf)
        converged = np.zeros(len(q), bool)
        active = np.arange(len(q))
        for iteration in range(max_iter + 1):
            T, J = fk_jacobian(q[active])
            e, Je, pos_err, rot_err = _residual(T, J, goal[active], task, length)
            energy = 0.5 * np.sum(e * e, axis=1)
            error[active] = energy
            done = (pos_err < pos_tol) & (rot_err < rot_tol)
            converged[active[done]] = True
            keep = ~done
            if iteration % 10 == 0:
                # WHY drop a start that stalls: an out-of-reach target never converges, and LM
                # creeps toward the closest posture forever; one that has not cut its error by 10%
                # in ten steps has stopped being a branch candidate.
                keep &= energy < 0.9 * energy_10_ago[active]
                energy_10_ago[active] = energy
            if iteration == max_iter or not keep.any():
                break
            active, e, Je, energy = active[keep], e[keep], Je[keep] * joint_scale, energy[keep]
            JeT = np.swapaxes(Je, 1, 2)
            A = JeT @ Je + (energy + damping)[:, None, None] * eye
            step = np.linalg.solve(A, (JeT @ e[:, :, None]))[:, :, 0]
            biggest = np.max(np.abs(step), axis=1)
            step *= np.minimum(1.0, max_step / np.maximum(biggest, 1e-300))[:, None]
            q[active] = into_limits(q[active] + step * joint_scale, lower, upper, revolute)

        q = q.reshape(n, n_starts, dof)
        ok = converged.reshape(n, n_starts)
        for a in range(n_starts):
            for b in range(a + 1, n_starts):
                both = ok[:, a] & ok[:, b]
                if both.any():
                    gap = q[:, a] - q[:, b]
                    gap = np.where(revolute, (gap + np.pi) % TWO_PI - np.pi, gap) / joint_scale
                    ok[both & (np.max(np.abs(gap), axis=1) < merge_tol), b] = False
        order = np.argsort(~ok, axis=1, kind="stable")
        q_out = np.take_along_axis(q, order[:, :, None], axis=1)
        ok_out = np.take_along_axis(ok, order, axis=1)
        code = np.where(ok_out, int(Code.OK), int(Code.PAD)).astype(np.int8)
        empty = ~ok_out[:, 0]
        code[empty, 0] = Code.NO_SOLUTION
        closest = np.argmin(error.reshape(n, n_starts), axis=1)
        q_out[empty, 0] = q[empty, closest[empty]]
        return q_out, code

    def ik(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = np.asarray(T, float).reshape(-1, 4, 4)
        q = np.zeros((len(T), n_starts, dof))
        code = np.full((len(T), n_starts), int(Code.PAD), np.int8)
        for start in range(0, len(T), block):  # bounds peak memory: block * n_starts problems at once
            q[start : start + block], code[start : start + block] = solve_block(T[start : start + block])
        return q, code

    return ik


# --------------------------------------------------------------------------- meshes
def resolve_mesh(filename: str, base_dir: Path | None, package_dirs=()) -> Path | None:
    """A URDF mesh reference -> an existing file, or None. ``package://pkg/rest`` is looked up as
    ``<dir>/pkg/rest`` under `package_dirs`, then the URDF's folder, its ``assets/`` and two parents."""
    if filename.startswith("file://"):
        path = Path(filename[len("file://"):])
        return path if path.exists() else None
    if filename.startswith("package://"):
        package, _, rest = filename[len("package://"):].partition("/")
        roots = [Path(d) for d in package_dirs]
        if base_dir is not None:
            roots += [base_dir, base_dir / "assets", base_dir.parent, base_dir.parent.parent]
        for root in roots:
            for candidate in (root / package / rest, root / rest if root.name == package else None):
                if candidate is not None and candidate.exists():
                    return candidate
        return None
    path = Path(filename)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path if path.exists() else None


def _mesh_color(mesh) -> tuple[int, int, int] | None:
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    color = getattr(material, "main_color", None)
    if color is None and getattr(visual, "kind", None) in ("face", "vertex"):
        color = np.asarray(visual.face_colors).mean(axis=0) if visual.kind == "face" else np.asarray(visual.vertex_colors).mean(axis=0)
    return None if color is None else tuple(int(c) for c in np.asarray(color)[:3])


def _mesh_arrays(geometry: UrdfGeometry, base_dir: Path | None, package_dirs) -> list[tuple[np.ndarray, np.ndarray, tuple | None]]:
    """[(vertices (V, 3) in the link frame, faces (F, 3), colour or None)]: one entry per coloured
    sub-mesh of the file (a UR's .dae carries its grey body and blue caps). Raises with the reason."""
    import trimesh  # the viz extra: only drawing needs it

    if geometry.shape == "mesh":
        path = resolve_mesh(geometry.filename, base_dir, package_dirs)
        if path is None:
            raise FileNotFoundError(f"{geometry.filename} not found")
        loaded = trimesh.load(str(path))  # a .dae needs pycollada: ImportError
        parts = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
        out = []
        for mesh in parts:
            color = _mesh_color(mesh)
            mesh = trimesh.Trimesh(np.asarray(mesh.vertices, float) * np.asarray(geometry.scale, float), np.asarray(mesh.faces), process=True)
            out.append((mesh.vertices, mesh.faces, color))
    elif geometry.shape == "box":
        mesh = trimesh.creation.box(extents=geometry.size)
        out = [(mesh.vertices, mesh.faces, None)]
    elif geometry.shape == "cylinder":
        mesh = trimesh.creation.cylinder(radius=geometry.size[0], height=geometry.size[1], sections=24)
        out = [(mesh.vertices, mesh.faces, None)]
    else:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=geometry.size[0])
        out = [(mesh.vertices, mesh.faces, None)]
    R, t = geometry.origin[:3, :3], geometry.origin[:3, 3]
    return [(np.asarray(v, float) @ R.T + t, np.asarray(f, np.int64), geometry.color or c) for v, f, c in out]


def _merge_by_color(parts: list) -> list:
    """One (vertices, faces, colour) per colour: fewer scene nodes to pose per frame."""
    groups: dict = {}
    for v, f, color in parts:
        groups.setdefault(color, []).append((v, f))
    out = []
    for color, items in groups.items():
        offsets = np.cumsum([0] + [len(v) for v, _ in items[:-1]])
        out.append((np.concatenate([v for v, _ in items]), np.concatenate([f + o for (_, f), o in zip(items, offsets)]), color))
    return out


def urdf_meshes(model: UrdfModel, chain: Chain, base_dir: Path | None, package_dirs=()) -> tuple[tuple, Callable | None, str]:
    """(`ArmMesh` parts, ``mesh_poses(q) -> (N, M, 4, 4)``, a note on what could not be read).

    Visuals are drawn (`draw`, not `measure`); collisions are measured, not drawn. A link whose
    visuals all fail is drawn with its collision geometry instead, and the note names why."""
    from .arm import ArmMesh

    by_link: dict[str, dict[str, list]] = {}
    failures = []
    for geometry in model.geometries:
        try:
            parts = _mesh_arrays(geometry, base_dir, package_dirs)
        except Exception as exc:  # report, never hide: the note lists each failure
            failures.append(f"{geometry.link} {geometry.role}: {type(exc).__name__}: {exc}")
            continue
        by_link.setdefault(geometry.link, {"visual": [], "collision": []})[geometry.role].extend(parts)
    placed, poser = model.link_poser(chain, list(by_link))
    meshes, owners = [], []
    for link in placed:
        visual, collision = _merge_by_color(by_link[link]["visual"]), by_link[link]["collision"]
        draw_from_collision = not visual and bool(collision)
        for v, f, color in visual:
            meshes.append(ArmMesh(link, v, f, color, draw=True, measure=False))
            owners.append(link)
        for v, f, color in collision:
            meshes.append(ArmMesh(link, v, f, color, draw=draw_from_collision, measure=True))
            owners.append(link)
    note = ""
    if failures:
        fell_back = sorted({fail.split(" ")[0] for fail in failures if " visual:" in fail} & {m.part for m in meshes if m.draw})
        note = f"{len(failures)} geometry file(s) unreadable" + (f"; drawn with collision meshes: {', '.join(fell_back)}" if fell_back else "")
        note += f". First: {failures[0]}"
    if not meshes:
        return (), None, note or "no geometry in the URDF"
    link_index = {link: i for i, link in enumerate(placed)}
    columns = np.array([link_index[o] for o in owners], int)

    def mesh_poses(q: np.ndarray) -> np.ndarray:
        return poser(np.atleast_2d(np.asarray(q, float)))[:, columns]

    return tuple(meshes), mesh_poses, note


# --------------------------------------------------------------------------- front doors
def arm_from_urdf(
    urdf: str | Path,
    tip_link: str,
    base_link: str | None = None,
    *,
    task: Task | None = None,
    ik: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    n_seeds: int = 8,
    name: str | None = None,
    tool_offset=None,
    meshes: bool = False,
    package_dirs=(),
) -> Arm:
    """An `Arm` for the chain `base_link` (default: the root) -> `tip_link` of a URDF.

    `urdf` is a file path or the XML text. `task` defaults from the DOF (`default_task`). `ik`
    replaces the generic `numeric_ik` with an analytic solver in the `Arm` contract. `tool_offset`
    is a fixed 4x4 appended at the tip (the tool's +Z is the approach axis). `meshes` reads every
    link's visual and collision geometry (module docstring; needs trimesh) into `Arm.meshes`."""
    model, path = _read(urdf)
    offset = None if tool_offset is None else np.asarray(tool_offset, float).reshape(4, 4)
    chain = model.chain(tip_link, base_link, tool_offset=offset)
    if chain.dof == 0:
        raise ValueError(f"{model.name}: no movable joint between {chain.base_link!r} and {tip_link!r}")
    task = default_task(chain.dof) if task is None else task
    lower, upper, revolute = chain.lower, chain.upper, chain.revolute
    home = np.clip(np.zeros(chain.dof), lower, upper)
    if name is None:
        name = f"{model.name}:{tip_link}" if base_link is None else f"{model.name}:{base_link}->{tip_link}"
    solver = ik
    if solver is None:
        solver = numeric_ik(
            chain.fk_jacobian,
            lower,
            upper,
            revolute=revolute,
            task=task,
            n_seeds=n_seeds,
            home=home,
            length=chain.length,
        )
    meta = {
        "home": home.tolist(),
        "revolute": revolute.tolist(),
        "base_link": chain.base_link,
        "tip_link": tip_link,
        "links": list(chain.links),
        "frames": chain.link_frames,
    }
    arm_meshes, mesh_poses = (), None
    if meshes:
        base_dir = None if path is None else path.parent
        arm_meshes, mesh_poses, note = urdf_meshes(model, chain, base_dir, package_dirs)
        meta["mesh_note"] = note
    if path is not None:
        # WHY only for a file: a saved result rebuilds its arm from this recipe (analyze.rebuild),
        # and an in-memory string has no address to rebuild from, so it records none rather than
        # inline the XML. A passed `ik` is not recorded: the rebuilt arm poses identically (same
        # FK) and falls back to the numeric IK.
        kwargs = {
            "urdf": str(path),
            "tip_link": tip_link,
            "base_link": base_link,
            "task": task,
            "n_seeds": n_seeds,
            "name": name,
        }
        if meshes:
            kwargs["meshes"] = True
            kwargs["package_dirs"] = [str(d) for d in package_dirs]
        kwargs["tool_offset"] = None if offset is None else offset.tolist()
        meta["factory"] = ["robot_reach.urdf", "arm_from_urdf", kwargs]
    solver_text = "numeric IK" if ik is None else "given IK"
    return Arm(
        name=name,
        joint_names=chain.joint_names,
        lower=tuple(float(v) for v in lower),
        upper=tuple(float(v) for v in upper),
        fk=chain.fk,
        ik=solver,
        task=task,
        skeleton=chain.skeleton,
        plane=chain.plane(),
        description=f"{model.name} from URDF, {chain.base_link} -> {tip_link}, {chain.dof} DOF, {solver_text}",
        meta=meta,
        meshes=arm_meshes,
        mesh_poses=mesh_poses,
    )


def arm_from_description(desc, tip_link: str, base_link: str | None = None, **kw) -> Arm:
    """`arm_from_urdf` for anything with ``to_urdf_string()``, such as a bam_robot_descriptions
    ``RobotDescription`` (``to_urdf_string(self, provenance: bool = True) -> str``).

    WHY duck-typed: bam_robot_descriptions is a sibling leaf, and leaves never import each other at
    runtime. A description *dataclass* (one with ``build()``) needs ``.build()`` first."""
    if not hasattr(desc, "to_urdf_string"):
        hint = "; call .build() first" if hasattr(desc, "build") else ""
        raise TypeError(f"{type(desc).__name__} has no to_urdf_string(){hint}")
    return arm_from_urdf(desc.to_urdf_string(), tip_link, base_link, **kw)


def chains(urdf_or_desc) -> list[tuple[str, str, int]]:
    """(base, tip, dof) for every leaf link with at least one movable joint above it: the
    kinematic chains a caller can offer. Chains through a floating or planar joint are skipped."""
    model = load_urdf(urdf_or_desc)
    out = []
    for leaf in model.leaf_links():
        try:
            dof = model.chain(leaf).dof
        except ValueError:
            continue
        if dof >= 1:
            out.append((model.root, leaf, dof))
    return out


def analyze_chains(source, tips=None, *, floor_z: float | None = None, quick: bool = True, arm_kw: dict | None = None,
                   **analyze_kw) -> dict:
    """Reachability of every kinematic chain of a robot (Zach's 2026-11-19 pipeline, step 5: "with
    1 click I can see the reachability map and the workspace of the different kinematic chains").

    `source` is a URDF path, URDF XML, or a description with ``to_urdf_string()``. Returns
    ``{"<tip> (<dof> DOF)": ReachResult}``, ready for ``ReachViewer(results).run()``.

    WHY `quick` (default on): the generic numeric IK costs ~0.4 ms per target x 8 seeds, so the full
    132-orientation map at the default grid would take minutes per 6-DOF chain. `quick` uses 48
    orientations (60 deg tilt rings x 4 rolls) and a grid step of extent/12 — about half a minute
    for a 6-DOF chain — and anything passed explicitly wins. `arm_kw` goes to `arm_from_urdf` for
    every chain (``{"meshes": True}`` to draw the URDF's meshes in the viewer)."""
    from .analyze import analyze
    from .orientations import OrientationSet

    found = chains(source)
    if tips is not None:
        wanted = set(tips)
        found = [c for c in found if c[1] in wanted]
        missing = wanted - {c[1] for c in found}
        if missing:
            raise ValueError(f"no movable chain ends at {sorted(missing)}; chains: {[c[1] for c in chains(source)]}")
    results = {}
    for base, tip, dof in found:
        akw = dict(arm_kw or {})
        if hasattr(source, "to_urdf_string"):
            arm = arm_from_description(source, tip, base, **akw)
        else:
            arm = arm_from_urdf(source, tip, base, **akw)
        kw = dict(analyze_kw)
        if quick:
            if arm.task != "position":
                kw.setdefault("orientations", OrientationSet.sample(tilt_step_deg=60.0, n_rolls=4))
            if "step" not in kw and "box" not in kw:
                kw["step"] = _quick_step(arm)
        results[f"{tip} ({dof} DOF)"] = analyze(arm, floor_z=floor_z, **kw)
    return results


def _quick_step(arm: Arm) -> float:
    from .sampling import Box, fk_cloud

    pts = fk_cloud(arm, 4000).points
    box = Box.around(pts)
    flat = sum(1 for s in box.size if s == 0)
    return float(max(box.size)) / (30 if flat else 12)
