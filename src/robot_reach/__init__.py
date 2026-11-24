"""robot_reach — BAM Reach v2.

    import robot_reach as rr
    from robot_reach.robots import ur5e

    result = rr.analyze(ur5e())        # compute once
    result.save("out/ur5e")            # the result log
    rr.load("out/ur5e").display()      # view it in viser, any time later
    rr.load("out/ur5e").workspace().dexterous.sample(3, rng)   # checked poses downstream
"""

from .analyze import ReachResult, analyze, load
from .arm import Arm, ArmMesh
from .codes import CODE_COLORS, CODE_LABELS, LEVEL_NAMES, Code
from .joint_space import JointMap, JointRanges, joint_map, joint_ranges
from .orientations import OrientationSet
from .reach import ReachMap, reach_map
from .sampling import Box, Cylinder, FKCloud, fk_cloud
from .workspace import NearestPose, PoseSampler, PoseSamples, ReachableWorkspace, Regions, WorkspaceRegion

__all__ = [
    "Arm",
    "ArmMesh",
    "Box",
    "Code",
    "Cylinder",
    "NearestPose",
    "PoseSampler",
    "PoseSamples",
    "ReachableWorkspace",
    "Regions",
    "WorkspaceRegion",
    "CODE_COLORS",
    "CODE_LABELS",
    "FKCloud",
    "JointMap",
    "JointRanges",
    "LEVEL_NAMES",
    "OrientationSet",
    "ReachMap",
    "fk_cloud",
    "joint_map",
    "joint_ranges",
    "reach_map",
    "ReachResult",
    "analyze",
    "load",
]
