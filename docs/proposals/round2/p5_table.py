"""P5 · A table of IK solutions: one row per checked solution, filter it like a dataframe.

    t = solution_table(reach)                       # ~1M rows for the UR5e
    t[t.in_dexterous].sample(3, rng)
    b = t[(t.elbow == "up") & belt.contains(t.xyz) & t.o_hemisphere]
    b[b.complete("hemisphere")].sample_groups("configuration", n=3, k=2, rng=rng)
    t.to_pandas()                                   # when pandas is around; never required

robot_msgs: `SolutionTable` (manipulation/solution_table.py): struct-of-arrays columns + a small
header (chain, frame, joint names, voxel size, the orientation set, the configuration axes). The
columns are the vocabulary: x y z, position, orientation, tilt_deg, configuration and one column
per configuration axis (shoulder, elbow, wrist), q, and workspace flags. Several chains: a `chain`
column; "both arms reach it" is a group-by on the cell grid key.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import core
from core import RULE, ChainReach, PoseSamples


@dataclass(frozen=True, eq=False)
class SolutionTable:
    columns: dict  # name -> (K, ...) array
    header: dict  # chain-level facts: orientations (O, 3, 3), kind_orientations, labels, voxel_size, joint_names, frame

    def __len__(self) -> int:
        return len(self.columns["position"])

    def __getattr__(self, name: str) -> np.ndarray:
        cols = self.__dict__["columns"]
        if name in cols:
            return cols[name]
        raise AttributeError(f"{name!r}: columns are {list(cols)}")

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.columns[key]
        return SolutionTable({k: v[key] for k, v in self.columns.items()}, self.header)

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.columns["x"], self.columns["y"], self.columns["z"]], 1)

    def complete(self, kind: str) -> np.ndarray:
        """(K,) bool, recomputed on THESE rows: the row's grid point has every orientation `kind` asks for.
        WHY a method and not only the in_* columns: in_* were computed on the full table; after a filter
        (elbow up only) a voxel can lose an orientation, and a stale flag would overclaim."""
        need = self.header["kind_orientations"][kind]
        if RULE[kind] == "none":
            return np.ones(len(self), bool)
        key, o = self.columns["key"], self.columns["orientation"]
        uk, inv = np.unique(key, return_inverse=True)
        have = np.zeros((len(uk), len(need)), bool)
        have[inv, o] = True
        ok = have[:, need].any(1) if RULE[kind] == "any" else have[:, need].all(1)
        return ok[inv]

    def sample(self, n: int, rng: np.random.Generator) -> SolutionTable:
        """n rows, uniform over poses (a pose with 4 IK configurations is not 4x as likely)."""
        idx = core.draw(self.columns["pose_key"], self.columns["configuration_index"], n, 1, rng, False)[:, 0]
        return self[idx]

    def sample_groups(self, by: str, *, n: int, k: int, rng: np.random.Generator) -> SolutionTable:
        """n groups of k distinct poses sharing the value of column `by`; adds a `group` column (n*k rows)."""
        codes = np.unique(self.columns[by], return_inverse=True)[1]
        idx = core.draw(self.columns["pose_key"], codes, n, k, rng, True)
        out = self[idx.reshape(-1)]
        out.columns["group"] = np.repeat(np.arange(n), k)
        return out

    def counts(self, by: str) -> dict:
        v, c = np.unique(self.columns[by], return_counts=True)
        return {str(a): int(b) for a, b in zip(v, c)}

    def to_samples(self, k: int = 1) -> PoseSamples:
        T = np.zeros((len(self), 4, 4))
        T[:, :3, :3] = self.header["orientations"][self.columns["orientation"]]
        T[:, :3, 3] = self.xyz
        T[:, 3, 3] = 1
        shape = (len(self),) if k == 1 else (len(self) // k, k)
        return PoseSamples(self.header["chain"], T.reshape(*shape, 4, 4), self.columns["q"].astype(float).reshape(*shape, -1),
                           self.columns["configuration_index"].reshape(shape), self.header["labels"], self.header["base_xyz"])

    def head(self, n: int = 5, cols: tuple[str, ...] = ("x", "y", "z", "tilt_deg", "configuration")) -> str:
        """A plain-text peek at the first n rows (no pandas needed)."""
        rows = [" ".join(f"{c:>13}" for c in cols)]
        for i in range(min(n, len(self))):
            rows.append(" ".join(f"{self.columns[c][i]:>13.3f}" if np.issubdtype(self.columns[c].dtype, np.floating)
                                 else f"{self.columns[c][i]!s:>13}" for c in cols))
        return "\n".join(rows)

    def to_pandas(self):
        import pandas as pd  # optional: the table works without it

        flat = {k: v for k, v in self.columns.items() if v.ndim == 1}
        for j, name in enumerate(self.header["joint_names"]):
            flat[name] = self.columns["q"][:, j]
        return pd.DataFrame(flat)

    @staticmethod
    def concat(tables: list[SolutionTable]) -> SolutionTable:
        cols = {k: np.concatenate([t.columns[k] for t in tables]) for k in tables[0].columns}
        return SolutionTable(cols, {**tables[0].header, "chains": {t.header["chain"]: t.header for t in tables}})

    def shared_by(self, column: str = "chain") -> np.ndarray:
        """(K,) bool: the row's grid point appears under every value of `column` (every chain reaches it)."""
        vals = np.unique(self.columns[column])
        common = None
        for v in vals:
            k = np.unique(self.columns["key"][self.columns[column] == v])
            common = k if common is None else np.intersect1d(common, k)
        return np.isin(self.columns["key"], common)


def solution_table(reach: ChainReach) -> SolutionTable:
    """robot_reach's producer: one row per checked IK solution."""
    r = reach
    p, o = r.solution_pose // r.O, r.solution_pose % r.O
    xyz = r.positions[p]
    cfg = r.solution_configuration
    labels = np.array(r.configurations.labels)
    cols = {
        "chain": np.full(len(p), r.chain), "position": p, "orientation": o.astype(np.int16),
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "key": core.cell_keys(xyz, r.voxel_size),
        "pose_key": r.solution_pose, "tilt_deg": r.tilt_deg[o].astype(np.float32),
        "configuration_index": cfg, "configuration": labels[cfg],
    }
    for a, axis in enumerate(r.configurations.axes):  # one column per IK configuration question
        options = np.array([r.configurations.answers(i)[axis.name] for i in range(len(r.configurations))])
        cols[axis.name] = options[cfg]
    cols["q"] = r.solution_q
    for kind in ("top_down", "hemisphere"):  # this row's orientation is one the kind asks for
        cols[f"o_{kind}"] = r.kind_orientations[kind][o]
    header = {"chain": r.chain, "frame": r.frame, "joint_names": r.joint_names, "voxel_size": r.voxel_size, "base_xyz": r.base_xyz,
              "orientations": r.orientations, "kind_orientations": r.kind_orientations, "labels": r.configurations.labels}
    t = SolutionTable(cols, header)
    for kind in ("top_down", "hemisphere", "dexterous"):  # membership with every IK configuration, computed once
        cols[f"in_{kind}"] = t.complete(kind)
    return t
