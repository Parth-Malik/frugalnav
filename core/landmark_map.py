"""
The landmark / world-frame map.

Design decision from the plan (section 1, clarification 2): *the landmark map
defines the world frame.* Every marker pose is a surveyed constant in the world
frame, and the target B is a fixed point in that same frame. VIO gives relative
motion between fixes; markers re-anchor both the drone and B to the world frame,
so correcting drift never moves B.

Memory note (plan constraint 3): each marker stores only ID + pose (~24 B), NOT
image descriptors. ~800 markers ~= 20 KB, trivially inside GAP9's ~1.6 MB. This
class is the Python stand-in for that tiny, fixed-size table.

The map loads from a JSON config so a site can be re-surveyed with no code
change. JSON (not YAML) keeps the dependency footprint at zero -- stdlib only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from core.geometry import make_T, rot_z, rpy_to_R


@dataclass
class MarkerEntry:
    marker_id: int
    T_WM: np.ndarray                       # 4x4 pose of the marker in the world frame
    size_m: float                          # printed black-square edge length [m]


class LandmarkMap:
    """ID -> world pose, plus the shared camera intrinsics/extrinsic and target B."""

    def __init__(self, markers, target_B, K, dist, T_BC, default_size_m=0.10):
        self._markers = {int(m.marker_id): m for m in markers}
        self.target_B = np.asarray(target_B, dtype=float).reshape(2)
        self.K = np.asarray(K, dtype=float).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=float).reshape(-1)
        self.T_BC = np.asarray(T_BC, dtype=float).reshape(4, 4)   # camera in body
        self.default_size_m = float(default_size_m)

    # --- dict-like access ------------------------------------------------
    def __contains__(self, marker_id) -> bool:
        return int(marker_id) in self._markers

    def __len__(self) -> int:
        return len(self._markers)

    def get(self, marker_id) -> "MarkerEntry | None":
        return self._markers.get(int(marker_id))

    def ids(self):
        return list(self._markers.keys())

    def marker_world_xy(self, marker_id) -> np.ndarray:
        return self._markers[int(marker_id)].T_WM[:2, 3].copy()

    def all_world_xy(self) -> np.ndarray:
        return np.array([m.T_WM[:2, 3] for m in self._markers.values()])

    # --- loading ---------------------------------------------------------
    @staticmethod
    def from_json(path) -> "LandmarkMap":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        default_size = float(cfg.get("default_marker_size_m", 0.10))

        markers = []
        for m in cfg["markers"]:
            x = float(m["x"])
            y = float(m["y"])
            z = float(m.get("z", 0.0))
            yaw = np.deg2rad(float(m.get("yaw_deg", 0.0)))
            T_WM = make_T(rot_z(yaw), [x, y, z])
            markers.append(MarkerEntry(int(m["id"]), T_WM, float(m.get("size_m", default_size))))

        target_B = cfg["target_B_xy"]

        cam = cfg["camera"]
        K = [[cam["fx"], 0.0, cam["cx"]],
             [0.0, cam["fy"], cam["cy"]],
             [0.0, 0.0, 1.0]]
        dist = cam.get("dist", [0.0, 0.0, 0.0, 0.0, 0.0])

        ext = cfg.get("camera_extrinsic", {})
        rpy = ext.get("rpy_deg", [180.0, 0.0, 0.0])   # default: camera looks straight down
        t_BC = ext.get("xyz", [0.0, 0.0, 0.0])
        T_BC = make_T(rpy_to_R(*rpy), t_BC)

        return LandmarkMap(markers, target_B, K, dist, T_BC, default_size)
