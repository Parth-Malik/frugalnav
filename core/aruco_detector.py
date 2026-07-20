"""
ArUco detection + pose  ->  MarkerSighting.

Siddharth's detector from Weeks 1-2, wrapped so its output feeds the landmark
corrector directly. This is the ONE core module that uses OpenCV -- detection is
inherently an image operation. Everything downstream of a MarkerSighting is pure
NumPy, so the portable hot loop stays OpenCV-free.

It reuses exactly the modern OpenCV API from the earlier lesson scripts
(`ArucoDetector` + `solvePnP` with `SOLVEPNP_IPPE_SQUARE`).
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
    import cv2.aruco as aruco
except ImportError:                          # keep the package importable w/o cv2
    cv2 = None
    aruco = None

from core.types import MarkerSighting


def marker_object_points(size_m: float) -> np.ndarray:
    """The 4 marker corners in the marker's own frame (Z=0), ArUco corner order:
    top-left, top-right, bottom-right, bottom-left."""
    s = size_m / 2.0
    return np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float32)


class ArucoDetector:
    """Detect mapped markers in a frame and estimate each one's camera-frame pose."""

    def __init__(self, landmark_map, dict_id=None):
        if cv2 is None:
            raise RuntimeError("OpenCV not available; `pip install opencv-contrib-python`")
        self.map = landmark_map
        if dict_id is None:
            dict_id = aruco.DICT_4X4_50
        self._dict = aruco.getPredefinedDictionary(dict_id)
        self._detector = aruco.ArucoDetector(self._dict, aruco.DetectorParameters())

    def detect(self, frame, timestamp: float = 0.0):
        """Run detection + PnP on one image. Returns a list of MarkerSighting.

        `frame` may be BGR or grayscale. Markers not present in the map are still
        returned (using the map's default marker size) so the caller can measure
        marker detection/success rate; the corrector will drop the unmapped ones.
        """
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self._detector.detectMarkers(gray)

        sightings = []
        if ids is None:
            return sightings

        K, dist = self.map.K, self.map.dist
        for marker_corners, marker_id in zip(corners_list, ids.flatten()):
            marker_id = int(marker_id)
            entry = self.map.get(marker_id)
            size_m = entry.size_m if entry is not None else self.map.default_size_m

            obj_pts = marker_object_points(size_m)
            img_pts = marker_corners.reshape(-1, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                continue

            reproj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            err = float(np.linalg.norm(reproj.reshape(-1, 2) - img_pts, axis=1).mean())
            sightings.append(
                MarkerSighting(marker_id, rvec.ravel(), tvec.ravel(), timestamp, err)
            )
        return sightings
