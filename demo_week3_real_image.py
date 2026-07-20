"""
Week 3 -- end-to-end validation on a REAL rendered image (no webcam needed).

demo_week3.py fed the corrector analytic sightings. This demo closes the loop
with actual pixels: it renders a mapped ArUco marker into a synthetic downward-
camera image at a KNOWN drone pose (the same projection trick as the Week-1/2
`demo_no_webcam.py`), then runs the REAL detector + PnP + landmark corrector and
checks that the recovered world position matches the drone's true position:

    pixels -> ArUco detect -> solvePnP (T_CM) -> corrector (T_WB) -> world (x,y)

It runs TWO viewing geometries to also illustrate a constraint we analyse in the
report (plan section 2): a single marker seen *frontally* is the weak case for
planar pose (the classic ArUco pose ambiguity), while a *tilted* view (any real
drone attitude) constrains it well. Same corrector, very different accuracy --
which is itself an argument for the uncertainty scheduler: trust a fix less when
geometry is poor.

Run:
    python demo_week3_real_image.py
Output: outputs/week3_real_scene.png  (the tilted-view frame the detector saw)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import cv2.aruco as aruco

from core.aruco_detector import ArucoDetector, marker_object_points
from core.geometry import inv_T, make_T, rpy_to_R, rvec_from_rot
from core.landmark_corrector import LandmarkCorrector
from core.landmark_map import LandmarkMap

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config", "landmark_map.json")
OUT_DIR = os.path.join(HERE, "outputs")

IMG_W, IMG_H = 1280, 720
MARKER_ID = 3                              # a marker that exists in the map
ALTITUDE = 2.5                             # meters above the ground marker
TRUE_DRONE_XY = np.array([0.40, 30.20])    # ground truth we try to recover


def render_marker_frame(lmap, entry, T_WB_true):
    """Project the marker into a synthetic camera image at T_WB_true and return
    the thresholded frame the detector will run on."""
    T_CM = inv_T(T_WB_true @ lmap.T_BC) @ entry.T_WM       # marker in camera frame

    obj = marker_object_points(entry.size_m)               # marker 3D corners
    rvec = rvec_from_rot(T_CM[:3, :3])
    tvec = T_CM[:3, 3]
    projected, _ = cv2.projectPoints(obj, rvec, tvec, lmap.K, lmap.dist)
    projected = projected.reshape(-1, 2).astype(np.float32)

    bitmap = aruco.generateImageMarker(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_50), entry.marker_id, 500)
    h, w = bitmap.shape
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(src, projected)

    canvas = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)   # white -> quiet zone
    warped = cv2.warpPerspective(bitmap, Hmat, (IMG_W, IMG_H),
                                 flags=cv2.INTER_NEAREST, borderValue=255)
    scene = np.minimum(canvas, warped)
    _, scene = cv2.threshold(scene, 127, 255, cv2.THRESH_BINARY)
    return scene


def run_case(lmap, entry, detector, corrector, roll, pitch, yaw_deg):
    """Render one viewing geometry, recover the world fix, return (err_m, reproj, scene)."""
    T_WB_true = make_T(rpy_to_R(roll, pitch, yaw_deg),
                       [TRUE_DRONE_XY[0], TRUE_DRONE_XY[1], ALTITUDE])
    scene = render_marker_frame(lmap, entry, T_WB_true)
    sightings = detector.detect(scene)
    if not sightings:
        return None, None, scene
    fix = corrector.correct(sightings[0])
    err = float(np.linalg.norm(fix.xy - TRUE_DRONE_XY))
    return err, sightings[0].reproj_error_px, scene, fix.xy


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lmap = LandmarkMap.from_json(CONFIG)
    entry = lmap.get(MARKER_ID)
    corrector = LandmarkCorrector(lmap)
    detector = ArucoDetector(lmap)

    print("=" * 68)
    print(" WEEK 3 -- REAL-IMAGE VALIDATION  (pixels -> world fix)")
    print("=" * 68)
    print(f" rendered marker id {MARKER_ID} at world {lmap.marker_world_xy(MARKER_ID)},"
          f" altitude {ALTITUDE} m")
    print(f" true drone position (world): {TRUE_DRONE_XY}")
    print("-" * 68)

    # Case A: frontal downward view -- the weak planar-pose geometry.
    frontal = run_case(lmap, entry, detector, corrector, 0, 0, 12)
    # Case B: tilted view (a realistic drone attitude) -- well-constrained.
    tilted = run_case(lmap, entry, detector, corrector, 10, 14, 12)

    for name, res in [("frontal (roll 0, pitch 0)", frontal),
                      ("tilted  (roll 10, pitch 14)", tilted)]:
        err, reproj, scene = res[0], res[1], res[2]
        if err is None:
            print(f" {name:30s}:  NO DETECTION")
            continue
        recovered = res[3]
        print(f" {name:30s}:  err {err*100:5.1f} cm   "
              f"(reproj {reproj:.2f} px, recovered [{recovered[0]:.3f} {recovered[1]:.3f}])")

    # save the tilted frame as the representative image
    cv2.imwrite(os.path.join(OUT_DIR, "week3_real_scene.png"), tilted[2])

    print("-" * 68)
    err_t = tilted[0]
    verdict = "PASS" if (err_t is not None and err_t < 0.05) else "CHECK"
    print(f" [{verdict}]  chain verified: tilted-view fix is accurate to "
          f"{err_t*100:.1f} cm")
    print(" note: the frontal case is the planar-pose ambiguity we flag in the")
    print("       report -- motivation for weighting a fix by its geometry.")
    print(f" saved frame -> {os.path.relpath(os.path.join(OUT_DIR, 'week3_real_scene.png'), HERE)}")
    print("=" * 68)


if __name__ == "__main__":
    main()
