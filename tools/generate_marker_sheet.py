"""
Generate printable ArUco markers for every ID in the landmark map.

For real-world data collection: print these, lay them out, and measure/enter
their world (x, y) into config/landmark_map.json. Each PNG is labelled with its
ID so you don't mix them up. Uses the same DICT_4X4_50 as the rest of the repo.

Run:
    python tools/generate_marker_sheet.py
Output: outputs/markers/marker_<id>.png  (one per mapped marker)
"""
from __future__ import annotations

import os
import sys

import cv2
import cv2.aruco as aruco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.landmark_map import LandmarkMap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(HERE, "config", "landmark_map.json")
OUT_DIR = os.path.join(HERE, "outputs", "markers")

PIXELS = 600
QUIET = 90            # white border the detector needs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lmap = LandmarkMap.from_json(CONFIG)
    adict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    for mid in sorted(lmap.ids()):
        marker = aruco.generateImageMarker(adict, mid, PIXELS)
        img = cv2.copyMakeBorder(marker, QUIET, QUIET + 40, QUIET, QUIET,
                                 cv2.BORDER_CONSTANT, value=255)
        wx, wy = lmap.marker_world_xy(mid)
        label = f"id={mid}  world=({wx:.1f},{wy:.1f})m  size={lmap.get(mid).size_m}m"
        cv2.putText(img, label, (QUIET, img.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,), 2)
        path = os.path.join(OUT_DIR, f"marker_{mid}.png")
        cv2.imwrite(path, img)

    print(f"Wrote {len(lmap)} markers -> {os.path.relpath(OUT_DIR, HERE)}")
    print("Print them, place them, and record each world (x,y) in config/landmark_map.json.")
    print("IMPORTANT: after printing, MEASURE the black square edge and set size_m to match.")


if __name__ == "__main__":
    main()
