#!/usr/bin/env python3
"""
Generate real ArUco marker textures + a Gazebo material script, so the drone's
camera sees genuine detectable tags on the ground (DICT_4X4_50, ids 0..N-1).

    python tools/gen_aruco_media.py
writes:
    media/materials/textures/aruco<id>.png
    media/materials/scripts/aruco.material   (materials aruco/0 .. aruco/N-1)
"""
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
N = 24
PX = 600
BORDER = 80   # white quiet zone (ArUco needs it to detect)


def make_marker(dic, i):
    if hasattr(cv2.aruco, "generateImageMarker"):
        m = cv2.aruco.generateImageMarker(dic, i, PX)
    else:
        m = cv2.aruco.drawMarker(dic, i, PX)
    canvas = np.full((PX + 2 * BORDER, PX + 2 * BORDER), 255, np.uint8)
    canvas[BORDER:BORDER + PX, BORDER:BORDER + PX] = m
    return canvas


def main():
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tex = os.path.join(PKG, "media", "materials", "textures")
    scr = os.path.join(PKG, "media", "materials", "scripts")
    os.makedirs(tex, exist_ok=True)
    os.makedirs(scr, exist_ok=True)
    blocks = []
    for i in range(N):
        cv2.imwrite(os.path.join(tex, f"aruco{i}.png"), make_marker(dic, i))
        blocks.append(f"""material aruco/{i}
{{
  technique
  {{
    pass
    {{
      texture_unit
      {{
        texture aruco{i}.png
        filtering none
      }}
    }}
  }}
}}""")
    with open(os.path.join(scr, "aruco.material"), "w") as f:
        f.write("\n".join(blocks) + "\n")
    print(f"wrote {N} ArUco textures + aruco.material")


if __name__ == "__main__":
    main()
