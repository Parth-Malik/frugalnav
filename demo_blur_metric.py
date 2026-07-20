"""
Week 4 -- the blur term of U, demonstrated on REAL images (Siddharth).

demo_week4.py used a *modeled* blur signal inside the nav sim. This script
exercises the actual `blur_metric` on pixels: it takes a small ArUco marker
(aerial markers are small in frame), applies increasing Gaussian blur
(simulating motion / defocus), and shows that

    sharpness (Laplacian variance) falls  ->  blur_badness rises toward 1
    ->  and ArUco detection eventually FAILS at extreme blur.

The useful finding: ArUco is robust, so blur_badness SATURATES well before the
detector gives up -- i.e. it is a leading indicator of degradation. That is
exactly why it belongs in U: down-weight or defer a fix from a blurry frame
*before* you lose the marker outright, rather than trusting a soft pose.

Run:
    python demo_blur_metric.py
Output: outputs/week4_blur.png  (+ console table)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import cv2.aruco as aruco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.blur_metric import blur_badness, gaussian_blur, laplacian_sharpness

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

SIGMAS = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0]


def make_marker_scene(marker_id=3, px=80, quiet=60):
    adict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    marker = aruco.generateImageMarker(adict, marker_id, px)
    return cv2.copyMakeBorder(marker, quiet, quiet, quiet, quiet,
                              cv2.BORDER_CONSTANT, value=255)


def detect_ok(gray):
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
                                   aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(gray)
    return ids is not None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sharp = make_marker_scene()
    ref = laplacian_sharpness(sharp)                 # reference = the crisp frame

    rows = []
    thumbs = []
    for sigma in SIGMAS:
        img = gaussian_blur(sharp, sigma)
        s = laplacian_sharpness(img)
        bad = blur_badness(img, ref)
        ok = detect_ok(img)
        rows.append((sigma, s, bad, ok))
        thumbs.append(img)

    print("=" * 60)
    print(" WEEK 4 -- BLUR TERM ON REAL IMAGES  (Siddharth)")
    print("=" * 60)
    print(f" reference sharpness (crisp frame): {ref:.0f}")
    print("-" * 60)
    print(f" {'sigma':>6} | {'sharpness':>10} | {'blur_badness':>12} | detect")
    print("-" * 60)
    for sigma, s, bad, ok in rows:
        print(f" {sigma:6.1f} | {s:10.0f} | {bad:12.2f} | {'ok' if ok else 'FAIL'}")
    print("=" * 60)

    # figure: thumbnails on top, curves below
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, len(SIGMAS), height_ratios=[1, 1.4])
    for i, (img, (sigma, s, bad, ok)) in enumerate(zip(thumbs, rows)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"σ={sigma}\n{'detect' if ok else 'FAIL'}",
                     fontsize=8, color=("green" if ok else "red"))
        ax.axis("off")

    axb = fig.add_subplot(gs[1, :])
    sig = [r[0] for r in rows]
    bad = [r[2] for r in rows]
    axb.plot(sig, bad, "-o", color="#1f77b4", lw=2, label="blur_badness (feeds U)")
    axb.axhline(1.0, color="gray", ls=":", lw=1)
    # shade where detection fails
    for sigma, s, b, ok in rows:
        if not ok:
            axb.axvspan(sigma - 0.4, sigma + 0.4, color="#ffcccc", alpha=0.5)
    axb.set_xlabel("Gaussian blur sigma (pixels)")
    axb.set_ylabel("blur_badness  [0..1]")
    axb.set_title("blur_badness rises with blur; red band = ArUco detection fails")
    axb.set_ylim(-0.05, 1.1)
    axb.legend(loc="lower right")
    axb.grid(True, alpha=0.3)

    fig.suptitle("Week 4 -- image blur -> uncertainty term (real OpenCV)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "week4_blur.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f" saved: outputs/week4_blur.png")


if __name__ == "__main__":
    main()
