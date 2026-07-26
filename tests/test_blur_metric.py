"""
Unit tests for core/blur_metric.py (uses OpenCV).
    python tests/test_blur_metric.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import cv2.aruco as aruco

from core.blur_metric import blur_badness, gaussian_blur, laplacian_sharpness


def _marker_scene():
    adict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # generateImageMarker is OpenCV >= 4.7; older builds (e.g. 4.5) use drawMarker
    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(adict, 0, 300)
    else:
        marker = aruco.drawMarker(adict, 0, 300)
    return cv2.copyMakeBorder(marker, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255)


def test_blur_lowers_sharpness():
    sharp = _marker_scene()
    s0 = laplacian_sharpness(sharp)
    s_blur = laplacian_sharpness(gaussian_blur(sharp, 4.0))
    assert s0 > s_blur > 0


def test_badness_range_and_monotonic():
    sharp = _marker_scene()
    ref = laplacian_sharpness(sharp)
    prev = -1.0
    for sigma in [0.0, 1.0, 2.0, 4.0, 8.0]:
        bad = blur_badness(gaussian_blur(sharp, sigma), ref)
        assert 0.0 <= bad <= 1.0
        assert bad >= prev - 1e-9          # non-decreasing with more blur
        prev = bad


def test_reference_frame_is_not_blurry():
    sharp = _marker_scene()
    ref = laplacian_sharpness(sharp)
    assert blur_badness(sharp, ref) < 1e-6     # a frame vs itself: badness ~ 0


def test_heavy_blur_is_bad():
    sharp = _marker_scene()
    ref = laplacian_sharpness(sharp)
    assert blur_badness(gaussian_blur(sharp, 10.0), ref) > 0.8


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
