"""
Image-blur term for the uncertainty metric U -- Siddharth's image-domain input.

Sharpness = variance of the Laplacian, the standard focus measure: a sharp image
has strong high-frequency content -> large Laplacian variance; a blurry one ->
small. We map sharpness to a [0, 1] "blur badness" that FEEDS U (the α₄·Blur term
in the plan). The point: a blurry frame makes ArUco corner/pose estimation noisy,
so U should rise and the scheduler should distrust -- or defer -- a fix taken
from that frame.

This is one of the few Week-4 modules that touches OpenCV, because it operates
on pixels. The scalar it produces is what crosses into the portable core.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:                          # keep importable without cv2
    cv2 = None


def _as_gray(img):
    if cv2 is None:
        raise RuntimeError("OpenCV not available; `pip install opencv-contrib-python`")
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def laplacian_sharpness(img) -> float:
    """Focus measure: variance of the Laplacian. Higher = sharper."""
    gray = _as_gray(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blur_badness(img, ref_sharpness: float) -> float:
    """Blur term in [0, 1] relative to a known-sharp reference.

    0.0  = as sharp as (or sharper than) the reference frame
    ->1.0 = badly blurred (little high-frequency content left)
    """
    s = laplacian_sharpness(img)
    return float(np.clip(1.0 - s / max(ref_sharpness, 1e-9), 0.0, 1.0))


def gaussian_blur(img, sigma: float):
    """Convenience: blur an image by a Gaussian of the given sigma (for tests /
    the blur demo -- simulates motion/defocus blur on a frame)."""
    if cv2 is None:
        raise RuntimeError("OpenCV not available")
    if sigma <= 0:
        return img.copy()
    return cv2.GaussianBlur(img, (0, 0), sigma)
