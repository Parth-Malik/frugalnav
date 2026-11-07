"""
camera_calibration.py
---------------------
Measures, with the real OpenCV ArUco detector, how camera degradation affects
(a) the Laplacian-variance sharpness cue that FrugalNav's scheduler reads,
(b) the probability that a marker is detected, (c) the pose error of the fix,
and (d) the processor time of one detection + solvePnP.

Images are synthetic: a 0.30 m DICT_4X4_50 marker on a textured ground plane,
seen by a 1280x720 downward camera (fx = 800 px, as in FrugalNav) from a given altitude. Each
image is degraded with lens blur, velocity-dependent motion blur, sensor noise
and an illumination gain, exactly the four degradations named in the proposal.

Output: results/camera_lut.npz (lookup table used by frugalnav_hf.py) and
results/detector_timing.json.
"""
import json
import os
import time

import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

W, H, F = 1280, 720, 800.0   # FrugalNav config/landmark_map.json camera
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], dtype=np.float64)
MARKER_M = 0.30
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DET = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())


def ground_texture(rng):
    """Mid-grey ground with multi-scale structure (gravel to boulders): octaves
    of smoothed noise at 2-32 px, so blur removes detail gradually, as on real
    terrain, instead of all at once."""
    img = np.full((H, W), 0.5)
    for scale, amp in ((2, 0.05), (4, 0.05), (8, 0.05), (16, 0.05), (32, 0.05)):
        n = rng.normal(0, 1, (H // scale + 1, W // scale + 1))
        img += amp * cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)[:H, :W]
    return np.clip(img, 0, 1)


def render(alt, rng, offset_px=(0, 0)):
    img = ground_texture(rng)
    side = int(round(F * MARKER_M / alt))
    mk = cv2.aruco.generateImageMarker(DICT, 7, 200).astype(np.float64) / 255.0
    mk = cv2.resize(mk, (side, side), interpolation=cv2.INTER_AREA)
    pad = max(2, side // 8)                     # white quiet zone around the tag
    tile = np.ones((side + 2 * pad, side + 2 * pad))
    tile[pad:pad + side, pad:pad + side] = mk
    cx, cy = W // 2 + offset_px[0], H // 2 + offset_px[1]
    t = tile.shape[0]
    x0, y0 = cx - t // 2, cy - t // 2
    img[y0:y0 + t, x0:x0 + t] = tile
    corners = np.array([[x0 + pad, y0 + pad], [x0 + pad + side, y0 + pad],
                        [x0 + pad + side, y0 + pad + side], [x0 + pad, y0 + pad + side]],
                       dtype=np.float64)
    return img, corners


def degrade(img, lens_sigma, motion_px, noise_sigma, gain, rng):
    out = img.copy()
    if lens_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), lens_sigma)
    L = int(round(motion_px))
    if L >= 2:
        k = np.zeros((L, L))
        k[L // 2, :] = 1.0 / L                   # horizontal streak (flight direction)
        out = cv2.filter2D(out, -1, k)
    out = out * gain
    if noise_sigma > 0:
        out = out + rng.normal(0, noise_sigma, out.shape)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def laplacian_var(u8):
    return float(cv2.Laplacian(u8, cv2.CV_64F).var())


def laplacian_var_nc(u8, noise_sigma):
    """Noise-compensated sharpness. The 3x3 Laplacian kernel has squared-weight
    sum 20, so i.i.d. sensor noise of std s (in DN) adds 20*s^2 to the raw
    Laplacian variance. Raw variance therefore reads noisy frames as SHARP; we
    subtract that floor (s is estimable on-board from flat image regions)."""
    return laplacian_var(u8) - 20.0 * (noise_sigma * 255.0) ** 2


def detect_and_pose(u8, alt):
    corners, ids, _ = DET.detectMarkers(u8)
    if ids is None or 7 not in ids.flatten():
        return False, np.nan
    i = list(ids.flatten()).index(7)
    obj = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float64) * MARKER_M / 2
    ok, rvec, tvec = cv2.solvePnP(obj, corners[i].reshape(4, 2).astype(np.float64), K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return False, np.nan
    # true marker centre is directly below the camera centre (offset 0) at depth alt
    err = float(np.linalg.norm(tvec.flatten()[:2]))
    return True, err


def main():
    rng = np.random.default_rng(2026)
    lens = [0.0, 1.0, 2.0, 3.0]
    motion = [0, 4, 8, 12, 16, 24]
    noise = [0.0, 0.03, 0.08]
    gain = [1.0, 0.6, 0.3]
    alts = [3.0, 5.0]
    reps = 10
    rows = []
    for alt in alts:
        for ls in lens:
            for mp in motion:
                for ns in noise:
                    for g in gain:
                        hits, errs, lv, lvnc = 0, [], [], []
                        for _ in range(reps):
                            img, _ = render(alt, rng)
                            u8 = degrade(img, ls, mp, ns, g, rng)
                            lv.append(laplacian_var(u8))
                            lvnc.append(laplacian_var_nc(u8, ns))
                            ok, e = detect_and_pose(u8, alt)
                            hits += ok
                            if ok:
                                errs.append(e)
                        rows.append((alt, ls, mp, ns, g, hits / reps,
                                     np.nanmean(errs) if errs else np.nan, np.mean(lv),
                                     np.mean(lvnc)))
    arr = np.array(rows, dtype=np.float64)
    np.savez(os.path.join(OUT, "camera_lut.npz"), rows=arr,
             cols=np.array(["alt", "lens", "motion_px", "noise", "gain", "p_det", "pose_err_m", "lapvar", "lapvar_nc"]))
    print(f"calibration grid: {len(rows)} cells x {reps} images")

    # Timing: detection + PnP on a clean 1280x720 frame, 300 repetitions.
    img, _ = render(5.0, rng)
    u8 = degrade(img, 0.5, 0, 0.01, 1.0, rng)
    for _ in range(20):
        detect_and_pose(u8, 5.0)
    ts = []
    for _ in range(300):
        t0 = time.perf_counter()
        detect_and_pose(u8, 5.0)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3
    timing = {"host": "x86-64 laptop, Python/OpenCV " + cv2.__version__,
              "frame": "1280x720 grey", "n": 300,
              "median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}
    with open(os.path.join(OUT, "detector_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(timing)

    # a few headline numbers
    def cell(alt, ls, mp, ns, g):
        m = (arr[:, 0] == alt) & (arr[:, 1] == ls) & (arr[:, 2] == mp) & (arr[:, 3] == ns) & (arr[:, 4] == g)
        return arr[m][0]
    for c in [(5, 0, 0, 0, 1), (5, 1, 8, 0.03, 1), (5, 2, 16, 0.03, 0.6), (3, 2, 24, 0.08, 0.3)]:
        r = cell(*c)
        print(c, "p_det=%.2f err=%.3f lapvar=%.0f nc=%.1f" % (r[5], r[6], r[7], r[8]))


if __name__ == "__main__":
    main()
