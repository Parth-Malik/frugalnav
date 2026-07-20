"""
harness/check_sync.py
---------------------
Rohan's Week-1 deliverable: confirm a phone recording's IMU and camera timestamps
line up well enough to feed a VIO later.

Phone logger apps export an IMU stream (~100-500 Hz) and a video (~30 Hz), each with
timestamps on a shared clock. This script reads both, reports their rates, durations,
overlap, and jitter, and FLAGS the two things that quietly ruin VIO data:
  - variable frame rate (caused by auto-exposure / OIS) -> lock exposure, disable OIS
  - a clock offset / poor overlap between the IMU and video streams

Usage:
  python3 harness/check_sync.py --imu recordings/imu.csv --video recordings/video.mp4
  python3 harness/check_sync.py --imu recordings/imu.csv --frames recordings/frames.csv
  python3 harness/check_sync.py --selftest        # runs on generated fake data

The IMU CSV needs a time column (any of: time, timestamp, t, seconds_elapsed, ns, sec).
Units are auto-detected from magnitude (ns / us / ms / s).
"""
import argparse, csv, math, os, sys


def _detect_time_unit(values):
    """Guess the multiplier to convert raw timestamps to SECONDS, from magnitude."""
    span = max(values) - min(values)
    if span == 0:
        return 1.0, "unknown"
    # A recording is seconds-to-minutes long. Pick the unit that makes the span sane.
    for mult, name in [(1e-9, "ns"), (1e-6, "us"), (1e-3, "ms"), (1.0, "s")]:
        if 0.1 <= span * mult <= 7200:        # between 0.1 s and 2 hours
            return mult, name
    return 1.0, "s(assumed)"


def _read_time_column(path):
    """Read the first plausible time column from a CSV; return list of floats in raw units."""
    candidates = ["time", "timestamp", "t", "seconds_elapsed", "ns",
                  "sec", "time_ns", "host_time", "sensor_time"]
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header row found")
        lower = {c.lower(): c for c in reader.fieldnames}
        col = next((lower[c] for c in candidates if c in lower), None)
        if col is None:                        # fall back to the first numeric column
            col = reader.fieldnames[0]
        out = []
        for row in reader:
            try:
                out.append(float(row[col]))
            except (ValueError, TypeError):
                pass
    if not out:
        raise ValueError(f"{path}: could not parse any numbers from column '{col}'")
    return out, col


def _stats(times_s):
    """Rate stats from a sorted list of timestamps in seconds."""
    times_s = sorted(times_s)
    dur = times_s[-1] - times_s[0]
    n = len(times_s)
    dts = [t2 - t1 for t1, t2 in zip(times_s, times_s[1:]) if t2 > t1]
    mean_dt = sum(dts) / len(dts) if dts else float("nan")
    rate = 1.0 / mean_dt if mean_dt else float("nan")
    # jitter: std of dt
    if dts:
        var = sum((d - mean_dt) ** 2 for d in dts) / len(dts)
        jitter = math.sqrt(var)
        max_gap = max(dts)
    else:
        jitter = max_gap = float("nan")
    return dict(n=n, start=times_s[0], end=times_s[-1], dur=dur,
                rate=rate, mean_dt=mean_dt, jitter=jitter, max_gap=max_gap)


def _video_times(path):
    """Frame timestamps (s) from a video file using OpenCV. Returns list."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # OpenCV doesn't expose per-frame capture timestamps reliably, so we model an even
    # cadence from fps+count. (For TRUE per-frame stamps, export the app's frames CSV
    # and use --frames instead -- always preferred for VIO.)
    return [i / fps for i in range(max(n, 1))], fps


def report(imu_times_raw, cam_times_raw, cam_source):
    mult_i, unit_i = _detect_time_unit(imu_times_raw)
    imu = _stats([t * mult_i for t in imu_times_raw])

    print("=" * 60)
    print("IMU stream")
    print(f"  samples      : {imu['n']}")
    print(f"  time unit    : detected '{unit_i}'")
    print(f"  duration     : {imu['dur']:.2f} s")
    print(f"  rate         : {imu['rate']:.1f} Hz  (dt={imu['mean_dt']*1e3:.2f} ms)")
    print(f"  jitter (std) : {imu['jitter']*1e3:.2f} ms   max gap: {imu['max_gap']*1e3:.1f} ms")

    if cam_times_raw is not None:
        if cam_source == "frames_csv":
            mult_c, unit_c = _detect_time_unit(cam_times_raw)
            cam = _stats([t * mult_c for t in cam_times_raw])
        else:                                  # already seconds from video fps model
            cam = _stats(cam_times_raw)
            unit_c = "s(from fps)"
        print("Camera stream")
        print(f"  frames       : {cam['n']}")
        print(f"  source       : {cam_source}   unit: {unit_c}")
        print(f"  duration     : {cam['dur']:.2f} s")
        print(f"  rate         : {cam['rate']:.1f} fps  (dt={cam['mean_dt']*1e3:.2f} ms)")
        print(f"  jitter (std) : {cam['jitter']*1e3:.2f} ms   max gap: {cam['max_gap']*1e3:.1f} ms")

        # ---- verdicts ----
        print("-" * 60)
        overlap_start = max(imu["start"], cam["start"])
        overlap_end = min(imu["end"], cam["end"])
        overlap = max(0.0, overlap_end - overlap_start)
        union = max(imu["end"], cam["end"]) - min(imu["start"], cam["start"])
        print(f"  stream overlap : {overlap:.2f} s of {union:.2f} s "
              f"({100*overlap/union:.0f}%)")

        ok = True
        if overlap / union < 0.95:
            print("  [WARN] streams do not overlap well -> likely a clock-offset problem.")
            ok = False
        if cam["jitter"] * cam["rate"] > 0.15:     # jitter > 15% of a frame interval
            print("  [WARN] variable frame rate -> LOCK EXPOSURE and DISABLE OIS, re-record.")
            ok = False
        if imu["rate"] < 90:
            print(f"  [WARN] IMU rate {imu['rate']:.0f} Hz is low for VIO (want >=100 Hz).")
            ok = False
        if ok:
            print("  [OK] timestamps look consistent and well-aligned. Good to feed a VIO.")
    print("=" * 60)


def _selftest():
    """Generate fake IMU + frame timestamps (a clean recording) and check them."""
    import tempfile
    d = tempfile.mkdtemp()
    imu_p = os.path.join(d, "imu.csv")
    frm_p = os.path.join(d, "frames.csv")
    t0_ns = 1_700_000_000_000_000_000          # a realistic nanosecond epoch
    with open(imu_p, "w", newline="") as f:    # 200 Hz IMU for 5 s, in ns
        w = csv.writer(f); w.writerow(["time", "gx", "gy", "gz"])
        for i in range(1000):
            w.writerow([t0_ns + int(i * 5e6), 0.0, 0.0, 0.0])
    with open(frm_p, "w", newline="") as f:    # 30 fps frames for 5 s, in ns
        w = csv.writer(f); w.writerow(["timestamp"])
        for i in range(150):
            w.writerow([t0_ns + int(i * (1e9 / 30))])
    imu_t, _ = _read_time_column(imu_p)
    cam_t, _ = _read_time_column(frm_p)
    report(imu_t, cam_t, "frames_csv")
    print("\nSelf-test complete: this is what a CLEAN recording report looks like.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu")
    ap.add_argument("--video")
    ap.add_argument("--frames")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest(); return
    if not a.imu:
        ap.error("provide --imu (and --video or --frames), or use --selftest")

    imu_t, _ = _read_time_column(a.imu)
    if a.frames:
        cam_t, _ = _read_time_column(a.frames); src = "frames_csv"
    elif a.video:
        cam_t, fps = _video_times(a.video); src = f"video@{fps:.1f}fps"
    else:
        cam_t, src = None, None
    report(imu_t, cam_t, src)


if __name__ == "__main__":
    main()
