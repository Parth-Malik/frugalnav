# Rohan — Week 2 (FrugalNav)

**Week-2 goal (team):** a VIO running with a drift plot (estimate vs ground truth
diverging) and the glass-box signals exposed to the scheduler.

**Your de-risked version:** instead of waiting on the fragile ORB-SLAM3/OpenVINS build
(that's Parth's risk), you produce the drift plot and the glass-box signals from a
**drift scaffold over the real EuRoC dataset**, behind a clean **adapter** so the real
VIO drops in later without touching your scheduler. This also calibrates the kinematic
sim against real data, which kills the "isn't your drift faked?" critique.

## New files this week
```
core/vio_adapter.py        # the SEAM: VioSource interface + VioSignals. Pure/portable.
harness/euroc_reader.py    # reads EuRoC IMU + ground-truth CSVs (no images needed)
harness/drift_scaffold.py  # fallback VIO: real GT -> drifting estimate + glass-box cues
demo_week2.py              # scheduler on the drift scaffold; 3 policies; demo_week2.png
```

## Run it now (no download needed)
```powershell
python demo_week2.py        # uses a built-in synthetic EuRoC-style trajectory
pytest -q                   # 11 tests pass
```
You'll get a calibration report (drift ≈ 2.6% of distance — a realistic VIO figure),
the none/fixed/adaptive table, and `demo_week2.png`.

## Run it on REAL EuRoC (this is the actual Week-2 deliverable)
1. Download one sequence, e.g. **MH_01_easy**, from the EuRoC MAV dataset page
   (search "EuRoC MAV dataset ETH ASL"). Grab the **ASL .zip** format. Unzip it; you'll
   get a `mav0/` folder. You only need the CSVs, so the download size isn't a blocker.
2. Point the demo at it:
   ```powershell
   python demo_week2.py --euroc "D:\drone\datasets\MH_01_easy"
   ```
3. Save the printed calibration numbers and `demo_week2.png` — that's your deliverable.

## Then: calibrate the sim against the real number
The calibration report prints `drift_pct_of_distance` for the real sequence. Open
`harness/kinematic_sim.py`, run it, and adjust `SimConfig.gyro_bias_walk` until the sim's
own drift % matches EuRoC's. Now your sim is grounded in real data — note this in the report.

## Your ordered tasks
1. Run `demo_week2.py` (synthetic) and confirm the figure + 11 passing tests.
2. Download MH_01_easy, run `--euroc`, capture the real calibration numbers + plot.
3. Tune the sim's `gyro_bias_walk` to match the real drift %.
4. Sync with Parth: the `VioSource` interface in `core/vio_adapter.py` is the slot his
   real VIO fills (`class OpenVinsSource(VioSource)`). Agree on it so integration is free.
5. Sync with Siddharth: your `fix_every_m` virtual landmarks are the stand-in for his real
   ArUco markers — when his corrector is ready, his `apply_fix` replaces the scaffold's.

## Gotchas
- **Don't fight the VIO build.** If Parth's OpenVINS/ORB-SLAM3 isn't ready, you are not
  blocked — the scaffold gives the scheduler identical inputs. That's the whole point of
  the adapter.
- **You don't need the EuRoC images** for any of your Week-2 work, just the two CSVs
  (`imu0/data.csv`, `state_groundtruth_estimate0/data.csv`).
- **Keep results honest.** On the smooth synthetic path adaptive uses ~2 fixes vs 10;
  on real, more dynamic EuRoC motion it'll trigger more often. Report the real numbers,
  not the synthetic ones.
