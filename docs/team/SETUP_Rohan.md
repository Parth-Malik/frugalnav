# Rohan — Laptop Setup & First Tasks (FrugalNav)

You own: **the uncertainty metric U**, **the kinematic simulator**, **the optical-flow
obstacle module**, the **report**, and the **RISC-V study**. Your Week-1 deliverable is a
phone IMU+camera clip with timestamps that line up. None of your work needs a GPU — after we
dropped Gazebo, your laptop is the team's **reference machine**: if it runs for you, it runs
for everyone.

This folder already contains working, tested starter code for your two biggest pieces. Set up
the environment, run them, then do your Week-1 recording task.

---

## 1. Install (15 minutes, one time)

You need **Python 3.10+** and nothing exotic. Works on Windows, macOS, or Linux — no dual-boot,
no Ubuntu, no ROS.

```bash
# 1. check Python (need 3.10+). On Windows use 'py -3' instead of 'python3'.
python3 --version

# 2. make a project virtual environment and activate it
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. install the whole stack (CPU-only)
pip install -r requirements.txt
```

That's it. **Do NOT install** ROS, Gazebo, CUDA, PyTorch, or TensorFlow — nothing in this
project uses them, and they're the things that would actually punish a no-GPU laptop.

**Editor:** VS Code + the Python extension (free). **Version control:** `git`; clone the team
repo and put these files in it (layout below).

---

## 2. What's in this starter pack and how to run it

```
frugalnav/
├── core/
│   └── uncertainty_scheduler.py   # YOUR contribution: the U metric + trigger logic
├── harness/
│   ├── kinematic_sim.py           # YOUR testbed: drone + drift + markers + obstacles
│   └── check_sync.py              # YOUR Week-1 tool: phone timestamp checker
├── tests/
│   └── test_rohan_modules.py      # unit tests for your two modules
├── demo_week1.py                  # ties sim + U together -> demo_week1.png
├── requirements.txt
└── SETUP_Rohan.md                 # this file
```

Run each to confirm your environment works:

```bash
python3 core/uncertainty_scheduler.py     # prints 5 trigger-logic checks, all pass
python3 harness/kinematic_sim.py          # prints drift with/without markers
python3 harness/check_sync.py --selftest  # shows what a clean recording report looks like
python3 demo_week1.py                      # prints the policy table + writes demo_week1.png
pytest -q                                  # runs all 8 unit tests
```

`demo_week1.py` already reproduces the project's headline result in miniature:

```
policy     corrections  target_miss(m)  peak_err(m)  mean_err(m)
none                 0          4.11           4.12         2.72   <- homes on bad estimate
fixed               19          1.00           0.36         0.12
adaptive            13          0.99           0.52         0.22   <- same accuracy, 1/3 fewer fixes
```

That table + `demo_week1.png` is a strong thing to show at the first advisor meeting: it proves
the contribution works before the team's VIO is even wired in.

---

## 3. Your Week-1 task: record a phone clip and check the timestamps

**Pick a logger app** (any one — Android has more options):
- **Sensor Logger** (iOS + Android) — easiest; exports CSVs on a shared clock. Good first choice.
- **MARS logger** (OSU, Android + iOS) — SLAM-oriented; records focal length in pixels (useful later).
- **OpenCamera Sensors** (Skoltech, Android) — robotics-grade synced video+IMU.

**Record correctly** (these three settings prevent the data problems the checker flags):
1. **Lock the exposure** before recording — auto-exposure makes the frame rate wobble.
2. **Disable video stabilization (OIS/DVS)** — it changes the camera intrinsics every frame.
3. **Disable audio** if the app allows — it can add untimed frames.
Then: hold the phone, walk a slow ~20–30 s path (a hallway loop is perfect), keep it textured
(avoid blank walls), and export both the **IMU CSV** and the **video** (or a per-frame
timestamp CSV — always prefer that if the app offers it).

**Check it:**
```bash
python3 harness/check_sync.py --imu recordings/imu.csv --frames recordings/frames.csv
# or, if you only have a video file:
python3 harness/check_sync.py --imu recordings/imu.csv --video recordings/clip.mp4
```
You want: IMU ≈ 100–500 Hz, camera ≈ 30 fps, **>95% overlap**, low jitter, and the `[OK]` line.
If it warns about variable frame rate, re-record with exposure locked. Commit the report output
and the clip as your Week-1 deliverable.

---

## 4. What to do, in order (your next ~2 weeks)

1. **Now:** set up the venv, run all five commands above, confirm the demo figure appears.
2. **Sync with Parth on the interface names** before extending anything — the team freezes the
   data contracts (`SensorInput`, `LandmarkFix`, `VelocityCmd`) in Week 1. My module signatures
   are close; rename to match whatever the team locks so integration is painless.
3. **Week 1 deliverable:** record the phone clip, run `check_sync.py`, commit the report.
4. **Get ahead on U (your Week-4 headline):** the scheduler currently uses only `sigma_pos` +
   `feature_loss` (as the plan says to start). Add the `blur`, `sigma_head`, and `imu_bias`
   terms by giving them weight in `SchedulerConfig.weights` (keep the sum = 1), and add a unit
   test for each. Tune the bounds in `CueBounds` against logged sim runs.
5. **Your kinematic sim is the team's testbed** — tell Siddharth and Parth it exists so they can
   develop the corrector and controller against it now, instead of waiting for the VIO.

---

## 5. Things that will bite you (and the fix)

- **Forgetting to activate the venv** → `ModuleNotFoundError`. Re-run `source .venv/bin/activate`.
- **OIS / auto-exposure on the phone** → wobbly frame rate; the checker will warn. Lock both.
- **Timestamp units** → phones often log nanoseconds; `check_sync.py` auto-detects, but if a
  report looks 1000× too long/short, check the IMU CSV's time column by hand.
- **Tuning U to look too good** → don't. A result of "13 vs 19 corrections at equal accuracy" is
  more credible than "3 vs 30." Keep the demo honest; the advisor will respect it more.
- **Letting AI generate the U math wholesale** → generate the boilerplate, but keep the trigger
  logic and the normalization yours, with a unit test pinned to a number you computed by hand.
