# Week 4 — Uncertainty Scheduler (full U) + Mid-Term

**Owner:** Rohan · **Status:** built + tested (15/15 pass) on real EuRoC MH_01_easy.

This is the headline week. The metric *U* graduated from the Week-1 starter
(`sigma_pos + feature_loss`) to the **full 5-cue U**, with cue bounds and weights
**tuned to the real MH_01 signal ranges** — not guessed.

---

## What changed in the code

| File | Change |
|---|---|
| `core/uncertainty_scheduler.py` | `CueBounds` retuned to real MH_01 ranges (blur `150–300`, imu_bias `0–0.15`); default `weights` now the full 5-cue set (sums to 1.0). |
| `demo_week4.py` | **New.** Policy comparison + ablation + glass-box figure. |
| `tests/test_rohan_modules.py` | +4 tests: full-5-cue default, and one per newly-activated cue (σ_head, blur, imu_bias). |

Run it:
```powershell
D:\drone\.venv\Scripts\python.exe demo_week4.py --euroc "D:\drone\datasets\MH_01_easy"   # real data
D:\drone\.venv\Scripts\python.exe demo_week4.py                                            # synthetic
D:\drone\.venv\Scripts\python.exe -m pytest -q
```

---

## The numbers (your mid-term result)

**Full U = 0.45·σ_pos + 0.20·σ_head + 0.20·blur + 0.10·feature_loss + 0.05·imu_bias**
(weights set by how discriminative each cue measured on real MH_01.)

**Real EuRoC MH_01_easy** (drift 1.98 % of distance — a realistic VIO figure):

| policy | corrections | final err | peak err |
|---|---|---|---|
| none | 0 | 1.42 m | 1.48 m |
| fixed | 11 | 0.12 m | 0.26 m |
| adaptive | 11 | 0.12 m | 0.26 m |

**Synthetic (structured easy/hard) — where frugality shows:**

| policy | corrections | final err |
|---|---|---|
| none | 0 | 1.63 m |
| fixed | 10 | 0.05 m |
| **adaptive** | **2** | **0.12 m** |

Ablation (adaptive, same τ): covariance-only = 5 corrections; **multi-cue = 2** — at near-equal accuracy.

---

## How to present this honestly (this is what makes it strong)

1. **Lead with the mechanism + glass-box figure.** Panel 3 of `demo_week4.png`
   decomposes U(t) into per-cue contributions — you can literally see which signal
   drives each correction. That interpretability *is* the contribution.

2. **Be upfront that MH_01 is easy.** On it, every correcting policy ties
   (11 fixes, 0.12 m) and the ablation ties too — because the sequence is gentle
   enough that the covariance hard-floor alone catches the trouble. **Say this.**
   It's the honest reason the frugality gap shows on the structured/synthetic run,
   not on easy real data. The Week-6 evaluation across A/B/C × difficulties is
   exactly where the multi-cue advantage gets measured properly.

3. **The imu_bias finding.** On MH_01 the ground-truth gyro bias is near-constant
   (~0.10), so imu_bias can't *discriminate* on this sequence — it gets the smallest
   weight (0.05) and earns its keep only on sequences with real bias drift. Reporting
   this is the kind of thing that makes an advisor trust the rest of your numbers.

4. **Don't tune the weights to manufacture a bigger gap.** "2 vs 10" honestly is
   better than a tuned "1 vs 30."

### Likely questions
- *"Why these weights?"* → set by measured signal spread on real MH_01 (p5–p95);
  cues that don't vary get little weight. Not hand-picked to win.
- *"Did adding cues actually help?"* → on easy data it ties covariance-only (honest);
  the value is in adversarial vision (blur/feature-loss spikes that covariance lags),
  which Week 6 tests across difficulties.
- *"How do you pick τ?"* → it's swept on an accuracy-vs-corrections curve in Week 6;
  the covariance hard-floor bounds worst-case regardless of τ.

---

## Mid-term deliverable checklist (from the plan)
- [x] drift plot (none) + drift-with-correction (fixed/adaptive) — `demo_week4.png`, `demo_week2.png`
- [x] uncertainty-triggered vs fixed-period vs none — policy table above
- [x] full multi-cue U implemented + tuned to real data + tested
- [ ] architecture diagram — *one slide; ask Parth for the locked module diagram from §1 of the plan*
- [ ] Weeks 5–8 plan — already in the plan doc; restate the 4 bullets on a slide

---

## Next (Week 5, also yours)
Optical-flow obstacle module (time-to-contact → evasion vector) + the ~100-line
kinematic sim for the control merge. Zero teammate dependency — pure phone-video/sim.
Coordinate only on: the `VelocityCmd` evasion-vector interface with Parth's controller.
