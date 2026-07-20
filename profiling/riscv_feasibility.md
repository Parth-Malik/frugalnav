# RISC-V Feasibility Study (Week 7)

**Question.** Can the FrugalNav navigation core run on an ultra-low-power RISC-V
SoC, and *where* does the budget actually go? Target reference: **GreenWaves
GAP9** — a 9-core RISC-V (CV32E40P-class, in-order, single-issue, FP32) compute
cluster plus a fabric-controller core, up to **~370 MHz**, **~1.6 MB** on-chip L2
SRAM, in the **sub-200 mW** envelope. (We use GAP9 as a concrete stand-in for the
class; the argument is not GAP9-specific.)

Every number below is tagged **[MEASURED]** (from the compiled C++ benchmark
`cpp/main.cpp` or the Python profiler `profile_core.py`, both on an x86 dev host)
or **[MODELED]** (an operation-count / clock-derating projection to GAP9). We do
not present a modeled number as if it were measured.

---

## 1. Method

The system splits cleanly into two tiers (this split is the whole point of the
`core/` vs `harness/` layout):

* **Front-ends** — VIO (OpenVINS/ORB-SLAM3) and the ArUco detector — turn pixels
  and IMU into small structs. Compute-bound; these dominate the frame budget.
* **The decision hot loop** (`core/navigator.py`, ported in `cpp/`) — scheduler,
  state fusion, controller, obstacle-avoidance policy. Scalar / 2×2 math; runs
  every frame but is essentially free.

We profile each module in isolation, then project to GAP9 by counting the scalar
FP operations and dividing by the target clock with a conservative **IPC = 0.7**
derating (in-order single-issue with load-use and FP latency stalls).

---

## 2. Measured latencies

### Decision hot loop — C++ port, `-O2`, x86 host  **[MEASURED]**

| Path | ns / step |
|---|---:|
| uncertainty scheduler only | **77** |
| full decision hot loop (predict→schedule→fuse→avoid→command) | **93** |

### Per-module relative cost — Python + NumPy, x86 host  **[MEASURED]**

| Module | µs / call | Notes |
|---|---:|---|
| uncertainty scheduler | 5.7 | ~40 scalar FP ops (the contribution) |
| target-centric controller | 15.8 | vector + speed clamp |
| obstacle avoidance (TTC) | 18.0 | hysteresis + perpendicular evasion |
| state fusion (2×2 EKF) | 52.6 | predict + Kalman update, 2×2 inverse |
| landmark corrector | 68.3 | one 4×4 chain + 2×2, **per sighting only** |
| **full `navigator.step()`** | **94.0** | the whole decision hot loop |

The Python numbers are dominated by per-call NumPy dispatch overhead on tiny
arrays; they are used only for the **relative** breakdown. The **absolute** hot-loop
cost is the C++ number (~93 ns), a ~1000× gap that is exactly why the C++/RISC-V
port matters.

---

## 3. Modeled projection to GAP9

**Scheduler.** ~40 scalar FP ops/call. At 370 MHz, IPC 0.7 → ~40 / (370e6·0.7) ≈
**0.15 µs/call** **[MODELED]**. Even at a conservative 250 MHz fabric clock, ≈
**0.23 µs/call**. Against a 30 Hz frame (33 ms) or a 60 Hz frame (16.7 ms), the
scheduler is **~0.001 %** of the budget.

**Full decision hot loop.** ~93 ns on the x86 host; the host runs ~3–4 ops/ns, so
≈ 300 scalar ops/step. On GAP9: ~300 / (370e6·0.7) ≈ **1.2 µs/step** **[MODELED]**.
Still **< 0.01 %** of a 30 Hz frame.

**Front-ends (the real cost).** From the published literature, not measured here:
ArUco marker detection + PnP on a small frame is **~1–5 ms** on an embedded
Cortex-class core **[MODELED, literature]**; a sparse VIO front-end (MSCKF-style)
update is **~2–8 ms/frame** on comparable hardware **[MODELED, literature]**. These
are what set the achievable frame rate, and both are event-gated by the scheduler
(ArUco/PnP only runs when a fix is requested), so **fewer scheduler fires directly
buys back front-end compute** — the frugality result in Chapter *Evaluation* is a
compute/power result, not just an accuracy one.

---

## 4. Memory budget  **[MODELED]**

| Item | Size | Note |
|---|---:|---|
| Landmark map (ID + 4×4 pose + size) | ~24 B / marker → **~20 KB / 800 markers** | plan constraint 3; no image descriptors |
| VIO sliding-window state (MSCKF, ~20 poses) | ~10–50 KB | front-end owned |
| ArUco dictionary (4×4_50) + detector work | ~10–40 KB | front-end owned |
| Decision-core state (fused NavState, scheduler, avoider) | **< 1 KB** | fixed structs, no heap |
| **Total core-relevant** | **well under 200 KB** | vs GAP9 **~1.6 MB** L2 |

The map is the item that could have blown up (satellite descriptors were the
~400 KB estimate in an earlier draft); storing **ID + pose only** collapses it to
~20 KB, which is what makes the whole thing comfortable.

---

## 5. Per-module verdict

| Module | Compute | Memory | Verdict |
|---|---|---|---|
| **Uncertainty scheduler** | ~0.15 µs/call [MODELED] | < 1 KB | ✅ **fits trivially** — the contribution is essentially free |
| State fusion (2×2 EKF) | ~sub-µs [MODELED] | < 1 KB | ✅ fits trivially |
| Target-centric controller | ~sub-µs [MODELED] | < 1 KB | ✅ fits trivially |
| Obstacle avoidance (TTC policy) | ~sub-µs [MODELED] | < 1 KB | ✅ fits (the *policy*; optical-flow front-end is separate) |
| Landmark corrector | ~sub-µs per sighting [MODELED] | < 1 KB | ✅ fits; runs only on a fix |
| Landmark map | — | ~20 KB [MODELED] | ✅ fits (1.3 % of L2) |
| **ArUco detect + PnP** (front-end) | ~1–5 ms/frame [MODELED, lit.] | ~10–40 KB | ⚠️ **feasible but the cost driver**; gate with the scheduler |
| **VIO front-end** (front-end) | ~2–8 ms/frame [MODELED, lit.] | ~10–50 KB | ⚠️ **the binding constraint** on frame rate; needs the cluster, careful port |
| Optical-flow expansion (front-end) | ~1–3 ms/frame [MODELED, lit.] | small | ⚠️ feasible on the cluster; the TTC math itself is free |

**Bottom line.** The *entire contribution and the decision hot loop fit with
enormous margin* — microseconds against a multi-millisecond frame, kilobytes
against 1.6 MB. Feasibility is **gated by the front-ends** (VIO, ArUco, optical
flow), which is exactly why the plan does not reinvent them and why the scheduler
adds value: by firing the AVL/ArUco path only when needed, it removes the biggest
avoidable slice of front-end compute, extending flight time on the same battery.

---

## 6. Threats to validity

* GAP9 projections are **modeled**, not run on silicon (no board available — a
  hardware bring-up is explicit Future Work). We mark them so and keep derating
  conservative.
* Front-end costs are cited from the literature on comparable embedded cores, not
  re-measured here; treat as order-of-magnitude.
* The host↔target op-count mapping assumes the compiler emits similar scalar FP;
  the actual RISC-V codegen (no FMA on CV32E40P base, e.g.) could shift the
  decision-loop estimate by ~2×, which does not change any verdict.
