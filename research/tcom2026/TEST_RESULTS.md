# FrugalNav v2: test results

Run on 10 November 2026 on branch `frugalnav-v2`. Raw output is in [`test_logs/`](test_logs/).

**Environment:** Windows 11, Python 3.12.5, pytest 9.2.0, NumPy 2.2.4, OpenCV 5.0.0,
g++ 6.3.0 (MinGW, `-std=c++14 -O2`).

## Summary

| Check | Command | Result |
|---|---|---|
| Python unit tests | `python -m pytest -v` | **90 passed**, 0 failed |
| Scheduler self-test (v1 cases + v2 cooldown/emergency) | `python core/uncertainty_scheduler.py` | passed |
| State-fusion self-test | `python -m core.state_fusion` | passed |
| C++ v2 cross-check against the Python numbers | `cd cpp && g++ -std=c++14 -O2 -I. v2_check.cpp -o v2_check && ./v2_check` | passed (variance 0.29000 = expected 0.29000) |
| Existing C++ core self-test, unchanged behaviour | `cd cpp && g++ -std=c++14 -O2 -I. main.cpp -o frugalnav && ./frugalnav` | passed ("FrugalNav C++ core OK.") |

## Python tests by file

| File | Tests | Result |
|---|---:|---|
| `tests/test_frugalnav_v2.py` (new) | 11 | all passed |
| `tests/test_rohan_modules.py` | 16 | all passed |
| `tests/test_new_core_modules.py` | 12 | all passed |
| `tests/test_eval_sim.py` | 7 | all passed |
| `tests/test_geometry.py` | 7 | all passed |
| `tests/test_pipeline.py` | 6 | all passed |
| `tests/test_policies.py` | 6 | all passed |
| `tests/test_detour_tracking.py` | 5 | all passed |
| `tests/test_landmark_corrector.py` | 5 | all passed |
| `tests/test_metrics.py` | 5 | all passed |
| `tests/test_vector_to_target.py` | 5 | all passed |
| `tests/test_blur_metric.py` | 4 | all passed |
| `tests/test_groundtruth.py` | 1 | all passed |
| **Total** | **90** | **all passed** |

The 79 tests that existed before v2 all still pass, so v1 behaviour is unchanged when the v2
options are off.

## What the new v2 tests check

- v1 filter growth is unchanged; v2 growth is quadratic in distance since the last fix.
- Poor image quality adds variance; a fix resets the distance; `predicted_sigma` looks ahead.
- v1 safety floors still override the cooldown; v2 floors wait for it.
- v2's emergency override still fires through the cooldown at 2x the budget.
- v2's position term alone can trigger at the budget (v1's could not).
- `distance_to_next_marker` returns the right marker on the surveyed map.
- `build_navigator` uses v2 by default and v1 on request.
- Raw Laplacian variance is fooled by sensor noise, and the compensated cue is not.

## Study results

The simulation results behind the TCOM paper are separate from these unit tests. They are in
[`results/`](results/) (4,500 test flights plus held-out tuning runs), with figures in
[`figures/`](figures/). See [`README.md`](README.md) for how to regenerate them.

| Policy (4,500 test flights) | RMSE | Detector calls per flight |
|---|---:|---:|
| Always-correct | 0.37 m | 316 |
| Fixed timer (5 s) | 0.44 m | 8.4 |
| **FrugalNav v2** | **0.46 m** | **9.1** |
| FrugalNav v1 (first release) | 1.54 m | 19.0 |
