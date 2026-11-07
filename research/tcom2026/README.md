# FrugalNav-HF: TCOM 2026 study

Code and results for *When Is a Correction Worth Its Cost?* (TCOM research paper, IIIT-Delhi,
November 2026). FrugalNav-HF is a closed-loop planar quadrotor simulator (attitude lag, drag,
gusting wind, battery draw) with a camera model measured using the real OpenCV ArUco detector.

`core/` holds an unmodified copy of FrugalNav v1's scheduler and filter as the study used them;
`core_ext.py` and the `repaired` branch of `frugalnav_hf.py` hold the v2 logic that is now
merged into the main `core/` package.

| Step | Script | Output |
|---|---|---|
| Measure the detector | `python camera_calibration.py` | `results/camera_lut.npz`, `results/detector_timing.json` |
| Fly 3,060 flights (always, timer, v1) | `python run_experiments.py` | `results/runs.csv` |
| Trigger tiers + v1 matched-cost tests | `python reasons_and_tests.py` | `results/trigger_reasons.csv`, `results/matched_tests.csv` |
| Tune v2 on held-out seeds 100-119 | `python tune_repaired.py`, `python diag_repaired.py` | `results/repaired_tuning.json`, `results/repaired_diag.csv` |
| Fly 1,440 v2 test flights (seeds 0-29) | `python run_repaired.py` | `results/runs_repaired.csv` |
| v2 matched-cost tests | `python repaired_matched.py` | `results/repaired_matched.csv` |
| Tables and figures | `python analyze.py` | `results/summary_main.csv`, `figures/*.png` |

Needs numpy, scipy, pandas, matplotlib and opencv-contrib-python. Run each script from this
folder. The full run takes about 45 minutes on an 8-core laptop.
