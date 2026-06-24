# FrugalNav Architecture & Data Flow

This document defines the canonical data flow for the FrugalNav pipeline. All modules must adhere to these contracts.

## 1. The Core Loop
The system operates on a single hot loop driven by the `SensorInput` stream.

`SensorInput` -> `FrugalPipeline` -> `VelocityCmd`

## 2. Component Responsibilities

* **Harness Readers (`dataset_reader.py`, `phone_reader.py`)**: Parse raw data (CSV, phone logs, video) and yield `SensorInput` objects strictly ordered by timestamp.
* **VioAdapter Protocol**: Intercepts `SensorInput` and outputs a `VioOutput` (relative delta pose and internal uncertainty metrics). This abstracts away whether we are using a synthetic drift scaffold or the real ORB-SLAM3/OpenVINS backend.
* **Pipeline (`core/pipeline.py`)**:
    * Integrates `VioOutput.delta_pose` into a running global state.
    * **State Fusion (W3)**: Will fuse `LandmarkFix` (from ArUco markers) to correct the global state.
    * **Controller (W3)**: Computes a target-centric `VelocityCmd` based on the fused state.
* **Uncertainty Scheduler (W4)**: Monitors `VioOutput.pos_std_m` and `blur` to govern when to actively seek landmark corrections.

## 3. Immutability Rules
* `core/` code must never import `harness/` code.
* `core/` must rely strictly on `numpy` to ensure portability to the RISC-V hardware. ROS and dataset paths are strictly forbidden in `core/`.
