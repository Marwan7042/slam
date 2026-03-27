
#Visual SLAM

**Visual-Inertial Odometry and Dense Reconstruction Pipeline for Subsea Environments**

[![Python]([https://img.shields.io/badge/Python-3.8%2B-blue.svg](https://www.python.org/))]()
[![Open3D]([https://img.shields.io/badge/Open3D-0.16%2B-lightgrey.svg](https://www.open3d.org/))]()
[![DepthAI]([https://img.shields.io/badge/DepthAI-OAK--D-orange.svg](https://docs.luxonis.com/software-v3/depthai/))]()
[![Numba]([https://img.shields.io/badge/Numba-JIT-green.svg](https://numba.pydata.org/))]()

---

## Overview

Triton is a Simultaneous Localization and Mapping (SLAM) system designed for underwater ROV navigation, addressing domain-specific challenges such as optical refraction, variable turbidity, and dynamic lighting. 

The pipeline fuses high-frequency inertial data with stereo vision utilizing a 15-Degree-of-Freedom (DOF) Local Error-State Kalman Filter (ESKF) for real-time state estimation. Offline, the estimated trajectory seeds a global pose graph, followed by Truncated Signed Distance Function (TSDF) volumetric integration to generate a dense 3D mesh.

---

## Architecture and Key Features

* **Hardware-Level Sensor Synchronization:** Utilizes Luxonis DepthAI spatial nodes to align RGB, depth, and VPU feature timestamps, mitigating Out-Of-Sequence Measurement (OOSM) errors.
* **15-DOF ESKF Formulation:** Estimates position, velocity, orientation ($SO(3)$), and IMU biases. Implements a local error-state Lie algebra formulation to prevent covariance singularities.
* **Dynamic Motion-Blur Gating:** Calculates estimated pixel blur as a function of focal length and real-time angular velocity. Frames exceeding the defined threshold are rejected prior to visual processing.
* **CPU Optical Flow Fallback:** In the event of VPU tracking failure (e.g., rapid illumination changes), the system initiates CPU-bound Lucas-Kanade optical flow to maintain feature tracking and state continuity.
* **Asynchronous Concurrency:** Implements isolated threads for hardware acquisition, visual processing, and disk I/O, utilizing zero-copy double-buffering and Numba JIT compilation to manage execution latency.

---

## Installation

### Prerequisites
* Python 3.8+
* Luxonis OAK-D S2 (or equivalent DepthAI hardware)
* Intel iGPU (Optional, required for OpenCL hardware-accelerated color correction)

### Setup
```bash
git clone [https://github.com/Marwan7042/slam.git](https://github.com/Marwan7042/slam.git)
cd triton-slam

pip install numpy opencv-contrib-python depthai open3d numba psutil
```

---

## Usage

The system operates in two distinct phases: real-time data acquisition and offline reconstruction.

### Phase 1: Real-Time VIO (Acquisition)
Execute this script during ROV operation. It initializes the sensor pipeline, runs the ESKF, and logs the trajectory and gated keyframes to the specified output directory.
```bash
python record.py
```
* **Diagnostic Interface:** Navigate to `http://<ROV_IP>:8080/` to access the real-time HUD, monitor EKF health status, and adjust camera parameters (Exposure/ISO/WB).
* **Terminal Controls:** * `r`: Initiate map recording.
  * `s`: Pause recording.
  * `q`: Terminate pipeline and export session telemetry.

### Phase 2: Global Reconstruction (Offline)
Execute this script post-mission on the host machine to process the acquired dataset, perform loop closure optimization, and extract the 3D mesh.
```bash
python reconstruct.py
```
* **Output:** `coral_mesh_mm.ply` (Dense 3D point cloud and mesh geometry).

---

## System Components

### 1. Acquisition and Gating (`record.py`)
Handles data ingestion, thread management, and initial data validation.
* **Statistical Gating:** Evaluates spatial point variance to filter unstructured feature tracking and applies the gyroscopic blur threshold.
* **Telemetry Watchdog:** Monitors hardware bus activity. Stalls exceeding 5.0 seconds trigger a safe pipeline termination and data flush.

### 2. State Estimator (`ekf.py`)
The primary numerical filter, with core matrix operations compiled via Numba.
* **IMU Kinematic Clamping:** Restricts input accelerations to physical ROV limits ($\pm 25 m/s^2$) to prevent integration instability during hull impacts.
* **Mahalanobis Gating:** Evaluates incoming visual measurements against the state covariance matrix ($\mathbf{P}$). Innovations exceeding the $\chi^2$ threshold are discarded.
* **Covariance Management:** In prolonged periods of visual denial, the filter marginally inflates the covariance matrix to ensure receptivity to future visual updates.

### 3. Reconstruction Backend (`reconstruct.py`)
Handles global trajectory optimization and volumetric mapping.
* **Color Equalization:** Optionally leverages OpenCL (`cv2.UMat`) for hardware-accelerated red-channel attenuation correction.
* **Pose Graph Optimization:** Utilizes Multi-Scale ICP for scan matching. Applies Singular Value Decomposition (SVD) on local point cloud normals to reject geometrically degenerate (planar) loop closures.
* **TSDF Integration:** Fuses the depth maps into a voxel grid, utilizing CUDA (`o3d.t.geometry.VoxelBlockGrid`) when compatible hardware is present.

---

## Configuration (`config.json`)

System parameters are defined externally in `config.json`. Key tunables include:

| Parameter | Description |
| :--- | :--- |
| `calibration_mode` | Select `"custom"` for underwater EEPROM or `"factory"` for standard air deployment. |
| `max_blur_pixels` | Rejection threshold for estimated gyroscopic blur. |
| `static_variance_threshold`| Maximum IMU variance permitted for gravity initialization. |
| `weight_depth` / `weight_blur`| Quality scoring weights for keyframe selection. |
| `voxel_tsdf_m` | Volumetric resolution of the final 3D mesh (e.g., `0.005` = 5mm). |
```
