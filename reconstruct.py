"""
RGB-D Reconstruction — OAK-D S2 — ROBUST & HIGH-PERFORMANCE VERSION
===================================================================
"""

import open3d as o3d
import numpy as np
import os
import json
import glob
import time
import gc
import sys
import threading
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# IMPORT OUR CUSTOM MODULES
from load_config import CFG, DECIMATE_FACTOR, ENABLE_INTEL_IGPU, DEPTH_SCALE
from utils import mem, ETA, PromiseLRU, HAS_NUMBA, _vram, _avail

# ============================================================
# THREAD CONTROL — before open3d internals spin up
# ============================================================
try:
    _ncpu = len(os.sched_getaffinity(0))
except AttributeError:
    import multiprocessing
    _ncpu = multiprocessing.cpu_count()

_phys = max(1, _ncpu // 2) if _ncpu <= 4 else _ncpu

os.environ["OMP_NUM_THREADS"]      = str(_phys)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = str(_phys)
os.environ["OMP_WAIT_POLICY"]      = "PASSIVE"

import open3d.core as o3c
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# ============================================================
# NUMBA EKF KERNELS
# ============================================================
if HAS_NUMBA:
    from numba import njit
else:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]): return args[0]
        def decorator(func): return func
        return decorator

@njit(cache=True)
def _inv6(A, r):
    B = A.copy()
    for i in range(6): 
        B[i, i] += r
    M = np.empty((6, 12))
    for i in range(6):
        for j in range(6): 
            M[i, j] = B[i, j]
            M[i, j+6] = 1.0 if i == j else 0.0
            
    for c in range(6):
        mr = c
        mv = abs(M[c, c])
        for rr in range(c+1, 6):
            if abs(M[rr, c]) > mv: 
                mv = abs(M[rr, c])
                mr = rr
        if mr != c:
            for j in range(12): 
                M[c, j], M[mr, j] = M[mr, j], M[c, j]
        p = M[c, c]
        if abs(p) < 1e-15: 
            p = 1e-15
        ip = 1.0 / p
        for j in range(12): 
            M[c, j] *= ip
        for rr in range(6):
            if rr != c:
                f = M[rr, c]
                for j in range(12): 
                    M[rr, j] -= f * M[c, j]
                    
    o = np.empty((6, 6))
    for i in range(6):
        for j in range(6): 
            o[i, j] = M[i, j+6]
    return o

if HAS_NUMBA: 
    _inv6(np.eye(6), 1e-6)

# ============================================================
# CONFIG INIT
# ============================================================
SCAN_DIR   = os.environ.get("TRITON_SCAN_DIR", CFG["paths"]["scan_dir"])
RGB_DIR    = os.path.join(SCAN_DIR, "rgb")
DEPTH_DIR  = os.path.join(SCAN_DIR, "depth")
POSES_FILE = os.path.join(SCAN_DIR, "poses.json")

if ENABLE_INTEL_IGPU:
    cv2.ocl.setUseOpenCL(True)
    if cv2.ocl.haveOpenCL():
        print("[HW] OpenCL Active: Hardware acceleration enabled.")
    else:
        print("[HW] OpenCL not available, falling back to CPU.")

DEPTH_TRUNC_DEFAULT = 3.0
DEPTH_TRUNC_PADDING = 0.3

DEPTH_QUANTILE = CFG["reconstruction"]["depth_quantile_pruning"]
VOXEL_TRACKING = CFG["reconstruction"]["voxel_tracking_m"]
VOXEL_LOOP     = CFG["reconstruction"]["voxel_loop_m"]
LOOP_CLOSURE_INTERVAL   = CFG["reconstruction"]["loop_closure_interval"]
MIN_LOOP_FRAME_DISTANCE = CFG["reconstruction"]["min_loop_frame_distance"]
ALLOW_MULTI_LOOPS       = CFG["reconstruction"]["allow_multi_loops"]
Z_REGULARIZE            = CFG["reconstruction"]["z_regularize"]
MIN_CLOUD_PTS           = CFG["reconstruction"]["min_cloud_points"]
LOOP_TOP_K              = CFG["reconstruction"]["loop_top_k"]

VOXEL_TSDF_BASE = CFG["reconstruction"]["voxel_tsdf_m"]
HAS_CUDA = o3c.cuda.is_available()
GPU_DEVICE = o3c.Device("CUDA:0") if HAS_CUDA else o3c.Device("CPU:0")
VOXEL_TSDF = VOXEL_TSDF_BASE if HAS_CUDA else max(0.015, VOXEL_TSDF_BASE)
SDF_TRUNC = VOXEL_TSDF * 4.0
TSDF_MODE = "TENSOR" if HAS_CUDA else "LEGACY"

# ICP Cascade & Trust Weights
ICP_TRACK_SCALES = [2.5, 1.5, 1.0]
ICP_TRACK_ITERS  = [60, 40, 25]
ICP_DIST_TRACKING = 0.10
TRACK_FIT_MIN     = 0.05
TRACK_RMSE_MAX    = 0.15
TRACK_GAP_SCALE_RATE = 0.25  
TRACK_GAP_SCALE_MAX  = 4.0   

ICP_LOOP_SCALES = [3.0, 2.0, 1.0]
ICP_LOOP_ITERS  = [50, 30, 20]
ICP_DIST_LOOP     = 0.05
ICP_DIST_LOOP_MAX = 0.30 
ICP_FIT_ACCEPT    = 0.50
ICP_RMSE_ACCEPT   = 0.025

SEQ_TRUST  = 1.0
LOOP_TRUST = 50.0
IMU_INFO_SCALE = 0.1
COARSE_FIT_REJECT = 0.10
VO_FALLBACK_THRESHOLD = 20.0
STATIONARY_THRESHOLD  = 0.001
Z_REG_MIN_DRIFT   = 0.05

FLOOR_REMOVAL_ENABLED     = True
FLOOR_RANSAC_DISTANCE     = 0.015
FLOOR_NORMAL_UP_THRESHOLD = 0.85
FLOOR_MIN_RATIO           = 0.25
FLOOR_RANSAC_ITERATIONS   = 350
NORMAL_SEARCH_NN          = 20

USE_COLORED_ICP = True
COLORED_ICP_LAMBDA_GEOMETRIC  = 0.99
COLORED_ICP_IMPROVEMENT_MIN   = 0.03
LOOP_DRIFT_RATE = 0.001
APERTURE_DISTINCTIVENESS_THR  = 0.05
APERTURE_IMU_BOOST            = 5.0

print(f"╔═════════════════════════════════════════════════════════╗")
print(f"║  reconstruct.py — Final Boss Edition                    ║")
print(f"║  Device : {'CUDA:0 (GPU)' if HAS_CUDA else 'CPU only':44s}  ║")
print(f"║  TSDF   : {TSDF_MODE:6s} @ {VOXEL_TSDF*1000:.0f}mm                                     ║")
print(f"║  Colored ICP : {'ON' if USE_COLORED_ICP else 'OFF':3s}                                        ║")
print(f"║  Decimation  : {DECIMATE_FACTOR}x (Hardware Benchmarked)                           ║")
print(f"║  {mem():53s}║")
print(f"╚═════════════════════════════════════════════════════════╝")

# ============================================================
# LOAD CALIBRATION & VERIFY FILES
# ============================================================
with open(os.path.join(SCAN_DIR, "intrinsics.json")) as f:
    cal = json.load(f)

W = int(cal["width"]) // DECIMATE_FACTOR
H = int(cal["height"]) // DECIMATE_FACTOR
fx = float(cal["fx"]) / DECIMATE_FACTOR
fy = float(cal["fy"]) / DECIMATE_FACTOR
cx = float(cal["cx"]) / DECIMATE_FACTOR
cy = float(cal["cy"]) / DECIMATE_FACTOR

intr_leg = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

if HAS_CUDA:
    intr_t = o3c.Tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1.]],
        dtype=o3c.Dtype.Float64, device=GPU_DEVICE
    )

cal_src = cal.get("calibration_source", "UNKNOWN")
print(f"  Calibration: {cal_src} (Scaled for {DECIMATE_FACTOR}x Decimation)")
print(f"  Scaled K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} ({W}x{H})")

rgb_files   = sorted(glob.glob(os.path.join(RGB_DIR, "*.jpg")))
depth_files = sorted(glob.glob(os.path.join(DEPTH_DIR, "*.png")))
n_frames    = min(len(rgb_files), len(depth_files))
print(f"\nFrames: {n_frames}")
if n_frames < 2: 
    raise RuntimeError("Need ≥2 frames to reconstruct.")

# ============================================================
# ZERO-COPY iGPU & CPU LOADERS (THREAD SAFE)
# ============================================================
ACTIVE_LOADER_MODE = "CPU"

_thread_local = threading.local()
_igpu_lock = threading.Lock()  # GRANDMASTER FIX: Prevents cv2.UMat segfaults across threads

def get_clahe():
    if not hasattr(_thread_local, "clahe"):
        _thread_local.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _thread_local.clahe

def _loader_igpu(rgb_path, depth_path, quantile=0.0):
    rgb_raw = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    if rgb_raw is None or depth_raw is None:
        return None, None

    # Locking OpenCL Context Operations to prevent thread collision segfaults
    with _igpu_lock:
        rgb_gpu = cv2.UMat(rgb_raw)
        depth_gpu = cv2.UMat(depth_raw)

        lab_gpu = cv2.cvtColor(rgb_gpu, cv2.COLOR_BGR2LAB)
        l_gpu, a_gpu, b_gpu = cv2.split(lab_gpu)
        
        clahe = get_clahe()
        l_clahe = clahe.apply(l_gpu)
        
        lab_clahe = cv2.merge([l_clahe, a_gpu, b_gpu])
        rgb_gpu_eq = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        if DECIMATE_FACTOR > 1:
            rgb_resized = cv2.resize(rgb_gpu_eq, (W, H), interpolation=cv2.INTER_AREA)
            depth_resized = cv2.resize(depth_gpu, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            rgb_resized = rgb_gpu_eq
            depth_resized = depth_gpu

        rgb_final = cv2.cvtColor(rgb_resized.get(), cv2.COLOR_BGR2RGB)
        depth_final = depth_resized.get()

    if quantile > 0.0 and quantile < 1.0:
        valid = depth_final[depth_final > 0]
        if len(valid) > 100:
            max_val = np.quantile(valid, quantile)
            depth_final[depth_final > max_val] = 0

    return o3d.geometry.Image(rgb_final), o3d.geometry.Image(depth_final)

def _loader_cpu(rgb_path, depth_path, quantile=0.0):
    rgb_raw = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    if rgb_raw is None or depth_raw is None:
        return None, None

    lab = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = get_clahe()
    l_clahe = clahe.apply(l)
    
    rgb_eq = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

    if DECIMATE_FACTOR > 1:
        rgb_resized = cv2.resize(rgb_eq, (W, H), interpolation=cv2.INTER_AREA)
        depth_resized = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        rgb_resized = rgb_eq
        depth_resized = depth_raw

    rgb_final = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
    depth_final = depth_resized

    if quantile > 0.0 and quantile < 1.0:
        valid = depth_final[depth_final > 0]
        if len(valid) > 100:
            max_val = np.quantile(valid, quantile)
            depth_final[depth_final > max_val] = 0

    return o3d.geometry.Image(rgb_final), o3d.geometry.Image(depth_final)

def load_and_decimate(rgb_path, depth_path, quantile=0.0):
    if ACTIVE_LOADER_MODE == "IGPU":
        return _loader_igpu(rgb_path, depth_path, quantile)
    return _loader_cpu(rgb_path, depth_path, quantile)

# ============================================================
# HARDWARE BENCHMARK & DEPTH AUTO-DETECT
# ============================================================
print(f"\n── Runtime Benchmark & Depth Auto-Detect ──")

sample_indices = np.linspace(0, n_frames - 1, min(20, n_frames), dtype=int)
all_maxes = []
cpu_times = []
igpu_times = []

if cv2.ocl.haveOpenCL():
    _loader_igpu(rgb_files[0], depth_files[0], 0.0)

for si in sample_indices:
    rgb_path = rgb_files[si]
    depth_path = depth_files[si]

    t0 = time.perf_counter()
    _loader_cpu(rgb_path, depth_path, 0.0)
    cpu_times.append(time.perf_counter() - t0)

    has_opencl = cv2.ocl.haveOpenCL()
    if has_opencl:
        t0 = time.perf_counter()
        d_img_tuple = _loader_igpu(rgb_path, depth_path, DEPTH_QUANTILE)
        igpu_times.append(time.perf_counter() - t0)
        d_img = d_img_tuple[1] if d_img_tuple[1] is not None else None
    else:
        d_img_tuple = _loader_cpu(rgb_path, depth_path, DEPTH_QUANTILE)
        d_img = d_img_tuple[1] if d_img_tuple[1] is not None else None

    if d_img is not None:
        d_arr = np.asarray(d_img).astype(np.float64)
        valid = d_arr[d_arr > 0] / DEPTH_SCALE
        if len(valid) > 100:
            all_maxes.append(float(np.percentile(valid, 95)))

avg_cpu_ms = (sum(cpu_times) / len(cpu_times)) * 1000 if cpu_times else 9999.0
avg_igpu_ms = (sum(igpu_times) / len(igpu_times)) * 1000 if igpu_times else 9999.0

if cv2.ocl.haveOpenCL() and avg_igpu_ms < avg_cpu_ms:
    ACTIVE_LOADER_MODE = "IGPU"
    print(f"  [WINNER] iGPU (OpenCL) selected! ({avg_igpu_ms:.1f}ms vs CPU {avg_cpu_ms:.1f}ms per frame)")
else:
    ACTIVE_LOADER_MODE = "CPU"
    if cv2.ocl.haveOpenCL():
        print(f"  [WINNER] CPU selected! Native instructions were faster ({avg_cpu_ms:.1f}ms vs iGPU {avg_igpu_ms:.1f}ms)")
    else:
        print(f"  [WINNER] CPU selected! (OpenCL drivers not available. {avg_cpu_ms:.1f}ms per frame)")

if len(all_maxes) >= 3:
    median_p95 = float(np.median(all_maxes))
    DEPTH_TRUNC = min(median_p95 + DEPTH_TRUNC_PADDING, 5.0)
    DEPTH_TRUNC = max(DEPTH_TRUNC, 0.5)
    print(f"  Depth 95th percentile: {median_p95:.2f}m")
    print(f"  DEPTH_TRUNC set to: {DEPTH_TRUNC:.2f}m")
else:
    DEPTH_TRUNC = DEPTH_TRUNC_DEFAULT
    print(f"  Could not auto-detect — using default: {DEPTH_TRUNC:.2f}m")

# ============================================================
# LOAD POSES & TELEMETRY
# ============================================================
with open(POSES_FILE) as f:
    pose_records = json.load(f)

if len(pose_records) != n_frames:
    n_frames = min(len(pose_records), n_frames)
    rgb_files   = rgb_files[:n_frames]
    depth_files = depth_files[:n_frames]

raw_poses = [np.array(r["pose"], dtype=np.float64) for r in pose_records[:n_frames]]
imu_covs  = [np.array(r["cov6"], dtype=np.float64) for r in pose_records[:n_frames]]
ekf_indices = [r.get("ekf_frame_idx", idx) for idx, r in enumerate(pose_records[:n_frames])]
quality_scores = [r.get("quality_score", 1.0) for r in pose_records[:n_frames]]
quality_states = [r.get("quality_state", "GOOD") for r in pose_records[:n_frames]]

positions   = np.array([p[:3, 3] for p in raw_poses])
traj_extent = np.max(positions, 0) - np.min(positions, 0)
max_extent  = float(np.max(traj_extent))

print(f"\n── EKF Trajectory Analysis ──")
print(f"  Extent: [{traj_extent[0]:.3f}, {traj_extent[1]:.3f}, {traj_extent[2]:.3f}] m")
print(f"  Max:    {max_extent:.3f} m")

for cov in imu_covs:
    np.fill_diagonal(cov, np.minimum(np.diag(cov), 1.0))

if max_extent > VO_FALLBACK_THRESHOLD:
    USE_VO = True
    print(f"  ✗ REJECTED ({max_extent:.1f}m > {VO_FALLBACK_THRESHOLD}m) → VO fallback")
elif max_extent < STATIONARY_THRESHOLD:
    USE_VO = True
    print(f"  ✗ REJECTED (stationary) → VO fallback")
else:
    USE_VO = False
    print(f"  ✓ ACCEPTED — EKF poses will seed ICP initial guesses")

TSDF_STEP = 3 if HAS_CUDA else max(5, n_frames // 150)
if USE_VO and not HAS_CUDA:
    TSDF_STEP = max(TSDF_STEP, n_frames // 150)

# ============================================================
# MATH & CLOUD HELPERS
# ============================================================
def cov2info(c6, sc=IMU_INFO_SCALE):
    # --- GRANDMASTER FIX: COVARIANCE ORDERING BOMB ---
    # EKF outputs Covariance in [Translation, Rotation] order (X,Y,Z, r,p,y).
    # Open3D's Global Optimizer absolutely requires [Rotation, Translation] order.
    # We must mathematically swap the quadrants before inversion.
    c6_o3d = np.zeros((6, 6), dtype=np.float64)
    c6_o3d[:3, :3] = c6[3:6, 3:6]   # Top-Left: Rotation Covariance
    c6_o3d[3:6, 3:6] = c6[:3, :3]   # Bottom-Right: Translation Covariance
    c6_o3d[:3, 3:6] = c6[3:6, :3]   # Top-Right: Cross-Covariance
    c6_o3d[3:6, :3] = c6[:3, 3:6]   # Bottom-Left: Cross-Covariance
    
    if HAS_NUMBA: 
        info = _inv6(c6_o3d, 1e-6)
    else: 
        info = np.linalg.inv(c6_o3d + np.eye(6) * 1e-6)
        
    info = 0.5 * (info + info.T)
    evals, evecs = np.linalg.eigh(info)
    evals = np.clip(evals, 1e-6, 1e6)
    info_spd = evecs @ np.diag(evals) @ evecs.T
    return info_spd * sc

def norm_info(raw, trust):
    tr = np.trace(raw)
    if tr < 1e-10: 
        return np.eye(6) * trust
    return raw * (6.0 / tr) * trust

def compute_cloud_distinctiveness(pcd):
    if pcd is None or not pcd.has_normals():
        return 0.5
        
    normals = np.asarray(pcd.normals)
    norms   = np.linalg.norm(normals, axis=1)
    valid   = normals[norms > 0.5]
    
    if len(valid) < 20:
        return 0.5
        
    try:
        C  = np.cov(valid.T)
        sv = np.linalg.svd(C, compute_uv=False)
        sv = np.abs(sv)
        if sv[0] < 1e-12:
            return 0.5
        return float(np.clip(sv[2] / sv[0], 0.0, 1.0))
    except Exception:
        return 0.5

def remove_floor_plane(pcd):
    if not FLOOR_REMOVAL_ENABLED: 
        return pcd
        
    n_pts = len(pcd.points)
    if n_pts < MIN_CLOUD_PTS * 2: 
        return pcd
        
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=FLOOR_RANSAC_DISTANCE,
            ransac_n=3, 
            num_iterations=FLOOR_RANSAC_ITERATIONS
        )
    except RuntimeError: 
        return pcd

    a, b, c, _d = plane_model
    plane_normal = np.array([a, b, c])
    plane_normal /= (np.linalg.norm(plane_normal) + 1e-12)
    inlier_ratio = len(inliers) / n_pts
    
    if inlier_ratio < FLOOR_MIN_RATIO: 
        return pcd

    up_candidates = [
        np.array([0., 1., 0.]), np.array([0., -1., 0.]),
        np.array([0., 0., 1.]), np.array([0., 0., -1.]),
    ]
    is_horizontal = any(abs(np.dot(plane_normal, up)) > FLOOR_NORMAL_UP_THRESHOLD for up in up_candidates)
    
    if not is_horizontal: 
        return pcd

    non_floor = pcd.select_by_index(inliers, invert=True)
    if len(non_floor.points) < MIN_CLOUD_PTS: 
        return pcd
        
    return non_floor

def _make_cloud(index, voxel, keep_color=False):
    img_tuple = load_and_decimate(rgb_files[index], depth_files[index], quantile=DEPTH_QUANTILE)
    if img_tuple[0] is None or img_tuple[1] is None:
        return None
        
    color, depth = img_tuple
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=DEPTH_SCALE,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=(not keep_color)
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_leg)
    del rgbd, color, depth

    down = pcd.voxel_down_sample(voxel)
    del pcd
    
    if len(down.points) < MIN_CLOUD_PTS:
        del down
        return None

    _, ind = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    clean = down.select_by_index(ind)
    del down
    
    if len(clean.points) < MIN_CLOUD_PTS:
        del clean
        return None

    clean = remove_floor_plane(clean)
    if clean is None or len(clean.points) < MIN_CLOUD_PTS:
        return None

    clean.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=NORMAL_SEARCH_NN)
    )
    clean.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
    
    return clean

def tracking_cloud(index): 
    return _make_cloud(index, VOXEL_TRACKING, keep_color=USE_COLORED_ICP)

def loop_cloud(index): 
    return _make_cloud(index, VOXEL_LOOP, keep_color=False)

def extract_fpfh(pcd, voxel_size):
    radius_feature = voxel_size * 5
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, 
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

def compute_cloud_and_fpfh(index):
    r = loop_cloud(index)
    if r is None: 
        return None
    f = extract_fpfh(r, VOXEL_LOOP)
    return (r, f)

def get_loop_cloud_and_fpfh(index, cache):
    return cache.get_or_compute(index, compute_cloud_and_fpfh)

# ============================================================
# MULTI-SCALE ICP WITH COLORED REFINEMENT
# ============================================================
def _ms_icp_geometric(src, tgt, fine_dist, init_T, scales, iters):
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    cr  = o3d.pipelines.registration.ICPConvergenceCriteria
    radii = [fine_dist * s for s in scales]
    
    res = o3d.pipelines.registration.registration_icp(
        src, tgt, radii[0], init_T, est, cr(max_iteration=iters[0])
    )
    
    if res.fitness < COARSE_FIT_REJECT:
        return init_T, res.fitness, res.inlier_rmse
        
    T = res.transformation
    for r, mi in zip(radii[1:], iters[1:]):
        res = o3d.pipelines.registration.registration_icp(
            src, tgt, r, T, est, cr(max_iteration=mi)
        )
        T = res.transformation
        
    return T, res.fitness, res.inlier_rmse

def _colored_icp_refine(src, tgt, fine_dist, init_T):
    if (not src.has_colors()) or (not tgt.has_colors()): 
        return init_T, -1.0, -1.0
        
    try:
        est = o3d.pipelines.registration.TransformationEstimationForColoredICP(
            lambda_geometric=COLORED_ICP_LAMBDA_GEOMETRIC
        )
        res = o3d.pipelines.registration.registration_colored_icp(
            src, tgt, fine_dist, init_T, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30
            )
        )
        return res.transformation, res.fitness, res.inlier_rmse
    except Exception: 
        return init_T, -1.0, -1.0

def tracking_icp(src, tgt, fine_dist, init_T=None):
    # GRANDMASTER FIX: Guard against assert failures on empty/stripped point clouds
    if src is None or tgt is None or len(src.points) < 30 or len(tgt.points) < 30: 
        return np.eye(4), 0.0, 1.0
        
    if init_T is None: 
        init_T = np.eye(4)
        
    T, fit, rmse = _ms_icp_geometric(src, tgt, fine_dist, init_T, ICP_TRACK_SCALES, ICP_TRACK_ITERS)
    
    if fit < COARSE_FIT_REJECT: 
        return T, fit, rmse
        
    if USE_COLORED_ICP and fit >= TRACK_FIT_MIN:
        T_c, fit_c, rmse_c = _colored_icp_refine(src, tgt, fine_dist, T)
        improvement = fit_c - fit
        rmse_ok = (rmse_c <= rmse * 1.15) or (rmse_c < 0.0)
        
        if improvement >= COLORED_ICP_IMPROVEMENT_MIN and rmse_ok:
            T, fit, rmse = T_c, fit_c, rmse_c
            
    return T, fit, rmse

def tracking_icp_retry(src, tgt, fine_dist, init_T):
    if src is None or tgt is None or len(src.points) < 30 or len(tgt.points) < 30:
        return init_T, 0.0, 1.0, False

    T, f, r = tracking_icp(src, tgt, fine_dist, init_T)
    if f >= TRACK_FIT_MIN and r <= TRACK_RMSE_MAX: 
        return T, f, r, True
        
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    cr  = o3d.pipelines.registration.ICPConvergenceCriteria
    coarse_radius = fine_dist * ICP_TRACK_SCALES[0]
    
    coarse_check = o3d.pipelines.registration.registration_icp(
        src, tgt, coarse_radius, np.eye(4), est, cr(max_iteration=ICP_TRACK_ITERS[0])
    )
    
    if coarse_check.fitness < 0.02: 
        return init_T, max(f, coarse_check.fitness), r, False
        
    T2, f2, r2 = tracking_icp(src, tgt, fine_dist, np.eye(4))
    if f2 > f and f2 >= TRACK_FIT_MIN: 
        return T2, f2, r2, True
        
    return init_T, max(f, f2), min(r, r2), False

def loop_icp(src, tgt, fine_dist, T_init=None):
    if src is None or tgt is None or len(src.points) < 30 or len(tgt.points) < 30: 
        return np.eye(4), 0.0, 1.0
        
    if T_init is None: 
        T_init = np.eye(4)
        
    return _ms_icp_geometric(src, tgt, fine_dist, T_init, ICP_LOOP_SCALES, ICP_LOOP_ITERS)

# ============================================================
t_total = time.perf_counter()

# ============================================================
# PHASE 1: POSE GRAPH (ASYNC PRE-FETCHING)
# ============================================================
print(f"\n{'='*60}")
print(f"PHASE 1: Pose Graph (Self-Healing Active)")
print(f"  Tracking voxel  : {VOXEL_TRACKING*1000:.0f}mm")
print(f"  ICP radius base : {ICP_DIST_TRACKING*1000:.0f}mm (auto-scales with gaps)")
print(f"  ICP cascade     : {ICP_TRACK_SCALES}")
print(f"  Colored ICP     : {'ON' if USE_COLORED_ICP else 'OFF'}")
print(f"  Floor removal   : {'ON' if FLOOR_REMOVAL_ENABLED else 'OFF'}")
print(f"  Depth truncation: {DEPTH_TRUNC:.2f}m")
print(f"{'='*60}")

graph = o3d.pipelines.registration.PoseGraph()

T_w = raw_poses[0].copy()
imu_poses = [T_w.copy()]

fa = np.array([])
n_aperture = 0

prev = tracking_cloud(0)
_start = 0
while prev is None and _start < n_frames - 1:
    _start += 1
    prev = tracking_cloud(_start)
    if prev is not None:
        print(f"  [P1] First valid cloud at frame {_start} (skipped dark frames)")

for _ in range(_start):
    imu_poses.append(imu_poses[-1].copy())

icp_ok = 0
icp_fail = 0
fitness_log = []
edge_data = []

eta1 = ETA(n_frames - 1, "ICP-P1", 50)

executor = ThreadPoolExecutor(max_workers=2)

future_curr = None
if _start + 1 < n_frames:
    future_curr = executor.submit(tracking_cloud, _start + 1)

for i in range(_start + 1, n_frames):
    curr = future_curr.result() if future_curr else None
    
    if i + 1 < n_frames:
        future_curr = executor.submit(tracking_cloud, i + 1)

    gap = ekf_indices[i] - ekf_indices[i-1]
    gap_scale = float(np.clip(1.0 + (gap - 1) * TRACK_GAP_SCALE_RATE, 1.0, TRACK_GAP_SCALE_MAX))
    
    base_dist = ICP_DIST_TRACKING * gap_scale
    q_state = quality_states[i]
    
    if q_state == "BAD":
        effective_dist = base_dist * 1.5  
        trust_multiplier = 0.1            
    elif q_state == "WEAK":
        effective_dist = base_dist * 1.25 
        trust_multiplier = 0.5            
    else:
        effective_dist = base_dist
        trust_multiplier = 1.0            

    if USE_VO:
        Rd = raw_poses[i-1][:3, :3].T @ raw_poses[i][:3, :3]
        Ti = np.eye(4)
        Ti[:3, :3] = Rd
    else:
        Ti = np.linalg.inv(raw_poses[i-1]) @ raw_poses[i]

    T_rel, fit, rmse, ok = tracking_icp_retry(curr, prev, effective_dist, Ti)
    fitness_log.append(fit)

    dist = compute_cloud_distinctiveness(curr) if curr is not None else 0.5

    info_mat = None
    if ok and curr is not None and prev is not None:
        try:
            raw_info = np.asarray(o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                curr, prev, effective_dist, T_rel
            ))
            q = float(np.clip(fit / max(TRACK_FIT_MIN, 0.01), 1.0, 5.0))
            info_mat = norm_info(raw_info, SEQ_TRUST * q * trust_multiplier)
        except Exception:
            pass

    edge_data.append((T_rel, fit, rmse, ok, dist, info_mat))

    if ok: 
        icp_ok += 1
    else:  
        icp_fail += 1

    imu_poses.append(imu_poses[-1] @ T_rel)
    
    del prev
    prev = curr
    eta1.tick()

executor.shutdown(wait=True)
del prev
gc.collect()

tot = icp_ok + icp_fail
if tot > 0:
    fa = np.array(fitness_log)
    n_aperture = sum(1 for _, _, _, _, d, _ in edge_data if d < APERTURE_DISTINCTIVENESS_THR)
    
    print(f"\n  Phase 1 ICP complete:")
    print(f"    Mode    : {'VO (rotation-only seed)' if USE_VO else 'EKF-guided (rot+trans seed)'}")
    print(f"    OK      : {icp_ok} ({100*icp_ok/tot:.1f}%)")
    print(f"    Failed  : {icp_fail} ({100*icp_fail/tot:.1f}%)")
    print(f"    Fitness : med={np.median(fa):.3f} mean={np.mean(fa):.3f} min={np.min(fa):.3f}")
    print(f"    Aperture-prone frames: {n_aperture} / {len(edge_data)}")

positions   = np.array([p[:3, 3] for p in imu_poses])
traj_extent = np.max(positions, 0) - np.min(positions, 0)
max_extent  = float(np.max(traj_extent))

SPATIAL_GATE = float(np.clip(max_extent * 0.15, 0.3, 5.0))
print(f"    ICP extent: [{traj_extent[0]:.3f}, {traj_extent[1]:.3f}, {traj_extent[2]:.3f}]m")
print(f"    Spatial gate: {SPATIAL_GATE:.2f}m  [{mem()}]")

for T in imu_poses:
    graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T.copy()))

for i in range(1, n_frames):
    Tpc = np.linalg.inv(imu_poses[i-1]) @ imu_poses[i]

    data_idx = i - _start - 1
    if 0 <= data_idx < len(edge_data):
        _, fit, rmse, ok, dist, info_mat = edge_data[data_idx]
        aperture_prone = (dist < APERTURE_DISTINCTIVENESS_THR)

        if ok and not aperture_prone:
            if info_mat is not None:
                info = info_mat
            else:
                q = float(np.clip(fit / max(TRACK_FIT_MIN, 0.01), 1.0, 5.0))
                info = np.eye(6) * SEQ_TRUST * q
        elif not USE_VO and i < len(imu_covs):
            trust_mult = APERTURE_IMU_BOOST if aperture_prone else 1.0
            info = norm_info(cov2info(imu_covs[i]), SEQ_TRUST * trust_mult)
        else:
            info = np.eye(6) * SEQ_TRUST
    else:
        info = np.eye(6) * SEQ_TRUST * 0.01

    graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i-1, i, Tpc, info, uncertain=False))

print(f"\nNodes: {len(graph.nodes)} | Seq edges: {n_frames-1}")
print(f"Trust: seq={SEQ_TRUST} loop={LOOP_TRUST} ({LOOP_TRUST/SEQ_TRUST:.0f}:1)")

# ============================================================
# PHASE 2: LOOP CLOSURE (PARALLEL EVALUATION)
# ============================================================
print(f"\n{'='*60}")
print(f"PHASE 2: Loop Closure (Parallel FGR Enabled)")
print(f"  Loop voxel   : {VOXEL_LOOP*1000:.0f}mm")
print(f"  ICP radius   : {ICP_DIST_LOOP*1000:.0f}mm (auto-scales with true gaps)")
print(f"  ICP cascade  : {ICP_LOOP_SCALES}")
print(f"  Spatial gate : {SPATIAL_GATE:.2f}m")
print(f"  Top-K Eval   : {LOOP_TOP_K} closest frames")
print(f"  [{mem()}]")
print(f"{'='*60}")

t2 = time.perf_counter()
n_loops = 0
n_skip = 0
n_reject = 0
n_try = 0

lc = PromiseLRU(60) 

positions_p1 = np.array([n.pose[:3, 3] for n in graph.nodes])
candidate_indices = np.arange(0, n_frames, LOOP_CLOSURE_INTERVAL)

def eval_candidate(i, ti, sd, sd_fpfh, dyn_dist):
    td_data = get_loop_cloud_and_fpfh(ti, lc)
    if td_data is None: 
        return None
        
    td, td_fpfh = td_data
    
    # --- FIX 3: DEGENERACY GATE ---
    # Reject loops on flat/featureless geometry where ICP will slide
    dist_s = compute_cloud_distinctiveness(sd)
    dist_t = compute_cloud_distinctiveness(td)
    if dist_s < APERTURE_DISTINCTIVENESS_THR or dist_t < APERTURE_DISTINCTIVENESS_THR:
        return None
    # ------------------------------
    
    try:
        fgr_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            sd, td, sd_fpfh, td_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=dyn_dist * 2.0
            )
        )
        if fgr_result.fitness > 0.1:
            T_init = fgr_result.transformation
        else:
            T_init = np.linalg.inv(imu_poses[ti]) @ imu_poses[i]
    except Exception:
        T_init = np.linalg.inv(imu_poses[ti]) @ imu_poses[i]

    T_icp, fit, rmse = loop_icp(sd, td, dyn_dist, T_init)

    if fit > ICP_FIT_ACCEPT and rmse < ICP_RMSE_ACCEPT:
        raw = np.asarray(o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            sd, td, dyn_dist, T_icp
        ))
        if np.trace(raw) > 1e-6: 
            return (ti, T_icp, norm_info(raw, LOOP_TRUST), fit, rmse)
            
    return None

with ThreadPoolExecutor(max_workers=_phys) as pool:
    for i in range(MIN_LOOP_FRAME_DISTANCE, n_frames, LOOP_CLOSURE_INTERVAL):
        
        pi = positions_p1[i]
        max_ti = i - MIN_LOOP_FRAME_DISTANCE
        valid_candidates = candidate_indices[candidate_indices <= max_ti]
        
        if len(valid_candidates) == 0: 
            continue
            
        dists = np.linalg.norm(positions_p1[valid_candidates] - pi, axis=1)
        
        mask = dists <= (SPATIAL_GATE * 1.5)
        valid_ti = valid_candidates[mask]
        valid_dists = dists[mask]
        
        if len(valid_ti) > LOOP_TOP_K:
            sorted_idx = np.argsort(valid_dists)
            valid_ti = valid_ti[sorted_idx[:LOOP_TOP_K]]
            
        n_skip += len(valid_candidates) - len(valid_ti)
        
        if len(valid_ti) == 0: 
            continue
            
        sd_data = get_loop_cloud_and_fpfh(i, lc)
        if sd_data is None: 
            continue
            
        sd, sd_fpfh = sd_data
        
        futures = []
        for ti in reversed(valid_ti):
            n_try += 1
            true_frame_gap = ekf_indices[i] - ekf_indices[ti]
            drift_est = float(np.clip(
                true_frame_gap * LOOP_DRIFT_RATE, 0.0, ICP_DIST_LOOP_MAX - ICP_DIST_LOOP
            ))
            dynamic_loop_dist = ICP_DIST_LOOP + drift_est
            
            fut = pool.submit(eval_candidate, i, ti, sd, sd_fpfh, dynamic_loop_dist)
            futures.append(fut)
            
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                ti_match, T_icp, info, fit, rmse = res
                graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                    i, ti_match, T_icp, info, uncertain=True
                ))
                n_loops += 1
                print(f"  Loop: {i:4d}→{ti_match:4d} fit={fit:.3f} rmse={rmse:.4f}")
                
                if not ALLOW_MULTI_LOOPS:
                    for f in futures: 
                        f.cancel()
                    break
            else:
                n_reject += 1

        if (i // LOOP_CLOSURE_INTERVAL) % 50 == 0 and i > 0:
            print(f"  [LC] {i}/{n_frames} cache={len(lc._d)} [{mem()}]")

lc.clear()
gc.collect()

print(f"\nLoops: {n_loops} | Tried: {n_try} | Rejected: {n_reject} | Skipped: {n_skip}")
print(f"Phase 2: {time.perf_counter()-t2:.1f}s")

# ============================================================
# PHASE 3: OPTIMIZATION
# ============================================================
print(f"\n{'='*60}")
print(f"PHASE 3: Optimization")
print(f"{'='*60}")

pos_pre = np.array([n.pose[:3, 3] for n in graph.nodes])

o3d.pipelines.registration.global_optimization(
    graph,
    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
    o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=ICP_DIST_LOOP,
        edge_prune_threshold=0.25,
        reference_node=0
    )
)

pos_post = np.array([n.pose[:3, 3] for n in graph.nodes])
post_ext = pos_post.max(0) - pos_post.min(0)
deltas   = np.linalg.norm(pos_post - pos_pre, axis=1)

print(f"\n  Post-opt extent (before Z correction): [{post_ext[0]:.3f}, {post_ext[1]:.3f}, {post_ext[2]:.3f}]m")
print(f"  Shift: mean={np.mean(deltas)*100:.1f}cm max={np.max(deltas)*100:.1f}cm")

if np.max(deltas) < 1e-4 and n_loops > 0: 
    print(f"  ⚠ {n_loops} loops had NO effect!")
elif n_loops > 0: 
    print(f"  ✓ Adjusted by {n_loops} loops")
else: 
    print(f"  ℹ No loops found — trajectory unchanged")

if Z_REGULARIZE and not USE_VO:
    all_z   = np.array([graph.nodes[i].pose[2, 3] for i in range(n_frames)])
    z_range = float(np.max(all_z) - np.min(all_z))
    
    if z_range > Z_REG_MIN_DRIFT:
        t_idx    = np.arange(n_frames, dtype=float)
        poly_deg = min(2, n_frames - 1)
        z_poly   = np.polyfit(t_idx, all_z, deg=poly_deg)
        z_drift  = np.polyval(z_poly, t_idx) - np.polyval(z_poly, 0.0)
        
        for i in range(n_frames):
            corrected_pose = graph.nodes[i].pose.copy()
            corrected_pose[2, 3] -= z_drift[i]
            graph.nodes[i].pose = corrected_pose
            
        all_z_post   = np.array([graph.nodes[i].pose[2, 3] for i in range(n_frames)])
        z_range_post = float(np.max(all_z_post) - np.min(all_z_post))
        z_max_corr   = float(np.max(np.abs(z_drift)))
        print(f"\n  Z-drift correction: range {z_range*100:.1f}cm → {z_range_post*100:.1f}cm "
              f"| max correction: {z_max_corr*100:.1f}cm")
    else:
        print(f"\n  Z-drift: {z_range*100:.1f}cm — below {Z_REG_MIN_DRIFT*100:.0f}cm threshold, skipped")

pos_post = np.array([n.pose[:3, 3] for n in graph.nodes])
post_ext  = pos_post.max(0) - pos_post.min(0)

# ============================================================
# PHASE 3.5: POINT CLOUD PREVIEW & GATE
# ============================================================
print(f"\n{'='*60}")
print(f"PHASE 3.5: Point Cloud Preview & Gate")
print(f"{'='*60}")

print("  Building sparse global point cloud for preview...")
preview_cloud = o3d.geometry.PointCloud()

fi = list(range(0, n_frames, TSDF_STEP))
for i in fi:
    pcd = loop_cloud(i) 
    if pcd is not None:
        pcd.transform(graph.nodes[i].pose)
        preview_cloud += pcd
        
    if i % (TSDF_STEP * 10) == 0:
        preview_cloud = preview_cloud.voxel_down_sample(VOXEL_LOOP)
        
preview_cloud = preview_cloud.voxel_down_sample(VOXEL_LOOP)

NON_INTERACTIVE = os.environ.get("TRITON_NONINTERACTIVE", "0") == "1"

if not NON_INTERACTIVE:
    print("\n  [PREVIEW] An Open3D window has opened.")
    print("  Inspect the alignment, then CLOSE THE WINDOW to continue.")
    try:
        o3d.visualization.draw_geometries(
            [preview_cloud, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)],
            window_name="Phase 3.5: Pose Graph Preview (Close to continue)"
        )
    except Exception as e:
        print(f"  ⚠ Preview visualization failed: {e}")

    while True:
        choice = input("\n  Proceed to dense TSDF mesh generation? (y/n): ").strip().lower()
        if choice == 'y':
            print("  Proceeding to Phase 4...")
            break
        elif choice == 'n':
            pc_path = os.path.join(SCAN_DIR, "optimized_sparse_cloud_mm.ply")
            preview_cloud.scale(1000.0, center=preview_cloud.get_center())
            o3d.io.write_point_cloud(pc_path, preview_cloud)
            print(f"\n  [EXIT] Mesh generation aborted.")
            print(f"  Saved optimized point cloud to: {pc_path}")
            sys.exit(0)
        else:
            print("  Invalid input. Please enter 'y' or 'n'.")
else:
    print("\n  [PREVIEW] TRITON_NONINTERACTIVE=1 detected. Auto-proceeding to Phase 4 TSDF.")

del preview_cloud
gc.collect()

# ============================================================
# PHASE 4: TSDF (ASYNC DOUBLE-BUFFERED & QUARANTINED)
# ============================================================
fi = [i for i in range(0, n_frames, TSDF_STEP) if quality_states[i] != "BAD"]
skipped_bad = len(range(0, n_frames, TSDF_STEP)) - len(fi)

if not fi:
    print(f"\n⚠ [FATAL] All {len(range(0, n_frames, TSDF_STEP))} sampled TSDF frames were quarantined as 'BAD'.")
    print("  The scan quality is too low to safely generate a mesh. Try lowering 'score_weak_threshold'.")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"PHASE 4: TSDF  [{TSDF_MODE}] (Double-Buffered Integration)")
print(f"  Voxel: {VOXEL_TSDF*1000:.0f}mm | SDF: {SDF_TRUNC*1000:.0f}mm | Depth trunc: {DEPTH_TRUNC:.2f}m")
print(f"  Quarantined: {skipped_bad} BAD frames skipped")
print(f"  [{mem()}]")
print(f"{'='*60}")

inv_ext = {i: np.linalg.inv(graph.nodes[i].pose) for i in fi}
eta4 = ETA(len(fi), "TSDF", 30)
t4   = time.perf_counter()

tsdf_exec = ThreadPoolExecutor(max_workers=2)

fut_frame = None
if fi:
    fut_frame = tsdf_exec.submit(load_and_decimate, rgb_files[fi[0]], depth_files[fi[0]], DEPTH_QUANTILE)

if HAS_CUDA:
    bc = max(10_000, min(
        int(np.prod(np.ceil((post_ext + 2*DEPTH_TRUNC) /
            (16*VOXEL_TSDF)).astype(int)) * 0.2), 500_000))
    print(f"  block_count = {bc}")

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=VOXEL_TSDF, block_resolution=16,
        block_count=bc, device=GPU_DEVICE
    )

    for idx, i in enumerate(fi):
        if fut_frame is None: continue
        
        color, depth = fut_frame.result()
        
        if idx + 1 < len(fi):
            fut_frame = tsdf_exec.submit(load_and_decimate, rgb_files[fi[idx+1]], depth_files[fi[idx+1]], DEPTH_QUANTILE)
        else:
            fut_frame = None
        
        if color is not None and depth is not None:
            dt = o3d.t.geometry.Image(np.asarray(depth)).to(GPU_DEVICE)
            ct = o3d.t.geometry.Image(np.asarray(color)).to(GPU_DEVICE)
            et = o3c.Tensor(inv_ext[i], dtype=o3c.Dtype.Float64, device=GPU_DEVICE)

            fb = vbg.compute_unique_block_coordinates(
                dt, intr_t, et, depth_scale=DEPTH_SCALE, depth_max=DEPTH_TRUNC
            )
                
            vbg.integrate(
                fb, dt, ct, intr_t, et,
                depth_scale=DEPTH_SCALE, depth_max=DEPTH_TRUNC,
                trunc_voxel_multiplier=SDF_TRUNC/VOXEL_TSDF
            )
            
            del dt, ct, et, fb

        if eta4.i % 10 == 0:
            gc.collect()
            o3c.cuda.release_cache()

        if eta4.i % 60 == 0:
            v = _vram()
            if v: 
                print(f"    [{v}]")
                
        eta4.tick()

    print(f"  Extracting mesh... [{mem()}]")
    mesh_t = vbg.extract_triangle_mesh(weight_threshold=1.0)
    del vbg
    gc.collect()
    mesh = mesh_t.to_legacy()
    del mesh_t
    gc.collect()

else:
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_TSDF, sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for idx, i in enumerate(fi):
        if fut_frame is None: continue
        
        color, depth = fut_frame.result()
        
        if idx + 1 < len(fi):
            fut_frame = tsdf_exec.submit(load_and_decimate, rgb_files[fi[idx+1]], depth_files[fi[idx+1]], DEPTH_QUANTILE)
        else:
            fut_frame = None
            
        if color is not None and depth is not None:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=DEPTH_SCALE, depth_trunc=DEPTH_TRUNC,
                convert_rgb_to_intensity=False
            )
            vol.integrate(rgbd, intr_leg, inv_ext[i])
            del rgbd, color, depth

        if eta4.i % 10 == 0:
            gc.collect()

        if eta4.i % 60 == 0 and _avail() < 2.0:
            print(f"  ⚠ Low memory — stopping early")
            break
            
        eta4.tick()

    print(f"  Extracting mesh... [{mem()}]")
    mesh = vol.extract_triangle_mesh()
    del vol
    gc.collect()

tsdf_exec.shutdown(wait=True)
del inv_ext
print(f"TSDF: {eta4.i} frames in {time.perf_counter()-t4:.1f}s")

# ============================================================
# PHASE 5: MESH
# ============================================================
print(f"\n{'='*60}")
print(f"PHASE 5: Mesh")
print(f"{'='*60}")

mesh.compute_vertex_normals()
mesh.remove_degenerate_triangles()
mesh.remove_unreferenced_vertices()

nv = len(mesh.vertices)
nt = len(mesh.triangles)
print(f"  {nv} verts, {nt} tris")

if nv == 0 or nt == 0:
    print("⚠ Empty mesh")
    print(f"TOTAL: {time.perf_counter()-t_total:.1f}s")
    raise SystemExit(0)

bb = mesh.get_axis_aligned_bounding_box()
bb.color = (1, 0, 0)
d  = bb.get_max_bound() - bb.get_min_bound()
b  = int(np.argmax(d))

print(f"\n  BBox: X={d[0]*100:.1f}cm Y={d[1]*100:.1f}cm Z={d[2]*100:.1f}cm")
print(f"  → {'XYZ'[b]}: {d[b]*100:.1f}cm")
print(f"\nTOTAL: {time.perf_counter()-t_total:.1f}s")
print(f"[{mem()}]")

mp = os.path.join(SCAN_DIR, "coral_mesh_mm.ply")
mesh.scale(1000.0, center=mesh.get_center())
o3d.io.write_triangle_mesh(mp, mesh)
print(f"Saved: {mp}")

# ============================================================
# PHASE 6: METRICS EXPORT
# ============================================================
metrics_path = os.path.join(SCAN_DIR, "reconstruction_metrics.json")
metrics_data = {
    "timing": {
        "total_seconds": time.perf_counter() - t_total
    },
    "phase_1_tracking": {
        "frames_processed": n_frames,
        "icp_successes": icp_ok,
        "icp_failures": icp_fail,
        "median_fitness": float(np.median(fa)) if tot > 0 else 0,
        "aperture_prone_frames": n_aperture
    },
    "phase_2_loop_closure": {
        "loops_found": n_loops,
        "candidates_tried": n_try,
        "candidates_rejected": n_reject,
        "spatial_gate_skipped": n_skip
    },
    "phase_3_optimization": {
        "mean_shift_m": float(np.mean(deltas)),
        "max_shift_m": float(np.max(deltas))
    },
    "phase_4_tsdf": {
        "frames_integrated": len(fi),
        "frames_quarantined": skipped_bad
    },
    "phase_5_mesh": {
        "vertices": nv,
        "triangles": nt
    }
}

with open(metrics_path, "w") as f:
    json.dump(metrics_data, f, indent=4)
print(f"  Metrics saved: {metrics_path}")

try:
    if not NON_INTERACTIVE:
        o3d.visualization.draw_geometries(
            [mesh, bb, o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)],
            window_name="Triton Coral — robustReconstruct (mm scale)"
        )
except Exception as e:
    print(f"  Viz skipped: {e}")
    print(f"  Open in MeshLab: {mp}")