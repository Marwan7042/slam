import math
import numpy as np
import threading
from utils import njit, RunningVariance
from load_config import CFG

# Import EKF constants securely from CFG
STATIC_VAR_THR   = CFG["ekf_tuning"]["static_variance_threshold"]
STATIC_WIN       = CFG["ekf_tuning"]["static_variance_window"]
MIN_GRAV_SAMPLES = CFG["ekf_tuning"]["min_gravity_samples"]
MIN_FEAT_UPDATE  = CFG["ekf_tuning"]["min_feature_update"]
DEPTH_PATCH_R    = CFG["ekf_tuning"]["depth_patch_radius"]
DEPTH_MIN_MM     = CFG["ekf_tuning"]["depth_min_mm"]
DEPTH_MAX_MM     = CFG["ekf_tuning"]["depth_max_mm"]

# --- IMU NOISE PARAMETERS ---
# Baseline noise densities (e.g., standard MEMS IMU on the OAK-D)
_BASE_ACCEL_ND  = 160e-6 * 9.81
_BASE_GYRO_ND   = np.deg2rad(0.007)

# Bias Random Walk (Internal silicon drift, largely unaffected by vibration)
ACCEL_BRW = 40e-6 * 9.81
GYRO_BRW  = np.deg2rad(0.5 / 3600.0)

# Apply the structural vibration multiplier from config
VIB_MULTIPLIER = CFG["ekf_tuning"].get("imu_vibration_multiplier", 15.0)

ACCEL_ND = _BASE_ACCEL_ND * VIB_MULTIPLIER
GYRO_ND  = _BASE_GYRO_ND * VIB_MULTIPLIER

# Visual Noise Baselines
VIS_NOISE_P   = (0.010)**2
VIS_NOISE_PHI = (np.deg2rad(1.0))**2
REORTHO_INTERVAL = 500

@njit(cache=True)
def _rodrigues_jit(wx,wy,wz,out):
    t2=wx*wx+wy*wy+wz*wz; t=math.sqrt(t2)
    if t<1e-9:
        out[0,0]=1.0;out[0,1]=-wz;out[0,2]=wy;out[1,0]=wz;out[1,1]=1.0;out[1,2]=-wx;out[2,0]=-wy;out[2,1]=wx;out[2,2]=1.0;return
    
    # FIX: Gimbal lock / division by zero protection near π
    if t > 3.13: 
        t = 3.13
        scale = 3.13 / math.sqrt(t2)
        wx *= scale; wy *= scale; wz *= scale
        
    c=math.cos(t);s=math.sin(t);tc=1.0-c;it=1.0/t;x=wx*it;y=wy*it;z=wz*it
    out[0,0]=tc*x*x+c;out[0,1]=tc*x*y-s*z;out[0,2]=tc*x*z+s*y;out[1,0]=tc*x*y+s*z;out[1,1]=tc*y*y+c;out[1,2]=tc*y*z-s*x;out[2,0]=tc*x*z-s*y;out[2,1]=tc*y*z+s*x;out[2,2]=tc*z*z+c

@njit(cache=True)
def _mat3_mul(A,B,out):
    for i in range(3):
        for j in range(3):
            s=0.0
            for k in range(3): s+=A[i,k]*B[k,j]
            out[i,j]=s

@njit(cache=True)
def _mat3_vec(A,v,out):
    out[0]=A[0,0]*v[0]+A[0,1]*v[1]+A[0,2]*v[2];out[1]=A[1,0]*v[0]+A[1,1]*v[1]+A[1,2]*v[2];out[2]=A[2,0]*v[0]+A[2,1]*v[1]+A[2,2]*v[2]

@njit(cache=True)
def _triple_product_15(F,P,Qd,out):
    tmp=np.empty((15,15))
    for i in range(15):
        for j in range(15):
            s=0.0
            for k in range(15): s+=F[i,k]*P[k,j]
            tmp[i,j]=s
    for i in range(15):
        for j in range(15):
            s=0.0
            for k in range(15): s+=tmp[i,k]*F[j,k]
            out[i,j]=s+Qd[i,j]

@njit(cache=True)
def _symmetrise_15(P):
    for i in range(15):
        for j in range(i+1,15): avg=0.5*(P[i,j]+P[j,i]);P[i,j]=avg;P[j,i]=avg

@njit(cache=True)
def _build_F_and_Qd_jit(F,Qd,R,a_b,w_b,dt,na_var,ng_var,nba_var,nbg_var):
    for i in range(15):
        for j in range(15): F[i,j]=0.0;Qd[i,j]=0.0
        F[i,i]=1.0
    F[0,3]=dt;F[1,4]=dt;F[2,5]=dt;ax,ay,az=a_b[0],a_b[1],a_b[2]
    
    for i in range(3):
        c0 = -R[i,1]*az + R[i,2]*ay
        c1 =  R[i,0]*az - R[i,2]*ax
        c2 = -R[i,0]*ay + R[i,1]*ax
        F[3+i, 6] = c0*dt
        F[3+i, 7] = c1*dt
        F[3+i, 8] = c2*dt

    wx,wy,wz=w_b[0],w_b[1],w_b[2]
    F[6,6]=1.0;F[6,7]=wz*dt;F[6,8]=-wy*dt;F[7,6]=-wz*dt;F[7,7]=1.0;F[7,8]=wx*dt;F[8,6]=wy*dt;F[8,7]=-wx*dt;F[8,8]=1.0
    F[6,12]=-dt;F[7,13]=-dt;F[8,14]=-dt
    Qd[3,3]=Qd[4,4]=Qd[5,5]=na_var*dt;Qd[6,6]=Qd[7,7]=Qd[8,8]=ng_var*dt;Qd[9,9]=Qd[10,10]=Qd[11,11]=nba_var*dt;Qd[12,12]=Qd[13,13]=Qd[14,14]=nbg_var*dt

@njit(cache=True)
def _propagate_state_jit(p,v,R,ba,bg,accel_raw,gyro_raw,dt,gravity_world,F,Qd,P,dR,dR_half,R_mid,a_w_mid,a_b,w_b,w_dt,na_var,ng_var,nba_var,nbg_var,step_count,reortho_interval):
    a_b[0]=accel_raw[0]-ba[0];a_b[1]=accel_raw[1]-ba[1];a_b[2]=accel_raw[2]-ba[2];w_b[0]=gyro_raw[0]-bg[0];w_b[1]=gyro_raw[1]-bg[1];w_b[2]=gyro_raw[2]-bg[2]
    w_dt[0]=w_b[0]*dt*0.5;w_dt[1]=w_b[1]*dt*0.5;w_dt[2]=w_b[2]*dt*0.5;_rodrigues_jit(w_dt[0],w_dt[1],w_dt[2],dR_half);_mat3_mul(R,dR_half,R_mid);_mat3_vec(R_mid,a_b,a_w_mid)
    a_w_mid[0]-=gravity_world[0];a_w_mid[1]-=gravity_world[1];a_w_mid[2]-=gravity_world[2];dt2h=0.5*dt*dt
    p[0]+=v[0]*dt+a_w_mid[0]*dt2h;p[1]+=v[1]*dt+a_w_mid[1]*dt2h;p[2]+=v[2]*dt+a_w_mid[2]*dt2h;v[0]+=a_w_mid[0]*dt;v[1]+=a_w_mid[1]*dt;v[2]+=a_w_mid[2]*dt
    w_dt[0]=w_b[0]*dt;w_dt[1]=w_b[1]*dt;w_dt[2]=w_b[2]*dt;_rodrigues_jit(w_dt[0],w_dt[1],w_dt[2],dR);_mat3_mul(R,dR,dR_half)
    for i in range(3):
        for j in range(3): R[i,j]=dR_half[i,j]
    step_count+=1
    if step_count%reortho_interval==0:
        n0=math.sqrt(R[0,0]**2+R[0,1]**2+R[0,2]**2)
        if n0>1e-12: R[0,0]/=n0;R[0,1]/=n0;R[0,2]/=n0
        d=R[1,0]*R[0,0]+R[1,1]*R[0,1]+R[1,2]*R[0,2];R[1,0]-=d*R[0,0];R[1,1]-=d*R[0,1];R[1,2]-=d*R[0,2]
        n1=math.sqrt(R[1,0]**2+R[1,1]**2+R[1,2]**2)
        if n1>1e-12: R[1,0]/=n1;R[1,1]/=n1;R[1,2]/=n1
        R[2,0]=R[0,1]*R[1,2]-R[0,2]*R[1,1];R[2,1]=R[0,2]*R[1,0]-R[0,0]*R[1,2];R[2,2]=R[0,0]*R[1,1]-R[0,1]*R[1,0]
    _build_F_and_Qd_jit(F,Qd,R,a_b,w_b,dt,na_var,ng_var,nba_var,nbg_var);_triple_product_15(F,P,Qd,P);_symmetrise_15(P)
    return step_count

@njit(cache=True)
def _batch_depth_lookup_jit(depth_map,xs,ys,r,h,w,min_mm,max_mm):
    n=len(xs);result=np.zeros(n,dtype=np.float64)
    for idx in range(n):
        xi=int(round(xs[idx]));yi=int(round(ys[idx]))
        if xi<r: xi=r
        if xi>=w-r: xi=w-r-1
        if yi<r: yi=r
        if yi>=h-r: yi=h-r-1
        vc=0;ps=(2*r+1)*(2*r+1);vals=np.empty(ps,dtype=np.float64)
        for dy in range(-r,r+1):
            for dx in range(-r,r+1):
                d=float(depth_map[yi+dy,xi+dx])
                if d>=min_mm and d<=max_mm: vals[vc]=d;vc+=1
        if vc>0:
            for i in range(vc):
                for j in range(i+1,vc):
                    if vals[j]<vals[i]: vals[i],vals[j]=vals[j],vals[i]
            result[idx]=vals[vc//2]
    return result

@njit(cache=True)
def _mat_to_rotvec_jit(R):
    val=(R[0,0]+R[1,1]+R[2,2]-1.0)*0.5
    if val>1.0: val=1.0
    if val<-1.0: val=-1.0
    theta=math.acos(val);out=np.zeros(3)
    if theta<1e-9: return out
    k=theta/(2.0*math.sin(theta))
    out[0]=(R[2,1]-R[1,2])*k;out[1]=(R[0,2]-R[2,0])*k;out[2]=(R[1,0]-R[0,1])*k;return out

import cv2

class VIO_EKF:
    def __init__(self):
        self._lock = threading.Lock()
        self.p=np.zeros(3); self.v=np.zeros(3)
        self.R=np.eye(3); self.ba=np.zeros(3); self.bg=np.zeros(3)
        self.P=np.diag([1e-6]*3+[1e-4]*3+[1e-6]*3+[1e-4]*3+[1e-4]*3).astype(np.float64)
        self._R_vis=np.diag([VIS_NOISE_P]*3+[VIS_NOISE_PHI]*3).astype(np.float64)
        self._I15=np.eye(15); self._I3=np.eye(3)
        self._F=np.zeros((15,15)); self._Qd=np.zeros((15,15))
        self._dR=np.eye(3); self._dR_half=np.eye(3); self._R_mid=np.eye(3)
        self._a_w_mid=np.zeros(3); self._a_b=np.zeros(3)
        self._w_b=np.zeros(3); self._w_dt=np.zeros(3)
        self._tmp33=np.zeros((3,3)); self._P_tmp=np.zeros((15,15))
        self.gravity_world=None; self.gravity_ready=False
        self._var_tracker=RunningVariance(STATIC_WIN)
        self._still_accels=[]; self.last_imu_ts=None
        self._kf_p=np.zeros(3); self._kf_R=np.eye(3); self._kf_set=False
        self._step_count=0
        
        self._starvation_ticks = 0 # COVARIANCE INFLATION TRACKER
        
        self._last_v_p = None
        self._last_v_R = None
        self.residual_log = [] 

    def feed_imu(self, a, g, ts):
        with self._lock: 
            # --- IMU SHOCK ABSORBER ---
            a_clipped = np.clip(a, -25.0, 25.0)
            g_clipped = np.clip(g, -5.0, 5.0)
            
            self._propagate(a_clipped, g_clipped, ts)
            
            # --- NaN QUARANTINE ---
            if np.isnan(self.p).any() or np.isnan(self.R).any():
                print("  [CRITICAL] NaN detected in IMU Propagation! Reverting state.")
                if self._last_v_p is not None:
                    self.p[:] = self._last_v_p
                    self.R[:] = self._last_v_R
                    self.v[:] = np.zeros(3)

    def _propagate(self, accel_raw, gyro_raw, ts):
        norm=math.sqrt(accel_raw[0]**2+accel_raw[1]**2+accel_raw[2]**2)
        self._var_tracker.push(norm)
        
        if not self.gravity_ready:
            is_s = self._var_tracker.is_full() and self._var_tracker.variance() < STATIC_VAR_THR
            if is_s:
                self._still_accels.append(accel_raw.copy())
                if len(self._still_accels) >= MIN_GRAV_SAMPLES:
                    # FIX: RANSAC style outlier rejection for gravity
                    samples = np.array(self._still_accels)
                    norms = np.linalg.norm(samples, axis=1)
                    median_norm = float(np.median(norms))
                    
                    # Keep samples within +/- 5% of median
                    inlier_mask = np.abs(norms - median_norm) < 0.05 * median_norm
                    inliers = samples[inlier_mask]
                    
                    if len(inliers) >= MIN_GRAV_SAMPLES // 2:
                        gb = np.mean(inliers, axis=0)
                        gm = float(np.linalg.norm(gb))
                        if 9.5 <= gm <= 10.5:
                            gu = gb / gm
                            zd = np.array([0., 0., -1.])
                            v = np.cross(gu, zd)
                            s = np.linalg.norm(v)
                            c = np.dot(gu, zd)
                            if s < 1e-8: 
                                Ra = np.eye(3) if c > 0 else np.diag([1., -1., -1.])
                            else:
                                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                                Ra = np.eye(3) + vx + vx@vx * ((1. - c) / (s * s))
                            self.R[:] = Ra
                            self.gravity_world = np.array([0., 0., -gm])
                            self.gravity_ready = True
                            print(f"\n  [EKF] Gravity: ‖g‖={gm:.4f} m/s^2 ({len(inliers)}/{len(samples)} inliers)")
                        else:
                            print(f"  [WARN] Gravity magnitude {gm:.2f} m/s^2 unrealistic. Recalibrating...")
                            self._still_accels = []
                    else:
                        print(f"  [WARN] Too many gravity outliers. Recollecting...")
                        self._still_accels = []
            else:
                self._still_accels = []
            self.last_imu_ts = ts
            return
            
        if self.last_imu_ts is None: 
            self.last_imu_ts = ts
            return
            
        dt = ts - self.last_imu_ts
        self.last_imu_ts = ts
        
        if dt <= 0 or dt > 0.1: 
            return
            
        self._step_count = _propagate_state_jit(
            self.p, self.v, self.R, self.ba, self.bg, accel_raw, gyro_raw, dt,
            self.gravity_world, self._F, self._Qd, self.P,
            self._dR, self._dR_half, self._R_mid, self._a_w_mid,
            self._a_b, self._w_b, self._w_dt,
            ACCEL_ND**2, GYRO_ND**2, ACCEL_BRW**2, GYRO_BRW**2,
            self._step_count, REORTHO_INTERVAL
        )

    def set_keyframe(self):
        with self._lock: 
            self._kf_p[:] = self.p
            self._kf_R[:] = self.R
            self._kf_set = True
        
    def get_angular_velocity(self):
        with self._lock: 
            return self._w_b.copy()

    def update_visual(self, pts_p, pts_c, depth_p, K, T_ic, T_ci):
        if len(pts_p) < MIN_FEAT_UPDATE: 
            with self._lock:
                self._starvation_ticks += 1
                if self._starvation_ticks > 500:
                    # FIX: Additive position uncertainty inflation during extended dropout
                    self.P[0:3, 0:3] += np.eye(3) * (0.01**2)
            return False, 0
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        h, w = depth_p.shape
        pxp, pyp = pts_p[:, 0], pts_p[:, 1]
        
        dmm = _batch_depth_lookup_jit(depth_p, pxp, pyp, DEPTH_PATCH_R, h, w, DEPTH_MIN_MM, DEPTH_MAX_MM)
        
        valid = dmm > 0
        if valid.sum() < MIN_FEAT_UPDATE: 
            with self._lock:
                self._starvation_ticks += 1
                if self._starvation_ticks > 500:
                    self.P[0:3, 0:3] += np.eye(3) * (0.01**2)
            return False, 0
        
        p_val = pts_p[valid]
        c_val = pts_c[valid]
        d_val = dmm[valid] / 1000.0

        # --- GEOMETRIC DEGENERACY GATE ---
        if len(p_val) > 0:
            var_x = np.var(p_val[:, 0])
            var_y = np.var(p_val[:, 1])
            if var_x < 1000.0 or var_y < 1000.0:
                with self._lock:
                    self._starvation_ticks += 1
                    if self._starvation_ticks > 500:
                        self.P[0:3, 0:3] += np.eye(3) * (0.01**2)
                return False, 0
        
        X = (p_val[:, 0] - cx) * d_val / fx
        Y = (p_val[:, 1] - cy) * d_val / fy
        Z = d_val
        p3d = np.column_stack([X, Y, Z])
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            p3d, c_val, K, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=3.0, iterationsCount=100
        )
        
        n_inliers = len(inliers) if inliers is not None else 0
        if not success or n_inliers < MIN_FEAT_UPDATE: 
            with self._lock:
                self._starvation_ticks += 1
                if self._starvation_ticks > 500:
                    self.P[0:3, 0:3] += np.eye(3) * (0.01**2)
            return False, n_inliers
            
        ni = n_inliers
        
        R_c_p, _ = cv2.Rodrigues(rvec)
        R_p_c = R_c_p.T
        t_p_c = (-R_p_c @ tvec).flatten()

        # --- EXTRINSIC FRAME ALIGNMENT ---
        T_c_curr_c_prev = np.eye(4)
        T_c_curr_c_prev[:3, :3] = R_p_c
        T_c_curr_c_prev[:3, 3] = t_p_c

        T_i_curr_i_prev = T_ic @ T_c_curr_c_prev @ T_ci
        R_p_c_imu = T_i_curr_i_prev[:3, :3]
        t_p_c_imu = T_i_curr_i_prev[:3, 3]
        
        # --- DYNAMIC NOISE SCALING ---
        inlier_idx = inliers.flatten()
        mean_depth = float(np.mean(d_val[inlier_idx]))
        depth_scale = max(1.0, mean_depth**2)
        
        with self._lock:
            if self._last_v_p is None:
                self._last_v_p = self.p.copy()
                self._last_v_R = self.R.copy()
                return True, ni
                
            t_meas_world = self._last_v_p + (self._last_v_R @ t_p_c_imu)
            dpi_world = t_meas_world - self.p
            
            R_meas_world = self._last_v_R @ R_p_c_imu
            
            # --- PURE LOCAL ERROR STATE FORMULATION ---
            R_err = self.R.T @ R_meas_world
            dphi_body = _mat_to_rotvec_jit(R_err)
            
            innov = np.concatenate([dpi_world, dphi_body])
            
            trans_err_m = float(np.linalg.norm(dpi_world))
            rot_err_deg = float(np.linalg.norm(dphi_body)) * (180.0 / math.pi)
            self.residual_log.append({"tick": self._step_count, "inliers": ni, "trans_err_m": trans_err_m, "rot_err_deg": rot_err_deg})
            
            H = np.zeros((6, 15))
            H[:3, :3] = self._I3
            H[3:6, 6:9] = self._I3
            
            R_vis_dyn = self._R_vis.copy()
            R_vis_dyn[0, 0] *= depth_scale
            R_vis_dyn[1, 1] *= depth_scale
            R_vis_dyn[2, 2] *= depth_scale
            
            S = H @ self.P @ H.T + R_vis_dyn
            
            # FIX: Tikhonov regularization for numerical stability
            lambda_reg = 1e-6
            S_reg = S + np.eye(6) * lambda_reg
            
            try: 
                Si = np.linalg.inv(S_reg)
            except np.linalg.LinAlgError: 
                print("  [EKF] Covariance matrix is singular. Skipping update.")
                self._starvation_ticks += 1
                return False, ni
            
            # --- MAHALANOBIS GATE ---
            mahalanobis_sq = innov.T @ Si @ innov
            if mahalanobis_sq > 16.81: # FIX: Stricter gate (16.81 is 99% confidence for 6DOF)
                self._starvation_ticks += 1
                return False, ni 
            
            K_gain = self.P @ H.T @ Si
            dx = K_gain @ innov
            
            self.p += dx[:3]; self.v += dx[3:6]
            _rodrigues_jit(dx[6], dx[7], dx[8], self._tmp33)
            
            _mat3_mul(self.R, self._tmp33, self._dR)
            self.R[:] = self._dR
            
            self.ba += dx[9:12]; self.bg += dx[12:15]
            
            IKH = self._I15 - K_gain @ H
            self.P[:] = IKH @ self.P @ IKH.T + K_gain @ R_vis_dyn @ K_gain.T
            _symmetrise_15(self.P)
            
            self._starvation_ticks = 0 # Update successful, reset starvation
            
            # --- NaN QUARANTINE (VISUAL) ---
            if np.isnan(self.p).any() or np.isnan(self.R).any() or np.isnan(self.P).any():
                print("  [CRITICAL] NaN detected in Visual Update! Reverting state.")
                self.p[:] = self._last_v_p
                self.R[:] = self._last_v_R
                self.P[:] = np.eye(15) * 1e-3 # Reset covariance safely
                return False, ni

            self._last_v_p = self.p.copy()
            self._last_v_R = self.R.copy()
            
        return True, ni

    def get_pose(self):
        with self._lock:
            T = np.eye(4)
            T[:3,:3] = self.R.copy(); T[:3,3] = self.p.copy()
            idx = [0,1,2,6,7,8]
            c6 = self.P[np.ix_(idx,idx)].copy()
        return T, c6

    def is_ready(self):
        with self._lock: return self.gravity_ready
