import time
import psutil
import threading
import numpy as np
import cv2
# Numba Auto-Detect
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]): return args[0]
        def decorator(func): return func
        return decorator

def _rss():  
    return psutil.Process().memory_info().rss / (1024**3)

def _avail(): 
    return psutil.virtual_memory().available / (1024**3)

def _vram():
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
            capture_output=True, text=True, timeout=2
        )
        if r.returncode == 0:
            u, t = r.stdout.strip().split(', ')
            return f"VRAM={int(u)}/{int(t)}MB"
    except Exception: 
        pass
    return ""

def mem():
    s = f"RSS={_rss():.1f}GB avail={_avail():.1f}GB"
    v = _vram()
    if v: s += f" {v}"
    return s

def fast_underwater_restore(frame, r_max=3.0, g_max=1.2):
    """Fast channel scaling to recover red/green with soft clipping."""
    b, g, r = cv2.split(frame)

    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)

    # Calculate dynamic scale, bounded by the config limits
    r_scale = min((b_mean / max(r_mean, 1.0)), r_max)
    g_scale = min((b_mean / max(g_mean, 1.0)), g_max)

    # Use float32 for math to prevent integer wrap-around/harsh clipping
    r_restored = np.clip(r.astype(np.float32) * r_scale, 0, 255).astype(np.uint8)
    g_restored = np.clip(g.astype(np.float32) * g_scale, 0, 255).astype(np.uint8)

    return cv2.merge([b, g_restored, r_restored])

class ETA:
    def __init__(self, n, nm="", ev=20):
        self.n = n
        self.nm = nm
        self.t0 = time.perf_counter()
        self.i = 0
        self.ev = ev
        
    def tick(self):
        self.i += 1
        if self.i % self.ev == 0 or self.i == self.n:
            dt = time.perf_counter() - self.t0
            r = self.i / dt if dt > 0 else 1
            l = (self.n - self.i) / r if r > 0 else 0
            print(f"  [{self.nm}] {self.i}/{self.n} {dt:.0f}s ~{l:.0f}s left [{mem()}]")

class PromiseLRU:
    def __init__(self, capacity=40): 
        self._d = {}
        self._q = []
        self._c = capacity
        self._lock = threading.Lock()
        self._in_progress = {} 
        
    def get_or_compute(self, key, compute_func):
        with self._lock:
            if key in self._d:
                self._q.remove(key)
                self._q.append(key)
                return self._d[key]
            
            if key not in self._in_progress:
                self._in_progress[key] = threading.Event()
                is_first = True
            else:
                is_first = False

        if not is_first:
            self._in_progress[key].wait() 
            with self._lock: return self._d.get(key) 

        result = None
        try:
            result = compute_func(key)
        except Exception as e:
            print(f"  [CACHE ERROR] Failed to compute key {key}: {e}")
        finally:
            with self._lock:
                if result is not None:
                    if len(self._d) >= self._c:
                        oldest = self._q.pop(0)
                        self._d.pop(oldest, None)
                    self._d[key] = result
                    self._q.append(key)
                if key in self._in_progress:
                    self._in_progress[key].set() 
                    del self._in_progress[key]
        return result

    def clear(self): 
        with self._lock:
            self._d.clear()
            self._q.clear()
            self._in_progress.clear()

class DepthDoubleBuffer:
    def __init__(self, h, w):
        self._bufs = [np.zeros((h,w), dtype=np.uint16), np.zeros((h,w), dtype=np.uint16)]
        self._write_idx = 0
        self._read_idx = -1
        self._lock = threading.Lock()
        
    def write(self, frame):
        wi = self._write_idx
        with self._lock:
            np.copyto(self._bufs[wi], frame)
            self._read_idx = wi
            self._write_idx = 1 - wi
            
    def read(self):
        with self._lock: ri = self._read_idx
        return self._bufs[ri] if ri >= 0 else None

class RunningVariance:
    __slots__ = ('_buf','_n','_idx','_sum','_sum2','_full')
    def __init__(self, window):
        self._buf=np.zeros(window,dtype=np.float64)
        self._n=window
        self._idx=0
        self._sum=0.0
        self._sum2=0.0
        self._full=False
        
    def push(self, val):
        old = self._buf[self._idx]
        self._buf[self._idx] = val
        self._idx = (self._idx+1) % self._n
        if not self._full:
            self._sum += val
            self._sum2 += val*val
            if self._idx == 0: self._full = True
        else: 
            self._sum += val - old
            self._sum2 += val*val - old*old
            
    def is_full(self): return self._full
        
    def variance(self):
        if not self._full: return 1e9
        m = self._sum / self._n
        return self._sum2 / self._n - m*m
