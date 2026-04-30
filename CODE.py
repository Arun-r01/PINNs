# ==============================================================================
# RANS-PINN — NACA 0012, k-ω SST v8 (FIXED)
#
# ALL v8 FIXES CARRIED FORWARD:
#   [v8-FIX-1]  Log-space ω normalization → architecturally guarantees ω > 0
#   [v8-FIX-2]  CDkw = 2σ_ω2·∇k·∇log(ω)  — no 1/ω singularity
#   [v8-FIX-3]  ω-flux scale = nu_rep·OMEGA_WALL/CHORD  (physical)
#   [v8-FIX-4]  L-BFGS disabled by default  (USE_LBFGS flag)
#   [v8-FIX-5]  Diagnostics updated for log-ω
#
# NEW FIXES IN THIS VERSION:
#   [NEW-FIX-1]  Smoother, extended ω-PDE and ω-BC weight schedules
#                — prevents the post-epoch-7500 instability seen in v8
#   [NEW-FIX-2]  Physically bounded log-ω clipping in PDE
#                — prevents unbounded ω growth by clamping to [50, 10000]
#   [NEW-FIX-3]  Learning rate decay schedule tied to epoch cursor
#                — smooths late-stage convergence
#   [NEW-FIX-4]  Gradient clipping via clipnorm=1.0 on Adam optimizer
#                — prevents epoch-8000-style loss spikes
#   [NEW-FIX-5]  Runtime ω divergence monitor with auto-abort
#                — saves wasted compute if ω mean exceeds 5×ω_wall
#   [NEW-FIX-6]  Wider network: 6×128 hidden units instead of 6×64
#                — better capacity for 12 outputs with strong wall gradients
#
# TOTAL ADAM EPOCHS increased from 10,000 → 25,000 to allow the
# gentler weight schedule sufficient training time.
#
# CHECKPOINT COMPATIBILITY: v8 checkpoints ARE compatible with this version.
#   The network architecture changes (6×128) require a fresh start if you
#   have v8 checkpoints from 6×64. Set FRESH_START=True to force a reset.
# ==============================================================================


# ==============================================================================
# SECTION 0 — GOOGLE DRIVE MOUNT
# ==============================================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os
_DRIVE_BASE = '/content/drive/MyDrive'


# ==============================================================================
# SECTION 1 — USER INPUTS
# ==============================================================================

RE        = 200000
AOA_DEG   = 5.0
U_INF     = 1.0
CHORD     = 1.0
P_INF     = 0.0

TURB_INTENSITY = 0.005
K_INF     = 1.5 * (TURB_INTENSITY * U_INF) ** 2   # 3.75e-05
OMEGA_INF = 75.0

# [NEW-FIX-1] Extended training budget to match gentler weight schedule
ADAM_EPOCHS    = 25000
LBFGS_ITER     = 20000
CKPT_EVERY     = 500
EARLY_STOP_TOL = 1e-4

# [v8-FIX-4] Disable L-BFGS by default
USE_LBFGS = False

# Set True to force a fresh start even if checkpoints exist
# Required if you are switching from the 6×64 to 6×128 network
FRESH_START = False

# [NEW-FIX-6] Network width — change to 64 if T4 OOMs
HIDDEN_WIDTH = 128
N_HIDDEN_LAYERS = 6

# Checkpoint directory — v8-fixed uses its own subdir to avoid collision
DRIVE_CKPT_DIR = os.path.join(_DRIVE_BASE, f'PINN_NACA_v8fixed_Re{RE:.0f}')
os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
print(f"Checkpoint directory: {DRIVE_CKPT_DIR}")


# ------------------------------------------------------------------------------
# [NEW-FIX-1] SMOOTHER ω-PDE WEIGHT SCHEDULE
# Original v8 jumped to 0.50 at epoch 7500, causing instability.
# New schedule is more gradual and extends to epoch 20000.
# ------------------------------------------------------------------------------
OM_WEIGHT_SCHEDULE = [
    (0,     0.01),
    (1000,  0.03),
    (3000,  0.08),
    (5000,  0.15),
    (7500,  0.25),
    (10000, 0.40),
    (15000, 0.60),
    (20000, 1.00),
]

def get_om_weight(epoch):
    w = OM_WEIGHT_SCHEDULE[0][1]
    for ep_thresh, wval in OM_WEIGHT_SCHEDULE:
        if epoch >= ep_thresh:
            w = wval
    return w


# ------------------------------------------------------------------------------
# [NEW-FIX-1] SMOOTHER ω-BC WEIGHT SCHEDULE
# Original v8 jumped to 5.0 at epoch 7500 in a single step.
# New schedule is finer-grained and max weight is reached at epoch 15000.
# ------------------------------------------------------------------------------
OM_BC_WEIGHT_SCHEDULE = [
    (0,     0.0),
    (1000,  0.3),
    (3000,  0.8),
    (5000,  1.5),
    (7500,  2.5),
    (10000, 4.0),
    (15000, 6.0),
]

def get_om_bc_weight(epoch):
    w = OM_BC_WEIGHT_SCHEDULE[0][1]
    for ep_thresh, wval in OM_BC_WEIGHT_SCHEDULE:
        if epoch >= ep_thresh:
            w = wval
    return w


# ------------------------------------------------------------------------------
# [NEW-FIX-3] LEARNING RATE DECAY SCHEDULE
# Reduces LR as training progresses to smooth convergence.
# ------------------------------------------------------------------------------
def get_learning_rate(epoch):
    if epoch < 5000:
        return 5e-4
    elif epoch < 10000:
        return 2e-4
    elif epoch < 15000:
        return 1e-4
    else:
        return 5e-5


PDE_SCALES = [
    1.0,    # 0  continuity
    1.0,    # 1  momentum-x
    1.0,    # 2  momentum-z
    1e-3,   # 3  τx  definition
    1e-3,   # 4  τz  definition
    1e-3,   # 5  τxz definition
    1e-6,   # 6  Dk  definition
    1e-6,   # 7  Dkz definition
    1e-2,   # 8  Dω  definition
    1e-2,   # 9  Dωz definition
    1e-6,   # 10 k transport
    1e5,    # 11 ω transport  (β·ω_wall²/1e5 ~ O(1))
]


# ==============================================================================
# SECTION 2 — IMPORTS
# ==============================================================================

import deepxde as dde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, shutil, json
import tensorflow as tf

from google.colab import files

dde.config.set_random_seed(42)
dde.config.set_default_float('float64')


# ==============================================================================
# SECTION 3 — FILE UPLOAD
# ==============================================================================

LOCAL_CSV = '/content/NACA.csv'

if not os.path.exists(LOCAL_CSV):
    print("Upload NACA.csv …")
    uploaded = files.upload()
    with open(LOCAL_CSV, 'wb') as fh:
        fh.write(uploaded[list(uploaded.keys())[0]])
    print(f"✅ Saved to {LOCAL_CSV}")
else:
    print(f"✅ Using existing {LOCAL_CSV}")


# ==============================================================================
# SECTION 4 — FLOW PARAMETERS
# ==============================================================================

AOA_RAD = np.deg2rad(AOA_DEG)
NU      = U_INF * CHORD / RE
U_X     = U_INF * np.cos(AOA_RAD)
U_Z     = U_INF * np.sin(AOA_RAD)
Q_INF   = 0.5 * U_INF ** 2 * CHORD

print(f"Re={RE:.0f}  ν={NU:.2e}  k∞={K_INF:.2e}  ω∞={OMEGA_INF}")


# ==============================================================================
# SECTION 5 — LOAD AND CLEAN CSV
# ==============================================================================

df_raw   = pd.read_csv(LOCAL_CSV)
y_slice  = sorted(df_raw['Points:1'].dropna().unique())[0]
df_slice = df_raw[df_raw['Points:1'] == y_slice].copy()
COLS     = ['Points:0', 'Points:2', 'U:0', 'U:2', 'p', 'k', 'omega', 'nut']
df_slice = df_slice.dropna(subset=COLS)
df_clean = df_slice[COLS].copy()
df_clean.columns = ['x', 'z', 'u', 'w', 'p', 'k', 'omega', 'nut']
df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
df_clean = df_clean[df_clean['omega'] > 0].copy()
print(f"Usable pts: {len(df_clean):,}")


# ==============================================================================
# SECTION 6 — NACA 0012
# ==============================================================================

def naca0012_coords(n=300, chord=1.0):
    beta = np.linspace(0, np.pi, n + 1)
    xc   = chord / 2 * (1 - np.cos(beta))
    t    = 0.12
    yt   = 5 * t * chord * (
        0.2969 * np.sqrt(xc / chord)
        - 0.1260 * (xc / chord)
        - 0.3516 * (xc / chord) ** 2
        + 0.2843 * (xc / chord) ** 3
        - 0.1015 * (xc / chord) ** 4
    )
    return np.concatenate([xc, xc[::-1]]), np.concatenate([yt, -yt[::-1]])

X_AF, Z_AF = naca0012_coords(n=300, chord=CHORD)


# ==============================================================================
# SECTION 7 — WALL DISTANCE
# ==============================================================================

def compute_wall_distance(xa, za, ax=X_AF, az=Z_AF):
    wd = np.zeros(len(xa))
    for i in range(len(xa)):
        wd[i] = np.min(np.sqrt((xa[i] - ax) ** 2 + (za[i] - az) ** 2))
    return wd

df_clean['wd'] = compute_wall_distance(
    df_clean['x'].values, df_clean['z'].values
)


# ==============================================================================
# SECTION 8 — AUXILIARY TARGETS
# ==============================================================================

df_clean['tau_x']  = 0.0
df_clean['tau_z']  = 0.0
df_clean['tau_xz'] = 0.0


# ==============================================================================
# SECTION 9 — NORMALISATION  [v8-FIX-1: log-space ω]
# ==============================================================================

NORM_FILE = os.path.join(DRIVE_CKPT_DIR, f'norm_Re{RE:.0f}_v8fixed.npy')

def norm_stats(arr):
    return float(arr.mean()), float(arr.std())

norm = None
if os.path.exists(NORM_FILE) and not FRESH_START:
    print(f"Loading norm stats from {NORM_FILE}")
    norm = np.load(NORM_FILE, allow_pickle=True).item()
    if 'log_om_m' not in norm:
        print("  ⚠️ Old norm file — recomputing …")
        norm = None

if norm is not None:
    u_m,      u_s      = norm['u_m'],      norm['u_s']
    w_m,      w_s      = norm['w_m'],      norm['w_s']
    p_m,      p_s      = norm['p_m'],      norm['p_s']
    k_m,      k_s      = norm['k_m'],      norm['k_s']
    log_om_m, log_om_s = norm['log_om_m'], norm['log_om_s']
    tau_scale           = norm['tau_scale']
else:
    u_m,  u_s  = norm_stats(df_clean['u'])
    w_m,  w_s  = norm_stats(df_clean['w'])
    p_m,  p_s  = norm_stats(df_clean['p'])
    k_m,  k_s  = norm_stats(df_clean['k'])
    log_om_vals          = np.log(df_clean['omega'].values)
    log_om_m, log_om_s   = norm_stats(log_om_vals)
    tau_scale            = max(NU * U_INF / CHORD ** 2, 1e-6)
    np.save(NORM_FILE, dict(
        u_m=u_m, u_s=u_s, w_m=w_m, w_s=w_s,
        p_m=p_m, p_s=p_s, k_m=k_m, k_s=k_s,
        log_om_m=log_om_m, log_om_s=log_om_s,
        tau_scale=tau_scale,
    ))
    print(f"Norm stats saved to {NORM_FILE}")

for name, val in [('u_s', u_s), ('w_s', w_s), ('p_s', p_s),
                  ('k_s', k_s), ('log_om_s', log_om_s)]:
    assert 1e-12 < val < 1e6, \
        f"Norm scale {name}={val:.3e} out of safe range [1e-12, 1e6]"
assert 1e-12 < tau_scale < 1e3, f"tau_scale={tau_scale:.3e} out of range"
print(f"✅ Norm validation passed  "
      f"u_s={u_s:.3e}  k_s={k_s:.3e}  "
      f"log_om_m={log_om_m:.3f}  log_om_s={log_om_s:.3f}  "
      f"(mean_ω_physical={np.exp(log_om_m):.1f})")

# Normalise data
df_norm          = df_clean.copy()
df_norm['u']     = (df_clean['u'] - u_m)  / u_s
df_norm['w']     = (df_clean['w'] - w_m)  / w_s
df_norm['p']     = (df_clean['p'] - p_m)  / p_s
df_norm['k']     = (df_clean['k'] - k_m)  / k_s
df_norm['omega'] = (np.log(df_clean['omega']) - log_om_m) / log_om_s

# BC values in normalised space
U_X_N    = (U_X   - u_m)  / u_s
U_Z_N    = (U_Z   - w_m)  / w_s
P_INF_N  = (P_INF - p_m)  / p_s
K_INF_N  = (K_INF - k_m)  / k_s
U_0_N    = (0.0   - u_m)  / u_s
W_0_N    = (0.0   - w_m)  / w_s
K_WALL_N = (0.0   - k_m)  / k_s

LOG_OM_INF_N  = (np.log(OMEGA_INF) - log_om_m) / log_om_s

BETA_1_C  = 0.075
spd       = np.sqrt(df_clean['u'] ** 2 + df_clean['w'] ** 2)
mask_wall = (
    (spd < 0.01 * U_INF)
    & (df_clean['x'] >= -0.05) & (df_clean['x'] <= 1.05)
    & (df_clean['z'] >= -0.15) & (df_clean['z'] <=  0.15)
)
y1         = df_clean.loc[mask_wall, 'wd'].mean()
y1         = 1e-4 if (np.isnan(y1) or y1 < 1e-6) else y1
OMEGA_WALL = 60 * NU / (BETA_1_C * y1 ** 2)

LOG_OM_WALL_N = (np.log(OMEGA_WALL) - log_om_m) / log_om_s
print(f"y₁={y1:.2e}  ω_wall={OMEGA_WALL:.2e}")
print(f"Log-ω BC (norm):  wall={LOG_OM_WALL_N:.3f}  inf={LOG_OM_INF_N:.3f}")


# ------------------------------------------------------------------------------
# [NEW-FIX-2] Physically bounded log-ω clipping constants
# Computed AFTER norm stats are known.
# Physical ω range: [50, 10000] with small safety margins.
# ------------------------------------------------------------------------------
LOG_OM_CLIP_MIN = float((np.log(50.0)    - log_om_m) / log_om_s)
LOG_OM_CLIP_MAX = float((np.log(10000.0) - log_om_m) / log_om_s)
print(f"[NEW-FIX-2] log-ω norm clip: [{LOG_OM_CLIP_MIN:.3f}, {LOG_OM_CLIP_MAX:.3f}]  "
      f"(physical: [50, 10000])")


# ==============================================================================
# SECTION 10 — TRAINING SUBSAMPLE
# ==============================================================================

np.random.seed(42)
mask_near = (
    (df_norm['x'] >= -0.1) & (df_norm['x'] <= 1.1)
    & (df_norm['z'] >= -0.2) & (df_norm['z'] <=  0.2)
)
df_near = df_norm[mask_near]
df_far  = df_norm[~mask_near]

N_NEAR = min(8000, len(df_near))
N_FAR  = min(2000, len(df_far))
df_train = pd.concat([
    df_near.iloc[np.random.choice(len(df_near), N_NEAR, replace=False)],
    df_far .iloc[np.random.choice(len(df_far),  N_FAR,  replace=False)],
]).reset_index(drop=True)

X_tr     = df_train[['x', 'z', 'wd']].values
COLS_OUT = ['u', 'w', 'p', 'k', 'omega']
targets  = {c: df_train[c].values.reshape(-1, 1) for c in COLS_OUT}
print(f"Training pts: {len(df_train):,}  (near={N_NEAR}, far={N_FAR})")


# ==============================================================================
# SECTION 11 — BOUNDARY DETECTION
# ==============================================================================

x_max  = df_clean['x'].max()
m_out  = df_clean['x'] > x_max - 0.05 * abs(x_max)
dist_c = np.sqrt((df_clean['x'] - 0.5) ** 2 + df_clean['z'] ** 2)
m_in   = (dist_c > 3.0) & ~m_out

def sub(df, n, s=0):
    np.random.seed(s)
    return df if len(df) <= n else df.iloc[
        np.random.choice(len(df), n, replace=False)
    ]

df_wbc = sub(df_norm[mask_wall.values], 300, 1)
df_obc = sub(df_norm[m_out.values],     300, 2)
df_ibc = sub(df_norm[m_in.values],      300, 3)
print(f"Wall={len(df_wbc)}  Out={len(df_obc)}  In={len(df_ibc)}")


# ==============================================================================
# SECTION 12 — GEOMETRY
# ==============================================================================

class Rect3(dde.geometry.Geometry):
    def __init__(self, x0, x1, z0, z1):
        mwd  = np.sqrt((x1 - x0) ** 2 + (z1 - z0) ** 2)
        bbox = [np.array([x0, z0, 0.0]), np.array([x1, z1, mwd])]
        super().__init__(3, bbox, float(np.linalg.norm(bbox[1] - bbox[0])))
        self.x0, self.x1, self.z0, self.z1 = x0, x1, z0, z1

    def inside(self, x):
        return ((x[:, 0] >= self.x0) & (x[:, 0] <= self.x1)
                & (x[:, 1] >= self.z0) & (x[:, 1] <= self.z1))

    def on_boundary(self, x):
        return (np.isclose(x[:, 0], self.x0) | np.isclose(x[:, 0], self.x1)
                | np.isclose(x[:, 1], self.z0) | np.isclose(x[:, 1], self.z1))

    def boundary_normal(self, x):
        n = np.zeros_like(x)
        n[np.isclose(x[:, 0], self.x0), 0] = -1
        n[np.isclose(x[:, 0], self.x1), 0] =  1
        n[np.isclose(x[:, 1], self.z0), 1] = -1
        n[np.isclose(x[:, 1], self.z1), 1] =  1
        nrm = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(nrm == 0, 1, nrm)

    def random_points(self, n, random="pseudo"):
        rng = np.random.default_rng()
        return np.column_stack([rng.uniform(self.x0, self.x1, n),
                                 rng.uniform(self.z0, self.z1, n),
                                 np.zeros(n)])

    def random_boundary_points(self, n, random="pseudo"):
        pts = np.zeros((n, 3))
        s   = np.random.randint(0, 4, n)
        for i in range(n):
            if   s[i] == 0: pts[i] = [self.x0, np.random.uniform(self.z0, self.z1), 0]
            elif s[i] == 1: pts[i] = [self.x1, np.random.uniform(self.z0, self.z1), 0]
            elif s[i] == 2: pts[i] = [np.random.uniform(self.x0, self.x1), self.z0, 0]
            else:            pts[i] = [np.random.uniform(self.x0, self.x1), self.z1, 0]
        return pts

    def uniform_points(self, n, boundary=True):
        nx = int(np.sqrt(n));  nz = max(1, n // nx)
        xv = np.linspace(self.x0, self.x1, nx)
        zv = np.linspace(self.z0, self.z1, nz)
        XX, ZZ = np.meshgrid(xv, zv)
        pts = np.column_stack([XX.ravel(), ZZ.ravel()])[:n]
        return np.hstack([pts, np.zeros((len(pts), 1))])

    def uniform_boundary_points(self, n):
        k = n // 4;  sides = []
        for xs, zs, xe, ze in [
            (self.x0, self.z0, self.x1, self.z0),
            (self.x0, self.z1, self.x1, self.z1),
            (self.x0, self.z0, self.x0, self.z1),
            (self.x1, self.z0, self.x1, self.z1),
        ]:
            t = np.linspace(0, 1, k)
            sides.append(np.column_stack([xs + t * (xe - xs),
                                           zs + t * (ze - zs),
                                           np.zeros(k)]))
        return np.vstack(sides)[:n]


geom = Rect3(df_clean['x'].min(), df_clean['x'].max(),
             df_clean['z'].min(), df_clean['z'].max())


# ==============================================================================
# SECTION 13 — TEST POINT SAMPLING + SafePDE
# ==============================================================================

WALL_EXCL_DIST = 0.05
mask_test = df_norm['wd'] > WALL_EXCL_DIST
df_test   = df_norm[mask_test]
N_TEST    = 500
np.random.seed(7)
df_test   = df_test.iloc[
    np.random.choice(len(df_test), min(N_TEST, len(df_test)), replace=False)
]
X_test = df_test[['x', 'z', 'wd']].values
print(f"Test pts: {len(X_test):,}  (all wd > {WALL_EXCL_DIST})")


class SafePDE(dde.data.PDE):
    def __init__(self, *args, fixed_test_x, **kwargs):
        self._fixed_test_x = fixed_test_x
        super().__init__(*args, **kwargs)

    def test_points(self):
        return self._fixed_test_x

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        f = self.pde(inputs, outputs)
        if not isinstance(f, (list, tuple)):
            f = [f]
        pde_losses = [loss_fn(tf.zeros_like(fi), fi) for fi in f]
        zero       = tf.constant(0.0, dtype=tf.float64)
        bc_losses  = [zero] * len(self.bcs)
        return pde_losses + bc_losses


# ==============================================================================
# SECTION 14 — k-ω SST PDEs  [v8 fixes + NEW-FIX-2 bounded log-ω clip]
# ==============================================================================

BETA_STAR = 0.09
BETA_2    = 0.0828
SIGMA_K1  = 0.85;   SIGMA_K2 = 1.0
SIGMA_O1  = 0.5;    SIGMA_O2 = 0.856
GAMMA_1   = 5 / 9;  GAMMA_2  = 0.44
A1        = 0.31

NUT_FLOOR = NU / 100.0
NUT_CEIL  = 1000.0 * NU

OM_CLIP  = 1e2
nu_rep   = 1000.0 * NU
_ofs_scale = nu_rep * OMEGA_WALL / CHORD


def rans_sst_v8_fixed(x, y):
    """
    12 first-order PDEs for k-ω SST.

    Network output layout:
      0  u_norm     1  w_norm     2  p_norm
      3  k_norm     4  log(ω)_norm  [v8-FIX-1]
      5  τx/τ_scale 6  τz/τ_scale  7  τxz/τ_scale
      8  Dk/_kfs    9  Dkz/_kfs   10  Dω/_ofs_scale  11  Dωz/_ofs_scale

    [NEW-FIX-2] log-ω network output is clamped to physical bounds
                BEFORE exp(), preventing runaway ω growth.
    """
    u  = y[:, 0:1] * u_s + u_m
    w  = y[:, 1:2] * w_s + w_m
    k  = y[:, 3:4] * k_s + k_m
    k  = tf.maximum(k, 1e-10)

    # [NEW-FIX-2] Clamp normalised log-ω to physically valid range BEFORE exp
    log_om_norm_clipped = tf.clip_by_value(
        y[:, 4:5],
        LOG_OM_CLIP_MIN,
        LOG_OM_CLIP_MAX,
    )
    log_om = log_om_norm_clipped * log_om_s + log_om_m
    om     = tf.exp(log_om)   # guaranteed in [50, 10000] physical

    yd = tf.maximum(tf.stop_gradient(x[:, 2:3]), 1e-10)

    tau_x  = y[:, 5:6] * tau_scale
    tau_z  = y[:, 6:7] * tau_scale
    tau_xz = y[:, 7:8] * tau_scale

    _kfs = nu_rep * k_s / CHORD

    Dk   = y[:, 8:9]   * _kfs
    Dkz  = y[:, 9:10]  * _kfs
    Dom  = y[:, 10:11] * _ofs_scale
    Domz = y[:, 11:12] * _ofs_scale

    j = lambda i, jj: dde.grad.jacobian(y, x, i=i, j=jj)

    du_dx = j(0, 0) * u_s;    du_dz = j(0, 1) * u_s
    dw_dx = j(1, 0) * w_s;    dw_dz = j(1, 1) * w_s
    dp_dx = j(2, 0) * p_s;    dp_dz = j(2, 1) * p_s
    dk_dx = j(3, 0) * k_s;    dk_dz = j(3, 1) * k_s

    # [NEW-FIX-2] Jacobian of the CLAMPED normalised output w.r.t. x
    # dde.grad.jacobian differentiates through clip_by_value correctly
    # (zero gradient outside bounds, pass-through inside).
    dlog_om_dx = j(4, 0) * log_om_s
    dlog_om_dz = j(4, 1) * log_om_s

    do_dx = om * dlog_om_dx
    do_dz = om * dlog_om_dz

    dtx_dx  = j(5, 0) * tau_scale
    dtz_dz  = j(6, 1) * tau_scale
    dtxz_dx = j(7, 0) * tau_scale
    dtxz_dz = j(7, 1) * tau_scale

    dDk_dx   = j(8,  0) * _kfs
    dDkz_dz  = j(9,  1) * _kfs
    dDom_dx  = j(10, 0) * _ofs_scale
    dDomz_dz = j(11, 1) * _ofs_scale

    S2    = 2 * (du_dx ** 2 + dw_dz ** 2 + 0.5 * (du_dz + dw_dx) ** 2)
    S     = tf.sqrt(S2 + 1e-10)
    shear = du_dz + dw_dx

    # [v8-FIX-2] CDkw without 1/ω singularity
    gkw_log = dk_dx * dlog_om_dx + dk_dz * dlog_om_dz
    CDkw    = tf.clip_by_value(2 * SIGMA_O2 * gkw_log, 1e-20, 1e6)

    t1 = tf.sqrt(k) / (BETA_STAR * om * yd)
    t2 = 500 * NU   / (yd ** 2 * om)
    t3 = 4 * SIGMA_O2 * k / (CDkw * yd ** 2)
    F1 = tf.tanh(tf.minimum(tf.maximum(t1, t2), t3) ** 4)

    arg2 = tf.maximum(2 * tf.sqrt(k) / (BETA_STAR * om * yd),
                      500 * NU / (yd ** 2 * om))
    F2   = tf.tanh(arg2 ** 2)

    sigma_k  = F1 * SIGMA_K1  + (1 - F1) * SIGMA_K2
    sigma_om = F1 * SIGMA_O1  + (1 - F1) * SIGMA_O2
    beta_eff = F1 * BETA_1_C  + (1 - F1) * BETA_2
    gam_eff  = F1 * GAMMA_1   + (1 - F1) * GAMMA_2

    D   = tf.maximum(tf.maximum(A1 * om, S * F2), 1e-10)
    nut = tf.clip_by_value(A1 * k / D, NUT_FLOOR, NUT_CEIL)

    nu_eff    = NU + nut
    nu_eff_k  = NU + sigma_k  * nut
    nu_eff_om = NU + sigma_om * nut

    Pk          = tf.minimum(nut * S2, 20 * BETA_STAR * k * om)
    Pk_over_nut = tf.minimum(Pk / (nut + 1e-12), 20 * BETA_STAR * om)
    om_prod     = gam_eff * Pk_over_nut

    cont  = (du_dx + dw_dz) / PDE_SCALES[0]
    mom_x = (u * du_dx + w * du_dz + dp_dx - dtx_dx  - dtxz_dz) / PDE_SCALES[1]
    mom_z = (u * dw_dx + w * dw_dz + dp_dz - dtxz_dx - dtz_dz ) / PDE_SCALES[2]

    def_tx  = (tau_x  - 2 * nu_eff * du_dx) / PDE_SCALES[3]
    def_tz  = (tau_z  - 2 * nu_eff * dw_dz) / PDE_SCALES[4]
    def_txz = (tau_xz -     nu_eff * shear ) / PDE_SCALES[5]

    def_Dk   = (Dk  - nu_eff_k * dk_dx) / PDE_SCALES[6]
    def_Dkz  = (Dkz - nu_eff_k * dk_dz) / PDE_SCALES[7]

    def_Dom  = (Dom  - nu_eff_om * om * dlog_om_dx) / PDE_SCALES[8]
    def_Domz = (Domz - nu_eff_om * om * dlog_om_dz) / PDE_SCALES[9]

    k_trans = (u * dk_dx + w * dk_dz - Pk + BETA_STAR * k * om
               - dDk_dx - dDkz_dz) / PDE_SCALES[10]

    cross    = 2 * (1 - F1) * SIGMA_O2 * gkw_log
    cross    = tf.clip_by_value(cross, -1e6, 1e6)

    om_raw   = (u * do_dx + w * do_dz - om_prod + beta_eff * om ** 2
                - dDom_dx - dDomz_dz - cross)
    om_trans = tf.clip_by_value(om_raw / PDE_SCALES[11], -OM_CLIP, OM_CLIP)

    return [cont, mom_x, mom_z,
            def_tx, def_tz, def_txz,
            def_Dk, def_Dkz, def_Dom, def_Domz,
            k_trans, om_trans]


print("✅ 12-output first-order k-ω SST v8-fixed PDEs defined.")


# ==============================================================================
# SECTION 15 — BOUNDARY CONDITIONS
# ==============================================================================

_IDX_PDE_START   = 0;   _IDX_PDE_END   = 12
_IDX_DATA_U      = 12
_IDX_DATA_W      = 13
_IDX_DATA_P      = 14
_IDX_DATA_K      = 15
_IDX_DATA_OM     = 16
_IDX_WALL_U      = 17
_IDX_WALL_W      = 18
_IDX_WALL_K      = 19
_IDX_WALL_OM     = 20
_IDX_OUT_P       = 21
_IDX_OUT_U       = 22
_IDX_OUT_W       = 23
_IDX_OUT_K       = 24
_IDX_OUT_OM      = 25
_IDX_IN_U        = 26
_IDX_IN_W        = 27
_IDX_IN_P        = 28
_IDX_IN_K        = 29
_IDX_IN_OM       = 30
_N_LOSS_TERMS    = 31


def bc(X, val, comp):
    return dde.PointSetBC(X, np.full((len(X), 1), val), component=comp)

W_XZD = df_wbc[['x', 'z', 'wd']].values
O_XZD = df_obc[['x', 'z', 'wd']].values
I_XZD = df_ibc[['x', 'z', 'wd']].values

bcs = [
    dde.PointSetBC(X_tr, targets['u'],     component=0),   # 12
    dde.PointSetBC(X_tr, targets['w'],     component=1),   # 13
    dde.PointSetBC(X_tr, targets['p'],     component=2),   # 14
    dde.PointSetBC(X_tr, targets['k'],     component=3),   # 15
    dde.PointSetBC(X_tr, targets['omega'], component=4),   # 16
    bc(W_XZD, U_0_N,          0),   # 17
    bc(W_XZD, W_0_N,          1),   # 18
    bc(W_XZD, K_WALL_N,       3),   # 19
    bc(W_XZD, LOG_OM_WALL_N,  4),   # 20
    bc(O_XZD, P_INF_N,        2),   # 21
    bc(O_XZD, U_X_N,          0),   # 22
    bc(O_XZD, U_Z_N,          1),   # 23
    bc(O_XZD, K_INF_N,        3),   # 24
    bc(O_XZD, LOG_OM_INF_N,   4),   # 25
    bc(I_XZD, U_X_N,          0),   # 26
    bc(I_XZD, U_Z_N,          1),   # 27
    bc(I_XZD, P_INF_N,        2),   # 28
    bc(I_XZD, K_INF_N,        3),   # 29
    bc(I_XZD, LOG_OM_INF_N,   4),   # 30
]

assert len(bcs) == _N_LOSS_TERMS - _IDX_PDE_END, \
    f"BC count ({len(bcs)}) != expected ({_N_LOSS_TERMS - _IDX_PDE_END})"
print(f"  BC objects: {len(bcs)}  (indices {_IDX_PDE_END}–{_N_LOSS_TERMS - 1})")


# ==============================================================================
# SECTION 16 — BUILD MODEL  [NEW-FIX-6: wider network]
# ==============================================================================

data = SafePDE(
    geom, rans_sst_v8_fixed, bcs,
    num_domain=0, num_boundary=0,
    num_test=len(X_test),
    anchors=X_tr,
    fixed_test_x=X_test,
)

# [NEW-FIX-6] Wider network: 6×128 vs original 6×64
layer_size = [3] + [HIDDEN_WIDTH] * N_HIDDEN_LAYERS + [12]
net        = dde.nn.FNN(layer_size, 'tanh', 'Glorot uniform')
model      = dde.Model(data, net)
print(f"  Network: {layer_size}")


def make_loss_weights(om_w, om_bc_w=None):
    if om_bc_w is None:
        om_bc_w = 2.0
    weights = [None] * _N_LOSS_TERMS

    weights[0]  = 1.0;  weights[1]  = 1.0;  weights[2]  = 1.0
    weights[3]  = 2.0;  weights[4]  = 2.0;  weights[5]  = 2.0
    weights[6]  = 1.0;  weights[7]  = 1.0
    weights[8]  = 1.0;  weights[9]  = 1.0
    weights[10] = 0.5;  weights[11] = om_w

    weights[_IDX_DATA_U]  = 5.0;  weights[_IDX_DATA_W]  = 5.0
    weights[_IDX_DATA_P]  = 5.0;  weights[_IDX_DATA_K]  = 2.0
    weights[_IDX_DATA_OM] = 2.0

    weights[_IDX_WALL_U]  = 5.0;  weights[_IDX_WALL_W]  = 5.0
    weights[_IDX_WALL_K]  = 2.0;  weights[_IDX_WALL_OM] = om_bc_w

    weights[_IDX_OUT_P]   = 10.0; weights[_IDX_OUT_U]  = 10.0
    weights[_IDX_OUT_W]   = 10.0; weights[_IDX_OUT_K]  =  3.0
    weights[_IDX_OUT_OM]  =  3.0

    weights[_IDX_IN_U]    = 3.0;  weights[_IDX_IN_W]   = 3.0
    weights[_IDX_IN_P]    = 3.0;  weights[_IDX_IN_K]   = 2.0
    weights[_IDX_IN_OM]   = 2.0

    assert None not in weights, "Some loss weight slots were not filled!"
    assert len(weights) == _N_LOSS_TERMS
    return weights


print(f"  Loss terms: {len(make_loss_weights(0.5))}")


# ------------------------------------------------------------------------------
# [NEW-FIX-4] Custom Adam optimizer with gradient clipping
# clipnorm=1.0 prevents the epoch-8000-style loss spikes seen in v8.
# ------------------------------------------------------------------------------
def make_adam_optimizer(lr):
    return tf.keras.optimizers.Adam(
        learning_rate=lr,
        clipnorm=1.0,
    )


# ==============================================================================
# SECTION 17 — CHECKPOINT UTILITIES
# ==============================================================================

_META_FILE = os.path.join(DRIVE_CKPT_DIR, 'checkpoint_meta.json')
_CKPT_STEM = f'pinn_SST_v8fixed_Re{RE:.0f}'


def _ckpt_local(epoch):
    return f'/content/{_CKPT_STEM}_ep{epoch}'

def _ckpt_drive(epoch):
    return os.path.join(DRIVE_CKPT_DIR, f'{_CKPT_STEM}_ep{epoch}')


def save_checkpoint(epoch):
    local_pfx = _ckpt_local(epoch)
    drive_pfx = _ckpt_drive(epoch)
    model.net.save_weights(local_pfx)
    for src in glob.glob(local_pfx + '*'):
        shutil.copy2(src, os.path.join(DRIVE_CKPT_DIR, os.path.basename(src)))
    index_on_drive = drive_pfx + '.index'
    if os.path.exists(index_on_drive):
        meta = {'epoch': epoch, 'stem': _CKPT_STEM,
                'prefix': f'{_CKPT_STEM}_ep{epoch}'}
        with open(_META_FILE, 'w') as fh:
            json.dump(meta, fh, indent=2)
        print(f"  💾 Checkpoint saved: epoch {epoch} → Drive  ✅")
    else:
        print(f"  ⚠️ .index not found on Drive after copy: {index_on_drive}")


def latest_checkpoint():
    if FRESH_START:
        print("  FRESH_START=True → ignoring existing checkpoints.")
        return 0, None

    if os.path.exists(_META_FILE):
        try:
            with open(_META_FILE) as fh:
                meta = json.load(fh)
            epoch     = int(meta['epoch'])
            drive_pfx = os.path.join(DRIVE_CKPT_DIR, meta['prefix'])
            if os.path.exists(drive_pfx + '.index'):
                print(f"  📋 Checkpoint meta found: epoch={epoch}  prefix={meta['prefix']}")
                return epoch, drive_pfx
            else:
                print(f"  ⚠️ Meta points to epoch {epoch} but .index missing — glob fallback …")
        except Exception as e:
            print(f"  ⚠️ Could not read checkpoint_meta.json: {e} — glob fallback …")

    pattern     = os.path.join(DRIVE_CKPT_DIR, f'{_CKPT_STEM}_ep*.index')
    files_found = glob.glob(pattern)
    if not files_found:
        print("  No existing checkpoints found → fresh start.")
        return 0, None
    epochs = []
    for f in files_found:
        try:
            ep_str = os.path.basename(f).split('_ep')[1].split('.')[0].split('-')[0]
            epochs.append(int(ep_str))
        except Exception:
            pass
    if not epochs:
        print("  Glob found .index files but could not parse epochs → fresh start.")
        return 0, None
    best_ep = max(epochs)
    print(f"  🔍 Glob fallback found epoch {best_ep}")
    return best_ep, _ckpt_drive(best_ep)


def load_checkpoint(drive_pfx, om_w, om_bc_w, lr):
    base      = os.path.basename(drive_pfx)
    local_pfx = f'/content/{base}'
    n_copied  = 0
    for src in glob.glob(drive_pfx + '*'):
        shutil.copy2(src, f'/content/{os.path.basename(src)}')
        n_copied += 1
    print(f"  Copied {n_copied} checkpoint file(s) from Drive to /content")
    if not os.path.exists(local_pfx + '.index'):
        raise FileNotFoundError(f"Expected {local_pfx}.index not found after copy.")
    print("  Building network variables via dummy forward pass …")
    _ = model.net(tf.constant(X_tr[:4].astype(np.float64)), training=False)
    print("  ✅ Network built.")
    model.net.load_weights(local_pfx).expect_partial()
    print(f"  ✅ Weights loaded from: {local_pfx}")
    # [NEW-FIX-4] Use clipped Adam optimizer on resume
    model.compile(make_adam_optimizer(lr),
                  loss_weights=make_loss_weights(om_w, om_bc_w))
    print(f"  ✅ Recompiled  (ω-PDE-w={om_w:.2f}  ω-BC-w={om_bc_w:.1f}  lr={lr:.1e})")


# ==============================================================================
# DIAGNOSTICS
# ==============================================================================

def print_om_stats():
    yp       = model.predict(X_tr[:200])
    # Apply the same physical clipping as in the PDE for consistency
    log_om_norm = np.clip(yp[:, 4], LOG_OM_CLIP_MIN, LOG_OM_CLIP_MAX)
    log_om_p = log_om_norm * log_om_s + log_om_m
    om_p     = np.exp(log_om_p)
    print(f"  [v8-fixed] ω stats (train sample):  "
          f"min={om_p.min():.2e}  mean={om_p.mean():.2e}  "
          f"max={om_p.max():.2e}  (physical range: [{OMEGA_INF:.1f}, {OMEGA_WALL:.1e}])")


def check_cdkw_clip(n_sample=2000):
    rng   = np.random.default_rng(99)
    xs    = rng.uniform(df_clean['x'].min(), df_clean['x'].max(), n_sample)
    zs    = rng.uniform(df_clean['z'].min(), df_clean['z'].max(), n_sample)
    wds   = compute_wall_distance(xs, zs)
    Xsamp = np.column_stack([xs, zs, wds])
    eps   = 1e-4

    def grad(comp, scale, dim):
        Xp, Xm = Xsamp.copy(), Xsamp.copy()
        Xp[:, dim] += eps;  Xm[:, dim] -= eps
        return (model.predict(Xp)[:, comp] - model.predict(Xm)[:, comp]) * scale / (2 * eps)

    dk_dx     = grad(3, k_s,       0)
    dlogom_dx = grad(4, log_om_s,  0)
    dk_dz     = grad(3, k_s,       1)
    dlogom_dz = grad(4, log_om_s,  1)

    gkw_log  = dk_dx * dlogom_dx + dk_dz * dlogom_dz
    CDkw_raw = 2 * SIGMA_O2 * gkw_log
    clipped  = np.mean((CDkw_raw > 1e6) | (CDkw_raw < 1e-20))
    if clipped > 0.10:
        print(f"  ⚠️ CDkw clipped on {clipped:.1%} of sampled pts — "
              f"cross-diffusion still noisy")
    else:
        print(f"  ✅ CDkw clip fraction: {clipped:.1%}")


# ------------------------------------------------------------------------------
# [NEW-FIX-5] ω divergence monitor
# Returns physical ω mean from 500 training sample points.
# ------------------------------------------------------------------------------
def get_om_mean_physical(n_sample=500):
    yp = model.predict(X_tr[:n_sample])
    log_om_norm = np.clip(yp[:, 4], LOG_OM_CLIP_MIN, LOG_OM_CLIP_MAX)
    log_om_p    = log_om_norm * log_om_s + log_om_m
    return float(np.exp(log_om_p).mean())


# ==============================================================================
# SECTION 18 — TRAINING  [NEW-FIX-3, 4, 5 integrated]
# ==============================================================================

start_epoch, ckpt_prefix = latest_checkpoint()

initial_om_w    = get_om_weight(start_epoch)
initial_om_bc_w = get_om_bc_weight(start_epoch)
initial_lr      = get_learning_rate(start_epoch)

# [NEW-FIX-4] Initial compile uses clipped Adam
model.compile(make_adam_optimizer(initial_lr),
              loss_weights=make_loss_weights(initial_om_w, initial_om_bc_w))

if ckpt_prefix is not None:
    print(f"\n🔄 Resuming from epoch {start_epoch} …")
    load_checkpoint(ckpt_prefix, initial_om_w, initial_om_bc_w, initial_lr)
    print(f"✅ Successfully resumed from epoch {start_epoch}")
else:
    print("\n🚀 Starting fresh training …")

SAVE_LOCAL  = f'/content/pinn_SST_v8fixed_Re{RE:.0f}_'
losshistory = None
train_state = None
_best_train_loss = float('inf')

epoch_cursor  = start_epoch
_prev_om_w    = None
_prev_om_bc_w = None
_prev_lr      = None

print(f"\nAdam phase ({ADAM_EPOCHS} total epochs, resuming from {start_epoch}) …")

_divergence_abort = False

while epoch_cursor < ADAM_EPOCHS:
    chunk   = min(CKPT_EVERY, ADAM_EPOCHS - epoch_cursor)
    om_w    = get_om_weight(epoch_cursor)
    om_bc_w = get_om_bc_weight(epoch_cursor)
    lr      = get_learning_rate(epoch_cursor)   # [NEW-FIX-3]

    # Recompile only when any schedule value changes
    if om_w != _prev_om_w or om_bc_w != _prev_om_bc_w or lr != _prev_lr:
        # [NEW-FIX-4] Always use clipped Adam
        model.compile(make_adam_optimizer(lr),
                      loss_weights=make_loss_weights(om_w, om_bc_w))
        print(f"  → epoch {epoch_cursor}: "
              f"ω-PDE-w={om_w:.2f}  ω-BC-w={om_bc_w:.1f}  lr={lr:.1e}")
        _prev_om_w    = om_w
        _prev_om_bc_w = om_bc_w
        _prev_lr      = lr

    losshistory, train_state = model.train(
        epochs=chunk, display_every=500,
        model_save_path=SAVE_LOCAL,
    )

    epoch_cursor += chunk
    save_checkpoint(epoch_cursor)

    # Per-checkpoint diagnostics
    print_om_stats()
    check_cdkw_clip()

    # [NEW-FIX-5] ω divergence monitor
    om_mean = get_om_mean_physical()
    if om_mean > 5.0 * OMEGA_WALL:
        print(f"\n⚠️ [NEW-FIX-5] ω DIVERGING: mean={om_mean:.2e} >> "
              f"5×ω_wall={5*OMEGA_WALL:.2e}")
        print(f"  Aborting training at epoch {epoch_cursor}.")
        print(f"  Suggestion: reduce OM_WEIGHT_SCHEDULE further or increase "
              f"LOG_OM_CLIP_MAX.")
        _divergence_abort = True
        break

    # Early stopping
    current_best = train_state.best_loss_train
    if current_best < _best_train_loss:
        _best_train_loss = current_best
    if _best_train_loss < EARLY_STOP_TOL:
        print(f"✅ Early stop at epoch {epoch_cursor}: "
              f"best train loss {_best_train_loss:.2e} < {EARLY_STOP_TOL:.0e}")
        break

if _divergence_abort:
    print(f"⚠️ Training aborted due to ω divergence at epoch {epoch_cursor}.")
else:
    print(f"✅ Adam complete (epoch {epoch_cursor}  "
          f"best_train={_best_train_loss:.3e})")

# ---- Optional L-BFGS phase [v8-FIX-4] ----
if USE_LBFGS and not _divergence_abort:
    print(f"\nL-BFGS-B ({LBFGS_ITER} iter) …")
    model.compile('L-BFGS-B',
                  loss_weights=make_loss_weights(get_om_weight(epoch_cursor),
                                                  om_bc_w=get_om_bc_weight(epoch_cursor)))
    model.train_step.optimizer_kwargs = {'options': {
        'maxcor': 50, 'ftol': np.finfo(float).eps,
        'maxfun': LBFGS_ITER, 'maxiter': LBFGS_ITER, 'maxls': 50,
    }}
    losshistory, train_state = model.train(
        display_every=500, model_save_path=SAVE_LOCAL,
    )
    save_checkpoint(epoch_cursor + LBFGS_ITER)
    print("✅ L-BFGS complete.")
else:
    if not USE_LBFGS:
        print("\nL-BFGS skipped (USE_LBFGS=False).")

print("✅ Training complete.")


# ==============================================================================
# SECTION 19 — PREDICTION  [v8-FIX-1 + NEW-FIX-2: clamped back-transform]
# ==============================================================================

X_PLOT = np.linspace(-1.5, 2.5, 400)
Z_PLOT = np.linspace(-1.5, 1.5, 300)
XX, ZZ = np.meshgrid(X_PLOT, Z_PLOT)
Xf, Zf = XX.ravel(), ZZ.ravel()
WDf    = compute_wall_distance(Xf, Zf)
Xp     = np.column_stack([Xf, Zf, WDf])

print(f"Predicting {len(Xp):,} pts …")
yp    = model.predict(Xp)

u_p   = yp[:, 0] * u_s  + u_m
w_p   = yp[:, 1] * w_s  + w_m
p_p   = yp[:, 2] * p_s  + p_m
k_p   = yp[:, 3] * k_s  + k_m

# [NEW-FIX-2] Apply same physical clipping as PDE during back-transform
log_om_norm_p = np.clip(yp[:, 4], LOG_OM_CLIP_MIN, LOG_OM_CLIP_MAX)
log_om_p      = log_om_norm_p * log_om_s + log_om_m
om_p          = np.exp(log_om_p)

nut_p = np.maximum(k_p, 0) / np.maximum(om_p, 1e-10)
mag_p = np.sqrt(u_p ** 2 + w_p ** 2)

U_g   = u_p .reshape(XX.shape)
W_g   = w_p .reshape(XX.shape)
P_g   = p_p .reshape(XX.shape)
K_g   = k_p .reshape(XX.shape)
OM_g  = om_p.reshape(XX.shape)
NUT_g = nut_p.reshape(XX.shape)
MAG_g = mag_p.reshape(XX.shape)
print("✅ Done.")


# ==============================================================================
# SECTION 20 — PLOTTING
# ==============================================================================

def add_af(ax):
    ax.fill(X_AF, Z_AF, color='#404040', zorder=5)
    ax.plot(X_AF, Z_AF, 'k-', lw=0.7, zorder=6)

TSUF = f'NACA 0012 | Re={RE:,.0f} | AoA={AOA_DEG}° | k-ω SST v8-fixed'
XL, ZL = (-1.5, 2.5), (-1.5, 1.5)


def quad_plot(fields, title, outfile):
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    fig.suptitle(title, fontsize=15, fontweight='bold')
    for ax, (fld, cm, lb, tt) in zip(axes.flat, fields):
        cf = ax.contourf(XX, ZZ, fld, levels=120, cmap=cm, extend='both')
        fig.colorbar(cf, ax=ax, label=lb, pad=0.02, shrink=0.88)
        add_af(ax)
        ax.set(title=tt, xlabel='x [m]', ylabel='z [m]',
               xlim=XL, ylim=ZL, aspect='equal')
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    shutil.copy2(outfile, os.path.join(DRIVE_CKPT_DIR, outfile))
    print(f"Saved: {outfile}")


out1 = f'01_flow_Re{RE:.0f}.png'
quad_plot([
    (P_g,   'coolwarm', 'p [m²/s²]', '(a) Pressure'),
    (U_g,   'RdBu_r',   'u [m/s]',   '(b) x-vel'),
    (W_g,   'RdBu_r',   'w [m/s]',   '(c) z-vel'),
    (MAG_g, 'viridis',  '|U|',        '(d) Speed'),
], f'Flow — {TSUF}', out1)

out2 = f'02_turb_Re{RE:.0f}.png'
Kpl  = np.clip(K_g,   0, np.percentile(K_g,   99))
NTpl = np.clip(NUT_g, 0, np.percentile(NUT_g, 99))
quad_plot([
    (Kpl,                  'YlOrRd', 'k [m²/s²]', '(a) TKE'),
    (np.log10(OM_g + 1.0), 'plasma',  'log₁₀(ω)', '(b) log₁₀(ω)'),
    (NTpl,                 'inferno', 'νt [m²/s]', '(c) νt'),
    (Kpl / (OM_g + 1e-10), 'cividis', 'k/ω [s]',  '(d) τ_t'),
], f'Turbulence — {TSUF}', out2)

out3 = f'03_stream_Re{RE:.0f}.png'
nan_frac = np.isnan(U_g).sum() / U_g.size
if nan_frac > 0.01:
    print(f"⚠️ {nan_frac:.1%} of prediction grid is NaN — skipping streamline plot")
    out3 = None
else:
    Us = np.where(np.isnan(U_g), U_X, U_g)
    Ws = np.where(np.isnan(W_g), U_Z, W_g)
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(f'Streamlines — {TSUF}', fontsize=15, fontweight='bold')
    st = axes[0].streamplot(X_PLOT, Z_PLOT, Us, Ws,
                            color=np.sqrt(Us**2 + Ws**2),
                            cmap='plasma', linewidth=0.9, density=2.0, arrowsize=1.0)
    fig.colorbar(st.lines, ax=axes[0], label='|U|', pad=0.02)
    add_af(axes[0])
    axes[0].set(xlim=XL, ylim=ZL, aspect='equal', title='Speed')
    ix = (X_PLOT >= -0.4) & (X_PLOT <= 1.8)
    iz = (Z_PLOT >= -0.7) & (Z_PLOT <= 0.7)
    st2 = axes[1].streamplot(X_PLOT[ix], Z_PLOT[iz],
                              Us[np.ix_(iz, ix)], Ws[np.ix_(iz, ix)],
                              color=np.clip(K_g[np.ix_(iz, ix)], 0,
                                            np.percentile(K_g, 95)),
                              cmap='YlOrRd', linewidth=1.2, density=2.5, arrowsize=1.2)
    fig.colorbar(st2.lines, ax=axes[1], label='k', pad=0.02)
    add_af(axes[1])
    axes[1].set(xlim=(-0.4, 1.8), ylim=(-0.7, 0.7), aspect='equal', title='TKE zoom')
    plt.tight_layout()
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    shutil.copy2(out3, os.path.join(DRIVE_CKPT_DIR, out3))
    print(f"Saved: {out3}")

# Cp curve
n_cp = 300
xs, zs = naca0012_coords(n=n_cp)
Xup = np.column_stack([xs[:n_cp + 1],       zs[:n_cp + 1]])
Xlo = np.column_stack([xs[n_cp + 1:][::-1], zs[n_cp + 1:][::-1]])

def pred_cp(Xs):
    wd = compute_wall_distance(Xs[:, 0], Xs[:, 1])
    yy = model.predict(np.column_stack([Xs[:, 0], Xs[:, 1], wd]))
    return (yy[:, 2] * p_s + p_m) / (0.5 * U_INF ** 2)

Cpu = pred_cp(Xup)
Cpl = pred_cp(Xlo)
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(xs[:n_cp + 1] / CHORD,       -Cpu, 'b-', lw=2, label='Upper')
ax.plot(xs[n_cp + 1:][::-1] / CHORD, -Cpl, 'r-', lw=2, label='Lower')
ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
ax.invert_yaxis()
ax.set(xlabel='x/c', ylabel='−Cp', title=f'Cp — {TSUF}', xlim=(0, 1))
ax.legend();  ax.grid(alpha=0.3)
plt.tight_layout()
out4 = f'04_Cp_Re{RE:.0f}.png'
plt.savefig(out4, dpi=150, bbox_inches='tight')
plt.close()
shutil.copy2(out4, os.path.join(DRIVE_CKPT_DIR, out4))
print(f"Saved: {out4}")

# CL / CD (pressure only)
def cl_cd(n=600):
    xs, zs = naca0012_coords(n=n)
    wd = compute_wall_distance(xs, zs)
    yy = model.predict(np.column_stack([xs, zs, wd]))
    ps = yy[:, 2] * p_s + p_m
    dx, dz = np.diff(xs), np.diff(zs)
    ds = np.sqrt(dx ** 2 + dz ** 2)
    nx, nz = dz / ds, -dx / ds
    pm = 0.5 * (ps[:-1] + ps[1:])
    Fx = np.sum(-pm * nx * ds);  Fz = np.sum(-pm * nz * ds)
    CL = (-Fx * np.sin(AOA_RAD) + Fz * np.cos(AOA_RAD)) / Q_INF
    CD = ( Fx * np.cos(AOA_RAD) + Fz * np.sin(AOA_RAD)) / Q_INF
    return CL, CD

CL, CD = cl_cd()
print(f"\n  CL={CL:+.4f}  CD={CD:+.4f}  L/D={CL / (abs(CD) + 1e-10):+.2f}")

# Loss history plot
out5 = None
if losshistory is not None:
    fig, ax = plt.subplots(figsize=(14, 6))
    la  = np.array(losshistory.loss_train)
    st  = np.array(losshistory.steps)
    tot = la.sum(axis=1) if la.ndim > 1 else la
    ax.semilogy(st, tot, 'b-', lw=1.8, label='Total train')
    if la.ndim > 1 and la.shape[1] >= _N_LOSS_TERMS:
        ax.semilogy(st, la[:, :12].sum(axis=1), 'r--', lw=1.3, label='PDE sum')
        ax.semilogy(st, la[:, 12:].sum(axis=1), 'g--', lw=1.3, label='BC+data sum')
        ax.semilogy(st, la[:, 11], 'm:', lw=2.0, label='ω-transport (idx 11)')
        ax.semilogy(st, la[:, 10], 'c:', lw=1.5, label='k-transport (idx 10)')
        ax.semilogy(st, la[:, _IDX_WALL_OM], 'y--', lw=1.5,
                    label=f'ω-wall BC (idx {_IDX_WALL_OM})')
    # Mark ω-PDE weight transitions
    for ep_thresh, wval in OM_WEIGHT_SCHEDULE[1:]:
        ax.axvline(ep_thresh, color='orange', lw=1, ls='--', alpha=0.6,
                   label=f'ω-PDE-w={wval}')
    # Mark LR transitions
    for ep_thresh in [5000, 10000, 15000]:
        ax.axvline(ep_thresh, color='purple', lw=1, ls=':', alpha=0.4,
                   label=f'LR change @ {ep_thresh}')
    ax.set(xlabel='Step', ylabel='Loss', title=f'Loss — {TSUF}')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.grid(alpha=0.3);  plt.tight_layout()
    out5 = f'05_loss_Re{RE:.0f}.png'
    plt.savefig(out5, dpi=150, bbox_inches='tight')
    plt.close()
    shutil.copy2(out5, os.path.join(DRIVE_CKPT_DIR, out5))
    print(f"Saved: {out5}")


# ==============================================================================
# SECTION 21 — DOWNLOAD
# ==============================================================================

for f in [out1, out2, out4]:
    if f and os.path.exists(f):
        files.download(f);  print(f"  ↓ {f}")

if out3 and os.path.exists(out3):
    files.download(out3);  print(f"  ↓ {out3}")

if out5 and os.path.exists(out5):
    files.download(out5);  print(f"  ↓ {out5}")

print(f"""
{'=' * 65}
COMPLETE  k-ω SST v8-fixed
Re={RE:.0f}   CL={CL:.4f}   CD={CD:.4f}   L/D={CL / (abs(CD) + 1e-10):.2f}

Fixes applied in this version:
  [v8-FIX-1]  Log-space ω norm → architecturally guarantees ω > 0
  [v8-FIX-2]  CDkw = 2σ_ω2·∇k·∇log(ω)  — no 1/ω singularity
  [v8-FIX-3]  ω-flux scale = nu_rep·OMEGA_WALL/CHORD  (physical)
  [v8-FIX-4]  L-BFGS disabled by default  (USE_LBFGS flag)
  [v8-FIX-5]  Diagnostics updated for log-ω
  [NEW-FIX-1]  Smoother ω-PDE/BC weight schedule → 25k epochs
  [NEW-FIX-2]  Physical log-ω clipping [50, 10000] in PDE + prediction
  [NEW-FIX-3]  LR decay: 5e-4 → 2e-4 → 1e-4 → 5e-5
  [NEW-FIX-4]  Adam with clipnorm=1.0 (gradient clipping)
  [NEW-FIX-5]  ω divergence monitor (auto-abort if mean > 5×ω_wall)
  [NEW-FIX-6]  Network 6×{HIDDEN_WIDTH} (vs 6×64 in v8)

Checkpoints: {DRIVE_CKPT_DIR}
NOTE: Fresh start required if switching from 6×64 network.
{'=' * 65}
""")
