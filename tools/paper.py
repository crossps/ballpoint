"""The paper: a seamless 41 mm tile at 5 um a texel, built from individually laid cellulose fibres (a few hundred thousand
collapsed ribbons, machine-direction biased, deposited in layers, then calendered flat), plus filler speck.

Writes assets/paper_um.npy (height in micrometres, zero mean, row 0 at y = 0), assets/paper_h.png (16-bit) and
assets/paper_a.png (8-bit albedo factor). The PNGs are flipped so v = 0 is the first row in world y.

Run: python -I tools/paper.py
"""
import os
import sys

import numpy as np
import taichi as ti
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(os.path.dirname(HERE), 'assets')

N = 8192
TEXEL = .005                 # mm
LAYERS = 3.4
SEED = 5
STD_TARGET = 3.9             # um, surface after calendering (uncoated writing paper is about 3 to 6 um)
HSPAN = 40.0                 # um, what the 16-bit PNG spans

ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=SEED, verbose=False)

rng = np.random.default_rng(SEED)
area = (N * TEXEL) ** 2
L_mm = np.clip(rng.lognormal(np.log(.78), .45, size=400000), .25, 2.8)
W_um = np.clip(rng.normal(24, 6, size=400000), 12, 40)
FINES = 260000                            # short broken fibre fragments, which fill the web between the long fibres
L_fine = np.clip(rng.lognormal(np.log(.11), .5, size=FINES), .06, .4)
W_fine = np.clip(rng.normal(9.5, 3, size=FINES), 5, 18)
cum = np.cumsum(L_mm * W_um * 1e-3)
NF = int(np.searchsorted(cum, LAYERS * area))
L_mm, W_um = np.r_[L_mm[:NF], L_fine], np.r_[W_um[:NF], W_fine]
NF = len(L_mm)
print(f'{NF} fibres, {area:.0f} mm^2')

# --- polylines
STEP_PX = 2.0
counts = np.maximum((L_mm / TEXEL / STEP_PX).astype(int), 6)
TOT = int(counts.sum())
offs = np.r_[0, np.cumsum(counts)[:-1]]
print(TOT, 'samples')

x0 = rng.uniform(0, N, NF)
y0 = rng.uniform(0, N, NF)
th0 = rng.normal(0, np.radians(46), NF) + (rng.random(NF) < .32) * rng.uniform(-np.pi / 2, np.pi / 2, NF)
curl = rng.lognormal(np.log(.05), .55, NF)
zb = rng.random(NF)                       # the layer this fibre was laid in
shade = rng.normal(.0, 1.0, NF)

px = np.empty(TOT, np.float32)
py = np.empty(TOT, np.float32)
tx = np.empty(TOT, np.float32)
ty = np.empty(TOT, np.float32)
zt = np.empty(TOT, np.float32)
fid = np.empty(TOT, np.int32)
wid = np.empty(TOT, np.float32)
rs = np.random.default_rng(SEED + 1)
print('walking fibres')
for i in range(NF):
    n = counts[i]
    st = offs[i]
    dth = rs.normal(0, curl[i], n)
    if rs.random() < .55:                  # dried fibres kink
        dth[rs.integers(0, n, rs.integers(1, 3))] += rs.normal(0, .55)
    th = th0[i] + np.cumsum(dth)
    dx, dy = np.cos(th), np.sin(th)
    px[st:st + n] = x0[i] + STEP_PX * np.cumsum(dx)
    py[st:st + n] = y0[i] + STEP_PX * np.cumsum(dy)
    tx[st:st + n], ty[st:st + n] = dx, dy
    # the fibre bows in and out of the sheet along its length
    bw = np.sin(np.linspace(0, rs.uniform(1.5, 7), n) + rs.uniform(0, 6.3)) * rs.uniform(.05, .22)
    zt[st:st + n] = zb[i] + bw
    fid[st:st + n] = i
    wid[st:st + n] = W_um[i]

H = ti.field(float, shape=(N, N))
FI = ti.field(int, shape=(N, N))
PX = ti.ndarray(ti.f32, TOT); PY = ti.ndarray(ti.f32, TOT); TX = ti.ndarray(ti.f32, TOT); TY = ti.ndarray(ti.f32, TOT)
ZT = ti.ndarray(ti.f32, TOT); WD = ti.ndarray(ti.f32, TOT); FD = ti.ndarray(ti.i32, TOT)
for arr, src in ((PX, px), (PY, py), (TX, tx), (TY, ty), (ZT, zt), (WD, wid)):
    arr.from_numpy(src)
FD.from_numpy(fid)


@ti.kernel
def lay(px: ti.types.ndarray(), py: ti.types.ndarray(), tx: ti.types.ndarray(), ty: ti.types.ndarray(), zt: ti.types.ndarray(),
        wd: ti.types.ndarray(), fd: ti.types.ndarray(), tot: int):
    H.fill(-1e9)
    FI.fill(-1)
    ti.loop_config(serialize=True)
    for k in range(tot):
        cx, cy = px[k], py[k]
        tx0, ty0 = tx[k], ty[k]
        half = wd[k] * .5 / (TEXEL * 1000.0)        # half width in texels
        r = int(half + 2)
        ix, iy = int(ti.floor(cx)), int(ti.floor(cy))
        for dy_ in range(-r, r + 1):
            for dx_ in range(-r, r + 1):
                ox = (ix + dx_) + .5 - cx
                oy = (iy + dy_) + .5 - cy
                along = ox * tx0 + oy * ty0
                if abs(along) > STEP_PX * .62:
                    continue
                perp = -ox * ty0 + oy * tx0
                q = abs(perp) / half
                if q > 1.0:
                    continue
                # a collapsed ribbon: rounded section with a shallow trough down the middle
                prof = ti.sqrt(1 - q * q) * (1 - .3 * ti.exp(-(q / .32) ** 2))
                z = zt[k] * LAYERS * 5.2 + prof * 5.0
                wx = (ix + dx_) % N
                wy = (iy + dy_) % N
                if z > H[wx, wy]:
                    H[wx, wy] = z
                    FI[wx, wy] = fd[k]


print('laying')
lay(PX, PY, TX, TY, ZT, WD, FD, TOT)
h = H.to_numpy().T
fi = FI.to_numpy().T
hole = h < -1e8
print(f'holes {hole.mean() * 100:.3f}%')
h[hole] = h[~hole].min()

# calendering: the rollers flatten the peaks and press everything together
h = gaussian_filter(h, .8, mode='wrap')
h = (h - h.mean())
h = np.tanh(h / (2.0 * h.std())) * h.std() * 2.0
floc = gaussian_filter(rng.standard_normal((N, N)).astype(np.float32), 90, mode='wrap')
floc /= floc.std()
h += floc * .55 * h.std()
h = (h - h.mean()) / h.std() * STD_TARGET
print(f'height std {h.std():.2f} um, range {h.min():.1f} to {h.max():.1f}')

# --- albedo: per-fibre shade, filler speck, a touch of crevice darkening, rare dark specks
sh = shade[np.clip(fi, 0, NF - 1)] * .012
speck = gaussian_filter(rng.standard_normal((N, N)).astype(np.float32), .6, mode='wrap')
speck /= speck.std()
cre = np.clip(-h / (2.6 * STD_TARGET), 0, 1)
a = 1 + sh + speck * .006 - cre * .075
n_spot = 260
sx, sy, sr = rng.integers(0, N, n_spot), rng.integers(0, N, n_spot), rng.uniform(1, 5, n_spot)
spots = np.zeros((N, N), np.float32)
spots[sy, sx] = 1
spots = np.clip(gaussian_filter(spots, 1.4, mode='wrap') * 20, 0, 1)
a *= 1 - spots * .24
a = np.clip(a, 0, 1.05)

os.makedirs(A, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(A), 'work'), exist_ok=True)
np.save(os.path.join(A, 'paper_um.npy'), h.astype(np.float32))
Image.fromarray(np.flipud(((h + HSPAN / 2) / HSPAN * 65535).clip(0, 65535).astype(np.uint16))).save(os.path.join(A, 'paper_h.png'), compress_level=1)
Image.fromarray(np.flipud((a / 1.05 * 255).clip(0, 255).astype(np.uint8))).save(os.path.join(A, 'paper_a.png'), compress_level=1)
crop = h[1000:2000, 1000:2000]
Image.fromarray(((crop - crop.min()) / np.ptp(crop) * 255).astype(np.uint8)).save(os.path.join(os.path.dirname(A), 'work', 'paper_crop.png'))
print('done')
