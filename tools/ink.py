"""The ink. Lays the ballpoint line on the real paper surface, texel by texel at 5 um.

For every texel near a stroke: how far it is from the ball's path across the line (u, -1 to 1), when the ball passed, and
the paper height there. Ink transfers where the paper actually touches the ball: near the edge of the contact only the
fibre tops touch, so the edge is broken and ragged; at the centre the ball presses valleys up to it and the line is
solid. On top of that: parallel striations from the ball's own ink film, ink pushed to the shoulders of the groove, more
ink where the hand slows, and small goops at pen-downs, lifts and now and then along the line.

Writes assets/ink_t.png (arrival time / 22 s), ink_d.png (density / 1.6), ink_g.png ((groove um + 12) / 72), each 16-bit,
flipped so v = 0 is the first world-y row.

Run: python -I tools/ink.py
"""
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.dirname(HERE)
A = os.path.join(FILM, 'assets')

TEXEL = .005
X0, Y0 = -3.0, -3.5
W, H = 47.0, 14.5
DUR = 22.0
NX, NY = int(round(W / TEXEL)), int(round(H / TEXEL))
R_BAND = .46           # mm: texels further than this from the path get nothing
W0 = .185              # mm, the half width of the line at normal pressure (a 1.0 mm ball leaves about 0.37 mm)
DEPTH = 30.0           # um, the groove at normal pressure

hand = np.load(os.path.join(A, 'hand.npz'))
paper = np.load(os.path.join(A, 'paper_um.npy'))
TILE = paper.shape[0]
PSTD = float(paper.std())

ix0 = int(round(X0 / TEXEL))
iy0 = int(round(Y0 / TEXEL))
yi = (np.arange(NY) + iy0) % TILE
xi = (np.arange(NX) + ix0) % TILE
P = paper[np.ix_(yi, xi)] / PSTD                  # paper height in sigmas, on the ink grid, texel-aligned with the tile
PS = gaussian_filter(P, 2.2)                      # the ball rides over the finest detail, so contact reads a softened surface
PS /= PS.std()

T = np.ones((NY, NX), np.float32)                  # arrival / DUR of the first pass that actually laid ink here
TA = np.ones((NY, NX), np.float32)                 # arrival / DUR of the first pass of any kind, for the bare groove and shoulder
D = np.zeros((NY, NX), np.float32)
G = np.zeros((NY, NX), np.float32)
seen = np.zeros((NY, NX), bool)
DS = np.zeros((NY, NX), np.float32)


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def noise_s(s, cell, rng):
    n = int(s.max() / cell) + 8
    return np.interp(s, np.arange(n) * cell, rng.standard_normal(n))


n_strokes = int(hand['n'])
blobs_total = 0
for k in range(n_strokes):
    q, s, tan, t, vel, prs = (hand[f'k{k}_{f}'] for f in ('q', 's', 'tan', 't', 'vel', 'press'))
    rng = np.random.default_rng(1000 + k)
    L = s[-1]
    ends = np.minimum(s, L - s)
    w = (1 + .05 * noise_s(s, .35, rng)) * W0 * (.86 + .14 * prs) * (.8 + .2 * smooth(0, .45, L - s)) * (1 + .1 * np.exp(-(s / .22) ** 2))
    vn = vel / max(vel.max(), 1e-6)
    flow = (1 + .35 * (1 - vn) ** 1.5) * (1 + .1 * noise_s(s, .8, rng))
    flow *= 1 - .24 * np.clip(noise_s(s, 1.1, rng) - 1.5, 0, 1) * vn   # the ball runs a little dry now and then at speed

    phase = rng.uniform(0, 6.28, 4)
    fr = np.array([6.0, 10.5, 17.0, 24.0]) * rng.uniform(.9, 1.1, 4)
    amp = np.array([.05, .04, .03, .025])
    drifts = [noise_s(s, 1.6, rng) * 1.2 for _ in range(4)]

    # goops: at the pen-down, at the lift, and now and then along the stroke
    goops = [(.05, .0, .17 + .03 * rng.random(), 1.1), (max(L - .08, .05), .0, .11, .85)]
    pos = rng.uniform(1.5, 8)
    while pos < L - 1:
        goops.append((pos, rng.uniform(-.09, .09), rng.uniform(.05, .14), rng.uniform(.6, 1.0)))
        pos += rng.uniform(4, 14)
    if k == 6:                                      # the full stop: one pressed bead of ink
        goops = [(L * .5, 0, .22, 1.35)]
    blobs_total += len(goops)
    gp = np.array(goops, np.float32)

    chunk = 2.6                                     # mm of path per chunk, so retraces stay separate passes
    n_chunk = max(1, int(np.ceil(L / chunk)))
    DS[:] = 0
    for c in range(n_chunk):
        a = max(0, int((c * chunk - .4) / .01))
        b = min(len(s), int(((c + 1) * chunk + .4) / .01))
        Q = q[a:b]
        xs, ys = Q[:, 0], Q[:, 1]
        bx0 = int(np.floor((xs.min() - R_BAND - X0) / TEXEL))
        bx1 = int(np.ceil((xs.max() + R_BAND - X0) / TEXEL))
        by0 = int(np.floor((ys.min() - R_BAND - Y0) / TEXEL))
        by1 = int(np.ceil((ys.max() + R_BAND - Y0) / TEXEL))
        bx0, by0 = max(bx0, 0), max(by0, 0)
        bx1, by1 = min(bx1, NX), min(by1, NY)
        gx = X0 + (np.arange(bx0, bx1) + .5) * TEXEL
        gy = Y0 + (np.arange(by0, by1) + .5) * TEXEL
        XX, YY = np.meshgrid(gx, gy)
        tree = cKDTree(Q)
        dd, ii = tree.query(np.c_[XX.ravel(), YY.ravel()], distance_upper_bound=R_BAND, workers=-1)
        ok = np.isfinite(dd)
        idx = np.nonzero(ok)[0]
        if idx.size == 0:
            continue
        ii = ii[ok] + a
        # a texel whose nearest sample is the chunk's own end is a false round cap: leave it to the neighbouring chunk
        keep = np.ones(ii.size, bool)
        if a > 0:
            keep &= ii > a
        if b < len(s):
            keep &= ii < b - 1
        idx, ii = idx[keep], ii[keep]
        if idx.size == 0:
            continue
        dist = dd[ok][keep].astype(np.float32)
        px, py = XX.ravel()[idx], YY.ravel()[idx]
        cross = (px - q[ii, 0]) * tan[ii, 1] - (py - q[ii, 1]) * tan[ii, 0]
        d = np.sign(cross) * dist
        rows = (idx // (bx1 - bx0)) + by0
        cols = (idx % (bx1 - bx0)) + bx0
        hn = PS[rows, cols]
        wl = w[ii]
        u = d / wl
        au = np.abs(u)
        fl = flow[ii]
        S = s[ii]

        # contact: at the edge only fibre tops reach the ball; toward the centre the ball presses valleys up to it
        squeeze = np.clip(1 - au * au, 0, 1)
        thr = 1.05 + (-2.7 - 1.05) * squeeze ** .5
        contact = smooth(thr - .35, thr + .35, hn)
        contact *= au < 1.14
        contact *= 1 - smooth(1.0, 1.14, au)
        # the striations of the ball's own ink film
        st = 1 + sum(amp[j] * np.sin(2 * np.pi * fr[j] * (u + 1) / 2 + phase[j] + drifts[j][ii]) for j in range(4))
        # ink pushed to the shoulders of the groove
        edge = 1 + .18 * np.exp(-((au - .84) / .17) ** 2)
        Dk = contact * np.clip(fl, .4, 1.8) * st * edge
        # a whisper of wicking along the fibres just outside the line
        wick = np.exp(-(((au - 1.02) / .17) ** 2)) * smooth(.9, 1.9, hn) * .12
        Dk = np.maximum(Dk, wick * (au < 1.5))

        # goops bridge the valleys: they are round, thick and do not care about the paper
        for gs, gd, gr, ga in gp:
            rr = np.sqrt((S - gs) ** 2 + (d - gd) ** 2) / gr
            near = rr < 1.25
            if near.any():
                g_ = smooth(1.15, .72, rr[near]) * (1 + .05 * hn[near])
                Dk[near] = np.maximum(Dk[near], ga * 1.3 * g_)

        # the groove the ball leaves
        pr = np.clip(prs[ii], .4, 1.5)
        groove = DEPTH * pr * np.clip(1 - au ** 2, 0, 1) ** 1.25 * (au < 1)
        shoulder = -3.2 * np.exp(-((au - 1.18) / .17) ** 2)
        Gk = groove + shoulder

        Tk = (t[ii] / DUR).astype(np.float32)

        # composite this chunk into the stroke (first arrival, strongest layer)
        inked = (Dk > .02) & (Tk < T[rows, cols])
        T[rows[inked], cols[inked]] = Tk[inked]
        TA[rows, cols] = np.minimum(TA[rows, cols], Tk)
        G[rows, cols] = np.maximum(G[rows, cols], Gk.astype(np.float32))
        DS[rows, cols] = np.maximum(DS[rows, cols], Dk.astype(np.float32))
        seen[rows, cols] = True
    # strokes crossing lay ink twice
    m = DS > 0
    D[m] = np.maximum(D[m], DS[m]) + .16 * np.minimum(D[m], DS[m])      # the clock can't split a crossing, so it may only darken a little
    print(f'stroke {k}: {L:.1f} mm, {len(goops)} goops, {m.sum() / 1e6:.2f} Mt')

D = np.clip(D, 0, 1.6)
T = np.where(T < 1, T, TA)
print(f'density >0.02 on {(D > .02).mean() * 100:.2f}% of the sheet, {blobs_total} goops, max {D.max():.2f}')
print('groove', G.min(), G.max())


def save16(path, a):
    Image.fromarray(np.flipud((a * 65535 + .5).clip(0, 65535).astype(np.uint16))).save(path, compress_level=1)


os.makedirs(os.path.join(FILM, 'work'), exist_ok=True)
save16(os.path.join(A, 'ink_t.png'), T)
save16(os.path.join(A, 'ink_d.png'), D / 1.6)
save16(os.path.join(A, 'ink_g.png'), (G + 12) / 72)
np.save(os.path.join(FILM, 'work', 'ink_d_full.npy'), D.astype(np.float32))
with open(os.path.join(A, 'ink_grid.json'), 'w', encoding='utf8') as f:
    json.dump(dict(x0=X0, y0=Y0, w=W, h=H, dur=DUR, nx=NX, ny=NY, texel=TEXEL), f)

# --- a flat proof: the whole line, and a 1:1 crop of the start of the F
paper_w = np.array([.92, .91, .86])
ink = np.array([.03, .08, .42])
sh = 1 + (P * .06)
img = (np.clip(sh, 0, 1.2)[..., None] * paper_w) * (ink[None, None, :] ** np.clip(D[..., None] * .9, 0, 1.3))
img8 = (np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
im = Image.fromarray(np.flipud(img8))
im.resize((NX // 4, NY // 4), Image.LANCZOS).save(os.path.join(FILM, 'work', 'ink_proof.png'))
xc, yc = int((1.6 + 3) / TEXEL), int((7.6 + 3.5) / TEXEL)
Image.fromarray(np.flipud(img8[yc - 450:yc + 450, xc - 800:xc + 800])).save(os.path.join(FILM, 'work', 'ink_crop.png'))
print('done')
