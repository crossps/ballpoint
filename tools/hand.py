"""The hand. Writes the line in joined natural handwriting: authored centrelines (mm on the paper, baseline y = 0),
then real handwriting dynamics. Speed follows the two-thirds power law (v ~ curvature^-1/3), ramps from rest at each
pen-down, pressure rises as the hand slows, and pen-up travels are minimum-jerk arcs.

Everything downstream reads the result: tools/ink.py lays the line from the stroke samples, tools/rig.py moves the
pen from the time-uniform track.

Run: python -I tools/hand.py [preview.png]
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d as g1

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.dirname(HERE)
OUT = os.path.join(FILM, 'assets', 'hand.npz')

SLANT = .15          # forward lean of the hand, dx per dy
STEP = .010          # mm between stroke samples
RATE = 960           # Hz of the time-uniform track
BALL_R = .5          # mm

T_TOUCH = .24        # first contact
T_WRITE_END = 15.75  # the last lift-off

X5A, X5B = 26.8, 33.9


def shift(pts, dx=0, dy=0):
    return [(x + dx, y + dy) for x, y in pts]


FIVE_BODY = [(1.28, 7.25), (1.05, 5.65), (.72, 3.98), (1.72, 4.78), (3.05, 4.72), (4.08, 3.55), (4.22, 1.95), (3.32, .48), (1.85, -.04), (.55, .34), (.02, 1.12)]
FIVE_BAR = [(1.26, 7.32), (2.72, 7.52), (4.62, 7.9)]

STROKES = [
    ('F stem', [(2.32, 7.98), (1.86, 7.36), (1.5, 5.6), (1.2, 3.42), (.95, 1.35), (.6, -.38)], .1),
    ('F top', [(.32, 7.12), (1.6, 7.56), (3.42, 7.76), (5.22, 7.96), (6.88, 8.38)], .08),
    ('F mid', [(1.04, 3.54), (2.2, 3.86), (3.62, 3.96), (4.86, 4.22)], .16),
    ('able', [
        # a: bowl anticlockwise, up to the peak, back down the stem
        (9.86, 2.96), (9.06, 3.52), (8.0, 3.36), (7.25, 2.42), (7.0, 1.2), (7.45, .2), (8.4, .04), (9.36, .86), (9.96, 2.3), (10.16, 3.52),
        (10.06, 2.2), (10.1, .8), (10.56, .04), (11.32, .22),
        # b: up into the loop, down the stem, round the bowl, tie
        (13.16, 1.72), (14.16, 4.62), (14.76, 7.12), (14.56, 8.26), (13.84, 8.1), (13.44, 6.3), (13.16, 3.62), (13.0, 1.3), (13.22, .12),
        (14.16, -.03), (15.42, .36), (16.0, 1.62), (15.66, 2.96), (14.56, 3.52), (13.76, 3.1), (14.12, 2.8), (15.02, 2.96), (16.42, 3.32),
        # l: loop and stem
        (17.92, 4.62), (18.66, 6.92), (18.52, 8.22), (17.86, 8.06), (17.5, 5.72), (17.36, 2.62), (17.5, .62), (18.06, -.03),
        # e: crossbar up, round the back, and out
        (19.22, .26), (20.22, 1.2), (21.06, 1.76), (21.76, 2.66), (21.42, 3.46), (20.46, 3.26), (19.96, 2.06), (20.06, .7), (20.96, -.04), (22.22, .2), (23.26, .98),
    ], .3),
    ('5 body', shift(FIVE_BODY, X5A), .1),
    ('5 bar', shift(FIVE_BAR, X5A), .22),
    ('dot', [(32.22, .16), (32.34, .1), (32.42, .2), (32.3, .26), (32.24, .19)], .24),
    ('5 body', shift(FIVE_BODY, X5B), .1),
    ('5 bar', shift(FIVE_BAR, X5B), .0),
]


def catmull(pts, step=STEP):
    """Centripetal Catmull-Rom through the points, resampled to even arc length."""
    p = np.asarray(pts, float)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** .5
        t2 = t1 + np.linalg.norm(p2 - p1) ** .5
        t3 = t2 + np.linalg.norm(p3 - p2) ** .5
        n = max(16, int(np.linalg.norm(p2 - p1) / step * 2))
        t = np.linspace(t1, t2, n, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    q = np.vstack(out + [p[-2]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    S = np.arange(0, s[-1], step)
    return np.c_[np.interp(S, s, q[:, 0]), np.interp(S, s, q[:, 1])], S


def noise1(n, sigma, rng):
    x = g1(rng.standard_normal(n + int(8 * sigma)), sigma, mode='wrap')[:n]
    return x / max(x.std(), 1e-9)


def build():
    rng = np.random.default_rng(55)
    strokes = []
    for name, pts, pause in STROKES:
        q, s = catmull(pts)
        q[:, 0] += SLANT * q[:, 1]
        # a hand is not a plotter: a slow, tiny wander on top of the authored line
        n = len(q)
        sig = max(.9 / STEP, 8)
        q[:, 0] += .022 * noise1(n, sig, rng) * np.minimum(1, s[-1] / 2)
        q[:, 1] += .022 * noise1(n, sig, rng) * np.minimum(1, s[-1] / 2)
        d = g1(np.gradient(q, axis=0), 3, axis=0, mode='nearest')
        tan = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        ang = np.unwrap(np.arctan2(tan[:, 1], tan[:, 0]))
        kappa = np.abs(g1(np.gradient(ang) / STEP, .12 / STEP, mode='nearest'))
        strokes.append(dict(name=name, q=q, s=s, tan=tan, kappa=kappa, pause=pause))

    def timing(c):
        """The time of every sample of every stroke for a given speed constant."""
        for k in strokes:
            r = 1 / np.maximum(k['kappa'], 1e-3)
            v = c * np.clip(r, .22, 7.0) ** (1 / 3)
            v = g1(v, .35 / STEP, mode='nearest')
            lim = 95.0                                    # mm/s^2, the wrist's acceleration limit
            v[0] = .4
            for i in range(1, len(v)):
                v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * lim * STEP))
            v[-1] = min(v[-1], .4)
            for i in range(len(v) - 2, -1, -1):
                v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * lim * STEP))
            k['v0'] = v
        t = T_TOUCH + .06                                  # the ball sits a moment before it moves
        for j, k in enumerate(strokes):
            rh = np.random.default_rng(100 + j)
            v = k['v0'] * (1 + .1 * noise1(len(k['v0']), .9 / STEP, rh))
            dt = STEP / np.maximum(v, .05)
            k['t'] = t + np.r_[0, np.cumsum(dt[:-1])]
            k['vel'] = v
            t = k['t'][-1]
            if j + 1 < len(strokes):
                a, b = k['q'][-1], strokes[j + 1]['q'][0]
                dist = np.linalg.norm(b - a)
                k['air'] = float(np.clip(.2 + .05 * dist, .22, .85))
                t += .04 + k['air'] + k['pause']
                k['air_at'] = float(k['t'][-1] + .025)
        return t

    lo, hi = 2.0, 40.0
    for _ in range(48):
        c = (lo * hi) ** .5
        if timing(c) > T_WRITE_END:
            lo = c
        else:
            hi = c
    end = timing(c)

    for j, k in enumerate(strokes):
        rp = np.random.default_rng(300 + j)
        n = len(k['q'])
        v = k['vel']
        p = 1 + .14 * noise1(n, 1.3 / STEP, rp) - .22 * (v / max(v.max(), 1e-6) - .55)
        edge = np.minimum(1, np.minimum(k['s'] / .35, (k['s'][-1] - k['s']) / .35) + .2)
        k['press'] = p * (.55 + .45 * np.clip(edge, 0, 1))

    # --- the time-uniform track
    dur = 22.0
    tt = np.arange(0, dur, 1 / RATE)
    x = np.zeros_like(tt); y = np.zeros_like(tt); z = np.zeros_like(tt)
    press = np.zeros_like(tt); vel = np.zeros_like(tt); down = np.zeros_like(tt, dtype=np.int8); sid = -np.ones_like(tt, dtype=np.int16)

    first = strokes[0]
    k0 = tt < T_TOUCH + .06
    x[k0], y[k0] = first['q'][0]
    u = np.clip(tt[k0] / T_TOUCH, 0, 1)
    z[k0] = .62 * (1 - (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5))
    down[k0] = u >= 1
    press[k0] = .7 * (u >= 1)

    for j, k in enumerate(strokes):
        m = (tt >= k['t'][0]) & (tt <= k['t'][-1])
        x[m] = np.interp(tt[m], k['t'], k['q'][:, 0])
        y[m] = np.interp(tt[m], k['t'], k['q'][:, 1])
        press[m] = np.interp(tt[m], k['t'], k['press'])
        vel[m] = np.interp(tt[m], k['t'], k['vel'])
        down[m] = 1
        sid[m] = j
        if j + 1 < len(strokes):
            nxt = strokes[j + 1]
            a, b = k['q'][-1], nxt['q'][0]
            # settle, lift, travel, touch
            m = (tt > k['t'][-1]) & (tt < nxt['t'][0])
            t0, t1 = k['air_at'], k['air_at'] + k['air']
            u = np.clip((tt[m] - t0) / (t1 - t0), 0, 1)
            e = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            x[m] = a[0] + (b[0] - a[0]) * e
            y[m] = a[1] + (b[1] - a[1]) * e
            dist = np.linalg.norm(b - a)
            z[m] = np.minimum(1.5, .38 + .1 * dist) * np.sin(np.pi * u) ** 1.2
            down[m] = (u <= 0) | (u >= 1)
            press[m] = .7 * ((u <= 0) | (u >= 1))

    # --- the last lift: up and away
    last = strokes[-1]
    tl = last['t'][-1]
    m = tt > tl
    x[m], y[m] = last['q'][-1]
    u = np.clip((tt[m] - tl - .06) / 1.5, 0, 1)
    e = u * u * (3 - 2 * u)
    z[m] = 34 * e ** 1.5
    x[m] += 22 * e ** 2
    y[m] -= 8 * e ** 2

    # the small tremor every real hand has, on the tip and only while it moves
    rt = np.random.default_rng(7)
    tr = .0065 * g1(rt.standard_normal(len(tt)), RATE / 26)
    tr2 = .0065 * g1(rt.standard_normal(len(tt)), RATE / 26)
    tr /= max(tr.std() / .0065, 1e-9); tr2 /= max(tr2.std() / .0065, 1e-9)
    moving = g1((vel > .3).astype(float), RATE * .03)
    x += tr * moving; y += tr2 * moving

    # --- save
    data = dict(t=tt, x=x, y=y, z=z, press=press, vel=vel, down=down, sid=sid, n=np.array(len(strokes)),
                t_end=np.array(end), c=np.array(c), last_lift=np.array(tl), touch=np.array(T_TOUCH))
    names = []
    for j, k in enumerate(strokes):
        names.append(k['name'])
        for f in ('q', 's', 'tan', 't', 'vel', 'press'):
            data[f'k{j}_{f}'] = k[f]
    data['names'] = np.array(names)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **data)
    return data, strokes


def preview(strokes, path):
    from PIL import Image, ImageDraw
    ss = 40
    x0, y0 = -3, -3
    W, H = 47, 14.5
    im = Image.new('L', (int(W * ss), int(H * ss)), 245)
    d = ImageDraw.Draw(im)
    for k in strokes:
        q = k['q'][::3]
        pts = [((a - x0) * ss, (H + y0 - b) * ss) for a, b in q]
        d.line(pts, fill=30, width=max(2, int(.4 * ss)), joint='curve')
    im.convert('RGB').save(path)


if __name__ == '__main__':
    data, strokes = build()
    print(f"c={float(data['c']):.2f} end={float(data['t_end']):.3f} s, strokes {len(strokes)}, ink path {sum(k['s'][-1] for k in strokes):.1f} mm")
    for k in strokes:
        print(f"  {k['name']:8s} {k['s'][-1]:6.1f} mm  {k['t'][0]:6.2f} to {k['t'][-1]:6.2f}   v mean {k['vel'].mean():5.1f} max {k['vel'].max():5.1f} mm/s")
    if len(sys.argv) > 1:
        preview(strokes, sys.argv[1])
