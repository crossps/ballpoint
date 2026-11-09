"""Every moving thing as a per-frame track, from the hand. No bpy here, so plain Python can read it too.

World is millimetres, Z up, paper at z = 0. Text runs along +x, baseline y = 0.
"""
import math
import os

import numpy as np


def g1(a, sigma, mode='nearest'):
    """Gaussian smoothing with numpy only: Blender's Python has no scipy."""
    r = int(4 * sigma + .5)
    k = np.exp(-.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(a, r, mode='edge'), k, mode='valid')

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.dirname(HERE)
FPS, DUR = 30, 22.0
NF = int(round(DUR * FPS))
BALL_R = .5
LENS = 85.0                        # mm on a 36 mm sensor
TEXT_C = (20.6, 3.75)              # where the finished words sit


def smoother(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)


def norm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def quat_rotate_z_to(d):
    """Quaternion (w, x, y, z) taking +Z to each unit vector d."""
    z = np.array([0, 0, 1.0])
    c = np.cross(np.broadcast_to(z, d.shape), d)
    w = 1 + d[..., 2]
    q = np.concatenate([w[..., None], c], axis=-1)
    return norm(q)


def quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def tracks():
    h = np.load(os.path.join(FILM, 'assets', 'hand.npz'))
    t, x, y, z = h['t'], h['x'], h['y'], h['z']
    RATE = 960
    press, vel, down = h['press'], h['vel'], h['down']
    last = float(h['last_lift'])

    # --- the pen: the ball follows the hand; the top of the pen follows a slower wrist
    ball = np.c_[x, y, z + BALL_R]
    away = smoother((t - (last + 1.1)) / 1.6)[:, None] * np.array([150.0, -90.0, 420.0])   # gone for good, so its shadow goes too
    ball = ball + away
    LEAN_EL, LEAN_AZ = math.radians(59), math.radians(-34)     # leaning back over the writing hand
    ax0 = np.array([math.cos(LEAN_EL) * math.cos(LEAN_AZ), math.cos(LEAN_EL) * math.sin(LEAN_AZ), math.sin(LEAN_EL)])
    L = 100.0
    sw = .75 * RATE
    px = g1(x, sw, mode='nearest') * .9 + x * .1 + ax0[0] * L
    py = g1(y, sw, mode='nearest') * .9 + y * .1 + ax0[1] * L
    pz = z + BALL_R + ax0[2] * L
    px += away[:, 0]; py += away[:, 1]; pz += away[:, 2]
    # the slow sway of a real hand while it writes
    wobble = np.random.default_rng(3)
    for arr in (px, py):
        n = g1(wobble.standard_normal(len(t) + 8 * RATE), RATE * .45)[:len(t)]
        arr += n / n.std() * .55
    axis = norm(np.c_[px, py, pz] - ball)

    # --- the ball rolls: world rotation integrated from the contact path
    dt = 1 / RATE
    vx = np.gradient(x, dt) * down
    vy = np.gradient(y, dt) * down
    w = np.c_[-vy, vx, np.zeros_like(vx)] / BALL_R          # omega = z cross v / R
    qb = np.zeros((len(t), 4))
    q = np.array([1.0, .0, .0, .0])
    for i in range(len(t)):
        qb[i] = q
        th = np.linalg.norm(w[i]) * dt
        if th > 1e-9:
            a = w[i] / np.linalg.norm(w[i])
            dq = np.r_[math.cos(th / 2), a * math.sin(th / 2)]
            q = quat_mul(dq, q)

    # --- the camera
    xc = x.copy(); yc = y.copy()
    idx = t > last
    xc[idx] = x[np.argmax(idx)]; yc[idx] = y[np.argmax(idx)]
    fol = .62 * RATE
    fx = g1(xc, fol, mode='nearest'); fy = g1(yc, fol, mode='nearest')

    T0 = last + .4
    T1 = T0 + 2.75
    b = smoother((t - T0) / (T1 - T0))
    # the shot opens right on the ball, then settles into the slow follow
    bs = smoother(t / 1.7)
    fx = x + (fx - x) * bs
    fy = y + (fy - y) * bs
    tx = fx + (TEXT_C[0] - fx) * b
    ty = fy + (TEXT_C[1] - fy) * b

    keys_w = np.array([[0, 9.5], [1.8, 17.5], [8.6, 19.0], [10.4, 31.0], [T0, 34.0], [T1, 50.0], [DUR, 48.5]])
    keys_el = np.array([[0, 22.0], [1.6, 27.0], [8.6, 30.0], [10.4, 24.0], [T0, 24.0], [T1, 62.0], [DUR, 63.0]])
    keys_az = np.array([[0, -34.0], [1.6, -28.0], [T0, -22.0], [T1, -1.5], [DUR, -1.0]])

    def blend(k):
        out = np.interp(t, k[:, 0], k[:, 1])
        return g1(out, .35 * RATE, mode='nearest')

    wid = blend(keys_w); el = np.radians(blend(keys_el)); az = np.radians(blend(keys_az))

    nr = np.random.default_rng(11)
    def drift(amp, sigma):
        n = g1(nr.standard_normal(len(t) + 12 * RATE), sigma * RATE)[:len(t)]
        return n / n.std() * amp

    tgt = np.c_[tx + drift(.05, .9), ty + drift(.05, .9), np.zeros_like(t)]
    dist = wid * LENS / 36.0
    off = np.c_[np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)]
    cam = tgt + dist[:, None] * off
    fwd = norm(tgt - cam)
    roll = np.radians(drift(.09, 1.1))

    # focus: on the nib while it writes, then on the words
    bf = smoother((t - T0) / (T1 - T0 - .3))
    gx = g1(xc, .03 * RATE, mode='nearest'); gy = g1(yc, .03 * RATE, mode='nearest')
    focus = np.c_[gx + (TEXT_C[0] - gx) * bf, gy + (TEXT_C[1] - gy) * bf, .42 * (1 - bf)]
    fdist = np.sum((focus - cam) * fwd, axis=1)
    apert = np.interp(t, [0, 8.6, 10.4, T0, T1, DUR], [1.05, 1.05, 1.5, 1.5, 2.1, 2.1])    # entrance pupil diameter, mm
    apert = g1(apert, .2 * RATE, mode='nearest')

    # --- per frame
    idx = (np.arange(NF + 1) * RATE / FPS).astype(int).clip(0, len(t) - 1)
    return dict(t=t[idx], ball=ball[idx], axis=axis[idx], qb=qb[idx], cam=cam[idx], fwd=fwd[idx], roll=roll[idx], fdist=fdist[idx], apert=apert[idx],
                press=press[idx], vel=vel[idx], down=down[idx], tgt=tgt[idx], wid=wid[idx], last=last, T0=T0, T1=T1, x=x[idx], y=y[idx], z=z[idx])
