"""The last step of the picture: what a camera and a colourist would add to the Cycles frames, then the encode.

Whites lifted to a warm paper white, a little vignette, a hair of lateral chromatic aberration, soft halation on the
hottest speculars, sensor grain. Then H.264 at the film's native 30 fps.

  python -I tools/finish.py --src work/frames --out renders/ballpoint.mp4 --crf 12
  python -I tools/finish.py --src work/stills --stills work/graded       (grade PNG stills, no encode)
"""
import argparse
import glob
import os
import subprocess

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--out', default='')
ap.add_argument('--stills', default='')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--crf', type=int, default=14)
a = ap.parse_args()


def load(path):
    """16-bit PNG to float RGB, display-referred, no 8-bit detour."""
    data = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-'], capture_output=True, check=True).stdout, np.uint16)
    w, h = Image.open(path).size
    return data.reshape(h, w, 3).astype(np.float32) / 65535.0


def scale_about_centre(c, z):
    """One channel zoomed about the frame centre, the cheap form of lateral chromatic aberration."""
    im = Image.fromarray(c, 'F')
    w, h = im.size
    k = 1 / z
    return np.asarray(im.transform((w, h), Image.AFFINE, (k, 0, w / 2 * (1 - k), 0, k, h / 2 * (1 - k)), Image.BILINEAR), np.float32)


def blur_fast(c, f):
    im = Image.fromarray(c, 'F')
    w, h = im.size
    s = im.resize((max(1, w // f), max(1, h // f)), Image.BOX)
    return np.asarray(s.resize((w, h), Image.BICUBIC), np.float32)


VIG = {}


def grade(x, n):
    h, w = x.shape[:2]
    k = w / 1920
    # the paper to a warm white, a soft shoulder so the speculars still have somewhere to go
    x = x * np.array([1.058, 1.05, 1.022], np.float32)
    hi = x > .9
    x = np.where(hi, .9 + (1 - np.exp(-(x - .9) / .1)) * .1, x)
    # halation: the hottest speculars breathe a little warm light
    lum = x.max(2)
    glow = np.clip((lum - .965) / .035, 0, 1)
    g1 = blur_fast(glow.astype(np.float32), max(2, int(6 * k)))
    x = x + (g1 * .1)[..., None] * np.array([1.0, .78, .55], np.float32)
    # lateral chromatic aberration, about half a pixel at the corners
    z = .55 / (w / 2)
    x[..., 0] = scale_about_centre(x[..., 0], 1 + 2 * z)        # blue stays put and the others zoom in, so no channel samples off the frame edge
    x[..., 1] = scale_about_centre(x[..., 1], 1 + z)
    # vignette
    if (w, h) not in VIG:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (w / 2))
        VIG[(w, h)] = 1 - .075 * np.clip(r, 0, 1.2) ** 2.4
    x = x * VIG[(w, h)][..., None]
    # sensor grain: strongest in the mids, held off the bright paper so the codec keeps it
    r = np.random.default_rng(1000 + n)
    gr = r.standard_normal((h // 2 + 1, w // 2 + 1)).astype(np.float32)
    gr = np.asarray(Image.fromarray(gr, 'F').resize((w, h), Image.BILINEAR), np.float32)
    amp = .0075 * (1 - .55 * np.clip(x.mean(2), 0, 1) ** 2)
    x = x + (gr * amp)[..., None]
    return np.clip(x, 0, 1)


frames = sorted(glob.glob(os.path.join(FILM, a.src, 'f*.png')))
if a.stills:
    out = os.path.join(FILM, a.stills)
    os.makedirs(out, exist_ok=True)
    for f in frames:
        n = int(os.path.basename(f)[1:5])
        Image.fromarray((grade(load(f), n) * 255 + .5).astype(np.uint8)).save(os.path.join(out, os.path.basename(f)))
    print('graded', len(frames), 'stills to', out)
    raise SystemExit

fps = 30 / a.every
w, h = Image.open(frames[0]).size
dst = os.path.join(FILM, a.out)
os.makedirs(os.path.dirname(dst), exist_ok=True)
cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-framerate', f'{fps:g}', '-i', '-',
       '-vf', 'scale=in_range=full:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,format=yuv420p', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(a.crf),
       '-profile:v', 'high', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
       '-movflags', '+faststart', dst]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i, f in enumerate(frames):
    n = int(os.path.basename(f)[1:5])
    p.stdin.write((grade(load(f), n) * 255 + .5).astype(np.uint8).tobytes())
    if i % 60 == 0:
        print(i, '/', len(frames), flush=True)
p.stdin.close()
p.wait()
print('wrote', dst, os.path.getsize(dst) // 1024, 'KB')
