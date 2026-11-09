"""Render frames from work/pen.blend.

  blender -b work/pen.blend -P tools/render.py -- frames 0,300,500 [--res 1280] [--spp 128] [--out work/stills] [--set exposure=0.2]
  blender -b work/pen.blend -P tools/render.py -- range 0 659 [--res 1920] [--spp 256] [--out work/frames]
"""
import argparse
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index('--') + 1:]
ap = argparse.ArgumentParser()
ap.add_argument('mode')
ap.add_argument('a')
ap.add_argument('b', nargs='?')
ap.add_argument('--res', type=int, default=1920)
ap.add_argument('--spp', type=int, default=256)
ap.add_argument('--out', default='work/stills')
ap.add_argument('--blur', type=int, default=1)
ap.add_argument('--skip', type=int, default=1, help='skip frames already on disk')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--denoiser', default='')
ap.add_argument('--prefilter', default='')
ap.add_argument('--thresh', type=float, default=0.0)
ap.add_argument('--device', default='OPTIX', help='OPTIX, CUDA, HIP, METAL, ONEAPI or CPU')
args = ap.parse_args(argv)

scene = bpy.context.scene
c = scene.cycles
cp = bpy.context.preferences.addons['cycles'].preferences
if args.device == 'CPU':
    scene.cycles.device = 'CPU'
    c.denoising_use_gpu = False
else:
    cp.compute_device_type = args.device
    cp.get_devices()
    for d in cp.devices:
        d.use = d.type == args.device
    scene.cycles.device = 'GPU'
    c.denoising_use_gpu = True
c.samples = args.spp
c.adaptive_min_samples = min(48, args.spp // 3)
scene.render.resolution_x = args.res
scene.render.resolution_y = int(round(args.res * 9 / 16))
scene.render.use_motion_blur = bool(args.blur)
if args.denoiser == 'NONE':
    c.use_denoising = False
elif args.denoiser:
    c.denoiser = args.denoiser
if args.prefilter:
    c.denoising_prefilter = args.prefilter
if args.thresh:
    c.adaptive_threshold = args.thresh

film = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outdir = os.path.join(film, args.out) if not os.path.isabs(args.out) else args.out
os.makedirs(outdir, exist_ok=True)

if args.mode == 'frames':
    frames = [int(float(x)) for x in args.a.split(',')]
else:
    frames = list(range(int(args.a), int(args.b) + 1, args.every))

scene.render.filepath = os.path.join(outdir, 'f')
t_all = time.time()
for n, f in enumerate(frames):
    path = os.path.join(outdir, f'f{f:04d}.png')
    if args.skip and os.path.exists(path):
        continue
    scene.frame_set(f)
    t0 = time.time()
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print(f'FRAME {f} ({n + 1}/{len(frames)}) {time.time() - t0:.1f}s', flush=True)
print(f'DONE {len(frames)} frames in {time.time() - t_all:.0f}s', flush=True)
