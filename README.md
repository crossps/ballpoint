# Ballpoint

The complete source for a 22-second macro shot of a ballpoint pen writing on paper, rendered in Blender Cycles.

There is no footage, generated video, downloaded model or texture pack here. The pen, paper fibres, ink, handwriting, camera and lights all come from the Python in `tools/`. The only outside file is the `interior.exr` studio light that ships with Blender. Run the steps below and you get the frames.

## Requirements

- [Blender 4.5 LTS](https://www.blender.org/download/lts/4-5/) (`blender` on your PATH)
- Python 3.11
- ffmpeg, for the final encode
- About 6 GB of disk for the textures and 16-bit frames

```
pip install -r requirements.txt
```

## Build it

Run these from the repo root:

```
python -I tools/hand.py
python -I tools/paper.py
python -I tools/ink.py
blender -b -P tools/build.py
blender -b work/pen.blend -P tools/render.py -- range 0 659 --res 1920 --spp 96 --out work/frames
python -I tools/finish.py --src work/frames --out renders/ballpoint.mp4 --crf 12
```

To look at the scene without rendering all of it, open `work/pen.blend` after the `build.py` step, or render a few frames:

```
blender -b work/pen.blend -P tools/render.py -- frames 0,300,600 --spp 96 --out work/stills
```

`render.py` uses OptiX by default. Pass `--device CUDA`, `HIP`, `METAL`, `ONEAPI` or `CPU` for other hardware. On an RTX 4060 laptop, 1080p at 96 samples takes about 20 seconds a frame, so about 4 hours for all 660. An interrupted render resumes where it stopped.

## How it works

| Script | What it does |
|---|---|
| `tools/hand.py` | Authors the strokes as centrelines, then applies handwriting dynamics: two-thirds power law speed, wrist acceleration limit, pressure that rises as the hand slows, minimum-jerk pen-up arcs and a small tremor. Everything else reads its output. |
| `tools/paper.py` | Lays several hundred thousand individual cellulose fibres into a seamless 41 mm tile at 5 µm per texel, then calenders it flat. |
| `tools/ink.py` | Puts the line on that surface texel by texel. Ink lands where the paper actually touches the ball, so edges break up on the fibre tops. Also bakes the groove and each texel's arrival time. |
| `tools/rig.py` | Per-frame tracks: pen, wrist sway, camera, focus and the ball's rolling rotation, integrated from the path it draws. |
| `tools/build.py` | Builds the pen, sheet, lint, materials, lights and keyframes into `work/pen.blend`. |
| `tools/render.py` | Renders frames from the `.blend`. |
| `tools/finish.py` | Grade and encode: warm paper white, halation, lateral chromatic aberration, vignette and grain. |

World units are millimetres, with the paper at z = 0, text along +x and the baseline at y = 0.

## License

MIT
