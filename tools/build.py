"""Builds work/pen.blend: the pen, the sheet, the room and every key. Millimetres, Z up, paper at z = 0.

  blender -b -P tools/build.py
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.dirname(HERE)
A = os.path.join(FILM, 'assets')
sys.path.insert(0, HERE)
import rig  # noqa: E402

BALL_R = rig.BALL_R
GRID = dict(x0=-3.0, y0=-3.5, w=47.0, h=14.5, dur=22.0)
TILE = 40.96

scene = bpy.context.scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = .001
scene.unit_settings.length_unit = 'MILLIMETERS'


# ---------------------------------------------------------------- meshes
def mesh_from_grid(name, rings, closed=True, sharp=30.0):
    """rings: (M, N, 3). Quads between consecutive rings, wrapping around N."""
    M, N = rings.shape[:2]
    verts = rings.reshape(-1, 3)
    faces = []
    jmax = N if closed else N - 1
    for i in range(M - 1):
        for j in range(jmax):
            a = i * N + j
            b = i * N + (j + 1) % N
            faces.append((a, b, b + N, a + N))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces)
    return finish_mesh(me, sharp)


def finish_mesh(me, sharp=30.0, merge=1e-5):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    me.update()
    if sharp is not None:
        me.set_sharp_from_angle(angle=math.radians(sharp))
    return me


def add_obj(name, me, parent=None):
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    if parent is not None:
        o.parent = parent
    return o


def lathe(name, prof, n=256, rfun=None, sharp=30.0):
    """prof: (r, z) list, the closed half-section of a solid of revolution. r = 0 points close on the axis."""
    prof = np.asarray(prof, float)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rings = []
    for (r, z) in prof:
        R = r if rfun is None else rfun(r, z, ang)
        rings.append(np.c_[R * np.cos(ang), R * np.sin(ang), np.full(n, z)])
    return mesh_from_grid(name, np.array(rings), True, sharp)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def densify(pts, step):
    out = [np.asarray(pts[0], float)]
    for p in pts[1:]:
        p = np.asarray(p, float)
        d = np.linalg.norm(p - out[-1])
        n = max(1, int(d / step))
        for k in range(1, n + 1):
            out.append(out[-1] + (p - out[-1]) / (n - k + 1))
    return np.array(out)


def round_profile(pts, radius, n=6):
    """Rounds the corners of an (r, z) polyline with a small arc-ish Bezier so edges catch light."""
    pts = [np.asarray(p, float) for p in pts]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        p, a, b = pts[i], pts[i - 1], pts[i + 1]
        ra = min(radius, np.linalg.norm(a - p) * .45)
        rb = min(radius, np.linalg.norm(b - p) * .45)
        s = p + (a - p) / np.linalg.norm(a - p) * ra
        e = p + (b - p) / np.linalg.norm(b - p) * rb
        for k in range(n + 1):
            u = k / n
            out.append((1 - u) ** 2 * s + 2 * (1 - u) * u * p + u * u * e)
    out.append(pts[-1])
    return np.array(out)


def solid_of_revolution(name, outer, n=256, sharp=30.0, rfun=None):
    pts = [(0.0, outer[0][1])] + list(map(tuple, outer)) + [(0.0, outer[-1][1])]
    return lathe(name, pts, n, rfun, sharp)


# --- the machined point
def build_point():
    prof = [
        (.498, -.05), (.515, -.075), (.572, -.06), (.635, .02), (.705, .22), (.86, .85), (1.05, 1.8), (1.18, 2.9),
        (1.22, 3.1), (1.24, 3.22), (1.5, 3.5), (1.5, 6.0),
    ]
    outer = round_profile(prof, .035, 5)
    outer = densify(list(outer), .08)
    # the inside of the seat, hidden behind the ball, so the part stays one closed solid
    inner = [(.30, .36), (.16, .42), (0, .43)]
    full = np.vstack([np.array(inner[::-1]), outer])
    flutes = lambda r, z, a: r * (1 + .012 * np.cos(a * 5) * np.exp(-((z + .03) / .12) ** 2))
    return lathe('point', full, 360, flutes, 30.0)


def build_ink_ring():
    """The crust of ink where the ball meets the seat: irregular and gathered, never a clean torus."""
    n = 360
    rng = np.random.default_rng(4)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)

    def nz(sig):
        x = rng.standard_normal(n * 3)
        from numpy.fft import irfft, rfft
        f = rfft(x)
        f[int(n * 3 / sig):] = 0
        y = irfft(f, n * 3)[:n]
        return y / y.std()

    fat = np.clip(.5 + .35 * nz(36) + .22 * nz(90), .08, 1.3)
    rows = []
    prof = 14
    for k in range(prof + 1):
        v = k / prof * math.pi
        # a crescent sitting in the gap, swelling where the fat is
        rr = .503 + (.012 + .05 * fat) * math.sin(v)
        zz = -.066 - (.014 + .026 * fat) * math.cos(v)
        rows.append(np.c_[rr * np.cos(ang), rr * np.sin(ang), zz + .008 * np.sin(v) * nz(120)])
    me = mesh_from_grid('ink_ring', np.array(rows), True, None)
    return me


def hexr(a, af=7.6, rc=4.12, k=.42):
    seg = math.pi / 3
    a2 = (a + seg / 2) % seg - seg / 2
    rh = (af / 2) / np.cos(a2)
    h = np.clip(.5 + .5 * (rc - rh) / k, 0, 1)
    return rc + (rh - rc) * h - k * h * (1 - h)


def build_barrel():
    # outer skin from the nose lip back to the end, inner bore back to the nose, one closed solid
    n = 420
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    hexa = hexr(ang)
    uo = np.r_[np.linspace(5.32, 9, 26), np.linspace(9, 30, 60)[1:], np.linspace(30, 146, 30)[1:]]

    def cone(u):
        return np.interp(u, [5.3, 6.5, 12, 20, 30], [2.2, 2.48, 3.2, 3.82, 4.05])

    rings = []
    # nose lip, inner to outer, rounded
    lip = round_profile([(1.62, 5.5), (1.62, 5.2), (2.28, 5.2), (2.4, 5.5)], .09, 5)
    for r, z in lip:
        rings.append(np.c_[r * np.cos(ang), r * np.sin(ang), np.full(n, z)])
    for u in uo:
        b = smoothstep(14.0, 31.0, u)
        R = cone(u) * (1 - b) + hexa * b
        # the parting line of the mould, along two opposite corners
        R = R + .012 * np.exp(-(((ang + .0) % math.pi) ** 2) / .0006) * smoothstep(12, 34, u)
        rings.append(np.c_[R * np.cos(ang), R * np.sin(ang), np.full(n, u)])
    # back wall
    rings.append(np.c_[2.3 * np.cos(ang), 2.3 * np.sin(ang), np.full(n, 146.0)])
    # the bore, back toward the nose
    for u in np.r_[np.linspace(146, 30, 8), np.linspace(30, 12, 26)[1:], np.linspace(12, 5.5, 18)[1:]]:
        R = np.interp(u, [5.5, 8, 12, 30, 146], [1.62, 1.72, 2.05, 2.3, 2.3])
        rings.append(np.c_[R * np.cos(ang), R * np.sin(ang), np.full(n, u)])
    return mesh_from_grid('barrel', np.array(rings), True, 26.0)


def build_tube(name, r_out, r_in, z0, z1, n=128):
    prof = [(r_in, z0), (r_out, z0), (r_out, z1), (r_in, z1)]
    prof = round_profile(prof + [prof[0]], .05, 3)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rows = [np.c_[r * np.cos(ang), r * np.sin(ang), np.full(n, z)] for r, z in densify(list(prof), 3.0)]
    return mesh_from_grid(name, np.array(rows), True, 30.0)


def build_ink_column(r, z0, z1, n=96):
    me_rows = [(0.0, z0)]
    for rr, zz in densify([(r, z0), (r, z1 - .6)], 4.0):
        me_rows.append((rr, zz))
    for v in np.linspace(0, 1, 8)[1:]:               # the meniscus
        me_rows.append((r * (1 - v ** 2) ** .5, z1 - .6 + .6 * v))
    me_rows.append((0.0, z1))
    me_rows = [p for i, p in enumerate(me_rows) if i == 0 or p != me_rows[i - 1]]
    return lathe('ink_col', me_rows, n, None, 40.0)


# ---------------------------------------------------------------- node helper
class G:
    def __init__(self, mat):
        self.t = mat.node_tree
        self.t.nodes.clear()

    def __call__(self, kind, name=None, **kw):
        n = self.t.nodes.new(kind)
        if name:
            n.name = name
        for k, v in kw.items():
            if k.startswith('i_'):
                key = k[2:].replace('_', ' ')
                inp = n.inputs[key]
                if isinstance(v, bpy.types.NodeSocket):
                    self.t.links.new(v, inp)
                else:
                    inp.default_value = v
            else:
                setattr(n, k, v)
        return n

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.t.nodes.new('ShaderNodeMath')
        n.operation = op
        n.use_clamp = clamp
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, bpy.types.NodeSocket):
                self.t.links.new(v, n.inputs[i])
            else:
                n.inputs[i].default_value = v
        return n.outputs[0]

    def link(self, a, b):
        self.t.links.new(a, b)

    def color(self, r, g=None, b=None):
        n = self.t.nodes.new('ShaderNodeCombineColor')
        for i, v in enumerate((r, g if g is not None else r, b if b is not None else r)):
            if isinstance(v, bpy.types.NodeSocket):
                self.t.links.new(v, n.inputs[i])
            else:
                n.inputs[i].default_value = v
        return n.outputs[0]

    def smooth(self, a, b, x):
        n = self.t.nodes.new('ShaderNodeMapRange')
        n.interpolation_type = 'SMOOTHSTEP'
        n.inputs['From Min'].default_value = a
        n.inputs['From Max'].default_value = b
        self.t.links.new(x, n.inputs['Value'])
        return n.outputs[0]


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    g = G(m)
    return m, g


def image(path):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = 'Non-Color'
    return im


def srgb(v):
    return v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4


def rgba(r, g, b, a=1.0):
    return (srgb(r), srgb(g), srgb(b), a)


# ---------------------------------------------------------------- materials
def paper_material():
    m, g = new_mat('paper')
    geo = g('ShaderNodeNewGeometry')
    pos = geo.outputs['Position']
    tile = g('ShaderNodeMapping', vector_type='POINT', i_Vector=pos, i_Scale=(1 / TILE, 1 / TILE, 1))
    inkuv = g('ShaderNodeMapping', vector_type='POINT', i_Vector=pos, i_Scale=(1 / GRID['w'], 1 / GRID['h'], 1),
              i_Location=(-GRID['x0'] / GRID['w'], -GRID['y0'] / GRID['h'], 0))
    th = g('ShaderNodeTexImage', image=image(os.path.join(A, 'paper_h.png')), interpolation='Cubic', i_Vector=tile.outputs[0])
    ta = g('ShaderNodeTexImage', image=image(os.path.join(A, 'paper_a.png')), interpolation='Linear', i_Vector=tile.outputs[0])
    it = g('ShaderNodeTexImage', image=image(os.path.join(A, 'ink_t.png')), interpolation='Closest', extension='EXTEND', i_Vector=inkuv.outputs[0])
    id_ = g('ShaderNodeTexImage', image=image(os.path.join(A, 'ink_d.png')), interpolation='Cubic', extension='EXTEND', i_Vector=inkuv.outputs[0])
    ig = g('ShaderNodeTexImage', image=image(os.path.join(A, 'ink_g.png')), interpolation='Cubic', extension='EXTEND', i_Vector=inkuv.outputs[0])

    now = g('ShaderNodeValue', 'now')
    age = g.math('SUBTRACT', now.outputs[0], it.outputs[0])           # 0..1 clock
    rev = g.math('MULTIPLY', age, GRID['dur'] / .014, clamp=True)     # a 14 ms front right under the ball
    # hide everything until the clock passes: rev is 0 outside the line and before the ball arrives

    d = g.math('MULTIPLY', g.math('MULTIPLY', id_.outputs[0], 1.6), rev)
    gum = g.math('MULTIPLY', g.math('SUBTRACT', g.math('MULTIPLY', ig.outputs[0], 72.0), 12.0), rev)
    pum = g.math('MULTIPLY', g.math('SUBTRACT', th.outputs[0], .5), 40.0)

    press = g.math('MULTIPLY', g.math('DIVIDE', g.math('MAXIMUM', gum, 0.0), 36.0, clamp=True), .6)
    fill = g.math('SUBTRACT', 1.0, g.math('ADD', press, g.math('MULTIPLY', g.math('MINIMUM', d, 1.0), .42), clamp=True))
    hum = g.math('ADD', g.math('SUBTRACT', g.math('MULTIPLY', g.math('MULTIPLY', pum, 1.05), fill), gum), g.math('MULTIPLY', d, 2.6))
    hmm = g.math('MULTIPLY', hum, .001)
    bump = g('ShaderNodeBump', i_Height=hmm, i_Distance=1.0, i_Strength=1.0)

    # colour: Beer-Lambert through the ink film over the paper
    al = g.math('MULTIPLY', g.math('POWER', g.math('MULTIPLY', ta.outputs[0], 1.05), 2.3), 1.11)
    def chan(base, sig):
        return g.math('MULTIPLY', g.math('MULTIPLY', al, base), g.math('EXPONENT', g.math('MULTIPLY', d, -sig)))

    base = g.color(chan(.92, 4.4), chan(.905, 3.7), chan(.86, 1.4))

    fresh = g.math('SUBTRACT', 1.0, g.math('DIVIDE', g.math('MAXIMUM', age, 0.0), 1.8 / GRID['dur'], clamp=True))
    kd = g.smooth(.04, .55, d)
    rough = g.math('ADD', g.math('MULTIPLY', .74, g.math('SUBTRACT', 1.0, kd)), g.math('MULTIPLY', kd, g.math('SUBTRACT', .42, g.math('MULTIPLY', fresh, .1))))
    spec = g.math('ADD', .34, g.math('MULTIPLY', kd, .12))
    heavy = g.smooth(.85, 1.35, d)

    p = g('ShaderNodeBsdfPrincipled', **{'i_Base_Color': base, 'i_Roughness': rough, 'i_Specular_IOR_Level': spec,
                                         'i_Normal': bump.outputs[0], 'i_Coat_Weight': g.math('MULTIPLY', heavy, .1),
                                         'i_Coat_Roughness': .22, 'i_Coat_Tint': (1.0, .9, .97, 1), 'i_Coat_Normal': bump.outputs[0]})
    # fibres pass light sideways: a soft scatter that keeps the tooth from looking like plaster
    sss = g.math('MULTIPLY', .55, g.math('SUBTRACT', 1.0, kd))
    g.link(sss, p.inputs['Subsurface Weight'])
    p.inputs['Subsurface Radius'].default_value = (1.0, .85, .7)
    p.inputs['Subsurface Scale'].default_value = .085
    out = g('ShaderNodeOutputMaterial')
    g.link(p.outputs[0], out.inputs[0])
    return m


def brass_material():
    m, g = new_mat('brass')
    tc = g('ShaderNodeTexCoord')
    ob = tc.outputs['Object']
    sep = g('ShaderNodeSeparateXYZ', i_Vector=ob)
    z = sep.outputs['Z']
    # machined lines: frequency runs along the axis only
    rings = g('ShaderNodeTexNoise', i_Vector=g.color(0.0, 0.0, g.math('MULTIPLY', z, 1.0)), i_Scale=95.0, i_Detail=4.0, i_Roughness=.55)
    # fine scratches around the part
    sc = g('ShaderNodeTexNoise', i_Vector=ob, i_Scale=70.0, i_Detail=8.0, i_Roughness=.65)
    h = g.math('ADD', g.math('MULTIPLY', rings.outputs[0], .0036), g.math('MULTIPLY', sc.outputs[0], .0009))
    bump = g('ShaderNodeBump', i_Height=h, i_Distance=1.0, i_Strength=1.0)
    rough = g.math('ADD', .13, g.math('MULTIPLY', g.math('SUBTRACT', sc.outputs[0], .45), .26))

    # the last of the ink: a thin wet film at the lip that thins out in fine capillary fingers
    cr = g('ShaderNodeTexNoise', i_Vector=ob, i_Scale=22.0, i_Detail=6.0, i_Roughness=.55)
    thresh = g.math('ADD', z, g.math('MULTIPLY', g.math('SUBTRACT', cr.outputs[0], .5), .22))
    mask = g.math('SUBTRACT', 1.0, g.smooth(.02, .2, thresh))

    tang = g('ShaderNodeTangent', direction_type='RADIAL', axis='Z')     # turned on a lathe: the highlight fans around the axis
    brass = g('ShaderNodeBsdfPrincipled', i_Base_Color=rgba(.84, .69, .32), i_Metallic=1.0, i_Roughness=rough, i_Normal=bump.outputs[0],
              i_Anisotropic=.72, i_Tangent=tang.outputs[0], **{'i_Specular_IOR_Level': .5})
    ink = g('ShaderNodeBsdfPrincipled', i_Base_Color=(.004, .011, .07, 1), i_Roughness=.16, i_Normal=bump.outputs[0])
    mix = g('ShaderNodeMixShader', i_Fac=mask)
    g.link(brass.outputs[0], mix.inputs[1])
    g.link(ink.outputs[0], mix.inputs[2])
    out = g('ShaderNodeOutputMaterial')
    g.link(mix.outputs[0], out.inputs[0])
    return m


def ball_material():
    m, g = new_mat('ball')
    tc = g('ShaderNodeTexCoord')
    ob = tc.outputs['Object']
    wc = g('ShaderNodeTexNoise', i_Vector=ob, i_Scale=2.6, i_Detail=7.0, i_Roughness=.62)
    wf = g('ShaderNodeTexNoise', i_Vector=ob, i_Scale=26.0, i_Detail=5.0, i_Roughness=.55)
    cover = g.smooth(.3, .39, g.math('ADD', wc.outputs[0], g.math('MULTIPLY', g.math('SUBTRACT', wf.outputs[0], .5), .18)))
    dim = g('ShaderNodeTexNoise', i_Vector=ob, i_Scale=95.0, i_Detail=3.0)
    bump = g('ShaderNodeBump', i_Height=g.math('MULTIPLY', dim.outputs[0], .0017), i_Distance=1.0, i_Strength=1.0)
    wc_b = g('ShaderNodeBsdfPrincipled', i_Base_Color=(.62, .63, .66, 1), i_Metallic=1.0, i_Roughness=.1, i_Normal=bump.outputs[0])
    ink = g('ShaderNodeBsdfPrincipled', i_Base_Color=(.003, .009, .06, 1), i_Roughness=.08, **{'i_Coat_Weight': .6, 'i_Coat_Roughness': .05})
    mix = g('ShaderNodeMixShader', i_Fac=cover)
    g.link(wc_b.outputs[0], mix.inputs[1])
    g.link(ink.outputs[0], mix.inputs[2])
    out = g('ShaderNodeOutputMaterial')
    g.link(mix.outputs[0], out.inputs[0])
    return m


def clear_plastic(name, ior, rough, tint=(.985, .99, .995, 1)):
    m, g = new_mat(name)
    tc = g('ShaderNodeTexCoord')
    sc = g('ShaderNodeTexNoise', i_Vector=tc.outputs['Object'], i_Scale=2.2, i_Detail=10.0, i_Roughness=.68)
    r = g.math('ADD', rough, g.math('MULTIPLY', g.math('MAXIMUM', g.math('SUBTRACT', sc.outputs[0], .55), 0.0), .08))
    b = g('ShaderNodeBump', i_Height=g.math('MULTIPLY', sc.outputs[0], .004), i_Distance=1.0, i_Strength=.8)
    p = g('ShaderNodeBsdfPrincipled', i_Base_Color=tint, i_Roughness=r, i_IOR=ior, i_Normal=b.outputs[0],
          **{'i_Transmission_Weight': 1.0})
    out = g('ShaderNodeOutputMaterial')
    g.link(p.outputs[0], out.inputs[0])
    return m


def ink_column_material():
    m, g = new_mat('ink_col')
    p = g('ShaderNodeBsdfPrincipled', i_Base_Color=(1, 1, 1, 1), i_Roughness=.02, i_IOR=1.5, **{'i_Transmission_Weight': 1.0})
    v = g('ShaderNodeVolumeAbsorption', i_Color=(.02, .05, .78, 1), i_Density=3.6)
    out = g('ShaderNodeOutputMaterial')
    g.link(p.outputs[0], out.inputs[0])
    g.link(v.outputs[0], out.inputs[1])
    return m


def ink_gloss_material():
    m, g = new_mat('ink_gloss')
    p = g('ShaderNodeBsdfPrincipled', i_Base_Color=(.004, .011, .075, 1), i_Roughness=.08, **{'i_Coat_Weight': .7, 'i_Coat_Roughness': .05})
    out = g('ShaderNodeOutputMaterial')
    g.link(p.outputs[0], out.inputs[0])
    return m


def desk_material():
    m, g = new_mat('desk')
    tc = g('ShaderNodeTexCoord')
    n = g('ShaderNodeTexNoise', i_Vector=tc.outputs['Object'], i_Scale=.03, i_Detail=8.0)
    r = g.math('ADD', .3, g.math('MULTIPLY', n.outputs[0], .3))
    p = g('ShaderNodeBsdfPrincipled', i_Base_Color=(.12, .065, .03, 1), i_Roughness=r)
    out = g('ShaderNodeOutputMaterial')
    g.link(p.outputs[0], out.inputs[0])
    return m


def lint_material():
    m, g = new_mat('lint')
    t = g('ShaderNodeBsdfTranslucent', i_Color=(.9, .88, .82, 1))
    d = g('ShaderNodeBsdfDiffuse', i_Color=(.92, .9, .85, 1))
    mx = g('ShaderNodeMixShader', i_Fac=.4)
    g.link(d.outputs[0], mx.inputs[1])
    g.link(t.outputs[0], mx.inputs[2])
    out = g('ShaderNodeOutputMaterial')
    g.link(mx.outputs[0], out.inputs[0])
    return m


# ---------------------------------------------------------------- objects
col = bpy.context.collection
M = dict(paper=paper_material(), brass=brass_material(), ball=ball_material(), barrel=clear_plastic('barrel', 1.59, .012),
         tube=clear_plastic('tube', 1.49, .2, (.97, .975, .98, 1)), ink_col=ink_column_material(), gloss=ink_gloss_material(), desk=desk_material(), lint=lint_material())

# the sheet (A4, the words low on the page), and the desk
x_c, y_c = 20.0, 42.0
W_, H_ = 210.0, 297.0
pv = [(x_c - W_ / 2, y_c - H_ / 2, 0), (x_c + W_ / 2, y_c - H_ / 2, 0), (x_c + W_ / 2, y_c + H_ / 2, 0), (x_c - W_ / 2, y_c + H_ / 2, 0)]
me = bpy.data.meshes.new('paper')
me.from_pydata(pv, [], [(0, 1, 2, 3)])
me.materials.append(M['paper'])
paper = add_obj('paper', me)
me = bpy.data.meshes.new('desk')
me.from_pydata([(-900, -900, -.12), (900, -900, -.12), (900, 900, -.12), (-900, 900, -.12)], [], [(0, 1, 2, 3)])
me.materials.append(M['desk'])
add_obj('desk', me)

root = bpy.data.objects.new('pen', None)
col.objects.link(root)
root.rotation_mode = 'QUATERNION'


def part(name, me, mat):
    me.materials.append(M[mat])
    return add_obj(name, me, root)


part('point', build_point(), 'brass')
part('ink_ring', build_ink_ring(), 'gloss')
part('barrel', build_barrel(), 'barrel')
part('tube', build_tube('tube', 1.45, .86, 5.7, 132.0), 'tube')
part('ink_col', build_ink_column(.848, 5.7, 112.0), 'ink_col')

# the end plug, the blue cap of the stick
end = solid_of_revolution('plug', [(2.28, 141.0), (2.28, 148.0), (3.4, 148.0), (3.4, 150.0), (0.0, 150.5)], 96, 30.0)
end.materials.append(M['gloss'])
add_obj('plug', end, root)

bpy.ops.mesh.primitive_uv_sphere_add(segments=256, ring_count=128, radius=BALL_R)
ball = bpy.context.object
ball.name = 'ball'
ball.data.materials.append(M['ball'])
ball.rotation_mode = 'QUATERNION'
ball.data.polygons.foreach_set('use_smooth', [True] * len(ball.data.polygons))
ball.data.update()

# lint: fibres lifted off the sheet, the thing macro lenses always find
rng = np.random.default_rng(21)
curve = bpy.data.curves.new('lint', 'CURVE')
curve.dimensions = '3D'
curve.bevel_depth = .0055
curve.bevel_resolution = 1
curve.resolution_u = 4
curve.fill_mode = 'FULL'
curve.use_fill_caps = False
curve.materials.append(M['lint'])
for _ in range(900):
    x, y = rng.uniform(-14, 54), rng.uniform(-14, 22)
    L = rng.lognormal(math.log(.26), .5)
    th = rng.uniform(0, 2 * math.pi)
    lift = rng.uniform(.25, 1.0)
    k = rng.normal(0, .8)
    n_pts = 6
    sp = curve.splines.new('NURBS')
    sp.points.add(n_pts - 1)
    for i in range(n_pts):
        u = i / (n_pts - 1)
        a = th + k * u
        sp.points[i].co = (x + L * u * math.cos(a), y + L * u * math.sin(a), max(0.002, .25 * L * lift * math.sin(math.pi * u ** .7) * (1 - .35 * u)), 1)
    sp.use_endpoint_u = True
add_obj('lint', curve)

# camera
cd = bpy.data.cameras.new('cam')
cd.lens = rig.LENS
cd.sensor_width = 36
cd.clip_start = .5
cd.clip_end = 3000
cd.dof.use_dof = True
cd.dof.aperture_blades = 7
cd.dof.aperture_rotation = .2
cam = bpy.data.objects.new('cam', cd)
col.objects.link(cam)
cam.rotation_mode = 'QUATERNION'
scene.camera = cam


# light: a big soft window low on the left, a strip for the long highlights, a little room
def area(name, loc, target, sx, sy, power, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, 'AREA')
    d.shape = 'RECTANGLE'
    d.size, d.size_y = sx, sy
    d.energy = power
    d.color = color
    d.normalize = False
    o = bpy.data.objects.new(name, d)
    col.objects.link(o)
    o.location = loc
    o.rotation_mode = 'QUATERNION'
    o.rotation_quaternion = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y')
    return o


area('window', (-260, -260, 165), (15, 8, 0), 320, 400, 50.0, (1.0, .955, .9))
area('strip', (170, 40, 120), (15, 5, 8), 26, 300, 8.0, (.94, .97, 1.0))

world = bpy.data.worlds.new('room')
scene.world = world
world.use_nodes = True
gw = G(world)
env = gw('ShaderNodeTexEnvironment')
env.image = bpy.data.images.load(os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles', 'studiolights', 'world', 'interior.exr'))
tc = gw('ShaderNodeTexCoord')
map_ = gw('ShaderNodeMapping', vector_type='POINT', i_Vector=tc.outputs['Generated'], i_Rotation=(0, 0, math.radians(120)))
gw.link(map_.outputs[0], env.inputs[0])
bg = gw('ShaderNodeBackground', i_Color=env.outputs[0], i_Strength=.18)
wo = gw('ShaderNodeOutputWorld')
gw.link(bg.outputs[0], wo.inputs[0])

# ---------------------------------------------------------------- keys
bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
tr = rig.tracks()
scene.render.fps = rig.FPS
scene.frame_start = 0
scene.frame_end = rig.NF - 1

now_node = M['paper'].node_tree.nodes['now']
now_input = now_node.outputs[0]
axis0 = Vector((0, 0, 1))
prev = None
prev_c = None
prev_b = None
spin = Quaternion((0, 0, 1), math.radians(9.0))
for f in range(rig.NF + 1):
    t = tr['t'][f]
    d = Vector(tr['axis'][f])
    q = axis0.rotation_difference(d) @ spin
    if prev is not None and q.dot(prev) < 0:
        q = -q
    prev = q.copy()
    root.location = tr['ball'][f]
    root.rotation_quaternion = q
    root.keyframe_insert('location', frame=f)
    root.keyframe_insert('rotation_quaternion', frame=f)

    ball.location = tr['ball'][f]
    qb = Quaternion(tr['qb'][f])
    if prev_b is not None and qb.dot(prev_b) < 0:
        qb = -qb
    prev_b = qb.copy()
    ball.rotation_quaternion = qb
    ball.keyframe_insert('location', frame=f)
    ball.keyframe_insert('rotation_quaternion', frame=f)

    fw = Vector(tr['fwd'][f])
    up = Vector((0, 0, 1))
    qc = fw.to_track_quat('-Z', 'Y')
    qc = qc @ Quaternion((0, 0, 1), float(tr['roll'][f]))
    if prev_c is not None and qc.dot(prev_c) < 0:
        qc = -qc
    prev_c = qc.copy()
    cam.location = tr['cam'][f]
    cam.rotation_quaternion = qc
    cam.keyframe_insert('location', frame=f)
    cam.keyframe_insert('rotation_quaternion', frame=f)

    cd.dof.focus_distance = float(tr['fdist'][f])
    cd.dof.aperture_fstop = rig.LENS * 1e-3 / float(tr['apert'][f])
    cd.keyframe_insert('dof.focus_distance', frame=f)
    cd.keyframe_insert('dof.aperture_fstop', frame=f)

    now_input.default_value = float(t) / GRID['dur']
    now_input.keyframe_insert('default_value', frame=f)

# ---------------------------------------------------------------- render
c = scene.cycles
scene.render.engine = 'CYCLES'
c.device = 'GPU'
c.samples = 256
c.use_adaptive_sampling = True
c.adaptive_threshold = .02
c.adaptive_min_samples = 32
c.use_denoising = True
c.denoiser = 'OPENIMAGEDENOISE'
c.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
c.denoising_prefilter = 'FAST'
c.max_bounces = 40
c.diffuse_bounces = 4
c.glossy_bounces = 26
c.transmission_bounces = 26
c.transparent_max_bounces = 8
c.volume_bounces = 1
c.caustics_refractive = True
c.caustics_reflective = False
c.blur_glossy = .15
c.sample_clamp_indirect = 8.0
c.use_light_tree = True
c.use_animated_seed = False
scene.render.use_motion_blur = True
scene.render.motion_blur_shutter = .25
c.motion_blur_position = 'CENTER'
scene.render.use_persistent_data = True
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.color_depth = '16'
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'AgX - Medium High Contrast'
scene.view_settings.exposure = .42
scene.display_settings.display_device = 'sRGB'

os.makedirs(os.path.join(FILM, 'work'), exist_ok=True)
out = os.path.join(FILM, 'work', 'pen.blend')
bpy.ops.wm.save_as_mainfile(filepath=out, compress=False)
print('BUILT', out, 'frames', rig.NF)
