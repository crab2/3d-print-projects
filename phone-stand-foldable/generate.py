#!/usr/bin/env python3
"""折叠手机支架 —— 参数化模型生成脚本。

用法:
    pip install -r requirements.txt
    python generate.py

三个零件 (底板 / 靠背 / 撑杆) 都平放打印, 不需要支撑。
两处铰链用一段 1.75 mm 打印耗材当销轴。
折叠后三件叠成一块圆角薄板; 撑杆卡进底板上不同的槽里可以调角度。
"""
import math
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold
from shapely.geometry import MultiPoint, Point, box
from shapely.geometry.polygon import orient

# ---------------------------------------------------------------- 参数 (mm)
W = 70.0              # 宽度
L = 122.0             # 底板长度 (= 折叠后长度)
CORNER = 8.0          # 外轮廓圆角
BASE_T = 4.0          # 底板厚度
PLATE_T = 5.0         # 靠背 / 撑杆厚度
LIP_T = 3.0           # 前挡边厚度
LIP_H = 6.0           # 前挡边高度 (从底板上表面算)
HINGE_X = 25.0        # 靠背铰链轴到底板前缘的距离
KNUCKLE_R = 3.5       # 铰链筒半径
POST_W = 12.0         # 底板上每个铰链座的宽度 (共 3 个)
STRUT_AT = 38.0       # 撑杆铰链到靠背铰链的距离
STRUT_LEN = 50.0      # 撑杆长度 (铰链中心到撑脚中心)
STRUT_W = 30.0        # 撑杆宽度
ANGLES = (70, 60, 50) # 三档角度 (靠背与桌面夹角, 度)
PIN_GRIP = 2.1        # 夹住销轴的孔径 (1.75 mm 耗材)
PIN_FREE = 2.4        # 可自由转动的孔径
GAP = 0.4             # 活动件之间的间隙
STACK_GAP = 0.2       # 折叠时靠背与底板之间的间隙
OUT_DIR = Path(__file__).resolve().parent
# ---------------------------------------------------------------------------

BACK_LEN = L - HINGE_X                       # 靠背长度 (从铰链轴算起)
H1_Z = BASE_T + STACK_GAP + PLATE_T - KNUCKLE_R   # 靠背铰链轴高度 (底板坐标)
H2_OFF = KNUCKLE_R - PLATE_T / 2             # 撑杆铰链轴相对靠背铰链轴的法向偏移
FOOT_R = PLATE_T / 2
GROOVE_R = FOOT_R + 0.3
GROOVE_Z = BASE_T + 0.6
FOOT_Z = GROOVE_Z - (GROOVE_R - FOOT_R) + 0.02
PLATE_X0 = KNUCKLE_R + 0.7                   # 靠背板体起始位置 (离铰链轴)
SEG = 48                                     # 圆的分段数


# ------------------------------------------------------------ 几何小工具
def to_cs(poly):
    polys = getattr(poly, "geoms", [poly])
    rings = []
    for p in polys:
        p = orient(p, 1.0)
        rings.append(list(p.exterior.coords)[:-1])
        rings += [list(r.coords)[:-1] for r in p.interiors]
    return CrossSection(rings)


def prism_z(poly, z0, z1):
    """俯视轮廓 (x, y) 沿 Z 拉伸。"""
    return to_cs(poly).extrude(z1 - z0).translate([0, 0, z0])


def prism_y(poly, y0, y1):
    """侧面轮廓 (x, z) 沿 Y 拉伸。"""
    m = np.array([[1, 0, 0, 0], [0, 0, -1, y1], [0, 1, 0, 0]], dtype=float)
    return to_cs(poly).extrude(y1 - y0).transform(m)


def ycyl(x, z, r, y0, y1):
    return prism_y(Point(x, z).buffer(r, quad_segs=SEG // 4), y0, y1)


def rounded(poly, r):
    return poly.buffer(-r).buffer(r, quad_segs=SEG // 4)


def rot_y(dx, dz, origin):
    """绕平行于 Y 的轴旋转, 把 +X 方向转到 (dx, dz); 轴过 origin=(x, z)。"""
    ox, oz = origin
    return np.array([[dx, 0, -dz, ox], [0, 1, 0, 0], [dz, 0, dx, oz]], dtype=float)


def chain(*mats):
    """依次应用多个 3x4 仿射变换。"""
    out = np.eye(4)
    for m in mats:
        out = np.vstack([m, [0, 0, 0, 1]]) @ out
    return out[:3]


def knuckle_ranges():
    gap = (W - 3 * POST_W) / 2
    posts = [(0, POST_W), (POST_W + gap, 2 * POST_W + gap), (W - POST_W, W)]
    knuckles = [(POST_W + GAP, POST_W + gap - GAP), (2 * POST_W + gap + GAP, W - POST_W - GAP)]
    pockets = [(POST_W, POST_W + gap), (2 * POST_W + gap, W - POST_W)]
    return posts, knuckles, pockets


# ------------------------------------------------------------ 运动学
def strut_pivot(theta_deg):
    """靠背抬起 theta 时, 撑杆铰链轴的位置 (底板坐标 x, z)。"""
    t = math.radians(theta_deg)
    return (HINGE_X + STRUT_AT * math.cos(t) - H2_OFF * math.sin(t),
            H1_Z + STRUT_AT * math.sin(t) + H2_OFF * math.cos(t))


def foot_x(theta_deg):
    """给定角度, 撑脚应落在底板上的 x 位置。"""
    px, pz = strut_pivot(theta_deg)
    return px + math.sqrt(STRUT_LEN ** 2 - (pz - FOOT_Z) ** 2)


# ------------------------------------------------------------ 三个零件 (打印姿态)
def build_base():
    outline = rounded(box(0, 0, L, W), CORNER)
    posts, _, pockets = knuckle_ranges()

    plate = prism_z(outline, 0, BASE_T)

    # 前挡边: 顶部倒圆, 根部内侧加圆角
    top = BASE_T + LIP_H
    lip2d = box(0, -5, LIP_T, top).union(box(0, -5, 14, BASE_T))
    lip2d = lip2d.buffer(-1.2).buffer(1.2, quad_segs=8).buffer(1.5, quad_segs=8).buffer(-1.5)
    lip2d = lip2d.intersection(box(0, 0, 14, top))
    lip = prism_y(lip2d, 0, W) ^ prism_z(outline, 0, top)

    # 铰链座
    post2d = MultiPoint(
        list(Point(HINGE_X, H1_Z).buffer(KNUCKLE_R, quad_segs=SEG // 4).exterior.coords)
        + [(HINGE_X - KNUCKLE_R, 0), (HINGE_X + KNUCKLE_R, 0)]).convex_hull
    body = plate + lip
    for y0, y1 in posts:
        body += prism_y(post2d, y0, y1)

    # 给靠背铰链筒让位的凹槽
    pocket2d = box(HINGE_X - KNUCKLE_R - 1.0, 1.5, HINGE_X + PLATE_X0 + 3.5, 30)
    for y0, y1 in pockets:
        body -= prism_y(pocket2d, y0, y1)

    # 销孔
    body -= ycyl(HINGE_X, H1_Z, PIN_GRIP / 2, -1, W + 1)

    # 撑脚卡槽
    y0 = (W - STRUT_W) / 2 - 1.0
    # 槽口朝撑杆来的方向斜着敞开 (锯齿形): 撑杆杆身不会磕到槽沿, 后壁仍然挡住撑脚
    for ang in ANGLES:
        fx = foot_x(ang)
        px, pz = strut_pivot(ang)
        dx, dz = (px - fx) / STRUT_LEN, (pz - FOOT_Z) / STRUT_LEN
        g = Point(fx, GROOVE_Z).buffer(GROOVE_R, quad_segs=SEG // 4).union(
            Point(fx + dx * 6, GROOVE_Z + dz * 6).buffer(GROOVE_R, quad_segs=SEG // 4)).convex_hull
        body -= prism_y(g, y0, W - y0)
    return body


def build_backrest():
    """打印坐标: x 从铰链轴算起, 正面 (贴手机的一面) 朝下贴热床。"""
    _, knuckles, _ = knuckle_ranges()
    plan = rounded(box(BACK_LEN - 40, 0, BACK_LEN, W), CORNER).union(
        box(PLATE_X0, 0, BACK_LEN - 20, W))
    body = prism_z(plan, 0, PLATE_T)

    # 铰链筒: 贴床一侧带 45° 倒角, 避免悬垂
    r = KNUCKLE_R
    k2d = MultiPoint(
        list(Point(0, r).buffer(r, quad_segs=SEG // 4).exterior.coords)
        + [(-0.414 * r, 0), (PLATE_X0, 0), (PLATE_X0 + 2.5, 0),
           (PLATE_X0, PLATE_T), (PLATE_X0 + 2.5, PLATE_T)]).convex_hull
    for y0, y1 in knuckles:
        body += prism_y(k2d, y0, y1)
    body -= ycyl(0, r, PIN_FREE / 2, -1, W + 1)

    # 收纳撑杆的窗口
    wy = (W - STRUT_W) / 2 - GAP
    win = box(STRUT_AT - FOOT_R - 3.0, wy,   # 铰链下方多留 3 mm, 撑杆向后摆出时不碰窗沿
               STRUT_AT + STRUT_LEN + FOOT_R + 0.5, W - wy)
    body -= prism_z(win.buffer(-1).buffer(1, quad_segs=4), -1, PLATE_T + 1)

    # 撑杆销孔
    body -= ycyl(STRUT_AT, PLATE_T / 2, PIN_GRIP / 2, -1, W + 1)
    return body


def build_strut():
    """打印坐标: 铰链中心在 x=0, 撑脚中心在 x=STRUT_LEN。"""
    h = PLATE_T / 2
    prof = Point(0, h).buffer(h, quad_segs=SEG // 4).union(
        Point(STRUT_LEN, h).buffer(h, quad_segs=SEG // 4)).convex_hull
    body = prism_y(prof, 0, STRUT_W)
    body -= ycyl(0, h, PIN_FREE / 2, -1, STRUT_W + 1)
    return body


# ------------------------------------------------------------ 装配 (用于干涉检查和预览)
def backrest_tf(theta_deg):
    t = math.radians(theta_deg)
    fold = np.array([[1, 0, 0, HINGE_X], [0, -1, 0, W],
                     [0, 0, -1, BASE_T + STACK_GAP + PLATE_T]], dtype=float)
    to_axis = np.array([[1, 0, 0, -HINGE_X], [0, 1, 0, 0], [0, 0, 1, -H1_Z]], dtype=float)
    return chain(fold, to_axis, rot_y(math.cos(t), math.sin(t), (HINGE_X, H1_Z)))


def strut_tf(theta_deg):
    px, pz = strut_pivot(theta_deg)
    if theta_deg == 0:
        d = (1.0, 0.0)
    else:
        d = ((foot_x(theta_deg) - px) / STRUT_LEN, (FOOT_Z - pz) / STRUT_LEN)
    center = np.array([[1, 0, 0, 0], [0, 1, 0, (W - STRUT_W) / 2],
                       [0, 0, 1, -PLATE_T / 2]], dtype=float)
    return chain(center, rot_y(d[0], d[1], (px, pz)))


def assemble(parts, theta_deg):
    base, back, strut = parts
    return [base, back.transform(backrest_tf(theta_deg)), strut.transform(strut_tf(theta_deg))]


def phone_ghost(theta_deg, thick=9.0, width=74.0, length=150.0):
    """靠在支架上的手机示意体。"""
    t = math.radians(theta_deg)
    u = np.array([math.cos(t), math.sin(t)])
    n = np.array([-math.sin(t), math.cos(t)])
    p = np.array([HINGE_X, H1_Z]) + n * KNUCKLE_R
    p = p - u * ((p[1] - BASE_T) / u[1])            # 沿靠背正面滑到底板上表面
    # 手机底面垂直于背面, 后下角触底
    slab = Manifold.cube([length, width, thick]).translate([0, (W - width) / 2, 0])
    m = np.array([[u[0], 0, n[0], p[0]], [0, 1, 0, 0], [u[1], 0, n[1], p[1]]], dtype=float)
    # 前下角会低于底板, 沿靠背上移到刚好触底
    lift = thick * u[0] / u[1]
    return slab.translate([lift, 0, 0]).transform(m)


# ------------------------------------------------------------ 导出与预览
def to_trimesh(solid):
    m = solid.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(m.vert_properties)[:, :3],
                           faces=np.asarray(m.tri_verts), process=True)


def render(solids, colors, azim, elev, size=(900, 700), ss=2):
    """简单的正交 Z-buffer 渲染 (平面着色 + 轮廓线), 返回 RGB 数组。"""
    wpx, hpx = size[0] * ss, size[1] * ss
    az, el = math.radians(azim), math.radians(elev)
    c = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    r = np.array([-math.sin(az), math.cos(az), 0.0])
    up = np.cross(c, r)
    light = np.array([0.35, -0.5, 0.8]); light /= np.linalg.norm(light)

    tris, cols, ids = [], [], []
    for i, (s, col) in enumerate(zip(solids, colors)):
        m = s.to_mesh()
        t = np.asarray(m.vert_properties)[:, :3][np.asarray(m.tri_verts)]
        tris.append(t); cols += [col] * len(t); ids += [i + 1] * len(t)
    tris = np.concatenate(tris); cols = np.array(cols); ids = np.array(ids)

    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    keep = nrm @ c > 1e-6
    tris, cols, ids, nrm = tris[keep], cols[keep], ids[keep], nrm[keep]

    sx, sy, sz = tris @ r, tris @ up, tris @ c
    span = max(sx.max() - sx.min(), (sy.max() - sy.min()) * wpx / hpx)
    scale = 0.88 * wpx / span
    px = (sx - (sx.max() + sx.min()) / 2) * scale + wpx / 2
    py = hpx / 2 - (sy - (sy.max() + sy.min()) / 2) * scale

    zbuf = np.full((hpx, wpx), -np.inf)
    img = np.ones((hpx, wpx, 3))
    idb = np.zeros((hpx, wpx), dtype=int)
    nb = np.zeros((hpx, wpx, 3))
    shade = 0.38 + 0.62 * np.clip(nrm @ light, 0, 1)
    for k in range(len(tris)):
        x, y, z = px[k], py[k], sz[k]
        x0, x1 = max(int(x.min()), 0), min(int(x.max()) + 1, wpx - 1)
        y0, y1 = max(int(y.min()), 0), min(int(y.max()) + 1, hpx - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(den) < 1e-12:
            continue
        w0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den
        w1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den
        w2 = 1 - w0 - w1
        depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        hit = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & (depth > sub)
        sub[hit] = depth[hit]
        img[y0:y1 + 1, x0:x1 + 1][hit] = cols[k] * shade[k]
        idb[y0:y1 + 1, x0:x1 + 1][hit] = ids[k]
        nb[y0:y1 + 1, x0:x1 + 1][hit] = nrm[k]

    # 轮廓线: 零件边界 / 深度跳变 / 法线折角
    edge = np.zeros((hpx, wpx), dtype=bool)
    zf = np.where(np.isfinite(zbuf), zbuf, -1e6)
    for ax in (0, 1):
        a = [slice(None)] * 2; b = [slice(None)] * 2
        a[ax] = slice(1, None); b[ax] = slice(None, -1)
        a, b = tuple(a), tuple(b)
        e = (idb[a] != idb[b]) | (np.abs(zf[a] - zf[b]) > 1.2) | \
            ((np.sum(nb[a] * nb[b], axis=-1) < 0.75) & (idb[a] > 0))
        edge[a] |= e
    if ss > 1:
        edge[:-1] |= edge[1:]; edge[:, :-1] |= edge[:, 1:]
    img[edge] *= 0.35
    return img.reshape(size[1], ss, size[0], ss, 3).mean(axis=(1, 3))


def render_preview(parts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_base, c_back, c_strut = (0.22, 0.25, 0.30), (0.95, 0.55, 0.20), (0.98, 0.78, 0.35)
    c_phone = (0.80, 0.83, 0.87)
    mid = ANGLES[len(ANGLES) // 2]
    views = [
        (assemble(parts, mid) + [phone_ghost(mid)], [c_base, c_back, c_strut, c_phone],
         -125, 22, f"in use ({mid} deg)"),
        (assemble(parts, mid), [c_base, c_back, c_strut], -40, 25, "rear view"),
        (assemble(parts, 0), [c_base, c_back, c_strut], -125, 30, "folded"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (solids, cols, az, el, title) in zip(axes, views):
        ax.imshow(render(solids, [np.array(c) for c in cols], az, el, size=(700, 700)))
        ax.set_title(title, fontsize=13)
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=100)


def main():
    parts = [build_base(), build_backrest(), build_strut()]
    names = ["base", "backrest", "strut"]

    # 1) 每个零件都必须是封闭的单一实体
    for name, solid in zip(names, parts):
        mesh = to_trimesh(solid)
        assert mesh.is_watertight and mesh.is_winding_consistent, f"{name}: 网格不封闭"
        assert len(solid.decompose()) == 1, f"{name}: 不是单一实体"
        mesh.export(OUT_DIR / f"{name}.stl")
        e = mesh.extents
        print(f"{name:9s} {e[0]:6.1f} × {e[1]:5.1f} × {e[2]:4.1f} mm   {mesh.volume / 1000:5.1f} cm³")

    # 2) 折叠和各档展开姿态下, 零件之间不能互相穿插
    for theta in (0,) + tuple(ANGLES):
        asm = assemble(parts, theta)
        for i in range(3):
            for j in range(i + 1, 3):
                v = (asm[i] ^ asm[j]).volume()
                assert v < 0.05, f"{theta}°: {names[i]} 与 {names[j]} 干涉 {v:.2f} mm³"
    print("干涉检查通过: 折叠 +", "、".join(f"{a}°" for a in ANGLES))

    # 3) 三件排在一起的整盘文件
    plate = Manifold.compose([
        parts[0],
        parts[1].translate([-PLATE_X0 + 0.0, W + 8, 0]),
        parts[2].translate([BACK_LEN + 8, W + 8 + (W - STRUT_W) / 2, 0]),
    ])
    pm = to_trimesh(plate)
    pm.export(OUT_DIR / "plate_all.stl")
    e = pm.extents
    print(f"整盘      {e[0]:6.1f} × {e[1]:5.1f} × {e[2]:4.1f} mm")

    folded = assemble(parts, 0)
    lo = np.min([np.asarray(s.to_mesh().vert_properties)[:, :3].min(0) for s in folded], axis=0)
    hi = np.max([np.asarray(s.to_mesh().vert_properties)[:, :3].max(0) for s in folded], axis=0)
    print("折叠尺寸  " + " × ".join(f"{v:.1f}" for v in hi - lo) + " mm")
    print("卡槽位置 (离底板前缘): " + ", ".join(f"{a}°→{foot_x(a):.1f}" for a in ANGLES))

    render_preview(parts, OUT_DIR / "preview.png")


if __name__ == "__main__":
    main()
