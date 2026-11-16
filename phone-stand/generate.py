#!/usr/bin/env python3
"""手机支架 —— 参数化模型生成脚本。

用法:
    pip install -r requirements.txt
    python generate.py            # 生成 phone_stand.stl 和 preview.png

模型是一个"侧面轮廓拉伸体": 先在 2D 平面画出支架的侧面轮廓, 再沿宽度方向拉伸。
导出的 STL 已经是打印姿态(侧躺), 无需支撑, 层纹方向与受力方向一致, 强度最好。
"""
import math
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

# ---------------------------------------------------------------- 参数 (mm)
WIDTH = 70.0          # 支架宽度 (拉伸长度)
THICK = 4.0           # 壁厚
ANGLE = 65.0          # 靠背与桌面夹角 (度), 越小越后仰
SLOT = 14.0           # 卡槽净宽 = 可放入的手机最大厚度 (含壳)
LIP = 12.0            # 前挡边高度
CRADLE_Z = 30.0       # 托台离桌面高度, 给充电线留空间
BACK_LEN = 85.0       # 托台以上的靠背长度
STRUT_AT = 60.0       # 后撑杆连接到靠背的位置 (从托台起算)
BASE_LEN = 110.0      # 底板长度
FILLET = 2.0          # 内角圆角半径
CABLE_W = 16.0        # 充电线孔宽度 (沿支架宽度方向), 设为 0 则不开孔
OUT_DIR = Path(__file__).resolve().parent
# ---------------------------------------------------------------------------


def build_profile():
    """返回 (侧面轮廓, 充电线孔轮廓), 坐标系: x=前后, y=高度。"""
    h = THICK / 2
    a = math.radians(ANGLE)
    u = np.array([math.cos(a), math.sin(a)])       # 沿靠背向上
    n = np.array([-math.sin(a), math.cos(a)])      # 垂直靠背, 指向前方

    # 靠背中心线与底板中心线 (y = h) 的交点, 以及托台根部 p0
    foot = np.array([CRADLE_Z / math.tan(a) + h, h])
    p0 = foot + u * (CRADLE_Z / math.sin(a))
    top = p0 + u * BACK_LEN
    ledge_len = SLOT + THICK                       # 中心线长度 -> 净宽 SLOT
    ledge_end = p0 + n * ledge_len
    lip_end = ledge_end + u * LIP
    strut_top = p0 + u * STRUT_AT
    strut_foot = np.array([BASE_LEN - h, h])

    paths = [
        [(h, h), tuple(strut_foot)],                      # 底板
        [tuple(foot), tuple(top)],                        # 靠背 (含前支腿)
        [tuple(p0), tuple(ledge_end), tuple(lip_end)],    # 托台 + 前挡边
        [tuple(strut_top), tuple(strut_foot)],            # 后撑杆
    ]
    shape = unary_union([LineString(p).buffer(h, join_style="round") for p in paths])
    shape = shape.buffer(FILLET).buffer(-FILLET)          # 内角倒圆, 减少应力集中
    shape = shape.intersection(box(-1e3, 0, 1e3, 1e3))    # 底面保证平整

    # 充电线孔: 只切穿托台, 保留前挡边 -> 侧躺打印时是一段短桥接, 不需要支撑
    m = FILLET + 0.5
    c0, c1 = h + 1.0, ledge_len - h - 1.0
    cable = Polygon([tuple(p0 + n * c + u * s)
                     for c, s in ((c0, -h - m), (c1, -h - m), (c1, h + m), (c0, h + m))])
    return shape, cable


def to_cross_section(poly):
    polys = getattr(poly, "geoms", [poly])
    rings = []
    for p in polys:
        p = orient(p, 1.0)   # 外环逆时针, 孔顺时针 (manifold 的 Positive 填充规则要求)
        rings.append(list(p.exterior.coords)[:-1])
        rings += [list(r.coords)[:-1] for r in p.interiors]
    return CrossSection(rings)


def build_solid() -> Manifold:
    profile, cable = build_profile()
    solid = to_cross_section(profile).extrude(WIDTH)
    if CABLE_W > 0:
        cut = to_cross_section(cable).extrude(CABLE_W).translate([0, 0, (WIDTH - CABLE_W) / 2])
        solid = solid - cut
    return solid


def to_trimesh(solid: Manifold) -> trimesh.Trimesh:
    m = solid.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(m.vert_properties)[:, :3],
                           faces=np.asarray(m.tri_verts), process=True)


def render_preview(mesh: trimesh.Trimesh, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # 预览图转成"使用姿态": 轮廓的高度方向朝上
    v = mesh.vertices[:, [0, 2, 1]]
    tris = v[mesh.faces]
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.4, -0.7, 0.6])
    shade = 0.45 + 0.55 * np.clip(np.abs(nrm @ (light / np.linalg.norm(light))), 0, 1)
    colors = np.outer(shade, [0.20, 0.55, 0.85])

    fig = plt.figure(figsize=(12, 6))

    # 左: 侧面轮廓 (使用姿态), 虚线为一台 160×9 mm 的手机示意
    ax = fig.add_subplot(1, 2, 1)
    profile, cable = build_profile()
    ax.fill(*profile.exterior.xy, color=(0.20, 0.55, 0.85))
    for ring in profile.interiors:
        ax.fill(*ring.xy, color="white")
    ax.fill(*cable.exterior.xy, facecolor="none", edgecolor="crimson", hatch="///", lw=1)
    a = math.radians(ANGLE)
    u = np.array([math.cos(a), math.sin(a)])
    n = np.array([-math.sin(a), math.cos(a)])
    p0 = np.array([CRADLE_Z / math.tan(a) + THICK / 2, THICK / 2]) + u * (CRADLE_Z / math.sin(a))
    o = p0 + n * THICK / 2 + u * THICK / 2
    phone = np.array([o, o + n * 9, o + n * 9 + u * 160, o + u * 160, o])
    ax.plot(phone[:, 0], phone[:, 1], "k--", lw=1)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_aspect("equal")
    ax.set_title("side profile (mm) - red hatch: cable hole")

    # 右: 3D 视图
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.add_collection3d(Poly3DCollection(tris, facecolors=colors, edgecolors="none"))
    lo, hi = v.min(0), v.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    solid = build_solid()
    mesh = to_trimesh(solid)
    assert mesh.is_watertight and mesh.is_winding_consistent, "网格不是封闭流形"
    assert len(solid.decompose()) == 1,"模型不是单一实体"

    stl = OUT_DIR / "phone_stand.stl"
    mesh.export(stl)
    render_preview(mesh, OUT_DIR / "preview.png")

    size = mesh.extents
    print(f"已生成 {stl.name}: {len(mesh.faces)} 个三角面")
    print(f"打印姿态尺寸 (X×Y×Z): {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} mm")
    print(f"体积: {mesh.volume / 1000:.1f} cm³  (PLA 实心约 {mesh.volume / 1000 * 1.24:.0f} g)")


if __name__ == "__main__":
    main()
