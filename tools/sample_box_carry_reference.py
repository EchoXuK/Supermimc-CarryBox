"""Sample complete reference stages by table height and render G1 MP4 chapters.

Run with the repository's .venv/bin/python; this replays NPZ poses directly.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import random
import subprocess
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, SCENE_HEIGHT, FPS = 1280, 720, 544, 25
STAGES = {"01_approach_lift": "接近并抬起", "02_carry_move": "持箱移动", "03_lower_place": "放下、松手并起身"}
# Adjacent fine levels are pooled so every stage has >= 10 distinct captures.
BANDS = (
    ("00_floor", "地面 / 低支撑", -float("inf"), .08),
    ("01_low", "低桌 8–27 cm", .08, .27),
    ("02_mid_low", "中低桌 27–42 cm", .27, .42),
    ("03_mid_high", "中高桌 42–56 cm", .42, .56),
    ("04_high", "高桌 >56 cm", .56, float("inf")),
)


def log(event, **data):
    print(json.dumps(dict(event=event, **data), ensure_ascii=False), flush=True)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def npz_joint_names():
    tree = ast.parse((ROOT / "src/blinkmimic/box_carry.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NPZ_JOINT_NAMES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("NPZ_JOINT_NAMES not found")


def select(args):
    rows = list(csv.DictReader((args.dataset / "manifest.csv").open()))
    catalog = []
    for row in rows:
        if row["variant"] != "augmented":
            continue
        with np.load(args.dataset / row["clip_npz"], allow_pickle=False) as motion:
            assert bool(motion["tables_available"])
            tables = [motion[f"table{k}_pos_w"][0].astype(float).tolist() for k in (1, 2)]
            for k in (1, 2):
                assert np.allclose(motion[f"table{k}_pos_w"], tables[k - 1], atol=1e-6)
            frames = len(motion["joint_pos"])
            assert frames == int(row["output_frames"]) and float(motion["fps"].item()) == 50.
            if row["stage"] == "03_lower_place":
                assert bool(motion["recovery_complete"].item()) == (row["recovery_complete"] == "True")
                if row["recovery_complete"] != "True":
                    assert row.get("terminal_init_trimmed") == "True"
        active_table = 1 if row["stage"] == "01_approach_lift" else 2
        z = tables[active_table - 1][2]
        band = next(b for b in BANDS if b[2] < z <= b[3])
        catalog.append(dict(row, table_positions_m=tables, active_table=active_table,
                            table_height_m=z, height_band=band[0], frames=frames))
    rng = random.Random(args.seed)
    groups = []
    for stage, label in STAGES.items():
        for band_id, band_label, low, high in BANDS:
            pool = sorted((r for r in catalog if r["stage"] == stage and r["height_band"] == band_id),
                          key=lambda r: r["clip_npz"])
            by_capture = {}
            for row in pool:
                by_capture.setdefault(row["capture_id"], []).append(row)
            assert len(by_capture) >= 10, (stage, band_id, len(pool), len(by_capture))
            # Capture-uniform selection avoids counting size variants as independent captures.
            chosen = [rng.choice(by_capture[c]) for c in rng.sample(sorted(by_capture), 10)]
            group_id = f"{stage}__{band_id}"
            for i, row in enumerate(chosen, 1):
                row["demo"] = i
                row["npz_sha256"] = hashlib.sha256((args.dataset / row["clip_npz"]).read_bytes()).hexdigest()
            groups.append(dict(id=group_id, stage=stage, label=label, height_band=band_id,
                               height_label=band_label, lower_exclusive_m=None if not np.isfinite(low) else low,
                               upper_inclusive_m=None if not np.isfinite(high) else high,
                               available_clips=len(pool), available_captures=len(by_capture),
                               video=f"{group_id}_10_demos.mp4", demos=chosen))
    manifest = dict(dataset=str(args.dataset), variant="augmented", splits=["train", "val", "test"],
                    selection_seed=args.seed, sampling="Uniformly sample 10 captures without replacement per group, then one clip per capture",
                    grouping="table1 centre z for approach/lift; table2 centre z for carry/place; actual table poses unchanged",
                    source_excluded_reason="source has no recorded table positions",
                    replay="Direct kinematic replay of the entire curated NPZ stage, from first to final frame",
                    fps=FPS, width=WIDTH, height=HEIGHT, hold_seconds=1, groups=groups,
                    reference_video=str(ROOT / "logs/box_carry_2h_20261110/attempt_02/demos_10655_wm_20261110/01_approach_lift_10_demos_wm.mp4"))
    if (args.dataset / "init_cleanup.json").exists():
        manifest["init_cleanup"] = json.loads((args.dataset / "init_cleanup.json").read_text())
    if (args.dataset / "terminal_init_cleanup.json").exists():
        manifest["terminal_init_cleanup"] = json.loads((args.dataset / "terminal_init_cleanup.json").read_text())
    write_json(args.out / "selection.json", manifest)
    inventory = []
    for r in catalog:
        inventory.append({k: r[k] for k in ("stage", "clip_npz", "capture_id", "split", "active_table", "table_height_m", "height_band", "frames")})
    with (args.out / "height_inventory.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(inventory[0]))
        w.writeheader()
        w.writerows(inventory)
    log("selection_complete", videos=len(groups), trajectories=sum(len(g["demos"]) for g in groups),
        availability=[(g["id"], g["available_clips"], g["available_captures"]) for g in groups])


def make_model():
    asset = ROOT / "data/g1_carry/g1_29dof.xml"
    xml = ET.parse(asset).getroot()
    xml.find("compiler").set("meshdir", str(asset.parent / "meshes"))
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth=str(WIDTH), offheight=str(HEIGHT))
    ET.SubElement(visual, "quality", shadowsize="2048")
    assets = xml.find("asset")
    ET.SubElement(assets, "texture", name="floor_grid", type="2d", builtin="checker", rgb1=".92 .93 .94", rgb2=".80 .82 .85", width="512", height="512")
    ET.SubElement(assets, "material", name="grid", texture="floor_grid", texuniform="true", texrepeat="2 2", reflectance=".05")
    ET.SubElement(assets, "texture", name="sky", type="skybox", builtin="gradient", rgb1=".95 .96 .98", rgb2=".75 .8 .88", width="256", height="256")
    world = xml.find("worldbody")
    ET.SubElement(world, "light", pos="0 0 6", dir="0 0 -1", directional="true", diffuse=".5 .5 .5")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 .05", material="grid", contype="0", conaffinity="0")
    box = ET.SubElement(world, "body", name="carry_box")
    ET.SubElement(box, "freejoint", name="carry_box_joint")
    ET.SubElement(box, "geom", name="carry_box_geom", type="box", size=".15 .15 .15", mass="2", rgba=".8 .55 .25 1")
    for k, color in ((1, ".3 .5 .6 1"), (2, ".22 .57 .50 1")):
        table = ET.SubElement(world, "body", name=f"table{k}", pos="0 0 -5")
        ET.SubElement(table, "geom", name=f"table{k}_slab", type="box", size=".2 .2 .01", rgba=color)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    model.vis.headlight.ambient[:] = .25
    model.vis.headlight.diffuse[:] = .45
    return model


def render_group(args, manifest, group):
    model = make_model()
    data = mujoco.MjData(model)
    names = npz_joint_names()
    addresses = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    box_address = int(model.jnt_qposadr[model.joint("carry_box_joint").id])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 120, -24
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title_font, font, small = [ImageFont.truetype(font_path, n) for n in (27, 21, 18)]
    count, chapters, checks = 0, [], []
    raw = args.out / f"{group['id']}.encoding.mp4"
    encoder = None if args.preview else subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-threads", "2", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(raw)
    ], stdin=subprocess.PIPE)
    started = time.monotonic()
    try:
        with mujoco.Renderer(model, height=SCENE_HEIGHT, width=WIDTH) as renderer:
            for row in (group["demos"][:1] if args.preview else group["demos"]):
                path = args.dataset / row["clip_npz"]
                with np.load(path, allow_pickle=False) as d:
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["npz_sha256"]
                    n = len(d["joint_pos"])
                    assert n == row["frames"]
                    for key in ("base_pos_w", "base_quat_w", "joint_pos", "object_pos_w", "object_quat_w"):
                        assert len(d[key]) == n and np.isfinite(d[key]).all(), (path, key)
                    for key in ("base_quat_w", "object_quat_w"):
                        assert np.max(np.abs(np.linalg.norm(d[key], axis=-1) - 1)) < 1e-4
                    half = d["carry_box_selected_size_cm"].astype(float) / 200.
                    model.geom_size[model.geom("carry_box_geom").id] = half
                    tables = np.array(row["table_positions_m"])
                    for k in (1, 2):
                        model.body_pos[model.body(f"table{k}").id] = tables[k - 1]
                    # Fixed framing contains both tables and the full motion, including the feet.
                    points = np.concatenate((d["body_pos_w"].reshape(-1, 3), d["object_pos_w"] - half.max(),
                                             d["object_pos_w"] + half.max(), tables - [.25, .25, .02], tables + [.25, .25, .02]))
                    lo, hi = points.min(0), points.max(0)
                    camera.lookat[:] = (lo + hi) / 2
                    az, el = np.radians([camera.azimuth, camera.elevation])
                    toward_camera = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), -np.sin(el)])
                    right = np.array([-np.sin(az), np.cos(az), 0.])
                    up = np.cross(toward_camera, right)
                    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (min(0., lo[2]), hi[2])]) - camera.lookat
                    tan_y = np.tan(np.radians(model.vis.global_.fovy) / 2)
                    depth = corners @ toward_camera
                    distance_y = depth + np.abs(corners @ up) / (tan_y * .86)
                    distance_x = depth + np.abs(corners @ right) / (tan_y * WIDTH / SCENE_HEIGHT * .86)
                    camera.distance = max(3., float(max(distance_x.max(), distance_y.max())))
                    indices = list(range(0, n, 2))
                    if indices[-1] != n - 1:
                        indices.append(n - 1)
                    assert indices[0] == 0 and indices[-1] == n - 1
                    top = Image.new("RGB", (WIDTH, 112), (20, 25, 32))
                    draw = ImageDraw.Draw(top)
                    draw.text((24, 7), f"{group['label']}  |  {group['height_label']}  |  Demo {row['demo']:02d}/10", font=title_font, fill=(240, 243, 248))
                    draw.text((24, 48), Path(row["clip_npz"]).name, font=small, fill=(190, 204, 220))
                    draw.text((24, 77), f"table1 中心高 {tables[0, 2]*100:.1f} cm  ·  table2 中心高 {tables[1, 2]*100:.1f} cm  ·  箱体 {row['box_size_cm']} cm", font=font, fill=(118, 210, 196))
                    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
                    frame[:112] = np.asarray(top)
                    start = count
                    fk_errors = []
                    render_indices = [0, n // 2, n - 1] if args.preview else indices
                    for t in render_indices:
                        data.qpos[:3] = d["base_pos_w"][t]
                        data.qpos[3:7] = d["base_quat_w"][t]
                        data.qpos[addresses] = d["joint_pos"][t]
                        data.qpos[box_address:box_address + 3] = d["object_pos_w"][t]
                        data.qpos[box_address + 3:box_address + 7] = d["object_quat_w"][t]
                        mujoco.mj_forward(model, data)
                        if t in (0, indices[len(indices)//2], n - 1):
                            fk_errors.append({name: float(np.linalg.norm(data.xpos[model.body(name).id] - d["body_pos_w"][t, idx]))
                                              for name, idx in (("pelvis", 0), ("torso_link", 11), ("left_ankle_roll_link", 25), ("right_ankle_roll_link", 26))})
                        renderer.update_scene(data, camera=camera)
                        for k in (1, 2):
                            geom = renderer.scene.geoms[renderer.scene.ngeom]
                            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                                              tables[k - 1] + [0., 0., .09], np.eye(3).ravel(), np.array([.1, .2, .25, 1.]))
                            geom.label = f"table{k}"
                            renderer.scene.ngeom += 1
                        frame[112:112 + SCENE_HEIGHT] = renderer.render()
                        footer = Image.new("RGB", (WIDTH, 64), (20, 25, 32))
                        fd = ImageDraw.Draw(footer)
                        caption = "已去除放置后 init · 尽量保留起身" if row.get("terminal_init_trimmed")=="True" else "G1 参考轨迹完整回放  ·  1× 实时速度"
                        fd.text((24, 7), caption, font=font, fill=(230, 235, 243))
                        fd.text((850, 7), f"{t/50:5.2f} / {(n-1)/50:.2f} s", font=font, fill=(230, 235, 243))
                        fd.rectangle((24, 47, WIDTH - 24, 52), fill=(60, 73, 87))
                        fd.rectangle((24, 47, 24 + round((WIDTH - 48)*t/max(1, n-1)), 52), fill=(105, 216, 176))
                        frame[-64:] = np.asarray(footer)
                        if encoder:
                            encoder.stdin.write(frame.tobytes())
                            count += 1
                        if args.preview or t == indices[len(indices)//2]:
                            suffix = f"_{t:05d}" if args.preview else ""
                            Image.fromarray(frame).save(args.out / "previews" / f"{group['id']}_demo_{row['demo']:02d}{suffix}.jpg", quality=91)
                    if encoder:
                        for _ in range(FPS):
                            encoder.stdin.write(frame.tobytes())
                            count += 1
                    chapters.append(dict(demo=row["demo"], start_seconds=start/FPS, end_seconds=count/FPS,
                                         motion=Path(row["clip_npz"]).name, clip_npz=row["clip_npz"], capture_id=row["capture_id"]))
                    checks.append(dict(demo=row["demo"], source_frames=n, first_rendered_frame=indices[0],
                                       final_rendered_frame=indices[-1], rendered_motion_frames=len(indices),
                                       source_duration_seconds=(n-1)/50, fk_errors_m=fk_errors))
                    log("demo_complete", group=group["id"], demo=row["demo"], motion_frames=n,
                        wall_seconds=round(time.monotonic()-started, 1))
    finally:
        if encoder:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg encoding failed")
    if args.preview:
        return
    metadata = args.out / "metadata" / f"{group['id']}.ffmetadata"
    lines = [";FFMETADATA1", f"title={group['label']} - {group['height_label']} - 10 reference motions"]
    for c in chapters:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(c['start_seconds']*1000)}",
                  f"END={round(c['end_seconds']*1000)}", f"title=Demo {c['demo']:02d} - {c['motion']}"]
    metadata.write_text("\n".join(lines)+"\n")
    output = args.out / group["video"]
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw), "-i", str(metadata),
                    "-map_metadata", "1", "-map_chapters", "1", "-c", "copy", "-movflags", "+faststart", str(output)], check=True)
    raw.unlink()
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,nb_read_frames,avg_frame_rate,duration:chapter=start_time,end_time", "-of", "json", str(output)]))
    s = probe["streams"][0]
    assert (s["width"], s["height"], s["avg_frame_rate"], int(s["nb_read_frames"]), len(probe["chapters"])) == (WIDTH, HEIGHT, "25/1", count, 10)
    rec = dict(group_id=group["id"], video=group["video"], frames=count, duration_seconds=count/FPS,
               bytes=output.stat().st_size, chapters=chapters, trajectory_checks=checks, probe=probe)
    write_json(args.out / "metadata" / f"{group['id']}.json", rec)
    log("video_complete", group=group["id"], seconds=count/FPS, wall_seconds=round(time.monotonic()-started, 1))


def publish(args, manifest):
    records = [json.loads((args.out / "metadata" / f"{g['id']}.json").read_text()) for g in manifest["groups"]]
    assert len(records) == 15 and sum(len(r["chapters"]) for r in records) == 150
    write_json(args.out / "videos.json", records)
    lines = ["# G1 三阶段搬箱：按桌高抽样的完整参考轨迹", "",
             "15 个视频，共 150 条完整阶段轨迹；每个动作 × 桌高档位 10 条，不同 capture 无放回抽样。",
             "优先使用有完整 table1/table2 位置的 augmented，覆盖 train/val/test；source 因缺少桌面位置未使用。",
             "直接回放 NPZ 首帧至末帧的机器人及箱体姿态，不改变轨迹、桌面位置或箱体尺寸。第三类保留放下、松手、起身与末端稳定站姿。",
             "1280×720、25 fps、1× 实时速度；50 Hz 数据每两帧取一帧，强制保留首尾帧，每条结束定格 1 秒；每个视频 10 个章节。",
             "场景为 G1、箱体、地面和两块 40×40×2 cm 桌板，桌板与项目实际模型一致。", "",
             "部分轨迹的 table1/table2 位置几乎重合，画面按原数据保留。", "",
             "[视频浏览与章节跳转](index.html) · [抽样清单](selection.json) · [全部桌高分布](height_inventory.csv) · [视频与完整性核验](videos.json)", "",
             "按 table 中心的世界 z 分档：地面/低支撑 z≤0.08 m，低桌 (0.08,0.27] m，中低桌 (0.27,0.42] m，中高桌 (0.42,0.56] m，高桌 >0.56 m。桌板顶面比中心高 1 cm。",
             "原有细高度档中，接近抬起只有 7–9 条的档位不足以各抽 10 条，因此合并相邻高度，统一上述分组。",
             "接近抬起按起点 table1 分档；搬运和放置按目标 table2 分档。每条画面同时标出两张桌面的实际中心高度。",
             f"固定抽样种子：{manifest['selection_seed']}。组内 10 条均来自不同 capture；不同阶段/高度档之间允许同一 capture 出现。", "",
             "| 动作 | 桌高档 | 可用轨迹 / capture | 抽样 | 视频时长 | 视频 |", "|---|---|---:|---:|---:|---|"]
    cards = []
    if manifest.get("init_cleanup"):
        count = sum(int(r.get("init_trim_frames", 0)) > 0 for g in manifest["groups"] for r in g["demos"])
        lines[2:2] = [f"本批样例中 {count} 条已去除开局双臂张开的 init 及之前的无用前缀，视频从保留动作的首帧开始。", "",
                       "[全库清理报告](../box_carry_init_cleanup_20261111/README.md) · [裁剪边界对照图](../box_carry_init_cleanup_20261111/boundary_review.jpg)", ""]
    if manifest.get("terminal_init_cleanup"):
        count = sum(r.get("terminal_init_trimmed")=="True" for g in manifest["groups"] for r in g["demos"])
        lines = [s.replace("完整参考轨迹", "清理后参考轨迹").replace("完整阶段轨迹", "阶段轨迹")
                   .replace("第三类保留放下、松手、起身与末端稳定站姿。", "第三类保留放下、松手及尽可能多的起身；再次 init 截尾的轨迹以实际保留边界为准。") for s in lines]
        lines[2:2] = [f"本批 {count} 条放置样例已去除再次张臂的 init 尾段。保留所有样例；未达到连续 0.5 秒稳定站姿的按实际标记，不拼接补齐。", "",
                       "[本次全库复查与清理](../box_carry_terminal_init_cleanup_20261111/README.md) · [全部边界对照](../box_carry_terminal_init_cleanup_20261111/review.html)", ""]
    for group, rec in zip(manifest["groups"], records):
        lines.append(f"| {group['label']} | {group['height_label']} | {group['available_clips']} / {group['available_captures']} | 10 | {rec['duration_seconds']:.2f}s | [{group['video']}]({group['video']}) |")
        buttons = ''.join(f'<button data-time="{c["start_seconds"]}" title="{html.escape(c["motion"])}">{c["demo"]:02d} · {c["start_seconds"]:.2f}s</button>' for c in rec["chapters"])
        cards.append(f'<section data-stage="{group["stage"]}" data-band="{group["height_band"]}"><h2>{group["label"]} · {group["height_label"]}</h2>'
                     f'<video controls playsinline preload="none" poster="previews/{group["id"]}_demo_01.jpg" src="{group["video"]}"></video>'
                     f'<div class="chapters">{buttons}</div><p>10 条阶段轨迹 · {rec["duration_seconds"]:.2f} 秒 · <a href="{group["video"]}" download>下载 MP4</a></p></section>')
    for g, r in zip(manifest["groups"], records):
        lines += ["", f"## {g['label']} · {g['height_label']}", "", "| Demo | 视频起点 | 时长 | table1 / table2 高 (cm) | Split | NPZ |", "|---:|---:|---:|---:|---|---|"]
        for row, chapter in zip(g["demos"], r["chapters"]):
            t = row["table_positions_m"]
            lines.append(f"| {row['demo']} | {chapter['start_seconds']:.2f}s | {(row['frames']-1)/50:.2f}s | {t[0][2]*100:.1f} / {t[1][2]*100:.1f} | {row['split']} | [{Path(row['clip_npz']).name}]({os.path.relpath(args.dataset/row['clip_npz'], args.out)}) |")
    (args.out / "README.md").write_text("\n".join(lines)+"\n")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 三阶段搬箱 · 桌高分层抽样</title>
<style>body{margin:28px auto;padding:0 20px;max-width:1280px;background:#141920;color:#edf0f6;font:16px/1.6 system-ui}h1{font-size:28px}h2{font-size:22px}section{margin:30px 0}video{width:100%;display:block;background:#000}.chapters,nav{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}button,select{padding:8px 12px;background:#263348;border:1px solid #506680;color:#edf0f6;border-radius:6px;cursor:pointer}button:hover{background:#365278}a{color:#78b9ff}p{color:#becbdb}section[hidden]{display:none}</style>
<h1>G1 三阶段搬箱 · 每个动作 × 桌高档 10 条</h1><p>15 个视频 · 150 条完整参考轨迹 · 720p · 25 fps · 实时速度。场景包含 G1、箱体、地面、table1 和 table2。</p>
<p>抬起按 table1 分档，搬运与放置按目标 table2 分档。桌高连续变化，合并相邻细档后抽样；每条保留实际桌高。数据来自 augmented，每组 10 个不同 capture。</p><p><a href="README.md">完整说明和轨迹清单</a> · <a href="selection.json">抽样记录</a> · <a href="videos.json">视频核验</a></p><nav><label>动作 <select id="stage"><option value="">全部</option value="01_approach_lift">接近并抬起</option><option value="02_carry_move">持箱移动</option><option value="03_lower_place">放下、松手并起身</option></select></label><label>桌高 <select id="band"><option value="">全部</option>'''
    page += ''.join(f'<option value="{b[0]}">{b[1]}</option>' for b in BANDS)
    page += '</select></label></nav>'+''.join(cards)+'''<script>const stageFilter=document.getElementById('stage'),bandFilter=document.getElementById('band');document.querySelectorAll('button[data-time]').forEach(b=>b.addEventListener('click',()=>{const v=b.closest('section').querySelector('video');document.querySelectorAll('video').forEach(x=>{if(x!==v)x.pause()});v.currentTime=Number(b.dataset.time);v.play()}));document.querySelectorAll('select').forEach(s=>s.addEventListener('change',()=>{document.querySelectorAll('section').forEach(x=>{x.hidden=(stageFilter.value&&stageFilter.value!==x.dataset.stage)||(bandFilter.value&&bandFilter.value!==x.dataset.band);if(x.hidden)x.querySelector('video').pause()})}));</script></html>'''
    if manifest.get("init_cleanup"):
        page = page.replace('<nav>', '<p>已去除开局 init 及之前的无用前缀。<a href="../box_carry_init_cleanup_20261111/README.md">查看全库清理报告</a></p><nav>', 1)
    if manifest.get("terminal_init_cleanup"):
        page = page.replace("150 条完整参考轨迹", "150 条清理后参考轨迹")
        page = page.replace('<nav>', '<p>已去除放置后再次张臂的 init 尾段，保留所有轨迹及尽可能多的起身。部分轨迹在稳定站姿前结束。<a href="../box_carry_terminal_init_cleanup_20261111/README.md">清理报告</a> · <a href="../box_carry_terminal_init_cleanup_20261111/review.html">全部裁剪边界</a></p><nav>', 1)
    (args.out / "index.html").write_text(page)
    thumbs = []
    for g in manifest["groups"]:
        img = Image.open(args.out / "previews" / f"{g['id']}_demo_01.jpg")
        img.thumbnail((384, 216))
        thumbs.append(img.copy())
    overview = Image.new("RGB", (5*384, 3*216), (20, 25, 32))
    for i, thumb in enumerate(thumbs):
        overview.paste(thumb, ((i % 5)*384, (i // 5)*216))
    overview.save(args.out / "overview.jpg", quality=92)
    log("published", videos=len(records), trajectories=150, seconds=sum(r["duration_seconds"] for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "box_carry_three_stage_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "logs/box_carry_three_stage_v2_samples_20261111")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--mode", choices=("all", "select", "render", "publish"), default="all")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--group", default=None)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    args.dataset, args.out = args.dataset.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("previews", "metadata"):
        (args.out / name).mkdir(exist_ok=True)
    if args.mode in ("all", "select"):
        select(args)
    manifest = json.loads((args.out / "selection.json").read_text())
    if args.mode in ("all", "render"):
        for i, group in enumerate(manifest["groups"]):
            if i % args.workers == args.worker and (args.group is None or args.group == group["id"]):
                render_group(args, manifest, group)
    if args.mode in ("all", "publish"):
        publish(args, manifest)


if __name__ == "__main__":
    main()
