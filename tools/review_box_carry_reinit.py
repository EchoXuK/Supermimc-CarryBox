"""Render every reinit candidate at release, proposed endpoint, and original tail."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sample_box_carry_reference import make_model, npz_joint_names, ROOT


def comparison(report):
    applied = json.loads((report/"applied.json").read_text())
    root = Path(json.loads((report/"plan.json").read_text())["dataset"])
    record = next(r for r in applied["records"] if r["variant"]=="augmented" and "257_5806409_002_03_000" in r["clip_npz"])
    path = record["clip_npz"]
    fields = ("joint_pos","base_pos_w","base_quat_w","object_pos_w","object_quat_w","table1_pos_w","table2_pos_w")
    with np.load(report/"backup"/path,allow_pickle=False) as d:
        before = {k:d[k] for k in fields}
        half = d["carry_box_selected_size_cm"]/200.
    with np.load(root/path,allow_pickle=False) as d:
        after = {k:d[k] for k in fields}
    model = make_model()
    data = mujoco.MjData(model)
    addresses = [int(model.jnt_qposadr[model.joint(n).id]) for n in npz_joint_names()]
    box_adr = int(model.jnt_qposadr[model.joint("carry_box_joint").id])
    model.geom_size[model.geom("carry_box_geom").id] = half
    for k in (1,2):
        model.body_pos[model.body(f"table{k}").id] = before[f"table{k}_pos_w"][0]
    cam = mujoco.MjvCamera()
    cam.azimuth,cam.elevation,cam.distance = 120.,-18.,3.1
    cam.lookat[:] = (before["base_pos_w"][-1]+before["object_pos_w"][-1])/2+[0,0,-.14]
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",23)
    small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",18)
    output = report/"target_before_after.mp4"
    encoder = subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1280x720","-r","25","-i","-","-an","-c:v","libx264","-preset","veryfast","-crf","21","-pix_fmt","yuv420p","-movflags","+faststart",str(output)],stdin=subprocess.PIPE)
    n,kept = len(before["joint_pos"]),len(after["joint_pos"])
    indices = list(range(0,n,2))
    if indices[-1]!=n-1:
        indices.append(n-1)
    with mujoco.Renderer(model,height=544,width=640) as renderer:
        for t in indices:
            frame = Image.new("RGB",(1280,720),(20,25,32))
            draw = ImageDraw.Draw(frame)
            draw.text((20,5),"用户指出的放置后再次张臂：原轨迹 / 清理后",font=font,fill="white")
            draw.text((20,39),Path(path).name,font=small,fill=(190,204,220))
            draw.text((20,76),"原轨迹：放下双手后又进入 init",font=font,fill=(255,175,120))
            draw.text((660,76),"清理后：保留起身，到张臂前结束",font=font,fill=(115,235,175))
            for j,d in enumerate((before,after)):
                k = min(t,len(d["joint_pos"])-1)
                data.qpos[:3] = d["base_pos_w"][k]
                data.qpos[3:7] = d["base_quat_w"][k]
                data.qpos[addresses] = d["joint_pos"][k]
                data.qpos[box_adr:box_adr+3] = d["object_pos_w"][k]
                data.qpos[box_adr+3:box_adr+7] = d["object_quat_w"][k]
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=cam)
                frame.paste(Image.fromarray(renderer.render()),(j*640,112))
            draw.text((20,666),f"原片时间 {t/50:.2f} / {(n-1)/50:.2f} 秒",font=font,fill="white")
            caption = f"保留至 {(kept-1)/50:.2f} 秒；之后定格" if t>=kept else f"保留段时间 {t/50:.2f} 秒"
            draw.text((660,666),caption,font=font,fill=(115,235,175))
            encoder.stdin.write(np.asarray(frame).tobytes())
            if t==indices[-1]:
                frame.save(report/"target_before_after.jpg",quality=92)
        for _ in range(25):
            encoder.stdin.write(np.asarray(frame).tobytes())
    encoder.stdin.close()
    assert encoder.wait()==0
    info = dict(clip_npz=path,original_frames=n,retained_frames=kept,removed_frames=n-kept,
                retained_last_time_s=(kept-1)/50.,duration_s=(len(indices)+25)/25.,
                video=output.name,original_sha256=record["original_sha256"],cleaned_sha256=record["cleaned_sha256"])
    (report/"target_before_after.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(info,ensure_ascii=False),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=ROOT/"logs/box_carry_terminal_init_cleanup_20261111")
    p.add_argument("--comparison",action="store_true")
    args = p.parse_args()
    report = args.report.resolve()
    if args.comparison:
        return comparison(report)
    audit = json.loads((report/("plan.json" if (report/"plan.json").exists() else "audit.json")).read_text())
    root = Path(audit["dataset"])
    records = [r for r in audit["records"] if r["keep_end"]<r["original_frames"] or r.get("unexplained_suffix_spreads")]
    records += [r for r in audit["records"] if r["events"] and r["status"]=="unchanged" and not r.get("unexplained_suffix_spreads")]
    heights = {r["clip_npz"]: r for r in csv.DictReader((root/"placement_height_analysis.csv").open())}
    gallery = report/"review"
    gallery.mkdir(exist_ok=True)
    model = make_model()
    data = mujoco.MjData(model)
    addresses = [int(model.jnt_qposadr[model.joint(n).id]) for n in npz_joint_names()]
    box_adr = int(model.jnt_qposadr[model.joint("carry_box_joint").id])
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 120, -18, 3.
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 14)
    pages, cards = [], []
    with mujoco.Renderer(model, height=180, width=320) as renderer:
        for i, row in enumerate(records):
            original = report/"backup"/row["clip_npz"]
            path = original if original.exists() else root/row["clip_npz"]
            with np.load(path, allow_pickle=False) as d:
                a = {k: d[k] for k in ("joint_pos", "base_pos_w", "base_quat_w", "object_pos_w", "object_quat_w", "table1_pos_w", "table2_pos_w")}
                if "carry_box_selected_size_cm" in d:
                    half = d["carry_box_selected_size_cm"]/200.
                else:
                    half = np.array([float(s) for s in heights[row["clip_npz"]]["box_dimensions_m"].split("x")])/2
                recorded = bool(d["tables_available"])
            tables = np.stack([a["table1_pos_w"][0], a["table2_pos_w"][0]])
            if not recorded:
                tables[1, :2] = a["object_pos_w"][-1, :2]
                source_height = heights[row["clip_npz"]].get("support_surface_z_m")
                tables[1, 2] = float(source_height or 0.)
            model.geom_size[model.geom("carry_box_geom").id] = half
            for k in (1, 2):
                model.body_pos[model.body(f"table{k}").id] = tables[k-1]
            cut = row["keep_end"]
            peak = row.get("terminal_init", {}).get("peak_frame")
            if peak is None:
                peak = row["events"][-1]["end"]-1
            frames = [max(0, row["last_contact"]), min(len(a["joint_pos"])-1, cut-1), peak, len(a["joint_pos"])-1]
            tile = Image.new("RGB", (1280, 222), (20, 25, 32))
            draw = ImageDraw.Draw(tile)
            draw.text((7, 2), f"{i+1:03d}  {Path(row['clip_npz']).name}  |  {'裁去再次 init' if cut<len(a['joint_pos']) else '正常收臂，保留'}", font=font, fill="white")
            cam.lookat[:] = a["base_pos_w"][min(len(a["joint_pos"])-1, cut-1)]+[0., 0., -.14]
            for j, t in enumerate(frames):
                data.qpos[:3] = a["base_pos_w"][t]
                data.qpos[3:7] = a["base_quat_w"][t]
                data.qpos[addresses] = a["joint_pos"][t]
                data.qpos[box_adr:box_adr+3] = a["object_pos_w"][t]
                data.qpos[box_adr+3:box_adr+7] = a["object_quat_w"][t]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                tile.paste(Image.fromarray(renderer.render()), (j*320, 21))
                label = ["松手", "保留末帧", "张臂峰值", "原轨迹末帧"][j]
                draw.text((j*320+7, 203), f"{label} · {t/50:.2f}s", font=font, fill=(115, 235, 175) if j==1 else "white")
            tile.save(gallery/f"clip_{i+1:03d}.jpg", quality=87)
            cards.append(f'<figure><figcaption>{html.escape(row["clip_npz"])}</figcaption><img loading="lazy" src="review/clip_{i+1:03d}.jpg"></figure>')
            if i%10==0:
                sheet = Image.new("RGB", (1280, 222*min(10, len(records)-i)), (20, 25, 32))
            sheet.paste(tile, (0, (i%10)*222))
            if i%10==9 or i==len(records)-1:
                name = f"sheet_{i//10+1:02d}.jpg"
                sheet.save(gallery/name, quality=88)
                pages.append(name)
            if (i+1)%30==0:
                print(f"review rendered {i+1}/{len(records)}", flush=True)
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>全库再次 init 边界复核</title><style>body{background:#141920;color:#edf0f6;font:16px system-ui;margin:24px auto;max-width:1280px}img{width:100%}figure{margin:24px 0}figcaption{overflow-wrap:anywhere}a{color:#79bbff}</style><h1>全部候选边界复核</h1><p>每行依次为松手、候选保留末帧、张臂峰值、原末帧。source 的支撑板按清单中的推断高度示意。</p>'
    (report/"review.html").write_text(page+''.join(cards)+'</html>')
    (report/"review/records.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(dict(reviewed_candidates=len(records), sheets=pages)))


if __name__ == "__main__":
    main()
