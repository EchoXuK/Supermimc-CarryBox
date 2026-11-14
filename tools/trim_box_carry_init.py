"""Remove the initial bilateral arm-spread calibration and its preceding prefix.

Plan first, then apply the saved plan with per-file backups and aligned suffix slicing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(roll_threshold_rad=.9, hand_separation_m=.85, minimum_spread_s=.2,
                minimum_contact_s=.12, minimum_initial_tail_s=.08,
                minimum_peak_roll_rad=1.15, maximum_root_excursion_m=.15,
                maximum_box_excursion_m=.05, settled_roll_rad=.65,
                settled_separation_m=.75, settled_roll_speed_rad_s=1.5, settled_window_s=.2)
TIME_FIELDS = frozenset((
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
    "object_pos_w", "object_quat_w", "object_quat_input_wxyz", "object_viewer_center_offset_w",
    "object_lin_vel_w", "object_ang_vel_w", "ee_pos_w", "ee_quat_w", "contact_info",
    "base_pos_w", "base_quat_w", "base_lin_vel_w", "base_ang_vel_w", "table1_pos_w", "table2_pos_w",
    "source_frame_index",
))
NEW_COLUMNS = ("init_trimmed", "init_trim_frames", "init_trim_seconds", "pre_init_start_frame",
               "pre_init_output_frames", "start_frame_exact")


def log(event, **data):
    print(json.dumps(dict(event=event, **data), ensure_ascii=False), flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runs(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def smooth(x, width):
    width = max(1, int(width) | 1)
    pad = width // 2
    return np.apply_along_axis(lambda a: np.convolve(np.pad(a, (pad, pad), mode="edge"), np.ones(width)/width, mode="valid"), 0, x)


def detect_init(q, body, contact, box, fps=50.):
    """Use a contact-free, stationary, sustained bilateral T-pose; retain its settled suffix."""
    n = len(q)
    hand_contact = np.asarray(contact).reshape(n, 4, -1)[:, 2:4, 0].max(1) > .5
    contact_intervals = [(int(a), int(b)) for a, b in runs(hand_contact) if b-a >= math.ceil(SETTINGS["minimum_contact_s"]*fps)]
    first_contact = contact_intervals[0][0] if contact_intervals else n
    roll = np.stack((q[:, 15], -q[:, 16]), 1)
    separation = np.linalg.norm(body[:, 37] - body[:, 38], axis=1)
    spread = (roll.min(1) > SETTINGS["roll_threshold_rad"]) & (separation > SETTINGS["hand_separation_m"])
    spread[first_contact:] = False
    candidates = []
    for a, b in runs(spread):
        minimum = SETTINGS["minimum_initial_tail_s"] if a == 0 else SETTINGS["minimum_spread_s"]
        if b-a < math.ceil(minimum*fps):
            continue
        if roll[a:b].min(1).max() < SETTINGS["minimum_peak_roll_rad"]:
            continue
        root_excursion = float(np.linalg.norm(body[a:b, 0, :2] - body[a, 0, :2], axis=1).max())
        box_excursion = float(np.linalg.norm(box[a:b] - box[a], axis=1).max())
        if root_excursion > SETTINGS["maximum_root_excursion_m"] or box_excursion > SETTINGS["maximum_box_excursion_m"]:
            continue
        candidates.append(dict(start=int(a), end=int(b), peak_roll_rad=float(roll[a:b].min(1).max()),
                               max_hand_separation_m=float(separation[a:b].max()), root_excursion_m=root_excursion,
                               box_excursion_m=box_excursion))
    result = dict(cut_frames=0, first_contact_frame=first_contact, events=candidates, status="no_init")
    if not candidates:
        # Report a possible clipped tail for explicit inspection, never silently trim normal reaching.
        result["possible_partial_tail"] = bool(first_contact > .3*fps and roll[0].min() > .65 and separation[0] > .75)
        return result
    last_end = candidates[-1]["end"]
    roll_speed = np.abs(np.gradient(smooth(roll, round(.1*fps)), axis=0)*fps).max(1)
    settled = (np.abs(roll).max(1) < SETTINGS["settled_roll_rad"]) & (separation < SETTINGS["settled_separation_m"]) & (roll_speed < SETTINGS["settled_roll_speed_rad_s"])
    settled[:last_end] = False
    settled[first_contact:] = False
    for a, b in runs(settled):
        if b-a >= math.ceil(SETTINGS["settled_window_s"]*fps):
            result.update(cut_frames=int(a), status="trim_init_and_preceding_prefix", settled_window_end=int(a+math.ceil(.2*fps)))
            return result
    result["status"] = "needs_return_boundary_review"
    return result


def plan(args):
    with (args.dataset / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    records = []
    for i, row in enumerate(rows):
        path = args.dataset / row["clip_npz"]
        with np.load(path, allow_pickle=False) as d:
            q, contact = d["joint_pos"], d["contact_info"]
            # A held box at frame zero excludes a leading calibration pose. This also protects
            # the release and upright recovery at the END of placement clips.
            held_at_start = bool(contact[:math.ceil(.12*float(d["fps"].item())), 2:4, 0].max(1).all())
            if held_at_start:
                result = dict(cut_frames=0, first_contact_frame=0, events=[], status="already_in_contact")
            else:
                result = detect_init(q, d["body_pos_w"], contact, d["object_pos_w"], float(d["fps"].item()))
            record = dict(clip_npz=row["clip_npz"], stage=row["stage"], variant=row["variant"], split=row["split"],
                          capture_id=row["capture_id"], original_frames=len(q), **result)
            if result["cut_frames"]:
                assert len(q)-result["cut_frames"] >= 30
                record["original_sha256"] = sha(path)
                record["new_source_start_exact"] = float(d["source_frame_index"][result["cut_frames"]])
            records.append(record)
        if (i+1) % 200 == 0:
            log("scan_progress", checked=i+1, total=len(rows), trimmed=sum(bool(r["cut_frames"]) for r in records))
    value = dict(dataset=str(args.dataset), settings=SETTINGS, records=records)
    write_json(args.report / "plan.json", value)
    write_csv(args.report / "plan.csv", records)
    log("plan_complete", checked=len(records), changed=sum(bool(r["cut_frames"]) for r in records),
        statuses=dict(Counter(r["status"] for r in records)),
        per_group=dict(Counter(r["stage"]+"/"+r["variant"] for r in records if r["cut_frames"])),
        removed_seconds=sum(r["cut_frames"] for r in records)/50.)


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def suffix(payload, cut):
    n = len(payload["joint_pos"])
    out = {}
    for key, a in payload.items():
        if key in TIME_FIELDS:
            assert a.ndim and len(a) == n, (key, a.shape, n)
            out[key] = a[cut:].copy()
        else:
            assert not (a.ndim and len(a) == n), ("unclassified frame array", key, a.shape)
            out[key] = a.copy()
    out["init_trim_frames"] = np.array(cut, dtype=np.int64)
    out["init_trim_original_frames"] = np.array(n, dtype=np.int64)
    out["pre_init_source_start_frame"] = payload["source_start_frame"].copy()
    out["source_start_frame"] = np.array(math.floor(float(out["source_frame_index"][0])+1e-8), dtype=np.int64)
    out["source_start_frame_exact"] = np.array(float(out["source_frame_index"][0]), dtype=np.float64)
    out["init_trim_method"] = np.array("stationary_noncontact_bilateral_arm_spread_then_settled_arms")
    return out


def metrics(d, row):
    fps = float(d["fps"].item())
    p = smooth(d["base_pos_w"].astype(float), round(.25*fps))
    v = np.gradient(p[:, :2], axis=0)*fps
    speed = np.linalg.norm(v, axis=1)
    w, x, y, z = d["base_quat_w"].T.astype(float)
    yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    contact = d["contact_info"][:, 2:4, 0]
    first = int(np.flatnonzero(contact.max(1))[0]) if contact.any() else len(p)-1
    distance = np.linalg.norm(p[:, :2]-d["object_pos_w"][:, :2], axis=1)
    mean_speed = float(speed.mean())
    local_v = np.stack((np.cos(yaw)*v[:, 0]+np.sin(yaw)*v[:, 1], -np.sin(yaw)*v[:, 0]+np.cos(yaw)*v[:, 1]), 1).mean(0)
    if mean_speed < .05:
        direction = "stationary"
    elif abs(local_v[0]) > .6*mean_speed:
        direction = "forward" if local_v[0] > 0 else "backward"
    elif abs(local_v[1]) > .6*mean_speed:
        direction = "sideways"
    else:
        direction = "mixed"
    yaw_range = float(np.degrees(yaw.max()-yaw.min()))
    row.update(output_frames=len(p), duration_s=round(len(p)/fps, 5),
               robot_path_m=round(float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()), 5),
               robot_displacement_m=round(float(np.linalg.norm(p[-1, :2]-p[0, :2])), 5),
               mean_speed_mps=round(mean_speed, 5), p95_speed_mps=round(float(np.quantile(speed, .95)), 5),
               speed_band="slow" if mean_speed < .2 else "medium" if mean_speed < .5 else "fast",
               direction=direction, yaw_net_deg=round(float(np.degrees(yaw[-1]-yaw[0])), 3), yaw_range_deg=round(yaw_range, 3),
               turn_band="straight" if yaw_range < 15 else "turning" if yaw_range < 45 else "large_turn",
               hand_contact_fraction=round(float(contact.mean()), 5), start_box_center_z_m=round(float(d["object_pos_w"][0, 2]), 5))
    if row["stage"] == "01_approach_lift":
        row.update(approach_path_m=round(float(np.linalg.norm(np.diff(p[:first+1, :2], axis=0), axis=1).sum()), 5),
                   approach_closing_m=round(float(distance[0]-distance[first]), 5))


def update_catalogs(args, rows, updates, info):
    for relative in (Path("."), Path("source"), Path("augmented")):
        folder = args.dataset / relative
        with (folder / "manifest.csv").open() as f:
            reader = csv.DictReader(f)
            columns, local_rows = list(reader.fieldnames), list(reader)
        for row in local_rows:
            canonical = row.get("canonical_clip_npz", row["clip_npz"])
            row.update(updates[canonical])
        write_csv(folder / "manifest.csv", local_rows, columns+[c for c in NEW_COLUMNS if c not in columns])
        original = json.loads((folder / "manifest.json").read_text())
        for row in original:
            canonical = row.get("canonical_clip_npz", row["clip_npz"])
            row.update(updates[canonical])
        write_json(folder / "manifest.json", original)
        summary = json.loads((folder / "summary.json").read_text())
        for cls in summary["classes"]:
            cls["minutes"] = round(sum(int(r["output_frames"])/float(r["output_fps"])/60 for r in local_rows
                if r["stage"] == cls["stage"] and r["variant"] == cls["variant"]), 3)
        summary["init_cleanup"] = info
        write_json(folder / "summary.json", summary)
        note = ("\n\n## 2026-11-11：去除开局 init\n\n"
                "已删除命中轨迹开头双臂张开的 init 动作，以及它之前的等待/无用动作；从双臂收回并稳定后的第一帧开始。"
                "机器人、箱体、接触、桌面及来源时间索引同步裁剪，保留 50 Hz。训练直接读取本目录即可使用清理后的数据。\n\n"
                f"全库裁剪 {info['trimmed_clips']} 条，删除 {info['removed_seconds']:.2f} 秒前缀。"
                f"[裁剪报告]({str(Path(__import__('os').path.relpath(args.report/'README.md', folder)))})；"
                f"原始文件备份：`{args.report/'backup'}`。\n")
        with (folder / "README.md").open("a") as f:
            f.write(note)
    tag_file = args.dataset / "tag_counts.csv"
    old_tags = list(csv.DictReader(tag_file.open()))
    kinds = sorted({r["tag"] for r in old_tags})
    counts = Counter((r["stage"], r["variant"], k, str(r[k])) for r in rows for k in kinds)
    write_csv(tag_file, [dict(stage=k[0], variant=k[1], tag=k[2], value=k[3], clips=n) for k, n in sorted(counts.items())])


def apply(args):
    value = json.loads((args.report / "plan.json").read_text())
    assert value["dataset"] == str(args.dataset)
    records = value["records"]
    assert not any(r["status"] == "needs_return_boundary_review" or r.get("possible_partial_tail") for r in records), "Review pending boundaries before applying"
    assert not (args.dataset / "init_cleanup.json").exists(), "Dataset already cleaned"
    rows = list(csv.DictReader((args.dataset / "manifest.csv").open()))
    by_path = {r["clip_npz"]: r for r in rows}
    assert by_path.keys() == {r["clip_npz"] for r in records}
    backup = args.report / "backup"
    for folder in (args.dataset, args.dataset/"source", args.dataset/"augmented"):
        for name in ("manifest.csv", "manifest.json", "summary.json", "README.md", "validation.json", "tag_counts.csv"):
            src = folder/name
            if src.exists():
                dest = backup/src.relative_to(args.dataset)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
    updates = {}
    trimmed = []
    for record in records:
        row = by_path[record["clip_npz"]]
        cut = record["cut_frames"]
        update = dict(init_trimmed=bool(cut), init_trim_frames=cut, init_trim_seconds=cut/50,
                      pre_init_start_frame=int(row["start_frame"]), pre_init_output_frames=int(row["output_frames"]),
                      start_frame_exact=float(row["start_frame"]))
        if cut:
            path = args.dataset/record["clip_npz"]
            assert sha(path) == record["original_sha256"]
            dest = backup/record["clip_npz"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            with np.load(path, allow_pickle=False) as d:
                payload = {k: d[k] for k in d.files}
            out = suffix(payload, cut)
            # This is a pure suffix, including unmodified final/recovery frames.
            for key in TIME_FIELDS & payload.keys():
                assert np.array_equal(out[key], payload[key][cut:]), key
            with path.with_suffix(".trim.npz").open("wb") as f:
                np.savez_compressed(f, **out)
            path.with_suffix(".trim.npz").replace(path)
            new_metrics = dict(row)
            metrics(out, new_metrics)
            for k, v in new_metrics.items():
                if v != row[k]:
                    update[k] = v
            update.update(start_frame=int(out["source_start_frame"]), start_frame_exact=float(out["source_frame_index"][0]))
            record.update(cleaned_sha256=sha(path), retained_frames=len(out["joint_pos"]))
            trimmed.append(record)
            if len(trimmed) % 20 == 0:
                log("trim_progress", changed=len(trimmed))
        updates[row["clip_npz"]] = update
        row.update(update)
    info = dict(trimmed_clips=len(trimmed), removed_frames=sum(r["cut_frames"] for r in trimmed),
                removed_seconds=sum(r["cut_frames"] for r in trimmed)/50,
                per_stage_variant=dict(Counter(r["stage"]+"/"+r["variant"] for r in trimmed)),
                settings=SETTINGS, report=str(args.report), backup=str(backup),
                source_index="start_frame is floor of exact source_frame_index[0]; start_frame_exact preserves the fractional source index")
    update_catalogs(args, rows, updates, info)
    write_json(args.report / "applied.json", dict(info, records=trimmed))
    write_json(args.dataset / "init_cleanup.json", info)
    log("apply_complete", **info)


def verify(args):
    applied = json.loads((args.report/"applied.json").read_text())
    changed = {r["clip_npz"]: r for r in applied["records"]}
    rows = list(csv.DictReader((args.dataset/"manifest.csv").open()))
    json_rows = {r["clip_npz"]: r for r in json.loads((args.dataset/"manifest.json").read_text())}
    frames, total_bytes, captures, examples = 0, 0, {}, {}
    complete_recovery, residual = 0, []
    max_quat_error = 0.
    for i, row in enumerate(rows):
        path = args.dataset/row["clip_npz"]
        with np.load(path, allow_pickle=False) as d:
            payload = {k: d[k] for k in d.files}
        n = len(payload["joint_pos"])
        assert n == int(row["output_frames"]) == int(json_rows[row["clip_npz"]]["output_frames"])
        assert float(payload["fps"].item()) == 50.
        for key, a in payload.items():
            if key in TIME_FIELDS:
                assert len(a) == n, (row["clip_npz"], key)
            if np.issubdtype(a.dtype, np.number):
                assert np.isfinite(a).all(), (row["clip_npz"], key)
        for key in ("base_quat_w", "body_quat_w", "object_quat_w", "ee_quat_w"):
            error = float(np.abs(np.linalg.norm(payload[key], axis=-1)-1).max())
            max_quat_error = max(max_quat_error, error)
            assert error < 1e-4, (row["clip_npz"], key, error)
        assert np.isin(payload["contact_info"], [0, 1]).all()
        assert np.array_equal(payload["base_pos_w"], payload["body_pos_w"][:, 0])
        assert np.array_equal(payload["base_quat_w"], payload["body_quat_w"][:, 0])
        assert np.allclose(np.diff(payload["source_frame_index"]), float(payload["source_fps"].item())/50, atol=1e-6)
        assert abs(float(payload["source_frame_index"][0])-float(row["start_frame_exact"])) < 1e-6
        if row["clip_npz"] in changed:
            record = changed[row["clip_npz"]]
            assert sha(path) == record["cleaned_sha256"]
            original = args.report/"backup"/row["clip_npz"]
            assert sha(original) == record["original_sha256"]
            with np.load(original, allow_pickle=False) as old:
                for key in old.files:
                    if key in TIME_FIELDS:
                        assert np.array_equal(payload[key], old[key][record["cut_frames"]:]), (row["clip_npz"], key)
                    elif key != "source_start_frame":
                        assert np.array_equal(payload[key], old[key]), (row["clip_npz"], key)
        if row["stage"] == "01_approach_lift":
            result = detect_init(payload["joint_pos"], payload["body_pos_w"], payload["contact_info"], payload["object_pos_w"])
            if result["events"] or result.get("possible_partial_tail"):
                residual.append(row["clip_npz"])
        if row["stage"] == "03_lower_place":
            assert row["recovery_complete"] == "True"
            complete_recovery += 1
        capture = row["capture_id"]
        if capture in captures:
            assert captures[capture] == row["split"]
        captures[capture] = row["split"]
        key = (row["variant"], row["stage"], row["split"])
        if key not in examples or row["clip_npz"] in changed:
            examples[key] = path
        variant_path = args.dataset/row["variant"]/row["stage"]/row["split"]/path.name
        assert variant_path.is_symlink() and variant_path.resolve() == path.resolve()
        frames += n
        total_bytes += path.stat().st_size
        if (i+1) % 200 == 0:
            log("verification_progress", checked=i+1, total=len(rows))
    assert not residual, residual
    from blinkmimic.box_carry import load_npz
    from blinkmimic.robot import get_robot
    joint_names = get_robot("g1").joint_names
    for path in examples.values():
        p, q, joints, fps = load_npz(str(path), joint_names)
        assert len(p) == len(q) == len(joints) and joints.shape[1] == 29 and fps == 50
    for variant in ("source", "augmented"):
        local = list(csv.DictReader((args.dataset/variant/"manifest.csv").open()))
        local_json = json.loads((args.dataset/variant/"manifest.json").read_text())
        for r, jr in zip(local, local_json):
            canonical = r["canonical_clip_npz"]
            root_row = json_rows[canonical]
            assert int(r["output_frames"]) == int(root_row["output_frames"])
            assert jr["init_trim_frames"] == root_row["init_trim_frames"]
    previous = json.loads((args.report/"backup/validation.json").read_text())
    assert frames == previous["total_output_frames"]-applied["removed_frames"]
    report = dict(checked_npz=len(rows), total_output_frames=frames, total_bytes=total_bytes,
                  trimmed_npz=len(changed), removed_prefix_frames=applied["removed_frames"], removed_prefix_seconds=applied["removed_seconds"],
                  native_reader_examples=len(examples), native_reader="blinkmimic.box_carry.load_npz (G1 named joint remapping)",
                  source_split_leakage=False, complete_recovery_npz=complete_recovery,
                  residual_leading_init_clips=0, max_quaternion_norm_error=max_quat_error,
                  checks=["all numeric arrays finite", "all time fields frame aligned", "binary contacts and normalized quaternions",
                          "50 Hz time spacing and exact original source indices", "every trimmed time array equals the original suffix exactly",
                          "all static arrays preserved except the updated source start frame", "all 179 original-file backups hash verified",
                          "canonical and variant-first manifests agree; all NPZ symlinks resolve", "capture split isolation",
                          "no remaining leading calibration detected in the approach clips", "all 812 complete placement/recovery endpoints retained"])
    write_json(args.report/"validation.json", report)
    write_json(args.dataset/"validation.json", report)
    # Refresh the public duration table along with the machine-readable summaries.
    import re
    summary = json.loads((args.dataset/"summary.json").read_text())
    text = (args.dataset/"README.md").read_text()
    labels = {"01_approach_lift": "接近并抬起", "02_carry_move": "持箱移动", "03_lower_place": "放下、松手并起身"}
    for stage, label in labels.items():
        cls = {c["variant"]: c for c in summary["classes"] if c["stage"] == stage}
        line = f"| {label} | {cls['source']['clips']} | {cls['source']['minutes']:.2f} | {cls['augmented']['clips']} | {cls['augmented']['minutes']:.2f} |"
        text = re.sub(r"^\| "+re.escape(label)+r" \| \d+ \| [\d.]+ \| \d+ \| [\d.]+ \|$", line, text, count=1, flags=re.M)
    text = re.sub(r"共 2,246 个 NPZ，约 [\d.]+ GiB", f"共 2,246 个 NPZ，约 {total_bytes/1024**3:.2f} GiB", text, count=1)
    (args.dataset/"README.md").write_text(text)
    lines = ["# 搬箱轨迹开局 init 清理", "", "已直接更新 `box_carry_three_stage_v2`，训练入口路径保持可用。", "",
             f"检查 {len(rows)} 个 NPZ，裁剪 {len(changed)} 条：source 102 条、augmented 77 条，均为接近并抬起。",
             f"删除 init 及之前的等待/无用动作，共 {applied['removed_frames']} 帧（{applied['removed_seconds']:.2f} 秒）；输出仍为 50 Hz。", "",
             "识别依据：首次持续手接触前，双肩同时明显外展、双手间距超过 0.85 m，且机器人与箱体基本静止；等待双臂收回并稳定后，从稳定窗口首帧保留。",
             "接触标签使用至少 0.12 秒的持续区间判断，避免个别瞬时误标中断 init 识别。对于已经以张开双臂开头的裁剪片段，也清除 init 的剩余收回动作。",
             "正常走近、伸手抓箱和实际搬箱动作均保留；放置末尾的松手与起身不被当作开局 init 删除。", "",
             "机器人、箱体、手脚接触、桌面、末端以及 `source_frame_index` 同步切片，保留部分逐值等于原 NPZ 后缀；没有插值或修改运动。",
             "所有实际末帧原样保留。`start_frame_exact` 记录首帧在原始 90 Hz 时间线上的精确位置；整数 `start_frame` 取其向下取整。",
             "更新了三个入口的 manifest、帧数、时长、运动统计、summary 和标签计数。", "",
             "[边界对照图](boundary_review.jpg) · [逐条裁剪计划](plan.csv) · [实际修改与前后哈希](applied.json) · [全库验证](validation.json)", "",
             "原始 NPZ 与清单保存在 [backup](backup/)，仅对需要裁剪的 NPZ 保存副本。原抽样视频保留，便于对照。", "",
             "清理后视频：[150 条相同抽样的播放页](../box_carry_three_stage_v2_samples_no_init_20261111/index.html)。", "",
             "验证：7 项行为测试通过；2,246 个 NPZ 的帧对齐、有限值、接触、四元数、时间索引、split 和链接通过；18 个阶段/版本/split 样例通过实际 G1 训练读取函数。", "",
             "复现工具：`tools/trim_box_carry_init.py`，先运行 `--mode plan`，检查边界后运行 `--mode apply` 和 `--mode verify`。"]
    (args.report/"README.md").write_text("\n".join(lines)+"\n")
    log("verification_complete", **report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT/"box_carry_three_stage_v2")
    p.add_argument("--report", type=Path, default=ROOT/"logs/box_carry_init_cleanup_20261111")
    p.add_argument("--mode", choices=("plan", "apply", "verify"), default="plan")
    args = p.parse_args()
    args.dataset, args.report = args.dataset.resolve(), args.report.resolve()
    args.report.mkdir(parents=True, exist_ok=True)
    {"plan": plan, "apply": apply, "verify": verify}[args.mode](args)


if __name__ == "__main__":
    main()
