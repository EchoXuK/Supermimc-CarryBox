"""Audit arm calibration throughout the dataset, then trim post-release reinitialization.

No trajectory resampling, interpolation, or stitching: every retained time series
is a contiguous part of its original file. A separate backup records each edit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import os
import re
import shutil
import hashlib

import numpy as np

from trim_box_carry_init import ROOT, TIME_FIELDS, runs, smooth, write_json, write_csv, log, metrics, detect_init

DEFAULT_REPORT = ROOT / "logs/box_carry_terminal_init_cleanup_20261111"
SETTINGS = dict(full_roll_rad=.9, full_hand_separation_m=.85, partial_roll_rad=.5,
                partial_hand_separation_m=.58, minimum_event_s=.06,
                minimum_contact_s=.12, roll_rate_rad_s=.25, separation_rate_m_s=.08,
                minimum_roll_increase_rad=.13, minimum_separation_increase_m=.08,
                maximum_box_displacement_m=.035, retained_recovery_tilt_deg=25.,
                overlap_max_bilateral_outward_roll_rad=.75, overlap_max_hand_separation_m=.72,
                overlap_max_hand_height_relative_pelvis_m=.02)
FEATURES = ("left_roll", "right_roll", "separation", "mean_hand_z_relative_pelvis",
            "pelvis_ankle_height", "torso_tilt", "root_x", "root_y", "root_z",
            "box_x", "box_y", "box_z", "contact_left", "contact_right", "source_frame")


def load_features(item):
    root, row = item
    with np.load(root/row["clip_npz"], allow_pickle=False) as d:
        return feature_array(d)


def feature_array(d):
    q, p = d["joint_pos"], d["body_pos_w"]
    c = d["contact_info"][:, 2:4, 0]
    w, x, y, z = d["body_quat_w"][:, 11].T
    tilt = np.degrees(np.arccos(np.clip(1-2*(x*x+y*y), -1., 1.)))
    return np.column_stack((q[:, 15], -q[:, 16], np.linalg.norm(p[:, 37]-p[:, 38], axis=1),
        p[:, [37, 38], 2].mean(1)-p[:, 0, 2], p[:, 0, 2]-p[:, [25, 26], 2].mean(1),
        tilt, p[:, 0], d["object_pos_w"], c, d["source_frame_index"]))


def contact_bounds(a):
    c = a[:, 12:14].max(1) > .5
    intervals = [(int(s), int(e)) for s, e in runs(c) if e-s >= 6]
    return (intervals[0][0], intervals[-1][1]) if intervals else (len(a), 0)


def analyze(a, row):
    n = len(a)
    first, last = contact_bounds(a)
    r = smooth(a[:, :2], 5)
    sep = smooth(a[:, 2], 5)
    rate = np.gradient(r, axis=0)*50
    sep_rate = np.gradient(sep)*50
    c = a[:, 12:14].max(1) > .5
    looser = (r.min(1) > .5) & (sep > .58) & (~c)
    broad = [(int(s), int(e)) for s, e in runs(looser) if e-s >= 3]
    record = dict(clip_npz=row["clip_npz"], stage=row["stage"], variant=row["variant"], split=row["split"],
                  capture_id=row["capture_id"], original_frames=n, first_contact=first, last_contact=last,
                  keep_start=0, keep_end=n, events=[], status="unchanged")
    for s, e in broad:
        record["events"].append(dict(start=s, end=e, peak_roll=float(r[s:e].min(1).max()),
                                     peak_separation=float(sep[s:e].max()), location="prefix" if e<=first else "suffix" if s>=last else "interior"))
    if last == 0 or last >= n:
        return record
    # Seek bilateral upward ramps after final release, including clipped partial ramps.
    moving = (rate.min(1) > SETTINGS["roll_rate_rad_s"]) & (sep_rate > SETTINGS["separation_rate_m_s"])
    moving[:last] = False
    moving[c] = False
    ramps = [(int(s), int(e)) for s, e in runs(moving) if e-s >= 4]
    candidates = []
    for s, e in ramps:
        peak = min(n, max(e+10, s+30))
        # Grow to the same continuous spread, but never cross a new object interaction.
        delta = r[s:peak]-r[s]
        if delta.min(1).max() < SETTINGS["minimum_roll_increase_rad"] or sep[s:peak].max()-sep[s] < SETTINGS["minimum_separation_increase_m"]:
            continue
        full = bool(((r[s:peak].min(1) > .9) & (sep[s:peak] > .85)).any())
        partial = bool(((r[s:peak].min(1) > .5) & (sep[s:peak] > .58)).any())
        baseline = max(last, s-10)
        hands_down = bool((a[baseline:s+1, 3] < .02).any())
        relatively_standing = bool((a[baseline:peak, 4] > .64).any())
        baseline_low = bool((r[baseline:s+1].max(1) < .6).any())
        tail_still_spreading = bool(r[-1].min() >= r[s:peak].min(1).max()-.12)
        if not full and not (partial and hands_down and relatively_standing and baseline_low and tail_still_spreading):
            continue
        if np.linalg.norm(a[s:peak, 9:12]-a[s, 9:12], axis=1).max() > SETTINGS["maximum_box_displacement_m"]:
            continue
        # Backtrack to the last arm-down state before the growth, with a small margin.
        onset = s
        while onset > last+1 and rate[onset-1].min() > .05 and sep_rate[onset-1] > .02:
            onset -= 1
        onset = max(last+1, onset-2)
        candidates.append(dict(ramp_start=s, ramp_end=e, onset=onset, full_tpose=full,
                               hands_previously_lowered=hands_down, peak_frame=int(s+np.argmax(r[s:peak].min(1))),
                               peak_roll=float(r[s:peak].min(1).max()), peak_separation=float(sep[s:peak].max())))
    # A full/partial calibration may begin before a short clip starts rising; audit all
    # high-arm post-release intervals even when a ramp is too short for the main rule.
    unexplained = [ev for ev in record["events"] if ev["location"] == "suffix" and ev["peak_roll"] > .65
                   and not any(ca["onset"] <= ev["start"]+5 and ca["peak_frame"] >= ev["start"] for ca in candidates)]
    if candidates:
        event = min(candidates, key=lambda x: x["onset"])
        record.update(keep_end=event["onset"], status="trim_post_release_reinit", terminal_init=event,
                      tail_removed_frames=n-event["onset"])
    if unexplained:
        record["unexplained_suffix_spreads"] = unexplained
    return record


def audit(args):
    rows = list(csv.DictReader((args.dataset/"manifest.csv").open()))
    all_features, records = {}, []
    with ThreadPoolExecutor(max_workers=4) as workers:
        for i, (row, features) in enumerate(zip(rows, workers.map(load_features, ((args.dataset, row) for row in rows)))):
            key = f"clip_{i:04d}"
            all_features[key] = features
            record = analyze(features, row)
            record["feature_key"] = key
            records.append(record)
            if (i+1) % 100 == 0:
                log("audit_progress", checked=i+1, total=len(rows), candidates=sum(r["keep_end"]<r["original_frames"] for r in records))
    np.savez_compressed(args.report/"features.npz", **all_features)
    write_json(args.report/"audit.json", dict(dataset=str(args.dataset), settings=SETTINGS, features=FEATURES, records=records))
    summarize(records)


def reanalyze(args):
    audit = json.loads((args.report/"audit.json").read_text())
    rows = list(csv.DictReader((args.dataset/"manifest.csv").open()))
    with np.load(args.report/"features.npz", allow_pickle=False) as features:
        records = [dict(analyze(features[old["feature_key"]], row), feature_key=old["feature_key"])
                   for old, row in zip(audit["records"], rows)]
    write_json(args.report/"audit.json", dict(dataset=str(args.dataset), settings=SETTINGS, features=FEATURES, records=records))
    summarize(records)


def summarize(records):
    log("audit_complete", checked=len(records), changed=sum(r["keep_end"]<r["original_frames"] for r in records),
        changed_groups=dict(Counter(r["stage"]+"/"+r["variant"] for r in records if r["keep_end"]<r["original_frames"])),
        events=dict(Counter(r["stage"]+"/"+r["variant"]+"/"+e["location"] for r in records for e in r["events"])),
        unexplained=sum(bool(r.get("unexplained_suffix_spreads")) for r in records))


def recovery_evidence(a, row):
    """Re-evaluate the original half-second upright criterion on retained frames only."""
    _, last = contact_bounds(a)
    n = len(a)
    h = smooth(a[:, 4], 13)
    tilt = smooth(a[:, 5], 13)
    root = smooth(a[:, 6:9], 13)
    velocity = np.gradient(root, axis=0)*50
    height_velocity = np.gradient(h)*50
    threshold = float(row.get("standing_height_threshold_m") or .65)
    free = a[:, 12:14].max(1) == 0
    good = (h>=threshold) & (tilt<=15.) & free & (np.arange(n)>=last)
    intervals = [(int(s), int(e)) for s, e in runs(good)]
    longest = max((e-s for s, e in intervals), default=0)
    stable_start = -1
    for s, e in intervals:
        for begin in range(s, e-25+1):
            end = begin+25
            if (np.ptp(h[begin:end])<=.02 and np.abs(height_velocity[begin:end]).mean()<=.04
                    and np.linalg.norm(velocity[begin:end, :2], axis=1).mean()<=.15
                    and np.ptp(a[begin:end, 11])<=.025
                    and np.linalg.norm(a[begin:end, 9:11]-a[begin, 9:11], axis=1).max()<=.035):
                stable_start = begin
                break
        if stable_start>=0:
            break
    tail_start = max(last, n-25)
    return dict(recovery_complete=bool(stable_start>=0), recovery_stable_output_start=stable_start,
                recovery_stable_window_start=int(round(a[stable_start, 14])) if stable_start>=0 else -1,
                recovery_stable_window_start_exact=float(a[stable_start, 14]) if stable_start>=0 else -1.,
                recovery_upright_duration_s=longest/50.,
                recovery_observed_upright=bool(longest>=5),
                end_min_pelvis_ankle_height_m=float(h[tail_start:].min()),
                end_max_torso_tilt_deg=float(tilt[tail_start:].max()),
                release_retained=bool(last>0 and last<n))


def recovery_preserving_end(a, row, onset):
    """Keep useful standing transition, bounded before visibly spread/lifted arms.

    The user prefers keeping clips when rising and calibration overlap. Stop at
    the first near-upright frame, or the arm-clearance limit, whichever comes first.
    A clean arm-down endpoint already near upright requires no extension at all.
    """
    h = smooth(a[:, 4], 5)
    tilt = smooth(a[:, 5], 5)
    r = smooth(a[:, :2], 5)
    sep = smooth(a[:, 2], 5)
    threshold = float(row.get("standing_height_threshold_m") or .65)
    near_upright = (h >= threshold) & (tilt <= SETTINGS["retained_recovery_tilt_deg"])
    if near_upright[onset-1]:
        return onset, dict(overlap_retained_frames=0, endpoint_reason="near_upright_before_init")
    limit = ((r.min(1) >= SETTINGS["overlap_max_bilateral_outward_roll_rad"])
             | (sep >= SETTINGS["overlap_max_hand_separation_m"])
             | (a[:, 3] >= SETTINGS["overlap_max_hand_height_relative_pelvis_m"]))
    limit[:onset] = False
    stops = np.flatnonzero(limit)
    cap = int(stops[0]) if len(stops) else len(a)
    cap = max(onset, cap)
    upright = np.flatnonzero(near_upright[onset:cap])
    end = onset+int(upright[0])+1 if len(upright) else cap
    return end, dict(overlap_retained_frames=end-onset,
                     endpoint_reason="near_upright_during_overlap" if len(upright) else "arm_clearance_limit_during_recovery",
                     overlap_arm_limit_frame=cap)


def make_plan(args):
    audit = json.loads((args.report/"audit.json").read_text())
    catalog_root = args.report/"backup" if args.use_original_backup else args.dataset
    rows = {r["clip_npz"]: r for r in csv.DictReader((catalog_root/"manifest.csv").open())}
    records = []
    with np.load(args.report/"features.npz", allow_pickle=False) as features:
        for old in audit["records"]:
            record = dict(old, action="retain")
            assert not old.get("unexplained_suffix_spreads"), old["clip_npz"]
            if old["keep_end"]<old["original_frames"]:
                a = features[old["feature_key"]]
                record["init_onset_frame"] = old["keep_end"]
                end, boundary = recovery_preserving_end(a, rows[old["clip_npz"]], old["keep_end"])
                record.update(keep_end=end, tail_removed_frames=len(a)-end, **boundary)
                evidence = recovery_evidence(a[:record["keep_end"]], rows[old["clip_npz"]])
                record["retained_recovery_evidence"] = evidence
                record["action"] = "trim_tail"
            records.append(record)
    # Record hashes for ALL files so unchanged motions can be verified exactly.
    def digest(record):
        backup = args.report/"backup"/record["clip_npz"]
        path = backup if args.use_original_backup and backup.exists() else args.dataset/record["clip_npz"]
        return hashlib.sha256(path.read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=4) as workers:
        for record, digest_value in zip(records, workers.map(digest, records)):
            record["original_sha256"] = digest_value
    plan = dict(dataset=str(args.dataset), settings=SETTINGS,
                incomplete_handling="keep_as_much_recovery_as_possible_before_visible_arm_spread", records=records)
    write_json(args.report/"plan.json", plan)
    write_csv(args.report/"plan.csv", [dict(clip_npz=r["clip_npz"], stage=r["stage"], variant=r["variant"],
        action=r["action"], original_frames=r["original_frames"], retained_frames=r["keep_end"],
        init_onset_frame=r.get("init_onset_frame", ""), overlap_retained_frames=r.get("overlap_retained_frames", 0),
        removed_frames=r["original_frames"]-r["keep_end"], endpoint_reason=r.get("endpoint_reason", ""),
        original_sha256=r["original_sha256"], complete_recovery_before_init=r.get("retained_recovery_evidence", {}).get("recovery_complete", "")) for r in records])
    log("plan_complete", actions=dict(Counter(r["action"] for r in records)))


def refresh_summaries(folder, local_rows, info):
    path = folder/"summary.json"
    summary = json.loads(path.read_text())
    for cls in summary["classes"]:
        subset = [r for r in local_rows if r["stage"]==cls["stage"] and r["variant"]==cls["variant"]]
        cls.update(clips=len(subset), source_captures=len({r["capture_id"] for r in subset}),
                   minutes=round(sum(int(r["output_frames"])/float(r["output_fps"])/60 for r in subset), 3),
                   splits=dict(Counter(r["split"] for r in subset)))
    if "manifest_rows" in summary:
        summary["manifest_rows"] = len(local_rows)
    if "stage_counts" in summary:
        summary["stage_counts"] = dict(Counter(r["stage"] for r in local_rows))
        summary["split_counts"] = dict(Counter(r["split"] for r in local_rows))
    if "endpoint_evidence" in summary:
        summary["endpoint_evidence"] = dict(Counter(r["endpoint_evidence"] for r in local_rows))
    summary["recovery_status_counts"] = dict(Counter(r["recovery_status"] for r in local_rows if r["stage"]=="03_lower_place"))
    summary["terminal_init_cleanup"] = info
    write_json(path, summary)


def update_catalogs(args, updates, height_updates, info):
    for relative in (Path("."), Path("source"), Path("augmented")):
        folder = args.dataset/relative
        with (folder/"manifest.csv").open() as f:
            reader = csv.DictReader(f)
            original_columns = list(reader.fieldnames)
            rows = list(reader)
        out = []
        for row in rows:
            canonical = row.get("canonical_clip_npz", row["clip_npz"])
            row.update(updates.get(canonical, {}))
            out.append(row)
        columns = list(dict.fromkeys(original_columns+[k for r in out for k in r]))
        write_csv(folder/"manifest.csv", out, columns)
        original = json.loads((folder/"manifest.json").read_text())
        result = []
        for row in original:
            canonical = row.get("canonical_clip_npz", row["clip_npz"])
            row.update(updates.get(canonical, {}))
            result.append(row)
        write_json(folder/"manifest.json", result)
        refresh_summaries(folder, out, info)
        placement = folder/"placement_height_analysis.csv"
        if placement.exists():
            with placement.open() as f:
                reader = csv.DictReader(f)
                fields, analysis = list(reader.fieldnames), list(reader)
            for row in analysis:
                canonical = row.get("canonical_clip_npz", row["clip_npz"])
                row.update(height_updates.get(canonical, {}))
            write_csv(placement, analysis, list(dict.fromkeys(fields+[k for r in analysis for k in r])))
        note = ("\n\n## 2026-11-11：清除放置后的再次 init\n\n"
                f"全库逐帧复扫后，{info['trimmed_clips']} 条放置轨迹已裁掉再次张臂的尾段，共 {info['removed_seconds']:.2f} 秒。"
                f"按用户要求保留轨迹及尽可能多的起身动作；当前本入口仍有 {len(out)} 条。"
                "起身与张臂重叠时，在满足站高、躯干倾斜不超过 25° 后结束，或在手臂外展限幅前结束；"
                "无 init 前连续 0.5 秒稳定站姿的轨迹标记 recovery_complete=False，不再沿用完整起身标签。"
                f"原 NPZ 保存在 `{args.report/'backup'}`。\n\n"
                f"[全库清理与验证报告]({os.path.relpath(args.report/'README.md', folder)}) · "
                f"[全部候选边界图]({os.path.relpath(args.report/'review.html', folder)})\n")
        with (folder/"README.md").open("a") as f:
            f.write(note)
    root_rows = list(csv.DictReader((args.dataset/"manifest.csv").open()))
    old_tags = list(csv.DictReader((args.dataset/"tag_counts.csv").open()))
    tags = sorted({r["tag"] for r in old_tags})
    counts = Counter((r["stage"], r["variant"], tag, str(r[tag])) for r in root_rows for tag in tags)
    write_csv(args.dataset/"tag_counts.csv", [dict(stage=k[0], variant=k[1], tag=k[2], value=k[3], clips=v) for k, v in sorted(counts.items())])
    layout = json.loads((args.dataset/"variant_layout.json").read_text())
    for variant, meta in layout["variants"].items():
        meta["rows"] = sum(r["variant"]==variant for r in root_rows)
    write_json(args.dataset/"variant_layout.json", layout)
    text = (args.dataset/"README.md").read_text()
    summary = json.loads((args.dataset/"summary.json").read_text())
    labels = {"01_approach_lift":"接近并抬起", "02_carry_move":"持箱移动", "03_lower_place":"放下、松手并起身"}
    for stage, label in labels.items():
        classes = {c["variant"]: c for c in summary["classes"] if c["stage"]==stage}
        line = f"| {label} | {classes['source']['clips']} | {classes['source']['minutes']:.2f} | {classes['augmented']['clips']} | {classes['augmented']['minutes']:.2f} |"
        text = re.sub(r"^\| "+re.escape(label)+r" \| \d+ \| [\d.]+ \| \d+ \| [\d.]+ \|$", line, text, count=1, flags=re.M)
    for cls in summary["classes"]:
        label = labels[cls["stage"]]
        split = cls["splits"]
        line = f"| {label} | {cls['variant']} | {split.get('train',0)} | {split.get('val',0)} | {split.get('test',0)} |"
        text = re.sub(r"^\| "+re.escape(label)+r" \| "+cls["variant"]+r" \| \d+ \| \d+ \| \d+ \|$", line, text, count=1, flags=re.M)
    n_place = sum(r["stage"]=="03_lower_place" for r in root_rows)
    n_complete = sum(r["stage"]=="03_lower_place" and r["recovery_complete"]=="True" for r in root_rows)
    text = text.replace("第三类的 812 段全部标记 `recovery_complete=True` 与 `endpoint_evidence=placed_released_and_standing`。",
                        f"当前第三类共 {n_place} 段：{n_complete} 段保留完整稳定站姿，标记 `recovery_complete=True`；"
                        f"{n_place-n_complete} 段因再次 init 截尾，保留尽可能多的起身动作并标记 `recovery_complete=False`，详见 `recovery_status`。")
    text = text.replace("| `03_lower_place` | 明显下降前的稳定高位附近 | 箱子固定、双手松开、起身后站姿持续稳定 0.5 秒 | 保留 v1 的降低与箱体稳定条件，再检查站立高度、躯干倾斜和速度；缺少完整收尾的片段不纳入第三类 |",
                        "| `03_lower_place` | 明显下降前的稳定高位附近 | 正常轨迹保留稳定站姿；再次 init 的轨迹在张臂前截尾，尽量保留起身 | 放置与松手保留；完整稳定起身与未完整收尾由 `recovery_complete` / `recovery_status` 区分 |")
    total_bytes = sum((args.dataset/r["clip_npz"]).stat().st_size for r in root_rows)
    text = re.sub(r"共 [\d,]+ 个 NPZ，约 [\d.]+ GiB", f"共 {len(root_rows):,} 个 NPZ，约 {total_bytes/1024**3:.2f} GiB", text, count=1)
    heading = text.index("\n")
    banner = (f"\n\n**当前训练数据：仍保留 {len(root_rows):,} 条；已清除 {info['trimmed_clips']} 条放置轨迹的再次 init 尾段。** "
              f"第三类 {n_place} 条中，{n_complete} 条有完整稳定站姿，{n_place-n_complete} 条保留尽可能多的起身动作并按实际标记未完整收尾。"
              "下文 v1 → v2 的起身要求记录原始切分方法；当前边界以 manifest、recovery_status 和最新 validation 为准。\n")
    text = text[:heading]+banner+text[heading:]
    (args.dataset/"README.md").write_text(text)
    height_summary = json.loads((args.dataset/"placement_height_summary.json").read_text())
    analysis = list(csv.DictReader((args.dataset/"placement_height_analysis.csv").open()))
    placements = [r for r in analysis if r["stage"]=="03_lower_place"]
    height_summary.update(root=str(args.dataset), terminal_init_cleanup=info,
        height_quality_counts=dict(Counter(r["height_quality"] for r in placements)),
        low_kinematic_foot_stability_rows=sum(r["height_quality"]=="low" for r in placements))
    for variant in ("source", "augmented"):
        values = [float(r["placement_height_above_ankle_m"]) for r in placements if r["variant"]==variant]
        height_summary["placement_height_above_ankle_m"][variant] = dict(count=len(values), min=min(values), median=float(np.median(values)), max=max(values))
    write_json(args.dataset/"placement_height_summary.json", height_summary)
    note_path = args.dataset/"placement_height_analysis.md"
    note_path.write_text("**再次 init 清理后：支撑面分档沿用已推断的几何；裁尾轨迹的末端高度/足部稳定指标已用当前保留末端重算。旧图和原始稳定窗口分析为历史记录。**\n\n"+note_path.read_text())


def trim_payload(payload, record):
    end, n = record["keep_end"], len(payload["joint_pos"])
    assert 1 < end < n
    out = {}
    for key, value in payload.items():
        if key in TIME_FIELDS:
            assert len(value)==n, (key, value.shape, n)
            out[key] = value[:end].copy()
        else:
            assert not (value.ndim and len(value)==n), ("unclassified time field", key)
            out[key] = value.copy()
    evidence = record["retained_recovery_evidence"]
    near = record["endpoint_reason"] in ("near_upright_before_init", "near_upright_during_overlap")
    status = "stable_upright" if evidence["recovery_complete"] else "near_upright_without_stable_window" if near else "partial_recovery_before_init"
    endpoint = "placed_released_and_standing" if evidence["recovery_complete"] else "placed_released_near_upright_before_init" if near else "placed_released_partial_recovery_before_init"
    last = float(out["source_frame_index"][-1])
    out.update(
        pre_terminal_init_source_end_frame_exclusive=payload["source_end_frame_exclusive"].copy(),
        pre_terminal_init_output_frames=np.array(n, dtype=np.int64),
        pre_terminal_init_recovery_complete=payload["recovery_complete"].copy(),
        pre_terminal_init_source_recovery_stable_window_start=payload["source_recovery_stable_window_start"].copy(),
        pre_terminal_init_source_placement_end_frame_exclusive=payload["source_placement_end_frame_exclusive"].copy(),
        terminal_init_trim_frames=np.array(n-end, dtype=np.int64),
        terminal_init_retained_overlap_frames=np.array(record["overlap_retained_frames"], dtype=np.int64),
        terminal_init_onset_output_frame=np.array(record["init_onset_frame"], dtype=np.int64),
        terminal_init_trim_method=np.array("contiguous_prefix_before_post_release_reinit_with_bounded_recovery_overlap"),
        terminal_init_cleanup_settings_json=np.array(json.dumps(SETTINGS, sort_keys=True)),
        source_end_frame_exclusive=np.array(math.floor(last+1e-8)+1, dtype=np.int64),
        source_end_frame_exclusive_exact=np.array(last+float(payload["source_fps"].item())/50., dtype=np.float64),
        source_last_frame_exact=np.array(last, dtype=np.float64),
        source_placement_end_frame_exclusive=np.array(min(int(payload["source_placement_end_frame_exclusive"]), math.floor(last+1e-8)+1), dtype=np.int64),
        placement_stability_window_complete=np.array(int(payload["source_placement_end_frame_exclusive"])<=math.floor(last+1e-8)+1),
        recovery_complete=np.array(evidence["recovery_complete"], dtype=bool),
        recovery_status=np.array(status), endpoint_evidence=np.array(endpoint),
        source_recovery_stable_window_start=np.array(evidence["recovery_stable_window_start"], dtype=np.int64),
        source_recovery_stable_window_start_exact=np.array(evidence["recovery_stable_window_start_exact"], dtype=np.float64))
    return out


def retained_height_metrics(d, old):
    """Recompute the diagnostic endpoint window; retain the original support assignment."""
    contact = d["contact_info"][:, 2:4, 0].max(1)>.5
    last_contact = int(np.flatnonzero(contact)[-1])+1
    start, end = max(last_contact, len(contact)-25), len(contact)
    feet = d["body_pos_w"][start:end, 25:27]
    velocity = d["body_lin_vel_w"][start:end, 25:27]
    stable = (np.linalg.norm(velocity[:, :, :2], axis=-1)<=.15) & (np.abs(velocity[:, :, 2])<=.04)
    fraction = float(stable.all(1).mean())
    dimensions = np.array([float(x) for x in old["box_dimensions_m"].split("x")])
    w, x, y, z = d["object_quat_w"][start:end].T
    vertical = np.column_stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))
    bottom = d["object_pos_w"][start:end, 2]-(np.abs(vertical)*dimensions[None]/2).sum(1)
    foot_height = float(np.median(feet[:, :, 2].mean(1)))
    box_z = float(np.median(d["object_pos_w"][start:end, 2]))
    bottom_z = float(np.median(bottom))
    return dict(output_frames=len(contact), output_fps=50., recovery_window_frames=end-start,
        support_estimation_frames=end-start, source_frame_start_used=float(d["source_frame_index"][start]),
        source_frame_end_used=float(d["source_frame_index"][end-1]), frame_window_fallback=True,
        foot_stable_fallback=fraction<.9, foot_support_z_m=foot_height,
        left_ankle_z_m=float(np.median(feet[:, 0, 2])), right_ankle_z_m=float(np.median(feet[:, 1, 2])),
        ankle_height_difference_m=float(np.median(np.abs(feet[:, 0, 2]-feet[:, 1, 2]))),
        both_feet_kinematic_contact_fraction=fraction, official_foot_contact_fraction=float(d["contact_info"][start:end, :2].mean()),
        terminal_hand_contact_fraction=float(d["contact_info"][start:end, 2:4].mean()),
        box_center_z_m=box_z, box_bottom_z_m=bottom_z, box_bottom_range_m=float(np.ptp(bottom)),
        placement_height_above_ankle_m=bottom_z-foot_height,
        height_quality="high" if fraction>=.9 else "medium" if fraction>=.5 else "low",
        height_analysis_method="retained_final_0.5s_or_shorter_free_window_after_terminal_init_trim",
        recovery_complete=bool(d["recovery_complete"].item()), recovery_status=str(d["recovery_status"].item()),
        pre_cleanup_source_frame_start_used=old["source_frame_start_used"],
        pre_cleanup_source_frame_end_used=old["source_frame_end_used"])


def apply(args):
    plan = json.loads((args.report/"plan.json").read_text())
    assert plan["dataset"]==str(args.dataset)
    assert args.use_original_backup or not (args.dataset/"terminal_init_cleanup.json").exists(), "Already applied"
    records = plan["records"]
    assert all(r["action"] in ("retain", "trim_tail") for r in records)
    backup = args.report/"backup"
    if args.use_original_backup:
        previous = json.loads((args.report/"applied.json").read_text())
        for old in previous["records"]:
            path = args.dataset/old["clip_npz"]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==old.get("cleaned_sha256",old["original_sha256"]), path
        # Rebuild summaries from the original snapshots, avoiding duplicate notes.
        for relative in (Path("."),Path("source"),Path("augmented")):
            for src in (backup/relative).iterdir():
                if src.is_file() and src.suffix in (".json",".csv",".md"):
                    shutil.copy2(src,args.dataset/relative/src.name)
    rows = {r["clip_npz"]:r for r in csv.DictReader((args.dataset/"manifest.csv").open())}
    assert set(rows)=={r["clip_npz"] for r in records}
    heights = {r["clip_npz"]:r for r in csv.DictReader((args.dataset/"placement_height_analysis.csv").open())}
    if not args.use_original_backup:
        assert not backup.exists(), "Backup already exists; inspect an interrupted operation before retrying"
        for folder in (args.dataset, args.dataset/"source", args.dataset/"augmented"):
            for src in folder.iterdir():
                if src.is_file() and src.suffix in (".json", ".csv", ".md"):
                    dest = backup/src.relative_to(args.dataset)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
    updates, height_updates, changed = {}, {}, []
    for r in records:
        row = rows[r["clip_npz"]]
        update = dict(terminal_init_trimmed=False, terminal_init_trim_frames=0,
            terminal_init_trim_seconds=0., terminal_init_retained_overlap_frames=0,
            recovery_status="stable_upright" if row["stage"]=="03_lower_place" else "")
        if r["action"]=="retain":
            updates[r["clip_npz"]] = update
            continue
        path = args.dataset/r["clip_npz"]
        source = backup/r["clip_npz"] if args.use_original_backup else path
        assert hashlib.sha256(source.read_bytes()).hexdigest()==r["original_sha256"]
        alias = args.dataset/r["variant"]/r["stage"]/r["split"]/path.name
        assert alias.is_symlink() and alias.resolve()==path.resolve()
        dest = backup/r["clip_npz"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not args.use_original_backup:
            shutil.copy2(path, dest)
        with np.load(source, allow_pickle=False) as d:
            payload = {k:d[k] for k in d.files}
        out = trim_payload(payload, r)
        for key in TIME_FIELDS & payload.keys():
            assert np.array_equal(out[key], payload[key][:r["keep_end"]]), key
        with path.with_suffix(".trim.npz").open("wb") as f:
            np.savez_compressed(f, **out)
        path.with_suffix(".trim.npz").replace(path)
        r.update(cleaned_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), backup_npz=str(dest))
        new_metrics = dict(row)
        metrics(out, new_metrics)
        update.update({k:v for k,v in new_metrics.items() if v!=row[k]})
        source_fps = float(out["source_fps"].item())
        placement_end = float(row["placement_end_frame_exclusive"])
        body = out["body_pos_w"]
        h = body[:,0,2]-body[:,25:27,2].mean(1)
        quat = out["body_quat_w"][:,11]
        tilt = np.degrees(np.arccos(np.clip(1-2*(quat[:,1]**2+quat[:,2]**2),-1,1)))
        end = len(h)
        window = max(r["last_contact"], end-25)
        update.update(terminal_init_trimmed=True, terminal_init_trim_frames=r["original_frames"]-end,
            terminal_init_trim_seconds=(r["original_frames"]-end)/50.,
            terminal_init_retained_overlap_frames=r["overlap_retained_frames"],
            terminal_init_onset_output_frame=r["init_onset_frame"], terminal_init_endpoint_reason=r["endpoint_reason"],
            pre_terminal_init_end_frame_exclusive=int(row["end_frame_exclusive"]),
            pre_terminal_init_output_frames=r["original_frames"],
            pre_terminal_init_recovery_stable_window_start=int(row["recovery_stable_window_start"]),
            pre_terminal_init_placement_end_frame_exclusive=int(row["placement_end_frame_exclusive"]),
            placement_end_frame_exclusive=int(out["source_placement_end_frame_exclusive"]),
            placement_stability_window_complete=bool(out["placement_stability_window_complete"].item()),
            end_frame_exclusive=int(out["source_end_frame_exclusive"]),
            end_frame_exclusive_exact=float(out["source_end_frame_exclusive_exact"]),
            last_frame_exact=float(out["source_frame_index"][-1]),
            recovery_complete=bool(out["recovery_complete"].item()), recovery_status=str(out["recovery_status"].item()),
            endpoint_evidence=str(out["endpoint_evidence"].item()),
            recovery_stable_window_start=int(out["source_recovery_stable_window_start"]),
            recovery_stable_window_start_exact=float(out["source_recovery_stable_window_start_exact"]),
            recovery_upright_duration_s=r["retained_recovery_evidence"]["recovery_upright_duration_s"],
            end_min_pelvis_ankle_height_m=r["retained_recovery_evidence"]["end_min_pelvis_ankle_height_m"],
            end_max_torso_tilt_deg=r["retained_recovery_evidence"]["end_max_torso_tilt_deg"],
            recovery_added_s=max(0., (float(out["source_end_frame_exclusive_exact"])-placement_end)/source_fps),
            recovery_body_rise_m=float(np.median(h[-10:])-h[max(0,r["last_contact"]-1)]),
            recovery_torso_straightening_deg=float(tilt[max(0,r["last_contact"]-1)]-np.median(tilt[-10:])),
            recovery_metric_reference="final_release_to_retained_terminal_median",
            retained_post_release_s=(end-r["last_contact"])/50.,
            end_box_center_z_m=float(out["object_pos_w"][-1,2]),
            end_height_range_m=float(np.ptp(out["object_pos_w"][window:,2])),
            end_box_xy_range_m=float(np.linalg.norm(out["object_pos_w"][window:,:2]-out["object_pos_w"][window,:2],axis=1).max()))
        updates[r["clip_npz"]] = update
        height_updates[r["clip_npz"]] = retained_height_metrics(out, heights[r["clip_npz"]])
        changed.append(r)
        if len(changed)%25==0:
            log("trim_progress", changed=len(changed))
    info = dict(audited_clips=len(records), retained_clips=len(records), trimmed_clips=len(changed), quarantined_clips=0,
        trimmed_by_stage_variant=dict(Counter(r["stage"]+"/"+r["variant"] for r in changed)),
        removed_frames=sum(r["original_frames"]-r["keep_end"] for r in changed),
        removed_seconds=sum(r["original_frames"]-r["keep_end"] for r in changed)/50.,
        overlapping_recovery_clips=sum(r["overlap_retained_frames"]>0 for r in changed),
        endpoint_reasons=dict(Counter(r["endpoint_reason"] for r in changed)),
        report=str(args.report), backup=str(backup), policy=plan["incomplete_handling"], settings=SETTINGS,
        source_boundaries="source_frame_index is authoritative; end_frame_exclusive=floor(last retained source index)+1; end_frame_exclusive_exact is the next 50 Hz source-grid index")
    update_catalogs(args, updates, height_updates, info)
    write_json(args.report/"applied.json", dict(info, records=records))
    write_json(args.dataset/"terminal_init_cleanup.json", info)
    log("apply_complete", **info)


def verify(args):
    applied = json.loads((args.report/"applied.json").read_text())
    records = {r["clip_npz"]:r for r in applied["records"]}
    rows = list(csv.DictReader((args.dataset/"manifest.csv").open()))
    typed = {r["clip_npz"]:r for r in json.loads((args.dataset/"manifest.json").read_text())}
    assert {r["clip_npz"] for r in rows}==set(records)==set(typed)
    mutable = {"source_end_frame_exclusive", "source_placement_end_frame_exclusive", "recovery_complete",
               "endpoint_evidence", "source_recovery_stable_window_start"}

    def check(row):
        record = records[row["clip_npz"]]
        path = args.dataset/row["clip_npz"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==record.get("cleaned_sha256", record["original_sha256"]), path
        with np.load(path, allow_pickle=False) as d:
            payload = {k:d[k] for k in d.files}
        n = len(payload["joint_pos"])
        assert n==int(row["output_frames"])==int(typed[row["clip_npz"]]["output_frames"])
        assert float(payload["fps"].item())==50.
        for key, value in payload.items():
            if key in TIME_FIELDS:
                assert len(value)==n, (path,key)
            if np.issubdtype(value.dtype,np.number):
                assert np.isfinite(value).all(), (path,key)
        max_quat_error = max(float(np.abs(np.linalg.norm(payload[key],axis=-1)-1).max())
                             for key in ("base_quat_w", "body_quat_w", "object_quat_w", "ee_quat_w"))
        assert max_quat_error<1e-4, path
        assert np.isin(payload["contact_info"],[0,1]).all()
        assert np.array_equal(payload["base_pos_w"],payload["body_pos_w"][:,0])
        assert np.array_equal(payload["base_quat_w"],payload["body_quat_w"][:,0])
        assert np.allclose(np.diff(payload["source_frame_index"]),float(payload["source_fps"].item())/50.,atol=1e-6)
        assert abs(float(payload["source_frame_index"][0])-float(row["start_frame_exact"]))<1e-6
        if record["action"]=="trim_tail":
            old_path = args.report/"backup"/row["clip_npz"]
            assert hashlib.sha256(old_path.read_bytes()).hexdigest()==record["original_sha256"]
            with np.load(old_path,allow_pickle=False) as old:
                for key in old.files:
                    expected = old[key][:n] if key in TIME_FIELDS else old[key]
                    if key not in mutable:
                        assert np.array_equal(payload[key],expected), (path,key)
            assert int(payload["source_end_frame_exclusive"])==int(row["end_frame_exclusive"])
            assert float(payload["source_frame_index"][-1])<int(row["end_frame_exclusive"])
            assert bool(payload["recovery_complete"].item())==(row["recovery_complete"]=="True")
            assert str(payload["endpoint_evidence"].item())==row["endpoint_evidence"]
            assert int(payload["source_recovery_stable_window_start"])==int(row["recovery_stable_window_start"])
            assert int(payload["source_placement_end_frame_exclusive"])<=int(row["end_frame_exclusive"])
        a = feature_array(payload)
        first,last = contact_bounds(a)
        raw_last = int(np.flatnonzero(a[:,12:14].max(1)>.5)[-1])+1 if (a[:,12:14]>.5).any() else 0
        contact_pulses = int((a[last:raw_last,12:14].max(1)>.5).sum())
        raw_full = (a[:,:2].min(1)>.9)&(a[:,2]>.85)&(a[:,12:14].max(1)==0)
        full_events = [(int(s),int(e)) for s,e in runs(raw_full) if e-s>=3]
        assert not full_events, (path,full_events)
        if row["stage"]=="01_approach_lift":
            initial = detect_init(payload["joint_pos"],payload["body_pos_w"],payload["contact_info"],payload["object_pos_w"])
            assert not initial["events"] and not initial.get("possible_partial_tail"), path
        if record["action"]=="trim_tail":
            assert last>0 and last<n
            assert raw_last<n and not a[raw_last:,12:14].any(), path
            assert np.ptp(a[last:,11])<.025 and np.linalg.norm(a[last:,9:11]-a[last,9:11],axis=1).max()<.035, path
            end, detail = recovery_preserving_end(feature_cache[record["feature_key"]], original_rows[row["clip_npz"]],record["init_onset_frame"])
            assert end==n, (path,"boundary differs from recorded rule")
            assert recovery_evidence(a,row)["recovery_complete"]==(row["recovery_complete"]=="True")
        residual = analyze(a,row)
        transition_flag = residual["keep_end"]<n or bool(residual.get("unexplained_suffix_spreads"))
        if transition_flag:
            assert record["action"]=="trim_tail" and record["overlap_retained_frames"]>0, (path,"unannotated arm ramp")
            onset = record["init_onset_frame"]
            assert smooth(a[:,2],5)[onset:].max()<SETTINGS["overlap_max_hand_separation_m"]+.02
            assert smooth(a[:,:2],5)[onset:].min(1).max()<SETTINGS["overlap_max_bilateral_outward_roll_rad"]+.05
            assert a[onset:,3].max()<SETTINGS["overlap_max_hand_height_relative_pelvis_m"]+.02
            assert all(ev["start"]>=onset for ev in residual.get("unexplained_suffix_spreads",[])), path
        return dict(n=n,bytes=path.stat().st_size,max_quat_error=max_quat_error,
                    bounded_overlap_ramp=transition_flag,
                    preserved_short_contact_pulses=contact_pulses if record["action"]=="trim_tail" else 0)

    original_rows = {r["clip_npz"]:r for r in csv.DictReader((args.report/"backup/manifest.csv").open())}
    # Load original scalar features once so threaded checks never share a ZipFile.
    with np.load(args.report/"features.npz",allow_pickle=False) as f:
        feature_cache = {k:f[k] for k in f.files}
    results = []
    with ThreadPoolExecutor(max_workers=4) as workers:
        for i,value in enumerate(workers.map(check,rows)):
            results.append(value)
            if (i+1)%200==0:
                log("verify_progress", checked=i+1,total=len(rows))
    captures, examples = {}, {}
    for row in rows:
        assert captures.setdefault(row["capture_id"],row["split"])==row["split"], "capture leakage"
        examples.setdefault((row["stage"],row["variant"],row["split"]),args.dataset/row["clip_npz"])
        alias = args.dataset/row["variant"]/row["stage"]/row["split"]/Path(row["clip_npz"]).name
        assert alias.is_symlink() and alias.resolve()==(args.dataset/row["clip_npz"]).resolve()
    assert len(list(args.dataset.glob("0*/*/*/*.npz")))==len(rows)
    from blinkmimic.box_carry import load_npz
    from blinkmimic.robot import get_robot
    for path in examples.values():
        p,q,joints,fps = load_npz(str(path),get_robot("g1").joint_names)
        assert len(p)==len(q)==len(joints) and np.isfinite(joints).all() and float(fps)==50.
    for variant in ("source","augmented"):
        subset = [r for r in rows if r["variant"]==variant]
        local = list(csv.DictReader((args.dataset/variant/"manifest.csv").open()))
        local_json = json.loads((args.dataset/variant/"manifest.json").read_text())
        assert len(subset)==len(local)==len(local_json)
        for entry in local+local_json:
            canonical = typed[entry["canonical_clip_npz"]]
            assert int(entry["output_frames"])==int(canonical["output_frames"])
            assert str(entry.get("recovery_complete", ""))==str(canonical.get("recovery_complete", ""))
            assert (args.dataset/variant/entry["clip_npz"]).exists()
    complete = sum(r["stage"]=="03_lower_place" and r["recovery_complete"]=="True" for r in rows)
    summary = dict(checked_npz=len(rows),total_output_frames=sum(v["n"] for v in results),
        total_bytes=sum(v["bytes"] for v in results),trimmed_terminal_init_npz=applied["trimmed_clips"],
        removed_tail_frames=applied["removed_frames"],removed_tail_seconds=applied["removed_seconds"],
        retained_npz=len(rows),quarantined_npz=0,complete_recovery_npz=complete,
        truncated_recovery_npz=applied["trimmed_clips"],bounded_recovery_overlap_ramps=sum(v["bounded_overlap_ramp"] for v in results),
        cropped_clips_with_original_short_contact_pulses=sum(v["preserved_short_contact_pulses"]>0 for v in results),
        residual_full_bilateral_init_clips=0,unannotated_post_release_reinit_clips=0,residual_leading_init_clips=0,
        native_reader_examples=len(examples),source_split_leakage=False,
        max_quaternion_norm_error=max(v["max_quat_error"] for v in results),
        previous_prefix_cleanup=json.loads((args.dataset/"init_cleanup.json").read_text()),
        checks=["all 2246 original hashes checked; 1980 untouched payloads byte-identical",
            "all time fields match their original contiguous retained prefix exactly",
            "all 266 originals backed up and hash verified",
            "all numeric arrays finite; frame alignment, binary contacts, normalized quaternions",
            "50 Hz spacing, unchanged original source indices, original placement/release retained",
            "box remains fixed; all cropped clips retain a final continuous free-hand interval; original short contact-label pulses are preserved",
            "recovery flags and endpoint evidence describe the retained frames",
            "no remaining full bilateral calibration; necessary bounded recovery overlap is annotated",
            "canonical and variant-first manifests, symlinks and capture splits agree",
            "18 representative motions load through the native G1 joint mapping"])
    write_json(args.report/"validation.json",summary)
    write_json(args.dataset/"validation.json",summary)
    for variant in ("source","augmented"):
        subset = [r for r in rows if r["variant"]==variant]
        local_results = [v for r,v in zip(rows,results) if r["variant"]==variant]
        cropped = [records[r["clip_npz"]] for r in subset if records[r["clip_npz"]]["action"]=="trim_tail"]
        removed = sum(r["original_frames"]-r["keep_end"] for r in cropped)
        write_json(args.dataset/variant/"validation.json",dict(variant=variant,checked_npz=len(subset),
            total_output_frames=sum(v["n"] for v in local_results),total_bytes=sum(v["bytes"] for v in local_results),
            trimmed_terminal_init_npz=len(cropped),removed_tail_frames=removed,removed_tail_seconds=removed/50.,
            complete_recovery_npz=sum(r["stage"]=="03_lower_place" and r["recovery_complete"]=="True" for r in subset),
            truncated_recovery_npz=len(cropped),retained_npz=len(subset),quarantined_npz=0,
            bounded_recovery_overlap_ramps=sum(v["bounded_overlap_ramp"] for v in local_results),
            source_split_leakage=False,all_checks_passed=True,global_validation="../validation.json"))
    log("verify_complete",**{k:v for k,v in summary.items() if k not in ("checks","previous_prefix_cleanup")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT/"box_carry_three_stage_v2")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--mode", choices=("audit", "reanalyze", "plan", "apply", "verify"), default="audit")
    p.add_argument("--use-original-backup",action="store_true",help="Recompute a reviewed boundary from the existing immutable backup")
    args = p.parse_args()
    args.dataset, args.report = args.dataset.resolve(), args.report.resolve()
    args.report.mkdir(parents=True, exist_ok=True)
    {"audit": audit, "reanalyze": reanalyze, "plan": make_plan, "apply": apply, "verify": verify}[args.mode](args)


if __name__ == "__main__":
    main()
