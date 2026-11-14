"""Evaluation identities and caches shared by the single and batch evaluators."""

import csv
import hashlib
import json
import os
from pathlib import Path
import time


def evaluation_protocol(args, run_args, motion, *, evaluator, motion_dir):
    from .motion import goal_frame_offsets
    from .networks import check_pruned
    check_pruned(run_args, nets_only=True)
    if not hasattr(motion, "evaluation_fingerprint"):
        digest = hashlib.sha256()
        for file in getattr(motion, "files", []):
            digest.update(os.path.realpath(file).encode())
            with open(file, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        motion.evaluation_fingerprint = digest.hexdigest() if getattr(motion, "files", []) else None
    options = vars(args)
    return {
        "version": 6,
        "action_clip": 20.0,
        "gait_metrics_version": 2,
        "goal_frame_offsets": list(goal_frame_offsets(run_args)),
        "dataset_fingerprint": run_args.get("dataset_fingerprint"),
        "evaluation_data_fingerprint": motion.evaluation_fingerprint,
        "evaluator": evaluator,
        "robot": args.robot,
        "task": run_args.get("task", "tracking"),
        "motion": {"directory": os.path.realpath(motion_dir), "names": list(motion.names),
                   "lengths": motion.clip_len.tolist(), "frames": motion.num_frames},
        "split": options.get("split", "test"),
        "stages": sorted(args.stages.split(",")) if options.get("stages") else None,
        "max_clips": options.get("max_clips"),
        "repeats": options.get("repeats", options.get("reps", 1)),
        "seed": options.get("seed", 0),
        "dr_from": options.get("dr_from"),
        "dr_parts": options.get("dr_parts", "all"),
        "group": options.get("group", 1),
        "policy": {key: run_args.get(key, default) for key, default in (
            ("algorithm", "bptt"), ("amp", "none"), ("vision", "none"),
            ("goal_frames", 10), ("hist", 10), ("pi_ref_skip", False))},
        "reward": {key: run_args.get(key, default) for key, default in (
            ("carry_reward_profile", "legacy"), ("carry_residual_rate", 0.),
            ("carry_joint_vel", 0.), ("carry_residual_acc", 0.), ("carry_residual_acc_arms", 1.),
            ("carry_gait_profile", "legacy"), ("carry_gait_scale", 1.))},
    }


def protocol_key(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def result_path(checkpoint, protocol):
    return os.path.splitext(checkpoint)[0] + f"_eval_{protocol_key(protocol)}.json"


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class EvaluationCache:
    """Reuse results only when the protocol and actual evaluated clips match.

    New runs retain eval.csv/eval_rN.csv. Existing files without a matching
    protocol sidecar are preserved; the new protocol gets its own suffix.
    """

    header = "iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n"

    def __init__(self, run_dir, protocol):
        self.protocol = protocol
        root = Path(run_dir)
        repeats = protocol["repeats"]
        stem = "eval" if repeats == 1 else f"eval_r{repeats}"
        csv_path, meta = root / f"{stem}.csv", root / f"{stem}.protocol.json"
        if (csv_path.exists() or meta.exists()) and _read_json(meta) != protocol:
            stem += "_" + protocol_key(protocol)
            csv_path, meta = root / f"{stem}.csv", root / f"{stem}.protocol.json"
        self.csv_path, self.stem = csv_path, stem
        root.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(protocol, indent=2) + "\n")
        if not csv_path.exists():
            csv_path.write_text(self.header)

    def result_path(self, checkpoint):
        return Path(os.path.splitext(checkpoint)[0] + f"_{self.stem}.json")

    def completed(self):
        done = set()
        with self.csv_path.open() as stream:
            for row in csv.DictReader(stream):
                iteration = int(row["iteration"])
                result = _read_json(self.result_path(self.csv_path.parent / f"ckpt_{iteration:06d}.pt"))
                if (result is not None and result.get("evaluation_protocol") == self.protocol
                        and set(result.get("per_clip", {})) == set(self.protocol["motion"]["names"])
                        and all(float(row[key]) == round(result[key], 6)
                                for key in ("norm_return", "norm_return_framew", "n_failed"))):
                    done.add(iteration)
        return done

    def record(self, checkpoint, checkpoint_state, result):
        result["evaluation_protocol"] = self.protocol
        path = self.result_path(checkpoint)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=1) + "\n")
        temporary.replace(path)
        iteration = int(Path(checkpoint).stem[5:])
        row = dict(iteration=str(iteration), train_wall_h=f"{checkpoint_state['wall_time'] / 3600:.6f}",
                   saved_at=checkpoint_state["saved_at"], eval_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   norm_return=f"{result['norm_return']:.6f}", norm_return_framew=f"{result['norm_return_framew']:.6f}",
                   n_failed=f"{result['n_failed']:.6f}", failed="|".join(result["failed"]))
        with self.csv_path.open() as stream:
            rows = [old for old in csv.DictReader(stream) if int(old["iteration"]) != iteration]
        rows.append(row)
        # Re-evaluating an incomplete result replaces its CSV row.
        temporary = self.csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.header.strip().split(","))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda entry: int(entry["iteration"])))
        temporary.replace(self.csv_path)
