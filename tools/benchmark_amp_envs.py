"""Measure the complete carry training loop under one foreground trainreg job."""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", required=True)
parser.add_argument("--motion_dir", required=True)
parser.add_argument("--envs", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
parser.add_argument("--iterations", type=int, default=120)
parser.add_argument("--cuda_graph", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
os.chdir(root)
run = Path(args.run_dir).resolve()
run.mkdir(parents=True, exist_ok=True)
results = []
for candidate_index, num_envs in enumerate(args.envs):
    directory = run / f"envs_{num_envs}"
    if (directory / "args.json").exists():
        parser.error(f"benchmark output already exists: {directory}")
    command = ["bash", "scripts/train.sh", str(directory), "--motion_dir", args.motion_dir,
               "--num_envs", str(num_envs), "--seed", "0", "--iterations", str(args.iterations),
               "--wm_only_iters", "20", "--profile", "--log_every", "10",
               "--ckpt_light", "--ckpt_every", "10000000", "--ckpt_minutes", "0"]
    if args.cuda_graph:
        command += ["--cuda_graph", "--graph_early"]
    print(f"benchmark: {num_envs} environments, {args.iterations} iterations including actor/critic", flush=True)
    started = time.time()
    code = subprocess.call(command)
    result = dict(num_envs=num_envs, exit_code=code, elapsed_s=time.time() - started, command=command)
    if code == 0:
        with (directory / "diagnostics.csv").open() as handle:
            # The last 40 iterations run after compilation, actor warmup, and buffer filling.
            rows = list(csv.DictReader(handle))[-4:]
        with (directory / "train_log.csv").open() as handle:
            train = list(csv.DictReader(handle))[-4:]
        for key in ("env_steps_per_s", "collect_ms", "wm_ms", "policy_ms", "peak_allocated_gb", "device_used_gb"):
            result[key] = statistics.median(float(row[key]) for row in rows)
        result["measured_iterations"] = [int(row["iteration"]) for row in rows]
        result["nonfinite"] = max(int(row["nonfinite"]) for row in train)
        result["pi_gradnorm"] = statistics.median(float(row["pi_gradnorm"]) for row in train)
    else:
        output = (directory / "stdout.log").read_text(errors="replace") if (directory / "stdout.log").exists() else ""
        result["oom"] = "out of memory" in output.lower()
        result["error_tail"] = output[-2000:]
    results.append(result)
    successful = [r for r in results if r["exit_code"] == 0]
    best = max(successful, key=lambda r: r["env_steps_per_s"]) if successful else None
    (run / "benchmark.json").write_text(json.dumps(dict(results=results, best=best), indent=2) + "\n")
    print("benchmark result: " + json.dumps(result), flush=True)
    if code != 0:
        if not result["oom"]:
            sys.exit(code)
        # Descending sweeps continue with smaller buffers after an OOM.
        if all(n >= num_envs for n in args.envs[candidate_index + 1:]):
            break
print("benchmark complete: " + json.dumps(best), flush=True)
