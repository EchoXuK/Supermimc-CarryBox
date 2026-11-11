"""Foreground trainreg child: train and periodically validate on the same GPU."""

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--hours", type=float, default=2.)
parser.add_argument("--eval_minutes", type=float, default=10.)
parser.add_argument("train_args", nargs=argparse.REMAINDER)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
os.chdir(root)
run = Path(args.run_dir).resolve()
run.mkdir(parents=True, exist_ok=True)
if (run / "args.json").exists():
    parser.error("use a new directory for training from scratch")
extras = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
command = ["bash", "scripts/train.sh", str(run), "--seed", str(args.seed),
           "--max_hours", str(args.hours), "--iterations", "10000000", "--profile",
           "--ckpt_light", "--ckpt_initial", "--ckpt_every", "10000000", "--ckpt_minutes", "5",
           "--log_every", "25", *extras]
(run / "launch.json").write_text(json.dumps(dict(command=command, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                               launched_at=time.time()), indent=2) + "\n")
train = subprocess.Popen(command)
evaluation = None
evaluation_file = None
evaluation_ckpt = None
evaluated = set()
next_eval = 0.
stop = False


def request_stop(signum, frame):
    global stop
    stop = True
    if train.poll() is None:
        # train.sh contains the actual trainer, whose owned PID file is explicit.
        pid_file = run / "train_pid.json"
        if pid_file.exists():
            try:
                os.kill(json.loads(pid_file.read_text())["pid"], signal.SIGUSR1)
            except ProcessLookupError:
                pass


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
print(f"monitor: training seed {args.seed}; log directory {run}", flush=True)
while True:
    code = train.poll()
    if evaluation is not None and evaluation.poll() is not None:
        result_code = evaluation.returncode
        evaluation_file.close()
        evaluated.add(evaluation_ckpt.name)
        print(f"monitor: evaluation {evaluation_ckpt.name} exit={result_code}", flush=True)
        evaluation = None
    checkpoints = sorted(run.glob("ckpt_*.pt"))
    candidates = [p for p in checkpoints if p.name not in evaluated]
    if evaluation is None and candidates and not stop and (code is not None or time.time() >= next_eval):
        # First evaluate the untrained baseline, then the latest available policy.
        evaluation_ckpt = candidates[0] if "ckpt_000000.pt" not in evaluated else candidates[-1]
        evaluated.update(p.name for p in candidates if p < evaluation_ckpt)
        output = evaluation_ckpt.with_name(evaluation_ckpt.stem + "_val.json")
        evaluation_file = output.with_suffix(".log").open("w")
        evaluation = subprocess.Popen([sys.executable, "-u", "-m", "blinkmimic.evaluate", "--run_dir", str(run),
                                       "--ckpt", str(evaluation_ckpt), "--split", "val", "--output", str(output)],
                                      stdout=evaluation_file, stderr=subprocess.STDOUT)
        next_eval = time.time() + args.eval_minutes * 60
        print(f"monitor: validating {evaluation_ckpt.name} on all 146 val clips", flush=True)
    status = dict(updated_at=time.time(), training_exit_code=code,
                  evaluation_pid=evaluation.pid if evaluation is not None else None,
                  evaluation_checkpoint=evaluation_ckpt.name if evaluation is not None else None)
    for filename, key in (("train_log.csv", "training"), ("diagnostics.csv", "diagnostics")):
        path = run / filename
        if path.exists():
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                status[key] = rows[-1]
    (run / "monitor.json").write_text(json.dumps(status, indent=2) + "\n")
    if code is not None and evaluation is None:
        # The final checkpoint must be evaluated even if it arrives between polls.
        if stop or not any(p.name not in evaluated for p in run.glob("ckpt_*.pt")):
            break
    time.sleep(5)
sys.exit(code)
