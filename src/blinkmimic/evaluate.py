"""Checkpoint evaluation (SuperMimic paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.

Run as a watcher: ``python -m blinkmimic.evaluate --run_dir logs/<run> --watch``.

Robustness: ``--dr_from <robot>`` evaluates under that robot's training randomization and sensing
(``self`` for the evaluated robot's own), still one start per clip from its first frame;
``--repeats n`` runs every clip n times to average the randomness.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="g1", help="robot package under blinkmimic/ (g1)")
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
parser.add_argument("--output", type=str, default=None, help="write the complete evaluation result as JSON")
parser.add_argument("--motion_dir", type=str, default=None, help="defaults to the robot's motion directory")
parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="box_carry evaluation split")
parser.add_argument("--max_clips", type=int, default=None)
parser.add_argument("--stages", type=str, default=None)
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
parser.add_argument("--dr_from", type=str, default=None, help="evaluate under this robot's training randomization ('self': own)")
parser.add_argument("--dr_parts", type=str, default="all", choices=["all", "dynamics", "sensing"],
                    help="with --dr_from: take only its simulator randomization or only its sensing")
parser.add_argument("--repeats", type=int, default=1, help="rollouts per clip")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import dataclasses
import re

import numpy as np
import torch

import blinkmimic.networks as networks
from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.eval_protocol import EvaluationCache, evaluation_protocol, result_path
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MAX_STEP_REWARD, MDP
from blinkmimic.motion import MotionLib
from blinkmimic.networks import check_pruned, make_policy
from blinkmimic.robot import RobotSpec, get_robot
from blinkmimic.box_carry import build_task, robot_spec, STAGES


def transfer_dr(target: RobotSpec, source: RobotSpec, parts: str = "all") -> RobotSpec:
    """``source``'s randomization and/or sensing on ``target``. PD-gain ranges are given per actuator group,
    so each target group takes the range of the source entry that covers most of its joints."""
    if parts == "sensing":
        return dataclasses.replace(target, sensing=source.sensing)
    if parts == "dynamics":
        return dataclasses.replace(transfer_dr(target, source), sensing=target.sensing)
    if target.name == source.name:
        return target
    src_acts = source.robot_cfg().articulation.actuators
    tgt_acts = target.robot_cfg().articulation.actuators
    joints = lambda act, names: {n for n in names if any(re.fullmatch(p, n) for p in act.target_names_expr)}  # noqa: E731
    entries = [(set().union(*(joints(src_acts[i], source.joint_names) for i in ids)), kp, kd) for ids, kp, kd in source.dr.pd_gain_scale]
    groups: dict[int, list[int]] = {}
    for i, act in enumerate(tgt_acts):
        tj = joints(act, target.joint_names)
        best = max(range(len(entries)), key=lambda e: len(tj & entries[e][0]))
        groups.setdefault(best, []).append(i)
    pd = tuple((tuple(ids), entries[e][1], entries[e][2]) for e, ids in sorted(groups.items()))
    dr = dataclasses.replace(source.dr, pd_gain_scale=pd)
    return dataclasses.replace(target, dr=dr, sensing=source.sensing, com_range=source.com_range)


torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(args.seed)
dev = "cuda:0"
run_args = (torch.load(args.ckpt, map_location="cpu", weights_only=False)["args"] if args.ckpt else
            json.load(open(os.path.join(args.run_dir, "args.json"))))
check_pruned(run_args, nets_only=True, what=args.ckpt or args.run_dir)
TASK = run_args.get("task", "tracking")
VISION = run_args.get("vision", "none") == "latent"
robot = get_robot(args.robot)
if TASK == "box_carry":
    robot = robot_spec(robot)
if args.dr_from:
    robot = transfer_dr(robot, robot if args.dr_from == "self" else get_robot(args.dr_from), args.dr_parts)
    print("pd_gain_scale on this robot:", robot.dr.pd_gain_scale)
kin = Kinematics(robot, dev)
motion, mdp = build_task(TASK, robot, args.motion_dir or run_args.get("motion_dir"), kin, dev,
                         XiLayout(robot, dev).dim, args.split, args.stages.split(",") if args.stages else None, args.max_clips,
                         reward_options=run_args)
C = len(motion.names)
N = C * args.repeats
clip_of = torch.arange(N, device=dev) % C
PROTOCOL = evaluation_protocol(args, run_args, motion, evaluator="single",
                               motion_dir=args.motion_dir or run_args.get("motion_dir") or robot.motion_dir)


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    from blinkmimic.evaluation import evaluate_policy

    a_ = ck["args"]
    motion.configure_goal_sampling(a_)
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    pi = make_policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), robot.num_joints, robot, a_).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    return evaluate_policy(env, pi, motion, mdp, robot, a_, repeats=args.repeats,
                           split=args.split, protocol=PROTOCOL, record=record)


sim_cfg = make_sim(robot)
env = TrackingSim(sim_cfg, N, robot, motion, mdp, dev, eval_mode=True, randomize=args.dr_from is not None, vision=VISION)
if args.ckpt:
    ck = torch.load(args.ckpt, map_location=dev)
    res = evaluate(env, ck, args.record)
    if args.output:
        json.dump(res, open(args.output, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_clip", "evaluation_protocol")}, indent=1))
    if args.record is None and (args.dr_from or args.repeats > 1):
        path = result_path(args.ckpt, PROTOCOL)
        json.dump(res, open(path, "w"), indent=1)
        print(f"evaluation result: {path}", flush=True)
    sys.exit(0)
cache = EvaluationCache(args.run_dir, PROTOCOL)
out_csv = cache.csv_path
done = cache.completed()
print(f"evaluation results: {out_csv}", flush=True)
while True:
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))[:: args.every]
    todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
    for c in todo:
        it = int(os.path.basename(c)[5:11])
        ck = torch.load(c, map_location=dev)
        t0 = time.time()
        res = evaluate(env, ck)
        cache.record(c, ck, res)
        print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)", flush=True)
        done.add(it)
    if not args.watch:
        break
    time.sleep(60)
