"""Evaluate many checkpoints of a run at once (same protocol as blinkmimic.evaluate, App. F.1).

Each checkpoint gets its own block of C worlds (one per clip) in a single simulator, so M checkpoints
cost about as much wall-clock as one: evaluation is latency-bound at C = 40 worlds. Results are
written to ``<run_dir>/eval.csv`` (``eval_r<reps>.csv`` with --reps) in the format of ``blinkmimic.evaluate``.
Different evaluation protocols get separate suffixes; the selected path is printed.

    python tools/eval_batch.py --run_dir logs/<run> --group 26 --reps 4
"""

import argparse
import glob
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--robot", type=str, default="g1")
parser.add_argument("--group", type=int, default=13, help="checkpoints evaluated together")
parser.add_argument("--reps", type=int, default=1,
                    help="independent replicas of every clip per checkpoint (the simulator is not bitwise "
                         "reproducible, so borderline clips can flip); results are averaged in a protocol-specific CSV")
parser.add_argument("--split", choices=["train", "val", "test"], default="test")
parser.add_argument("--max_clips", type=int, default=None)
parser.add_argument("--stages", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import torch

import blinkmimic.networks as networks
from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.eval_protocol import EvaluationCache, evaluation_protocol
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MAX_STEP_REWARD, MDP
from blinkmimic.motion import MotionLib, goal_frame_offsets
from blinkmimic.networks import check_pruned, make_policy
from blinkmimic.robot import get_robot
from blinkmimic.box_carry import build_task, build_mdp, robot_spec

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(args.seed)
dev = "cuda:0"
_run_args = json.load(open(os.path.join(args.run_dir, "args.json"))) if os.path.exists(os.path.join(args.run_dir, "args.json")) else {}
TASK, VISION = _run_args.get("task", "tracking"), _run_args.get("vision") == "latent"
robot = get_robot(args.robot)
if TASK == "box_carry":
    robot = robot_spec(robot)
kin = Kinematics(robot, dev)
motion, mdp = build_task(TASK, robot, _run_args.get("motion_dir"), kin, dev, XiLayout(robot, dev).dim, args.split,
                         stages=args.stages.split(",") if args.stages else None, max_clips=args.max_clips,
                         reward_options=_run_args)
check_pruned(_run_args, nets_only=True, what=args.run_dir)
C = len(motion.names)


def load_policy(ck):
    a_ = ck["args"]
    pi = make_policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), robot.num_joints, robot, a_).to(dev)
    pi.load_state_dict(ck["pi"])
    return pi.eval()


@torch.no_grad()
def evaluate_group(env: TrackingSim, cks: list[dict]) -> list[dict]:
    M = len(cks)
    N = env.N
    a0 = cks[0]["args"]
    if any(goal_frame_offsets(c["args"]) != goal_frame_offsets(a0) for c in cks):
        raise ValueError("Batch reference sampling must match")
    motion.configure_goal_sampling(a0)
    pis = [load_policy(ck) for ck in cks]
    K, Hh = a0["goal_frames"], a0["hist"]
    assert all(ck["args"]["goal_frames"] == K and ck["args"]["hist"] == Hh for ck in cks)
    networks.AMP_DTYPE = torch.bfloat16 if a0.get("amp", "none") == "bf16" else None
    clip_of = torch.arange(N, device=dev) % C
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(torch.arange(N, device=dev), motion.clip_start[clip_of].clone())
    clip_end = motion.clip_end[clip_of]
    ret = torch.zeros(N, device=dev)
    hist = None
    a = torch.zeros(N, robot.num_joints, device=dev)
    while not bool(env.done_eval.all()):
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        g = motion.goal(env.f, K)
        for m, pi in enumerate(pis):
            sl = slice(m * C, (m + 1) * C)
            a[sl] = pi(hist[sl], g[sl], depth=env.depth[sl] if VISION else None)
        alive = ~env.done_eval.clone()
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
    clip_len = motion.clip_len[clip_of]
    norm = ret / ((clip_len - 1).float() * mdp.max_step_reward)
    res = []
    for m in range(M):
        sl = slice(m * C, (m + 1) * C)
        failed = env.failed_eval[sl]
        res.append({
            "norm_return": norm[sl].mean().item(),
            "norm_return_framew": (ret[sl].sum() / ((clip_len[sl] - 1).sum() * mdp.max_step_reward)).item(),
            "n_failed": int(failed.sum().item()),
            "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
            "per_clip": {motion.names[i]: round(norm[m * C + i].item(), 4) for i in range(C)},
        })  # fmt: skip
    return res


def merge_reps(res: list[dict]) -> list[dict]:
    """Average every ``args.reps`` consecutive results (replicas of one checkpoint); n_failed becomes the mean
    number of failed clips per replica and ``failed`` lists every clip that failed in at least one replica."""
    R = args.reps
    if R == 1:
        return res
    out = []
    for k in range(0, len(res), R):
        rs = res[k : k + R]
        out.append({
            "norm_return": sum(r["norm_return"] for r in rs) / R,
            "norm_return_framew": sum(r["norm_return_framew"] for r in rs) / R,
            "n_failed": sum(r["n_failed"] for r in rs) / R,
            "failed": sorted({n for r in rs for n in r["failed"]}),
            "per_clip": {n: round(sum(r["per_clip"][n] for r in rs) / R, 4) for n in rs[0]["per_clip"]},
            "reps": [{"norm_return": r["norm_return"], "n_failed": r["n_failed"], "failed": r["failed"]} for r in rs],
        })  # fmt: skip
    return out


# Mixed old/new preview checkpoints must never share a precomputed goal tensor or cache.
groups = {}
for path in sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt"))):
    options = torch.load(path, map_location="cpu", weights_only=False)["args"]
    check_pruned(options, nets_only=True, what=path)
    identity = (goal_frame_offsets(options), options["goal_frames"], options["hist"],
                options.get("amp", "none"),
                options.get("carry_reward_profile", "legacy"), options.get("carry_gait_profile", "legacy"),
                options.get("carry_gait_scale", 1.))
    groups.setdefault(identity, []).append(path)
for paths_all in groups.values():
    options = torch.load(paths_all[0], map_location="cpu", weights_only=False)["args"]
    motion.configure_goal_sampling(options)
    mdp = build_mdp(TASK, robot, kin, motion, dev, XiLayout(robot, dev).dim, options)
    PROTOCOL = evaluation_protocol(args, options, motion, evaluator="batch",
                                   motion_dir=options.get("motion_dir") or robot.motion_dir)
    cache = EvaluationCache(args.run_dir, PROTOCOL)
    done = cache.completed()
    todo = [p for p in paths_all if int(os.path.basename(p)[5:11]) not in done]
    print(f"{len(todo)} checkpoints, offsets={goal_frame_offsets(options)}, results={cache.csv_path}", flush=True)
    MOTION_ID = {"clips": C, "frames": motion.num_frames, "names": motion.names}
    env = None
    for s in range(0, len(todo), args.group):
        paths = todo[s:s + args.group]
        cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
        if env is None or env.N != C * len(paths) * args.reps:
            env = TrackingSim(make_sim(robot), C * len(paths) * args.reps, robot, motion, mdp, dev,
                              eval_mode=True, randomize=False, vision=VISION)
        t0 = time.time()
        for p, ck, res in zip(paths, cks, merge_reps(evaluate_group(env, [c for c in cks for _ in range(args.reps)]))):
            it = int(os.path.basename(p)[5:11])
            res["layout"] = {"n_worlds": env.N, "group": [int(os.path.basename(q)[5:11]) for q in paths],
                             "reps": args.reps, "motion": MOTION_ID}
            cache.record(p, ck, res)
            print(f"[eval] it {it}: return {res['norm_return']:.4f}, failed {res['n_failed']}/{C}", flush=True)
        print(f"group of {len(paths)} in {time.time() - t0:.0f}s", flush=True)
