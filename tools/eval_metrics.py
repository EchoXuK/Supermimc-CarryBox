"""Tracking-accuracy and contact metrics for a set of checkpoints (same rollout protocol as the evaluator:
each clip once from its first frame, friction 1.0, no randomization or noise, deterministic policy).

Per checkpoint, averaged over all alive control steps of the selected clips:
  body_pos_cm   RMS error of the 14 tracked bodies (reference re-anchored to the robot root, as in the reward)
  body_ori_deg  RMS orientation error of the tracked bodies
  root_pos_cm   global root position error (drift)
  joint_deg     mean absolute joint-angle error
  slip_cm_s     horizontal foot speed while the simulated foot is in contact (ankle link < CONTACT_Z)
  ref_slip_cm_s the same on the reference motion (retargeting artifacts set a floor)
  contact_acc   fraction of foot-frames whose contact state matches the reference
  pen_mm        mean ankle-height deficit below the flat-foot height (ground penetration proxy)
plus the checkpoint's normalized return, fixed original task return, carry
shaping costs and failed clips. Each checkpoint uses its own reward settings.

    uv run python tools/eval_metrics.py logs/<run_a>/ckpt_003500.pt logs/<run_b>/ckpt_003500.pt --out results/metrics
"""

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("ckpts", nargs="+")
parser.add_argument("--robot", type=str, default="g1")
parser.add_argument("--out", type=str, default="results/metrics")
parser.add_argument("--split", choices=["train", "val", "test"], default="test")
parser.add_argument("--max_clips", type=int, default=None)
parser.add_argument("--stages", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import torch

import blinkmimic.networks as networks
from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MAX_STEP_REWARD, MDP
from blinkmimic.motion import MotionLib
from blinkmimic.networks import check_pruned, make_policy
from blinkmimic.robot import get_robot
from blinkmimic.box_carry import build_mdp, build_task, robot_spec

CONTACT_Z = 0.045  # ankle-roll link origin height below which a foot counts as in contact (flat foot: ~0.035 m)
FLAT_Z = 0.033

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(args.seed)
dev = "cuda:0"
run_args = torch.load(args.ckpts[0], map_location="cpu", weights_only=False)["args"]
check_pruned(run_args, nets_only=True, what=args.ckpts[0])
TASK, VISION = run_args.get("task", "tracking"), run_args.get("vision") == "latent"
robot = get_robot(args.robot)
if TASK == "box_carry":
    robot = robot_spec(robot)
kin = Kinematics(robot, dev)
motion, mdp = build_task(TASK, robot, run_args.get("motion_dir"), kin, dev, XiLayout(robot, dev).dim, args.split,
                         stages=args.stages.split(",") if args.stages else None, max_clips=args.max_clips)
C = len(motion.names)
FEET = [robot.tracked_bodies.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]


def load_policy(ck):
    a_ = ck["args"]
    pi = make_policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), robot.num_joints, robot, a_).to(dev)
    pi.load_state_dict(ck["pi"])
    return pi.eval()


@torch.no_grad()
def run(cks):
    M = len(cks)
    N = C * M
    for ck in cks:
        for key, default in (("task", "tracking"), ("vision", "none"), ("goal_frames", 10), ("hist", 10), ("motion_dir", None)):
            if ck["args"].get(key, default) != cks[0]["args"].get(key, default):
                raise ValueError(f"Shared evaluation requires matching {key}")
    from blinkmimic.motion import goal_frame_offsets
    if any(goal_frame_offsets(ck["args"]) != goal_frame_offsets(cks[0]["args"]) for ck in cks):
        raise ValueError("Evaluate different goal_frame_offsets in separate groups")
    motion.configure_goal_sampling(cks[0]["args"])
    scoring_mdps = [build_mdp(TASK, robot, kin, motion, dev, XiLayout(robot, dev).dim, ck["args"]) for ck in cks]
    env = TrackingSim(make_sim(robot), N, robot, motion, mdp, dev, eval_mode=True, randomize=False, vision=VISION)
    pis = [load_policy(ck) for ck in cks]
    K, Hh = cks[0]["args"]["goal_frames"], cks[0]["args"]["hist"]
    clip_of = torch.arange(N, device=dev) % C
    env.reset(torch.arange(N, device=dev), motion.clip_start[clip_of].clone())
    clip_end = motion.clip_end[clip_of]
    keys = ("n", "ret", "task_ret", "grasp", "drop", "place", "pos", "ori", "root", "joint", "slip", "slip_n", "ref_slip", "ref_slip_n", "c_ok", "c_n", "pen")
    acc = {k: torch.zeros(N, device=dev) for k in keys}
    hist, a = None, torch.zeros(N, robot.num_joints, device=dev)
    while not bool(env.done_eval.all()):
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        g = motion.goal(env.f, K)
        for m, pi in enumerate(pis):
            networks.AMP_DTYPE = torch.bfloat16 if cks[m]["args"].get("amp", "none") == "bf16" else None
            sl = slice(m * C, (m + 1) * C)
            a[sl] = pi(hist[sl], g[sl], depth=env.depth[sl] if VISION else None)
        alive = ~env.done_eval.clone()
        previous2 = env.a_prev2.clone() if TASK == "box_carry" else None
        f1 = env.f + 1
        out = env.step(a)
        S1 = env.state()
        t = out["terms"]
        bp, _, bv, _ = kin.forward(*S1[:6])
        fz, fv = bp[:, FEET, 2], bv[:, FEET, :2].norm(dim=-1)
        rz, rv = motion.body_p[f1][:, FEET, 2], motion.body_v[f1][:, FEET, :2].norm(dim=-1)
        c_sim, c_ref = fz < CONTACT_Z, rz < CONTACT_Z
        acc["n"] += alive
        for m, scoring in enumerate(scoring_mdps):
            sl = slice(m * C, (m + 1) * C)
            reward_options = (dict(smooth_valid=out["ep_step"][sl] > 0, a_prev2=previous2[sl],
                                   smooth_acc_valid=out["ep_step"][sl] > 1) if TASK == "box_carry" else {})
            reward, detail = scoring.reward(tuple(x[sl] for x in S1), f1[sl], out["a"][sl], out["a_prev"][sl],
                                            terms=True, **reward_options)
            acc["ret"][sl] += torch.where(alive[sl], reward, 0.)
            acc["task_ret"][sl] += torch.where(alive[sl], detail.get("task_reward", reward), 0.)
            if TASK == "box_carry":
                for key in ("grasp", "drop", "place"):
                    acc[key][sl] += torch.where(alive[sl], detail[f"{key}_penalty"], 0.)
        acc["pos"] += torch.where(alive, t["e_body_pos"], 0.)
        acc["ori"] += torch.where(alive, t["e_body_ori"], 0.)
        acc["root"] += torch.where(alive, t["e_root_pos"].sqrt(), 0.)
        acc["joint"] += torch.where(alive, (S1[4] - motion.q[f1]).abs().mean(-1), 0.)
        acc["slip"] += torch.where(alive, (fv * c_sim).sum(-1), 0.)
        acc["slip_n"] += alive * c_sim.float().sum(-1)
        acc["ref_slip"] += alive * (rv * c_ref).sum(-1)
        acc["ref_slip_n"] += alive * c_ref.float().sum(-1)
        acc["c_ok"] += alive * (c_sim == c_ref).float().sum(-1)
        acc["c_n"] += alive * 2
        acc["pen"] += torch.where(alive, (FLAT_Z - fz).clamp(min=0).sum(-1) / 2, 0.)
        env.f = torch.minimum(env.f, clip_end - 1)
    clip_len = motion.clip_len[clip_of]
    res = []
    for m in range(M):
        s = {k: v[m * C : (m + 1) * C].sum().item() for k, v in acc.items()}
        n = s["n"]
        res.append({
            "norm_return": (acc["ret"][m * C : (m + 1) * C] / ((clip_len[m * C : (m + 1) * C] - 1).float() * mdp.max_step_reward)).mean().item(),
            "task_norm_return": (acc["task_ret"][m * C : (m + 1) * C] / ((clip_len[m * C : (m + 1) * C] - 1).float() * mdp.max_step_reward)).mean().item(),
            "carry_reward_profile": cks[m]["args"].get("carry_reward_profile", "legacy"),
            "mean_grasp_place_penalty": {key: s[key] / n for key in ("grasp", "drop", "place")},
            "n_failed": int(env.failed_eval[m * C : (m + 1) * C].sum().item()),
            "body_pos_cm": 100 * (s["pos"] / n) ** 0.5,
            "body_ori_deg": 57.2958 * (s["ori"] / n) ** 0.5,
            "root_pos_cm": 100 * s["root"] / n,
            "joint_deg": 57.2958 * s["joint"] / n,
            "slip_cm_s": 100 * s["slip"] / max(s["slip_n"], 1),
            "ref_slip_cm_s": 100 * s["ref_slip"] / max(s["ref_slip_n"], 1),
            "contact_acc": s["c_ok"] / s["c_n"],
            "pen_mm": 1000 * s["pen"] / n,
        })  # fmt: skip
    return res


cks = [torch.load(p, map_location=dev, weights_only=False) for p in args.ckpts]
for path, ck in zip(args.ckpts, cks):
    check_pruned(ck["args"], nets_only=True, what=path)
res = run(cks)
cols = ["task_norm_return", "norm_return", "carry_reward_profile", "n_failed", "body_pos_cm", "body_ori_deg", "root_pos_cm", "joint_deg", "slip_cm_s", "ref_slip_cm_s", "contact_acc", "pen_mm"]
lines = ["| checkpoint | train h | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
for p, ck, r in zip(args.ckpts, cks, res):
    name = f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"
    lines.append(f"| {name} | {ck['wall_time'] / 3600:.2f} | " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
table = "\n".join(lines)
print(table)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
open(args.out + ".md", "w").write(table + "\n")
json.dump({p: r for p, r in zip(args.ckpts, res)}, open(args.out + ".json", "w"), indent=1)
