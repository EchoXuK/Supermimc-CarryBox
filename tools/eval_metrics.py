"""Tracking-accuracy and contact metrics for a set of checkpoints (same rollout protocol as the evaluator:
each clip once from its first frame, friction 1.0, no randomization or noise, deterministic policy).

Per checkpoint, averaged over all alive control steps of all 40 clips:
  body_pos_cm   RMS error of the 14 tracked bodies (reference re-anchored to the robot root, as in the reward)
  body_ori_deg  RMS orientation error of the tracked bodies
  root_pos_cm   global root position error (drift)
  joint_deg     mean absolute joint-angle error
  slip_cm_s     horizontal foot speed while the simulated foot is in contact (ankle link < CONTACT_Z)
  ref_slip_cm_s the same on the reference motion (retargeting artifacts set a floor)
  contact_acc   fraction of foot-frames whose contact state matches the reference
  pen_mm        mean ankle-height deficit below the flat-foot height (ground penetration proxy)
plus normalized return and failed clips.

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
args = parser.parse_args()

import torch

import blinkmimic.networks as networks
from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MAX_STEP_REWARD, MDP
from blinkmimic.motion import MotionLib
from blinkmimic.networks import Policy, policy_kwargs
from blinkmimic.robot import get_robot
from blinkmimic.box_carry import build_task, robot_spec
from blinkmimic.amp import attach_motion_prior
from blinkmimic.evaluation_metrics import CarryEvalMetrics, evaluation_group_key

CONTACT_Z = 0.045  # ankle-roll link origin height below which a foot counts as in contact (flat foot: ~0.035 m)
FLAT_Z = 0.033

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
def load_policy(ck, mdp, motion, robot):
    a_ = ck["args"]
    widths = lambda key: tuple(int(x) for x in a_.get(key, "2048,1024,512,256").split(","))  # noqa: E731
    pi = Policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), robot.num_joints, hist=a_["hist"],
                enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"), **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
    pi.load_state_dict(ck["pi"])
    return pi.eval()


@torch.no_grad()
def run(cks):
    run_args = cks[0]["args"]
    task, vision = run_args.get("task", "tracking"), run_args.get("vision") == "latent"
    robot = get_robot(args.robot)
    if task == "box_carry":
        robot = robot_spec(robot)
    kin = Kinematics(robot, dev)
    motion, mdp = build_task(task, robot, run_args.get("motion_dir"), kin, dev, XiLayout(robot, dev).dim,
                             args.split, max_clips=args.max_clips, reward_options=run_args)
    C = len(motion.names)
    feet = [robot.tracked_bodies.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    M = len(cks)
    N = C * M
    attach_motion_prior(mdp, cks)
    env = TrackingSim(make_sim(robot), N, robot, motion, mdp, dev, eval_mode=True, randomize=False, vision=vision)
    pis = [load_policy(ck, mdp, motion, robot) for ck in cks]
    K, Hh = cks[0]["args"]["goal_frames"], cks[0]["args"]["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if cks[0]["args"].get("amp", "none") == "bf16" else None
    clip_of = torch.arange(N, device=dev) % C
    env.reset(torch.arange(N, device=dev), motion.clip_start[clip_of].clone())
    clip_end = motion.clip_end[clip_of]
    keys = ("n", "ret", "pos", "ori", "root", "joint", "slip", "slip_n", "ref_slip", "ref_slip_n", "c_ok", "c_n", "pen")
    acc = {k: torch.zeros(N, device=dev) for k in keys}
    carry_metrics = CarryEvalMetrics(mdp, clip_of) if task == "box_carry" else None
    def add(key, value, alive):
        acc[key].add_(torch.where(alive, value, 0.))
    hist, a = None, torch.zeros(N, robot.num_joints, device=dev)
    while not bool(env.done_eval.all()):
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        g = mdp.goal(env.state(), env.f, K)
        for m, pi in enumerate(pis):
            sl = slice(m * C, (m + 1) * C)
            a[sl] = pi(hist[sl], g[sl], depth=env.depth[sl] if vision else None)
        alive = ~env.done_eval
        f1 = env.f + 1
        out = env.step(a)
        S1 = env.state()
        t = out["terms"]
        bp, _, bv, _ = kin.forward(*S1[:6])
        fz, fv = bp[:, feet, 2], bv[:, feet, :2].norm(dim=-1)
        rz, rv = motion.body_p[f1][:, feet, 2], motion.body_v[f1][:, feet, :2].norm(dim=-1)
        c_sim, c_ref = fz < CONTACT_Z, rz < CONTACT_Z
        acc["n"] += alive
        add("ret", out["r"], alive)
        add("pos", t["e_body_pos"], alive)
        add("ori", t["e_body_ori"], alive)
        add("root", t["e_root_pos"].sqrt(), alive)
        add("joint", (S1[4] - motion.q[f1]).abs().mean(-1), alive)
        add("slip", (fv * c_sim).sum(-1), alive)
        add("slip_n", c_sim.float().sum(-1), alive)
        add("ref_slip", (rv * c_ref).sum(-1), alive)
        add("ref_slip_n", c_ref.float().sum(-1), alive)
        add("c_ok", (c_sim == c_ref).float().sum(-1), alive)
        add("c_n", torch.full_like(acc["n"], 2.), alive)
        add("pen", (FLAT_Z - fz).clamp(min=0).sum(-1) / 2, alive)
        if carry_metrics is not None:
            carry_metrics.update(out, alive)
        env.f = torch.minimum(env.f, clip_end - 1)
    clip_len = motion.clip_len[clip_of]
    res = []
    task_metrics = carry_metrics.per_world() if carry_metrics is not None else {}
    for m in range(M):
        s = {k: v[m * C : (m + 1) * C].sum().item() for k, v in acc.items()}
        n = s["n"]
        res.append({
            "norm_return": (acc["ret"][m * C : (m + 1) * C] / ((clip_len[m * C : (m + 1) * C] - 1).float() * mdp.max_step_reward)).mean().item(),
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
        if carry_metrics is not None:
            sl = slice(m * C, (m + 1) * C)
            res[-1].update({key: value[sl].mean().item() for key, value in task_metrics.items()})
            res[-1].update(imitation_mode=mdp.imitation_mode, split=args.split,
                           fail_rate=env.failed_eval[sl].float().mean().item())
    return res


cks = [torch.load(p, map_location="cpu", weights_only=False) for p in args.ckpts]
# Each simulator has one goal/history shape and one reward configuration. Keep
# compatible policies batched, and run tracking/AMP or differing windows separately.
groups = {}
for i, ck in enumerate(cks):
    groups.setdefault(evaluation_group_key(ck["args"]), []).append(i)
res = [None] * len(cks)
for indices in groups.values():
    for i, result in zip(indices, run([cks[i] for i in indices])):
        res[i] = result
cols = ["norm_return", "n_failed", "body_pos_cm", "body_ori_deg", "root_pos_cm", "joint_deg", "slip_cm_s", "ref_slip_cm_s", "contact_acc", "pen_mm"]
if any("carry_task_norm_return" in r for r in res):
    cols += ["carry_task_norm_return", "completion_fraction", "mean_object_error_m", "hand_contact_agreement",
             "release_success_rate", "release_stand_success_rate", "mean_control_penalty"]
if any("mean_amp_style" in r for r in res):
    cols += ["mean_amp_style", "mean_amp_score"]
lines = ["| checkpoint | train h | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
for p, ck, r in zip(args.ckpts, cks, res):
    name = f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"
    values = [r.get(c, "—") for c in cols]
    lines.append(f"| {name} | {ck['wall_time'] / 3600:.2f} | " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in values) + " |")
table = "\n".join(lines)
print(table)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
open(args.out + ".md", "w").write(table + "\n")
json.dump({p: r for p, r in zip(args.ckpts, res)}, open(args.out + ".json", "w"), indent=1)
