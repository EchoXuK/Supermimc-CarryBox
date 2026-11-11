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
parser.add_argument("--output", type=str, default=None, help="JSON output path for one --ckpt evaluation")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
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
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MAX_STEP_REWARD, MDP
from blinkmimic.motion import MotionLib
from blinkmimic.networks import Policy, policy_kwargs
from blinkmimic.robot import RobotSpec, get_robot
from blinkmimic.box_carry import build_task, robot_spec, STAGES
from blinkmimic.amp import attach_motion_prior
from blinkmimic.evaluation_metrics import CarryEvalMetrics
from blinkmimic.reward_audit import RewardAudit


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


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    attach_motion_prior(mdp, ck)
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    widths = lambda key, default: tuple(int(x) for x in a_.get(key, default).split(","))  # noqa: E731
    pi = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=Hh, enc_hidden=widths("enc_hidden", "2048,1024,512,256"),
                dec_hidden=widths("dec_hidden", "2048,1024,512,256"), **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
    pi.load_state_dict(ck["pi"])
    pi.eval()
    ids = torch.arange(N, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start[clip_of].clone())
    clip_end = motion.clip_end[clip_of]
    ret = torch.zeros(N, device=dev)
    task_ret = torch.zeros(N, device=dev)
    smooth_sum = torch.zeros(N, device=dev)
    steps = torch.zeros(N, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    object_error = torch.zeros(N, device=dev)
    if TASK == "box_carry":
        quality_q, quality_a = [env.state()[4].cpu()], []
        traj.update(object_p=[], object_quat=[])
        failure_names = ("robot_height", "robot_tilt", "box_position", "box_drop", "box_below_floor")
        failure_causes = torch.zeros(N, len(failure_names), dtype=torch.bool, device=dev)
        carry_metrics = CarryEvalMetrics(mdp, clip_of)
    reward_audit = RewardAudit(motion.stages[clip_of]) if TASK == "box_carry" and mdp.imitation_mode == "amp" else None
    rollout_steps, rollout_start = 0, time.time()
    while not bool(env.done_eval.all()):
        S = env.state()
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, mdp.goal(S, env.f, K), depth=env.depth)
        alive = ~env.done_eval.clone()
        if record:
            from blinkmimic.rotations import mat_to_quat

            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
            if TASK == "box_carry":
                traj["object_p"].append(S[6].cpu()), traj["object_quat"].append(mat_to_quat(S[7]).cpu())
        previous_action2 = env.a_prev2.clone() if reward_audit is not None else None
        out = env.step(a)
        if reward_audit is not None:
            reward_audit.update(out, alive, previous_action2)
        ret += torch.where(alive, out["r"], 0.0)
        steps += alive.float()
        if TASK == "box_carry":
            object_error += torch.where(alive, out["terms"]["object_pos"], 0.)
            task_ret += torch.where(alive, out["terms"]["task_reward"], 0.)
            smooth_sum += torch.where(alive, out["terms"]["smooth_penalty"], 0.)
            carry_metrics.update(out, alive)
            S1 = env.state()
            quality_q.append(S1[4].cpu())
            quality_a.append(a.cpu())
            f1 = env.f
            causes = torch.cat((
                ((motion.root_p[f1, 2] - S1[0][:, 2]).abs() > mdp.cfg.term_height).unsqueeze(-1),
                ((motion.root_R[f1, 2, 2] - S1[1][:, 2, 2]).abs() > mdp.cfg.term_tilt).unsqueeze(-1),
                mdp._object_failure_margins(S1, f1) > 0), -1)
            failure_causes |= causes & (alive & out["term"]).unsqueeze(-1)
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
        rollout_steps += 1
        if rollout_steps % 500 == 0:
            print(f"[eval] step {rollout_steps}, alive {int((~env.done_eval).sum())}/{N}, "
                  f"wall {time.time() - rollout_start:.1f}s", flush=True)
    clip_len = motion.clip_len[clip_of]
    norm = ret / ((clip_len - 1).float() * mdp.max_step_reward)
    norm_c = norm.view(args.repeats, C).mean(0)
    fail_c = env.failed_eval.view(args.repeats, C).float().mean(0)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((clip_len - 1).sum() * mdp.max_step_reward)).item(),
        "n_failed": int(env.failed_eval.sum().item()),
        "fail_rate": env.failed_eval.float().mean().item(),
        "failed": [motion.names[i] for i in (fail_c > 0).nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm_c[i].item(), 4) for i in range(C)},
        "per_clip_fail_rate": {motion.names[i]: round(fail_c[i].item(), 3) for i in range(C) if fail_c[i] > 0},
    }
    if TASK == "box_carry":
        from blinkmimic.quality import episode_quality, mean_quality

        qq, aa = torch.stack(quality_q).numpy(), torch.stack(quality_a).numpy()
        quality = []
        arms = [i for i, name in enumerate(robot.joint_names) if any(part in name for part in ("shoulder", "elbow", "wrist"))]
        groups = {"legs_waist": [i for i in range(robot.num_joints) if i not in arms], "arms": arms}
        group_quality = {name: [] for name in groups}
        scales, defaults = np.asarray(robot.action_scale_list()), np.asarray(robot.default_joint_pos_list())
        for n, c in enumerate(clip_of.tolist()):
            length = int(steps[n])
            start = int(motion.clip_start[c])
            actual, reference, actions = qq[:length + 1, n], motion.q[start:start + length + 1].cpu().numpy(), aa[:length, n]
            quality.append(episode_quality(actual, reference, actions, scales, defaults))
            for name, joints in groups.items():
                group_quality[name].append(episode_quality(actual[:, joints], reference[:, joints], actions[:, joints],
                                                          scales[joints], defaults[joints]))
        res["quality"] = mean_quality(quality)
        res["quality"]["by_joint_group"] = {name: mean_quality(values) for name, values in group_quality.items()}
        res["quality"]["per_stage"] = {name: mean_quality([quality[n] for n, c in enumerate(clip_of.tolist())
                                                           if int(motion.stages[c]) == s]) for s, name in enumerate(STAGES)}
        for s, name in enumerate(STAGES):
            res["quality"]["per_stage"][name]["by_joint_group"] = {
                group: mean_quality([values[n] for n, c in enumerate(clip_of.tolist()) if int(motion.stages[c]) == s])
                for group, values in group_quality.items()}
        res["quality"]["per_clip"] = {motion.names[c]: mean_quality([quality[n] for n in range(N) if n % C == c])
                                       for c in range(C)}
        res["task_norm_return"] = (task_ret / ((clip_len - 1).float() * mdp.max_step_reward)).mean().item()
        common = carry_metrics.per_world()
        if mdp.imitation_mode == "amp":
            res["task_norm_return"] = common["carry_task_norm_return"].mean().item()
        res.update({key: value.mean().item() for key, value in common.items()})
        res["imitation_mode"] = mdp.imitation_mode
        res["mean_smooth_penalty"] = (smooth_sum / steps.clamp(min=1)).mean().item()
        object_error_c = (object_error / steps.clamp(min=1)).view(args.repeats, C).mean(0)
        completion_c = (steps / (clip_len - 1).clamp(min=1)).view(args.repeats, C).mean(0)
        res["split"] = args.split
        res["mean_object_error_m"] = object_error_c.mean().item()
        res["completion_fraction"] = completion_c.mean().item()
        # A transition may cross several failure thresholds; counts can overlap.
        res["failure_causes"] = dict(zip(failure_names, failure_causes.sum(0).tolist()))
        res["per_clip_completion_fraction"] = {motion.names[i]: completion_c[i].item() for i in range(C)}
        clip_causes = failure_causes.view(args.repeats, C, -1).any(0).cpu()
        res["per_clip_failure_causes"] = {motion.names[i]: [name for name, yes in zip(failure_names, clip_causes[i].tolist()) if yes]
                                           for i in range(C) if bool(clip_causes[i].any())}
        res["per_clip_object_error_m"] = {motion.names[i]: round(object_error_c[i].item(), 4) for i in range(C)}
        res["per_stage"] = {name: {"clips": int((motion.stages == s).sum()),
                                  "norm_return": norm_c[motion.stages == s].mean().item(),
                                  "fail_rate": fail_c[motion.stages == s].mean().item(),
                                  "completion_fraction": completion_c[motion.stages == s].mean().item(),
                                  "mean_object_error_m": object_error_c[motion.stages == s].mean().item()}
                            for s, name in enumerate(STAGES) if bool((motion.stages == s).any())}
        for s, name in enumerate(STAGES):
            mask = motion.stages[clip_of] == s
            if name in res["per_stage"]:
                res["per_stage"][name].update({key: value[mask].mean().item() for key, value in common.items()})
                res["per_stage"][name]["failure_causes"] = dict(zip(failure_names, failure_causes[mask].sum(0).tolist()))
    if reward_audit is not None:
        print("[eval] auditing reward contributions and local gradients on CPU", flush=True)
        res["reward_audit"] = reward_audit.finish(robot, args.motion_dir or run_args.get("motion_dir"), args.split,
                                                 args.stages.split(",") if args.stages else None, args.max_clips, ck)
    if record:
        extra = {}
        if TASK == "box_carry":
            extra = dict(ref_object_p=motion.object_p.cpu().numpy(), ref_object_R=motion.object_R.cpu().numpy(),
                         scenes=motion.scenes.cpu().numpy(), task=np.array(TASK))
        np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()}, **extra,
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


sim_cfg = make_sim(robot)
env = TrackingSim(sim_cfg, N, robot, motion, mdp, dev, eval_mode=True, randomize=args.dr_from is not None, vision=VISION)
if args.ckpt:
    ck = torch.load(args.ckpt, map_location=dev)
    res = evaluate(env, ck, args.record)
    if args.output:
        with open(args.output, "w") as output:
            json.dump(res, output, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
    if args.record is None and (args.dr_from or args.repeats > 1):
        tag = f"dr-{args.dr_from or 'none'}{'' if args.dr_parts == 'all' else '-' + args.dr_parts}_x{args.repeats}"
        json.dump(res, open(args.ckpt.replace(".pt", f"_eval_{tag}.json"), "w"), indent=1)
    sys.exit(0)
out_csv = os.path.join(args.run_dir, "eval.csv")
done = set()
if os.path.exists(out_csv):
    done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
else:
    open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
while True:
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))[:: args.every]
    todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
    for c in todo:
        it = int(os.path.basename(c)[5:11])
        ck = torch.load(c, map_location=dev)
        t0 = time.time()
        res = evaluate(env, ck)
        line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
        open(out_csv, "a").write(line)
        json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
        print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)", flush=True)
        done.add(it)
    if not args.watch:
        break
    time.sleep(60)
