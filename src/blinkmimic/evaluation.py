"""Deterministic evaluation shared by the CLI and isolated training probes."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import torch

from .box_carry import STAGES


@torch.no_grad()
def evaluate_policy(env, pi, motion, mdp, robot, run_args, *, repeats=1, split="test", protocol=None, record=None):
    a_ = run_args
    K, Hh = a_["goal_frames"], a_["hist"]
    C, N, dev = len(motion.names), env.N, motion.device
    TASK = a_.get("task", "tracking")
    args = SimpleNamespace(repeats=repeats, split=split)
    PROTOCOL = protocol
    clip_of = torch.arange(N, device=dev) % C
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
        quality_q, quality_a = [env.state()[4].clone()], []
        quality_root = [(env.state()[0].clone(), env.state()[1].clone())]
        traj.update(object_p=[], object_quat=[])
        failure_names = ("robot_height", "robot_tilt", "box_position", "box_drop", "box_below_floor")
        failure_causes = torch.zeros(N, len(failure_names), dtype=torch.bool, device=dev)
        shaping_sum = torch.zeros(N, 3, device=dev)
        reach_sum = torch.zeros(N, 2, device=dev)
        release_count, release_speed, release_error = (torch.zeros(N, device=dev) for _ in range(3))
    rollout_steps, rollout_start = 0, time.time()
    while not bool(env.done_eval.all()):
        S = env.state()
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K), depth=env.depth)
        alive = ~env.done_eval.clone()
        if record:
            from blinkmimic.rotations import mat_to_quat

            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
            if TASK == "box_carry":
                traj["object_p"].append(S[6].cpu()), traj["object_quat"].append(mat_to_quat(S[7]).cpu())
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        steps += alive.float()
        if TASK == "box_carry":
            object_error += torch.where(alive, out["terms"]["object_pos"], 0.)
            task_ret += torch.where(alive, out["terms"]["task_reward"], 0.)
            smooth_sum += torch.where(alive, out["terms"]["smooth_penalty"], 0.)
            S1 = env.state()
            quality_q.append(S1[4].clone())
            quality_a.append(out["a"].clone())
            quality_root.append((S1[0].clone(), S1[1].clone()))
            f1 = env.f
            shaping_sum += torch.where(alive[:, None], torch.stack([out["terms"][key] for key in
                                        ("grasp_penalty", "drop_penalty", "place_penalty")], -1), 0.)
            reach_sum += torch.where(alive[:, None], torch.stack([out["terms"][key] for key in
                                        ("reach_root_penalty", "reach_hand_penalty")], -1), 0.)
            released = ((motion.stages[motion.clip_id[f1]] == 2)
                        & (motion.hand_contact[f1].amax(-1) < .5)
                        & (motion.object_v[f1].norm(dim=-1) <= .15) & alive)
            release_count += released.float()
            release_speed += torch.where(released, S1[8].norm(dim=-1), 0.)
            release_error += torch.where(released, out["terms"]["object_pos"], 0.)
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
        "success_rate": (~env.failed_eval).float().mean().item(),
        "failed": [motion.names[i] for i in (fail_c > 0).nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm_c[i].item(), 4) for i in range(C)},
        "per_clip_fail_rate": {motion.names[i]: round(fail_c[i].item(), 3) for i in range(C) if fail_c[i] > 0},
        "evaluation_protocol": PROTOCOL,
    }
    if TASK == "box_carry":
        from blinkmimic.quality import episode_quality, mean_quality, foot_quality, gait_quality, mean_gait

        qq, aa = torch.stack(quality_q).cpu().numpy(), torch.stack(quality_a).cpu().numpy()
        quality, leg_quality, feet_quality, gait_values = [], [], [], []
        mdp.prepare_gait_reference()
        leg_ids = [i for i, name in enumerate(robot.joint_names) if any(k in name for k in ("_hip_", "_knee_", "_ankle_"))]
        feet = [robot.tracked_bodies.index(side + "_ankle_roll_link") for side in ("left", "right")]
        root_p = torch.stack([r[0] for r in quality_root]).flatten(0, 1)
        root_R = torch.stack([r[1] for r in quality_root]).flatten(0, 1)
        joint_q = torch.stack(quality_q).flatten(0, 1)
        foot_positions, sole_positions = [], []
        for begin in range(0, len(joint_q), 8192):
            sl = slice(begin, begin + 8192)
            zero = torch.zeros_like(root_p[sl])
            bodies = mdp.kin.forward(root_p[sl], root_R[sl], zero, zero, joint_q[sl], torch.zeros_like(joint_q[sl]))
            foot_positions.append(bodies[0][:, feet].cpu())
            sole_positions.append(mdp.kin.sole_points(*bodies)[0].cpu())
        foot_positions = torch.cat(foot_positions).view(len(quality_q), N, 2, 3).numpy()
        sole_positions = torch.cat(sole_positions).view(len(quality_q), N, 2, 4, 3).numpy()
        reference_soles = motion.sole_p.cpu().numpy()
        for n, c in enumerate(clip_of.tolist()):
            length = int(steps[n])
            start = int(motion.clip_start[c])
            quality.append(episode_quality(qq[:length + 1, n], motion.q[start:start + length + 1].cpu().numpy(),
                                           aa[:length, n], robot.action_scale_list(), robot.default_joint_pos_list()))
            reference_q = motion.q[start:start + length + 1].cpu().numpy()
            leg_quality.append(episode_quality(qq[:length + 1, n][:, leg_ids], reference_q[:, leg_ids],
                                               aa[:length, n][:, leg_ids], np.asarray(robot.action_scale_list())[leg_ids],
                                               np.asarray(robot.default_joint_pos_list())[leg_ids]))
            end = int(motion.clip_end[c]) + 1
            feet_quality.append(foot_quality(foot_positions[:length + 1, n],
                                             motion.body_p[start:end, feet].cpu().numpy(),
                                             motion.body_v[start:end, feet].cpu().numpy()))
            gait_values.append(gait_quality(sole_positions[:length + 1, n], reference_soles[start:end]))
        res["quality"] = mean_quality(quality)
        res["leg_quality"] = mean_quality(leg_quality)
        valid_feet = [v for v in feet_quality if v is not None]
        res["feet_quality"] = ({k: float(np.mean([v[k] for v in valid_feet])) for k in valid_feet[0]}
                               if valid_feet else {})
        res["gait_quality"] = mean_gait(gait_values)
        res["gait_quality"]["per_stage"] = {
            name: mean_gait([gait_values[n] for n, c in enumerate(clip_of.tolist()) if int(motion.stages[c]) == s])
            for s, name in enumerate(STAGES)}
        res["gait_quality"]["per_clip"] = {
            motion.names[c]: [gait_values[n] for n in range(N) if n % C == c] for c in range(C)}
        res["quality"]["per_stage"] = {name: mean_quality([quality[n] for n, c in enumerate(clip_of.tolist())
                                                           if int(motion.stages[c]) == s]) for s, name in enumerate(STAGES)}
        res["quality"]["per_clip"] = {motion.names[c]: mean_quality([quality[n] for n in range(N) if n % C == c])
                                       for c in range(C)}
        res["task_norm_return"] = (task_ret / ((clip_len - 1).float() * mdp.max_step_reward)).mean().item()
        res["mean_smooth_penalty"] = (smooth_sum / steps.clamp(min=1)).mean().item()
        res["carry_reward_profile"] = a_.get("carry_reward_profile", "legacy")
        res["carry_gait_profile"] = a_.get("carry_gait_profile", "legacy")
        res["algorithm"] = a_.get("algorithm", "bptt")
        res["mean_grasp_place_penalty"] = dict(zip(("grasp", "drop", "place"),
                                                   (shaping_sum / steps.clamp(min=1)[:, None]).mean(0).tolist()))
        res["mean_reach_penalty"] = dict(zip(("root", "hand"),
                                             (reach_sum / steps.clamp(min=1)[:, None]).mean(0).tolist()))
        res["release_stability"] = {
            "rollouts_reaching_release": int((release_count > 0).sum()),
            "released_steps": int(release_count.sum()),
            "mean_speed_m_s": (release_speed.sum() / release_count.sum().clamp(min=1)).item(),
            "mean_object_error_m": (release_error.sum() / release_count.sum().clamp(min=1)).item(),
            "reference_window_coverage": {
                "complete": sum(str(r.get("placement_stability_window_complete", "unknown")).lower() == "true"
                                for r in motion.rows if r["stage"] == STAGES[2]),
                "partial": sum(str(r.get("placement_stability_window_complete", "unknown")).lower() == "false"
                               for r in motion.rows if r["stage"] == STAGES[2]),
                "unknown": sum(str(r.get("placement_stability_window_complete", "unknown")).lower() not in ("true", "false")
                               for r in motion.rows if r["stage"] == STAGES[2]),
            },
        }
        task_norm_c = (task_ret / ((clip_len - 1).float() * mdp.max_step_reward)).view(args.repeats, C).mean(0)
        object_error_c = (object_error / steps.clamp(min=1)).view(args.repeats, C).mean(0)
        completion_c = (steps / (clip_len - 1).clamp(min=1)).view(args.repeats, C).mean(0)
        res["split"] = args.split
        res["mean_object_error_m"] = object_error_c.mean().item()
        res["completion_fraction"] = completion_c.mean().item()
        # A transition may cross several failure thresholds; counts can overlap.
        res["failure_causes"] = dict(zip(failure_names, failure_causes.sum(0).tolist()))
        res["per_clip_object_error_m"] = {motion.names[i]: round(object_error_c[i].item(), 4) for i in range(C)}
        res["per_stage"] = {name: {"clips": int((motion.stages == s).sum()),
                                  "norm_return": norm_c[motion.stages == s].mean().item(),
                                  "task_norm_return": task_norm_c[motion.stages == s].mean().item(),
                                  "fail_rate": fail_c[motion.stages == s].mean().item(),
                                  "success_rate": (1 - fail_c[motion.stages == s]).mean().item(),
                                  "box_drop_count": int(failure_causes[:, 3][motion.stages[clip_of] == s].sum()),
                                  "completion_fraction": completion_c[motion.stages == s].mean().item(),
                                  "mean_object_error_m": object_error_c[motion.stages == s].mean().item()}
                            for s, name in enumerate(STAGES) if bool((motion.stages == s).any())}
    if record:
        extra = {}
        if TASK == "box_carry":
            extra = dict(ref_object_p=motion.object_p.cpu().numpy(), ref_object_R=motion.object_R.cpu().numpy(),
                         scenes=motion.scenes.cpu().numpy(), task=np.array(TASK))
        np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()}, **extra,
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res

