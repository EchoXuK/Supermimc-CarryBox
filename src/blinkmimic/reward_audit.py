"""Reward contributions and local gradients on states visited in validation."""

import torch

COMPONENTS = (
    "reward_amp", "reward_box_position", "reward_box_orientation", "reward_box_velocity",
    "reward_palms", "reward_contact", "reward_standing", "reward_approach_progress", "reward_action_rate",
    "reward_joint_limit", "reward_foot_slip", "reward_action_acc",
)


class RewardAudit:
    def __init__(self, stages, max_samples=128, stride=16):
        self.stages = stages
        self.sums = torch.zeros(3, len(COMPONENTS), device=stages.device)
        self.count = torch.zeros(3, device=stages.device)
        self.value_error = torch.zeros((), device=stages.device)
        self.samples, self.max_samples, self.stride, self.step = [], max_samples, stride, 0

    @torch.no_grad()
    def update(self, out, alive, previous_action2):
        values = torch.stack([out["terms"][key] for key in COMPONENTS], -1)
        values = torch.where(alive[:, None], values, 0.)
        error = torch.where(alive, (values.sum(-1) - out["r"]).abs(), 0.)
        self.value_error.copy_(torch.maximum(self.value_error, error.max()))
        for stage in range(3):
            mask = alive & (self.stages == stage)
            self.sums[stage] += torch.where(mask[:, None], values, 0.).sum(0)
            self.count[stage] += mask.sum()
        if self.step % self.stride == 0 and sum(len(s["f"]) for s in self.samples) < self.max_samples:
            selected = []
            for stage in range(3):
                ids = (alive & (self.stages == stage)).nonzero().flatten()
                if len(ids):
                    selected.append(ids[(self.step // self.stride) % len(ids)])
            if selected:
                ids = torch.stack(selected)
                sample = {key: out[key][ids].detach().cpu() for key in ("s", "s1", "f", "a", "a_prev", "ep_step", "r")}
                sample["a_prev2"] = previous_action2[ids].detach().cpu()
                self.samples.append(sample)
        self.step += 1

    @torch.enable_grad()
    def finish(self, robot, motion_dir, split, stages, max_clips, checkpoint):
        from .amp import attach_motion_prior
        from .box_carry import build_task, STAGES
        from .dr import XiLayout
        from .kinematics import Kinematics
        from .mdp import unflatten_state

        means = (self.sums / self.count[:, None].clamp(min=1)).cpu()
        result = dict(
            component_order=list(COMPONENTS),
            per_stage_mean_reward_per_live_step={name: dict(zip(COMPONENTS, means[s].tolist()))
                                                for s, name in enumerate(STAGES)},
            value_decomposition_max_error=float(self.value_error),
            gradient_sample_kind="policy_visited_validation_transitions",
        )
        if not self.samples:
            return dict(result, gradient_samples=0)
        data = {key: torch.cat([s[key] for s in self.samples])[:self.max_samples] for key in self.samples[0]}
        kin = Kinematics(robot, "cpu")
        motion, mdp = build_task("box_carry", robot, motion_dir, kin, "cpu", XiLayout(robot, "cpu").dim,
                                 split, stages, max_clips, reward_options=checkpoint["args"])
        attach_motion_prior(mdp, checkpoint)
        current = tuple(x.detach().requires_grad_() for x in unflatten_state(data["s1"], object_state=True))
        previous = tuple(x.detach().requires_grad_() for x in unflatten_state(data["s"], object_state=True))
        a, ap, ap2 = (data[key].detach().requires_grad_() for key in ("a", "a_prev", "a_prev2"))
        reward, terms = mdp.reward(current, data["f"] + 1, a, ap, terms=True, previous_state=previous,
                                    a_prev2=ap2, smooth_valid=data["ep_step"] > 0,
                                    smooth_acc_valid=data["ep_step"] > 1)
        leaves = (*current, *previous, a, ap, ap2)
        state_names = ("root_position", "root_rotation", "root_linear_velocity", "root_angular_velocity",
                       "joint_position", "joint_velocity", "box_position", "box_rotation",
                       "box_linear_velocity", "box_angular_velocity")
        input_names = (*state_names, *("previous_" + name for name in state_names), "action", "previous_action", "previous_action2")
        sample_stages = motion.stages[motion.clip_id[data["f"]]]
        gradients, joint_gradients, per_component = [], [], {}
        for key in COMPONENTS:
            raw = (torch.autograd.grad(terms[key].sum(), leaves, retain_graph=True, allow_unused=True)
                   if terms[key].requires_grad else (None,) * len(leaves))
            grads = [torch.zeros_like(x) if g is None else g for x, g in zip(leaves, raw)]
            gradients.append(grads)
            joints = torch.cat((grads[4], grads[len(current) + 4]), -1)
            joint_gradients.append(joints)
            per_component[key] = dict(
                all_gradients_finite=all(bool(torch.isfinite(g).all()) for g in grads),
                per_input_mean_gradient_norm={name: float(g.flatten(1).norm(dim=-1).mean())
                                              for name, g in zip(input_names, grads)},
                per_stage={name: dict(
                    samples=int((sample_stages == stage).sum()),
                    joint_gradient_mean_norm=float(joints[sample_stages == stage].norm(dim=-1).mean()) if bool((sample_stages == stage).any()) else None,
                    active_joint_gradient_fraction=float((joints[sample_stages == stage].norm(dim=-1) > 1e-8).float().mean()) if bool((sample_stages == stage).any()) else None,
                    action_gradient_mean_norm=float(grads[-3][sample_stages == stage].norm(dim=-1).mean()) if bool((sample_stages == stage).any()) else None,
                    active_reward_fraction=float((terms[key][sample_stages == stage].abs() > 1e-9).float().mean()) if bool((sample_stages == stage).any()) else None,
                ) for stage, name in enumerate(STAGES)},
            )
        combined = torch.autograd.grad(reward.sum(), leaves, allow_unused=True)
        gradient_error = max(float(((torch.zeros_like(x) if g is None else g) - sum(gs[i] for gs in gradients)).abs().max())
                             for i, (x, g) in enumerate(zip(leaves, combined)))
        all_finite = all(info["all_gradients_finite"] for info in per_component.values())
        if not all_finite:
            raise RuntimeError("Non-finite reward component gradient on visited validation states")
        # Compare joint gradients in radians; state/action fields have different units.
        stack = torch.stack(joint_gradients)
        norm = stack.norm(dim=-1)
        cosine, active_pairs = [], []
        for i in range(len(COMPONENTS)):
            row, counts = [], []
            for j in range(len(COMPONENTS)):
                active = (norm[i] > 1e-8) & (norm[j] > 1e-8)
                similarity = (stack[i] * stack[j]).sum(-1) / (norm[i] * norm[j]).clamp(min=1e-16)
                row.append(float(similarity[active].mean()) if bool(active.any()) else None)
                counts.append(int(active.sum()))
            cosine.append(row)
            active_pairs.append(counts)
        result.update(gradient_samples=len(data["f"]), components=per_component,
                      all_gradients_finite=all_finite,
                      live_state_recompute_max_error=float((reward.detach() - data["r"]).abs().max()),
                      gradient_decomposition_max_error=gradient_error,
                      joint_gradient_cosine=cosine, joint_gradient_active_pair_counts=active_pairs)
        # Locate approach failures in task space, where AMP has no translation target.
        with torch.no_grad():
            positions = kin.forward(*current[:6])[0]
            frames = data["f"] + 1
            clips = motion.clip_id[frames]
            starts = motion.clip_start[clips]
            palm_error = (positions[:, mdp.hands] - current[6][:, None]
                          - (motion.body_p[frames][:, mdp.hands] - motion.object_p[frames, None])).norm(dim=-1).mean(-1)
            result["approach_state_samples"] = [dict(
                clip=motion.names[int(clips[i])],
                frame_fraction=float((frames[i] - starts[i]) / (motion.clip_len[clips[i]] - 1)),
                root_target_error_m=float((current[0][i] - motion.root_p[frames[i]]).norm()),
                palm_target_error_m=float(palm_error[i]),
                actual_root_displacement_xy_m=float((current[0][i, :2] - motion.root_p[starts[i], :2]).norm()),
                reference_root_displacement_xy_m=float((motion.root_p[frames[i], :2] - motion.root_p[starts[i], :2]).norm()),
                box_target_error_m=float((current[6][i] - motion.object_p[frames[i]]).norm()),
            ) for i in (sample_stages == 0).nonzero().flatten().tolist()]
        return result
