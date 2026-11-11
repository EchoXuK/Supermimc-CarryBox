"""Shared task metrics and compatible groups for checkpoint evaluation."""

import torch


def evaluation_group_key(options):
    """Inputs, data and reward settings that must agree within one simulator."""
    defaults = (
        ("task", "tracking"), ("vision", "none"), ("motion_dir", None),
        ("goal_frames", 10), ("hist", 10), ("amp", "none"),
        ("imitation_mode", "tracking"), ("carry_residual_rate", 0.),
        ("carry_joint_vel", 0.), ("carry_residual_acc", 0.),
        ("carry_residual_acc_arms", 1.), ("carry_action_acc", 0.),
        ("carry_action_acc_arms", 0.), ("amp_reward_weight", 5.),
        ("carry_palm_far_weight", 0.), ("carry_palm_far_std", .5),
        ("carry_palm_far_kernel", "gaussian"),
        ("carry_goal_anchor", "reference"),
        ("carry_approach_progress_weight", 0.),
        ("amp_foot_slip_weight", .25),
    )
    return tuple(options.get(key, default) for key, default in defaults)


class CarryEvalMetrics:
    """Per-world carry metrics, with identical definitions across evaluators."""

    def __init__(self, mdp, clip_of):
        self.mdp, self.clip_of = mdp, clip_of
        self.steps = torch.zeros(len(clip_of), device=clip_of.device)
        self.sums = {key: torch.zeros_like(self.steps) for key in (
            "carry_task_reward", "amp_style", "amp_score", "control_penalty",
            "grasp_success", "released", "release_success", "release_stand", "object_pos", "approach_progress_reward")}

    def update(self, out, alive):
        terms = out["terms"]
        self.steps += alive
        for key, total in self.sums.items():
            value = (terms["release_success"] * terms["stand_success"]
                     if key == "release_stand" else terms[key])
            # Finished worlds keep stepping while other clips are alive. Their
            # subsequent states, including any nonfinite values, do not count.
            total.add_(torch.where(alive, value, 0.))

    def per_world(self):
        sums, steps = self.sums, self.steps.clamp(min=1)
        length = (self.mdp.motion.clip_len[self.clip_of] - 1).clamp(min=1)
        released = sums["released"].clamp(min=1)
        result = dict(
            carry_task_norm_return=sums["carry_task_reward"] / (length * self.mdp.max_task_reward),
            completion_fraction=self.steps / length,
            mean_object_error_m=sums["object_pos"] / steps,
            mean_control_penalty=sums["control_penalty"] / steps,
            mean_approach_progress_reward=sums["approach_progress_reward"] / steps,
            hand_contact_agreement=sums["grasp_success"] / steps,
            release_success_rate=sums["release_success"] / released,
            release_stand_success_rate=sums["release_stand"] / released,
        )
        if self.mdp.imitation_mode == "amp":
            result.update(mean_amp_style=sums["amp_style"] / steps, mean_amp_score=sums["amp_score"] / steps)
        return result
