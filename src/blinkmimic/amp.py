"""Differentiable, stage-conditioned adversarial motion prior for box carrying.

Mechanism adapted from OpenLegged/supermimic_AMP, supermimic branch,
commit 596dfe37e0f75769b635386cd589960cce7faaa2 (see THIRD_PARTY_NOTICES.md).
The discriminator learns from real simulator transitions, never imagined negatives.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .mdp import unflatten_state
from .networks import RunningNorm, mlp
from .rotations import mat_to_6d, mm3, mv3


class AMPFeatures:
    """Shared FK features for projected experts, simulator states and imagination."""

    def __init__(self, kin):
        self.kin = kin
        names = [kin.body_names[i] for i in kin.tracked_idx]
        self.anchor = names.index("torso_link")
        self.bodies = [i for i, name in enumerate(names) if name != "torso_link"]
        self.body_names = [names[i] for i in self.bodies]
        self.dim = 15 * len(self.bodies)

    def from_fk(self, positions, rotations, velocities, angular):
        R = rotations[..., self.bodies, :, :]
        anchor_Rt = rotations[..., self.anchor:self.anchor + 1, :, :].transpose(-1, -2)
        relative = mv3(anchor_Rt, positions[..., self.bodies, :] - positions[..., self.anchor:self.anchor + 1, :])
        orientation = mat_to_6d(mm3(anchor_Rt, R))
        return torch.cat((relative.flatten(-2), orientation.flatten(-2),
                          mv3(R.transpose(-1, -2), velocities[..., self.bodies, :]).flatten(-2),
                          mv3(R.transpose(-1, -2), angular[..., self.bodies, :]).flatten(-2)), -1)

    def __call__(self, state):
        return self.from_fk(*self.kin.forward(*state[:6]))


class MotionPrior(nn.Module):
    def __init__(self, kin):
        super().__init__()
        self.features = AMPFeatures(kin)
        self.norm = RunningNorm(self.features.dim)
        self.discriminator = mlp(2 * self.features.dim + 3, [1024, 512, 256], 1, act=nn.ReLU)
        # Frozen between discriminator updates; input derivatives remain available.
        self.discriminator.requires_grad_(False)

    def pair_input(self, state, next_state, stages):
        return torch.cat((self.norm(state.float()).clamp(-10, 10),
                          self.norm(next_state.float()).clamp(-10, 10),
                          F.one_hot(stages, 3).to(state.dtype)), -1)

    def forward(self, state, next_state, stages):
        score = self.discriminator(self.pair_input(state, next_state, stages)).squeeze(-1).float()
        return (1 - .25 * (score - 1).square()).clamp_min(0), score

    def checkpoint(self, cpu=False):
        state = self.state_dict()
        if cpu:
            state = {k: v.detach().to("cpu", copy=True) for k, v in state.items()}
        return {"state_dict": state, "body_names": self.features.body_names, "anchor": "torso_link"}

    def load_checkpoint(self, checkpoint):
        if checkpoint["body_names"] != self.features.body_names or checkpoint["anchor"] != "torso_link":
            raise ValueError("AMP checkpoint body order/anchor differs from this robot")
        self.load_state_dict(checkpoint["state_dict"])


class MotionPriorGroup(nn.Module):
    """One discriminator per contiguous policy block in batched evaluation."""

    def __init__(self, priors):
        super().__init__()
        self.priors = nn.ModuleList(priors)
        self.features = priors[0].features

    def forward(self, state, next_state, stages):
        if len(state) % len(self.priors):
            raise ValueError("Evaluation worlds must divide into equal policy blocks")
        results = [prior(s, ns, stage) for prior, s, ns, stage in zip(
            self.priors, state.chunk(len(self.priors)), next_state.chunk(len(self.priors)), stages.chunk(len(self.priors)))]
        return tuple(torch.cat([r[i] for r in results]) for i in range(2))


def attach_motion_prior(mdp, checkpoints):
    """Restore reward models for evaluators; old tracking checkpoints need none."""
    if getattr(mdp, "imitation_mode", "tracking") != "amp":
        return
    checkpoints = checkpoints if isinstance(checkpoints, list) else [checkpoints]
    priors = []
    for ck in checkpoints:
        if "motion_prior" not in ck:
            raise ValueError("AMP evaluation requires a checkpoint containing motion_prior")
        prior = MotionPrior(mdp.kin).to(mdp.default_q.device)
        prior.load_checkpoint(ck["motion_prior"])
        priors.append(prior.eval())
    mdp.motion_prior = priors[0] if len(priors) == 1 else MotionPriorGroup(priors)


class ExpertPairs:
    """Uniform clips within each requested stage, then uniform adjacent frames."""

    def __init__(self, motion):
        if any(row["split"] != "train" for row in motion.rows):
            raise ValueError("AMP experts must come exclusively from the train split")
        self.motion = motion
        groups = [(motion.stages == stage).nonzero().flatten() for stage in range(3)]
        self.counts = torch.tensor([len(g) for g in groups], device=motion.device)
        self.clips = torch.zeros(3, max(len(g) for g in groups), device=motion.device, dtype=torch.long)
        for stage, group in enumerate(groups):
            self.clips[stage, :len(group)] = group

    def sample_frames(self, stages):
        m = self.motion
        indices = (torch.rand(stages.shape, device=stages.device) * self.counts[stages]).long()
        clips = self.clips[stages, indices]
        offset = (torch.rand(stages.shape, device=stages.device) * (m.clip_len[clips] - 1)).long()
        return m.clip_start[clips] + offset

    def sample(self, stages, features):
        f = self.sample_frames(stages)
        m = self.motion
        get = lambda frames: features.from_fk(m.body_p[frames], m.body_R[frames], m.body_v[frames], m.body_w[frames])
        return get(f), get(f + 1)


class DiscriminatorUpdates:
    def __init__(self, prior, motion, lr=1e-5, batch=256, grad_penalty=10.):
        self.prior, self.experts = prior, ExpertPairs(motion)
        layers = list(prior.discriminator.children())
        self.optimizer = torch.optim.Adam([
            {"params": [p for layer in layers[:-1] for p in layer.parameters()], "weight_decay": 1e-3},
            {"params": layers[-1].parameters(), "weight_decay": .1},
        ], lr=lr)
        self.batch, self.grad_penalty, self.updates = batch, grad_penalty, 0

    def update(self, replay):
        w = replay.sample_transitions(self.batch)
        prior, m = self.prior, self.experts.motion
        with torch.no_grad():
            stages = m.stages[m.clip_id[w["f"]]]
            real = prior.features(unflatten_state(w["s"], object_state=True))
            next_real = prior.features(unflatten_state(w["s1"], object_state=True))
            expert, next_expert = self.experts.sample(stages, prior.features)
            # Statistics are estimated only from raw real/expert features and
            # remain fixed for the entire ensuing actor/critic update.
            prior.norm.update(torch.cat((real, next_real, expert, next_expert)))
            policy_input = prior.pair_input(real, next_real, stages)
            expert_input = prior.pair_input(expert, next_expert, stages)
        expert_input.requires_grad_(True)
        prior.discriminator.requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            policy_score = prior.discriminator(policy_input)
            expert_score = prior.discriminator(expert_input)
            classification = .5 * ((expert_score - 1).square().mean() + (policy_score + 1).square().mean())
            gradient = torch.autograd.grad(expert_score.sum(), expert_input, create_graph=True)[0]
            # Penalize continuous, normalized features, not the stage one-hot.
            penalty = self.grad_penalty * gradient[..., :2 * prior.features.dim].square().sum(-1).mean()
            loss = classification + penalty
            loss.backward()
            gradnorm = nn.utils.clip_grad_norm_(prior.discriminator.parameters(), 1.)
            self.optimizer.step()
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            prior.discriminator.requires_grad_(False)
        self.updates += 1
        return {"disc_loss": classification.detach(), "grad_penalty": penalty.detach(),
                "policy_score": policy_score.detach().mean(), "expert_score": expert_score.detach().mean(),
                "disc_gradnorm": gradnorm.detach()}

    def checkpoint(self):
        return {"amp_optimizer": self.optimizer.state_dict(), "amp_updates": self.updates}

    def load_checkpoint(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint["amp_optimizer"])
        self.updates = checkpoint["amp_updates"]
