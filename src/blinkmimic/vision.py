"""Depth camera of the live box/table/ground scene and the visual latent modules.

Rays use optical-axis depth, with torso +x forward, image-right -y and image-down -z.
Only real collection renders images; imagined rollouts propagate the WM's latent.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .networks import run_mlp
from .rotations import mv3, so3_exp

DEPTH_SHAPE = (1, 18, 32)
DEPTH_NEAR, DEPTH_FAR = .1, 3.


def latent_prediction_loss(prediction, target, mask):
    """Autoregressive error in units of the current valid target variance (Supermimic latent vision)."""
    weight = mask.unsqueeze(-1)
    count = weight.sum().clamp(min=1)
    mean = (target * weight).sum((0, 1)) / count
    scale = ((target - mean).square() * weight).sum() / (count * target.shape[-1])
    error = (prediction - target).square().mean(-1) / scale.detach().clamp(min=1e-6)
    return (error * mask).sum() / target.shape[0]


def reconstruction_indices(mask, limit):
    """Uniform valid images across original/mirrored windows, with a fixed size for CUDA capture.

    If fewer valid images exist, remaining slots get zero weight from the caller's mask.
    """
    score = torch.rand(mask.numel(), device=mask.device).masked_fill(mask.flatten() == 0, 2.)
    return score.argsort()[:min(limit, mask.numel())]


class DepthEncoder(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ELU(),
            nn.Flatten(), nn.Linear(32 * 5 * 8, 64), nn.ELU(), nn.Linear(64, z_dim),
        )

    def forward(self, depth):
        return run_mlp(self.net, depth.float())


class LatentDecoder(nn.Module):
    """Clean-depth reconstruction trains the encoder; latent targets are detached."""
    def __init__(self, z_dim=32):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(z_dim, 32 * 5 * 8), nn.ELU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=(0, 1)), nn.ELU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ELU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return run_mlp(self.dec, run_mlp(self.fc, z).reshape(-1, 32, 5, 8))


def ray_box(origin, directions, centre, rotation, half):
    """Optical depths for rays (B,P,3) against an oriented box, including inside rays."""
    Rt = rotation.transpose(-1, -2)
    o = mv3(Rt, origin - centre).unsqueeze(-2)
    d = mv3(Rt.unsqueeze(-3), directions)
    parallel = d.abs() < 1e-7
    safe = torch.where(parallel, torch.ones_like(d), d)
    t0, t1 = (-half.unsqueeze(-2) - o) / safe, (half.unsqueeze(-2) - o) / safe
    lo, hi = torch.minimum(t0, t1), torch.maximum(t0, t1)
    lo = torch.where(parallel, -torch.inf, lo)
    hi = torch.where(parallel, torch.inf, hi)
    enter, leave = lo.amax(-1), hi.amin(-1)
    hit = (leave >= enter.clamp(min=0)) & ~(parallel & (o.abs() > half.unsqueeze(-2))).any(-1)
    t = torch.where(enter >= 0, enter, leave)
    return torch.where(hit, t, DEPTH_FAR)


class DepthCamera:
    def __init__(self, kin, device):
        self.kin = kin
        self.torso = kin.body_names.index("torso_link")
        self.offset = torch.tensor([.1, 0., .32], device=device)
        self.mount = so3_exp(torch.tensor([0., math.radians(25), 0.], device=device))
        h, w = DEPTH_SHAPE[-2:]
        fx = (w / 2) / math.tan(math.radians(87) / 2)
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        self.rays = torch.stack((torch.ones_like(x), -(x + .5 - w / 2) / fx, -(y + .5 - h / 2) / fx), -1).flatten(0, 1).float()
        self.identity = torch.eye(3, device=device)
        self.table_half = torch.tensor([.2, .2, .01], device=device)

    @torch.no_grad()
    def render(self, S, scene=None):
        bp, bR, _, _ = self.kin.forward(*S[:6], all_bodies=True)
        R = bR[..., self.torso, :, :]
        origin = bp[..., self.torso, :] + mv3(R, self.offset)
        directions = mv3((R @ self.mount).unsqueeze(-3), self.rays)
        dz = directions[..., 2]
        t = -origin[..., 2:3] / torch.where(dz.abs() > 1e-7, dz, torch.ones_like(dz))
        depth = torch.where((dz < -1e-7) & (t > 0), t, DEPTH_FAR)
        if scene is not None:
            depth = torch.minimum(depth, ray_box(origin, directions, S[6], S[7], scene[..., :3]))
            for i in range(2):
                centre = scene[..., 3 + 3 * i:6 + 3 * i]
                depth = torch.minimum(depth, ray_box(origin, directions, centre, self.identity, self.table_half.expand_as(centre)))
        depth = (depth.clamp(DEPTH_NEAR, DEPTH_FAR) - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR)
        return depth.reshape(-1, *DEPTH_SHAPE)

    @staticmethod
    def noise(clean):
        noise = torch.randn_like(clean) * .005
        return torch.where(torch.rand_like(clean) < .01, torch.ones_like(clean), (clean + noise).clamp(0, 1))
