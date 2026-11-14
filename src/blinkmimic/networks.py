"""World model, policy (motion encoder + shared decoder), and critic (SuperMimic paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mdp import inv_dim, invariant_features
from .rotations import mm3, mv3, so3_exp, yaw_mat


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    # InstantTrack: when true, the freeze test runs on the GPU (torch.where) instead of a host-side `if`
    no_sync: bool = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if not RunningNorm.no_sync and self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        mean = self.mean + delta * n / tot
        var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        if RunningNorm.no_sync:
            live = self.count < self.max_count
            self.mean.copy_(torch.where(live, mean, self.mean))
            self.var.copy_(torch.where(live, var, self.var))
            self.count.copy_(torch.where(live, tot, self.count))
        else:
            self.mean.copy_(mean)
            self.var = var
            self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)




# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments.

    With ``use_prev_action`` the model also sees a_{t-1}: delayed actuators still track the previous
    action's target during the first physics steps of a control step.
    """

    def __init__(self, num_j: int, xi_dim: int, use_prev_action: bool = False, hidden=(2048, 2048, 2048),
                 contact_modes: bool = False, structured: bool = False, dt: float = 0.02,
                 object_state: bool = False, z_dim: int = 0, scene_dim: int = 0):  # fmt: skip
        """``contact_modes`` (InstantTrack): the last layer predicts the two feet's next-step ground contact
        and one increment head per contact mode (flight, left, right, double support); the increment is the
        mixture of the heads weighted by the predicted mode probabilities, so each head stays smooth within
        its mode instead of one MLP smoothing across touch-down and lift-off."""
        super().__init__()
        self.num_j = num_j
        self.object_state, self.z_dim, self.scene_dim = object_state, z_dim, scene_dim
        self.use_prev_action = use_prev_action
        self.contact_modes = contact_modes
        # ``structured`` = semi-implicit integration (positions follow the predicted velocities,
        # the net predicts velocity increments plus small position corrections)
        self.structured, self.dt = structured, dt
        self.in_dim = inv_dim(num_j, object_state) + num_j * (2 if use_prev_action else 1) + xi_dim + scene_dim
        self.out_dim = 12 + num_j + num_j + 12 * object_state
        n_out = (4 * self.out_dim + 2 if contact_modes else self.out_dim) + z_dim
        self.net = mlp(self.in_dim + z_dim, list(hidden), n_out, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)
        # scale of the plain frame-local increments, used to normalize the training loss (equal to out_norm
        # unless ``structured`` changes what the network outputs)
        self.inc_norm = RunningNorm(self.out_dim, eps=1e-4)
        if z_dim:
            self.register_buffer("z_mean", torch.zeros(z_dim))
            self.register_buffer("z_std", torch.ones(z_dim))
            self.register_buffer("dz_std", torch.ones(z_dim))

    def inputs(self, S, a, a_prev, xi_n, scene=None):
        if self.use_prev_action:
            x = torch.cat((invariant_features(S), a, a_prev, xi_n), -1)
        else:
            x = torch.cat((invariant_features(S), a, xi_n), -1)
        if self.scene_dim:
            from .box_carry import scene_features
            x = torch.cat((x, scene_features(S, scene)), -1)
        return x

    @torch.no_grad()
    def update_z_stats(self, z, dz, mask, rate=.01):
        z, dz, w = z.flatten(0, -2), dz.flatten(0, -2), mask.reshape(-1, 1)
        count = w.sum().clamp(min=1)
        mean = (z * w).sum(0) / count
        std = ((z - mean).square() * w).sum(0).div(count).sqrt().clamp(min=1e-3)
        self.z_mean.lerp_(mean, rate)
        self.z_std.lerp_(std, rate)
        self.dz_std.lerp_((dz.square() * w).sum(0).div(count).sqrt().clamp(min=1e-3), rate)

    # weight of the new velocity in the position drift: n semi-implicit substeps with a linearly changing
    # velocity give sum_k k/n / n = (n + 1) / (2 n); n = 4 substeps -> 0.625 (measured residual std of the
    # joint-position increment: 16 % with 0.5, 11 % with 0.625, 17 % with 1.0)
    DRIFT_W = 0.625

    def _drift(self, S, v1, w1, qd1):
        """Semi-implicit position update implied by the old and new velocities, in the frames of the
        increments: yaw frame for the root position, body frame for the rotation vector."""
        p, R, v, w, q, qd = S[:6]
        H = yaw_mat(R)
        a, b = self.dt * (1 - self.DRIFT_W), self.dt * self.DRIFT_W
        return (mv3(H.transpose(-1, -2), a * v + b * v1), mv3(R.transpose(-1, -2), a * w + b * w1), a * qd + b * qd1)

    def targets(self, S, S1):
        """Regression targets of the network: the frame-local increments, or with ``structured`` the
        velocity increments plus the position corrections on top of the semi-implicit drift."""
        inc = self.increments(S, S1)
        if not self.structured:
            return inc
        J = self.num_j
        robot_inc = inc[..., :12 + 2 * J]
        dp, dth, dv, dw, dq, dqd = robot_inc.split((3, 3, 3, 3, J, J), -1)
        gp, gth, gq = self._drift(S, S1[2], S1[3], S1[5])
        out = torch.cat((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)
        if self.object_state:
            op, oR, ov, ow = S[6:]
            ov1, ow1 = S1[8:]
            a, b = self.dt * (1 - self.DRIFT_W), self.dt * self.DRIFT_W
            od = inc[..., 12 + 2 * J:]
            out = torch.cat((out, od[..., :3] - mv3(yaw_mat(S[1]).transpose(-1, -2), a * ov + b * ov1),
                             od[..., 3:6] - mv3(oR.transpose(-1, -2), a * ow + b * ow1), od[..., 6:]), -1)
        return out

    @staticmethod
    def increments(S, S1):
        """Frame-local increments from S to S1."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S[:6]
        p1, R1, v1, w1, q1, qd1 = S1[:6]
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        out = torch.cat(
            (
                mv3(H.transpose(-1, -2), p1 - p),
                so3_log(mm3(Rt, R1)),
                mv3(Rt, v1 - v),
                mv3(Rt, w1 - w),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )
        if len(S) > 6:
            op, oR, ov, ow = S[6:]
            op1, oR1, ov1, ow1 = S1[6:]
            out = torch.cat((out, mv3(H.transpose(-1, -2), op1 - op), so3_log(mm3(oR.transpose(-1, -2), oR1)),
                             mv3(Rt, ov1 - ov), mv3(Rt, ow1 - ow)), -1)
        return out

    def step(self, S, a, a_prev, xi_n):
        return self.step_c(S, a, a_prev, xi_n)[0]

    def step_c(self, S, a, a_prev, xi_n):
        """Next state and next-step foot-contact logits (B, 2), or None without ``contact_modes``."""
        S1, logits, _ = self.step_z(S, a, a_prev, xi_n)
        return S1, logits

    def step_z(self, S, a, a_prev, xi_n, z=None, scene=None):
        x = self.in_norm(self.inputs(S, a, a_prev, xi_n, scene))
        if self.z_dim:
            x = torch.cat((x, (z - self.z_mean) / self.z_std), -1)
        y = run_mlp(self.net, x)
        z1 = z + y[..., -self.z_dim:] * self.dz_std if self.z_dim else None
        y = y[..., :-self.z_dim] if self.z_dim else y
        logits = None
        if self.contact_modes:
            logits = y[..., -2:]
            cl, cr = torch.sigmoid(logits).unbind(-1)
            mode_w = torch.stack(((1 - cl) * (1 - cr), cl * (1 - cr), (1 - cl) * cr, cl * cr), -1)
            y = (y[..., :-2].unflatten(-1, (4, self.out_dim)) * mode_w.unsqueeze(-1)).sum(-2)
        return self.integrate(S, y * self.out_norm.std() + self.out_norm.mean), logits, z1

    def integrate(self, S, d):
        """Next state from the denormalized network output ``d`` (the layout of ``targets``)."""
        p, R, v, w, q, qd = S[:6]
        H = yaw_mat(R)
        dim = 12 + 2 * self.num_j
        dp, dth, dv, dw, dq, dqd = d[..., :dim].split((3, 3, 3, 3, self.num_j, self.num_j), -1)
        v1 = v + mv3(R, dv)
        w1 = w + mv3(R, dw)
        qd1 = qd + dqd
        if self.structured:
            gp, gth, gq = self._drift(S, v1, w1, qd1)
            dp, dth, dq = dp + gp, dth + gth, dq + gq
        p1 = p + mv3(H, dp)
        R1 = mm3(R, so3_exp(dth))
        out = (p1, R1, v1, w1, q + dq, qd1)
        if self.object_state:
            op, oR, ov, ow = S[6:]
            dp, dr, dv, dw = d[..., dim:].split(3, -1)
            ov1, ow1 = ov + mv3(R, dv), ow + mv3(R, dw)
            if self.structured:
                a, b = self.dt * (1 - self.DRIFT_W), self.dt * self.DRIFT_W
                dp = dp + mv3(H.transpose(-1, -2), a * ov + b * ov1)
                dr = dr + mv3(oR.transpose(-1, -2), a * ow + b * ow1)
            out += (op + mv3(H, dp), mm3(oR, so3_exp(dr)), ov1, ow1)
        return out

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S[:6]
        ph, Rh, vh, wh, qh, qdh = S_hat[:6]
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        out = torch.cat(
            (
                mv3(H.transpose(-1, -2), ph - p),
                so3_log(mm3(Rt, Rh)),
                mv3(Rt, vh - v),
                mv3(Rt, wh - w),
                qh - q,
                qdh - qd,
            ),
            -1,
        )
        if len(S) > 6:
            op, oR, ov, ow = S[6:]
            ph, Rh, vh, wh = S_hat[6:]
            out = torch.cat((out, mv3(H.transpose(-1, -2), ph - op), so3_log(mm3(oR.transpose(-1, -2), Rh)),
                             mv3(Rt, vh - ov), mv3(Rt, wh - ow)), -1)
        return out


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh.

    ``ref_skip`` = (offset, q0, alpha) (--pi_ref_skip): reference-action skip connection,
    a = (q_ref(f+1) - q0) / alpha + dec(...), with q_ref(f+1) read from the raw (unnormalized) goal vector at
    ``offset``; the network only learns the correction to tracking the reference with the PD controller."""

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hist: int = 10, z_dim: int = 32,
                 enc_hidden=(2048, 1024, 512, 256), dec_hidden=(2048, 1024, 512, 256), ref_skip=None,
                 visual_dim: int = 0):  # fmt: skip
        super().__init__()
        self.hist = hist
        self.act_dim = act_dim
        self.visual_dim = visual_dim
        if visual_dim:
            from .vision import DepthEncoder
            self.depth_enc = DepthEncoder(visual_dim)
        self.ref_off = None
        if ref_skip is not None:
            self.ref_off = ref_skip[0]
            self.register_buffer("ref_q0", torch.tensor(ref_skip[1], dtype=torch.float32))
            self.register_buffer("ref_alpha", torch.tensor(ref_skip[2], dtype=torch.float32))
        self.enc = mlp(goal_dim, list(enc_hidden), z_dim)
        self.dec = mlp(z_dim + obs_dim * hist + visual_dim, list(dec_hidden), act_dim)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, z: torch.Tensor | None = None,
                visual: torch.Tensor | None = None, depth: torch.Tensor | None = None) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last. ``z`` (optional) is
        ``encode(goal)`` computed beforehand."""
        a = run_mlp(self.dec, self.condition(obs_hist, goal, z, visual, depth))
        if self.ref_off is not None:
            a = a + self.reference_action(goal)
        return a

    def condition(self, obs_hist, goal, z=None, visual=None, depth=None):
        o = self.obs_norm(obs_hist).flatten(-2)
        if z is None:
            z = self.encode(goal)
        parts = (z, o)
        if self.visual_dim:
            if visual is None:
                visual = self.depth_enc(depth)
            # E is trained only by reconstruction. Predicted z keeps its separate
            # WM dynamics path, while the actor sees it through a stop-gradient.
            parts += (visual.detach(),)
        return torch.cat(parts, -1)

    def reference_action(self, goal):
        if self.ref_off is None:
            return goal.new_zeros((*goal.shape[:-1], self.act_dim))
        q_ref = goal[..., self.ref_off : self.ref_off + self.act_dim]
        return (q_ref - self.ref_q0) / self.ref_alpha

    def encode(self, goal: torch.Tensor) -> torch.Tensor:
        """Motion latent z from the goal features (reference only, independent of the robot state)."""
        return torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))




def make_policy(obs_dim, goal_dim, act_dim, robot, run_args):
    """Build the ordinary policy shared by training, evaluation and export."""
    options = policy_kwargs(run_args, robot)
    widths = lambda key: tuple(int(x) for x in run_args.get(key, "2048,1024,512,256").split(","))
    return Policy(obs_dim, goal_dim, act_dim, hist=run_args.get("hist", 10),
                  enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"), **options)


def initialize_policy(policy, state, source_args):
    """Load a compatible ordinary policy without architecture conversion."""
    check_pruned(source_args, nets_only=True)
    policy.load_state_dict(state)


# Switches of the development code that this release does not have, with their "off" value. Run arguments (args.json,
# the "args" of a checkpoint) of that code still carry them; at the off value they are ignored, any other value
# describes a run that this code cannot rebuild or continue.
PRUNED_OFF = {
    "actor_lam": 1.0, "grad_decay": 1.0, "anchored": False, "demo_replay": False,
    "imag_action_noise": 0.0, "wm_prio": 0.0, "imag_prio": 0.0, "critic_alive": False, "critic_prior": False,
    "critic_grad": 0.0, "critic_ema": 0.0, "lookahead_k": 0, "wm_batches_early": 0, "wm_foot_weight": 0.0,
    "bc_sharp_weight": 0.0, "pi_grad_batch": 0, "no_critic": False, "wm_imag_ema": 0.0, "anchor_coef": 0.0,
    "boot_coef": 1.0, "grad_clip_step": 0.0, "det_frac": 0.0, "wm_load": None, "wm_freeze": False, "wm_ready": 0.0,
    # these also change the networks (inputs or architecture)
    "wm_pd": False, "pi_arch": "mlp", "pi_fourier": None, "pi_goal_feet": False, "wm_contact": 0.0,
}  # fmt: skip
PRUNED_NET = ("wm_pd", "pi_arch", "pi_fourier", "pi_goal_feet", "wm_contact")


def check_pruned(a_: dict, nets_only: bool = False, what: str = "this run") -> None:
    """Raise if the run arguments ``a_`` use a removed switch. ``nets_only``: only the switches that change the
    networks (enough to rebuild and evaluate a policy / world model); otherwise also every removed training switch
    (for tools that recompute the training objective)."""
    algorithm = a_.get("algorithm", "bptt")
    if algorithm != "bptt":
        raise ValueError(f"{what} uses unsupported algorithm {algorithm!r}; only bptt checkpoints are supported")
    keys = PRUNED_NET if nets_only else PRUNED_OFF
    bad = {k: a_[k] for k in keys if k in a_ and a_[k] != PRUNED_OFF[k] and not (PRUNED_OFF[k] is None and not a_[k])}
    if bad:
        raise ValueError(f"{what} uses switches that this code does not have: {bad}")


def policy_kwargs(a_: dict, robot) -> dict:
    """Policy keyword arguments of a run with the training arguments ``a_``."""
    check_pruned(a_, nets_only=True)
    return {"ref_skip": policy_ref_skip(a_, robot),
            "visual_dim": a_.get("visual_dim", 32) if a_.get("vision") == "latent" else 0}


def policy_ref_skip(a_: dict, robot):
    """Policy(ref_skip=...) argument of a run with the training arguments ``a_`` (None without --pi_ref_skip)."""
    if not a_.get("pi_ref_skip", False):
        return None
    from .motion import GOAL_Q_OFFSET

    return (GOAL_Q_OFFSET, robot.default_joint_pos_list(), robot.action_scale_list())


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
