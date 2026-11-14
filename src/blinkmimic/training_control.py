"""Real-transition budgets and isolated validation during training."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import threading
import time

import numpy as np
import torch


class TransitionBudget:
    def __init__(self, interval=0, warmup=None, updates=0):
        if interval < 0 or (warmup is not None and warmup < 0):
            raise ValueError("Transition budgets must be nonnegative")
        self.interval, self.warmup, self.updates = interval, warmup, updates

    def due(self, transitions, legacy_batches):
        return max(0, transitions // self.interval - self.updates) if self.interval else legacy_batches

    def ready(self, transitions, iteration, legacy_warmup):
        return transitions >= self.warmup if self.warmup is not None else iteration >= legacy_warmup


class ParameterUpdates:
    """Measure actual parameter movement between validation probes, without GPU storage."""
    def __init__(self, groups):
        self.groups = {name: tuple(parameters) for name, parameters in groups.items()}
        self.previous = {name: self._vector(parameters) for name, parameters in self.groups.items()}

    @staticmethod
    def _vector(parameters):
        return torch.cat([p.detach().float().cpu().flatten() for p in parameters])

    @torch.no_grad()
    def sample(self):
        result = {}
        for name, parameters in self.groups.items():
            current = self._vector(parameters)
            previous = self.previous[name]
            delta = (current - previous).norm()
            result[name] = dict(norm=float(current.norm()), delta_norm=float(delta),
                                relative_delta=float(delta / previous.norm().clamp(min=1e-20)),
                                finite=bool(torch.isfinite(current).all()))
            self.previous[name] = current
        return result


@torch.no_grad()
def training_summaries(aggregates, statistics, device):
    """Aggregate on-device and read once, preserving non-finite loss diagnostics."""
    def values(items):
        return (torch.stack([torch.as_tensor(v, device=device, dtype=torch.float32) for v in items])
                if items else torch.empty(0, device=device))

    def mean(items, episode=False):
        data = values(items)
        if episode:  # masked resets represent "no ended episode" with NaN
            valid = ~data.isnan()
            return torch.where(valid, data, 0.).sum() / valid.sum().clamp(min=1)
        return data.sum() / max(len(items), 1)

    result = [mean(aggregates[k]) for k in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")]
    result += [mean(statistics["r"]), mean(statistics["ep_len"], True), mean(statistics["ep_ret"], True)]
    result += [values(statistics[k]).sum() for k in ("term", "done")]
    return torch.stack(result).tolist()


class TrainingHeartbeat:
    """Report the active phase without GPU synchronization or per-step file I/O."""
    def __init__(self, directory):
        self.path = Path(directory) / "progress.json"
        self.stop = threading.Event()
        self.state = {}
        self.mark("initializing", 0, 0)
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def mark(self, phase, iteration, transitions):
        self.state = dict(phase=phase, iteration=iteration, sim_steps=transitions,
                          phase_started_at=time.time(), phase_started_monotonic=time.monotonic())

    def _write(self):
        while True:
            state = self.state.copy()
            state["phase_age_s"] = time.monotonic() - state["phase_started_monotonic"]
            state["updated_at"] = time.time()
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(state))
            os.replace(temp, self.path)
            if self.stop.wait(5):
                return

    def close(self):
        self.stop.set()
        self.thread.join()
        state = self.state.copy()
        state.update(phase_age_s=time.monotonic() - state["phase_started_monotonic"], updated_at=time.time())
        self.path.write_text(json.dumps(state))


@contextmanager
def evaluation_rng(device):
    """Validation must not consume the training RNG or change actor precision."""
    import blinkmimic.networks as networks

    py_state, np_state, amp = random.getstate(), np.random.get_state(), networks.AMP_DTYPE
    devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        networks.AMP_DTYPE = amp


class ValidationProbe:
    """Thirty fixed validation clips in a separate simulator; no replay writes."""
    def __init__(self, robot, kin, device, xi_dim, run_args):
        from .box_carry import build_task
        from .env import TrackingSim, make_sim

        self.robot, self.device, self.options = robot, device, dict(run_args)
        with evaluation_rng(device):
            self.motion, self.mdp = build_task("box_carry", robot, run_args.get("motion_dir"), kin, device,
                                              xi_dim, split="val", max_clips=30, reward_options=run_args)
            self.env = TrackingSim(make_sim(robot), len(self.motion.names), robot, self.motion, self.mdp,
                                   device, eval_mode=True, randomize=False, vision=run_args.get("vision") == "latent")

    def run(self, policy):
        from .evaluation import evaluate_policy

        training = policy.training
        try:
            with evaluation_rng(self.device):
                policy.eval()
                return evaluate_policy(self.env, policy, self.motion, self.mdp, self.robot,
                                       self.options, split="val")
        finally:
            policy.train(training)


@torch.no_grad()
def world_model_probe(wm, encoder, buffer, normalize_xi, device, batch=128, horizon=32, kin=None):
    """Open-loop error against real transitions, with real actions and reset masks."""
    from .mdp import unflatten_state

    with evaluation_rng(device):
        data = buffer.sample_windows(batch, horizon)
        state = unflatten_state(data["s"][:, 0], object_state=wm.object_state)
        z = encoder(data["depth"][:, 0].float()) if encoder is not None else None
        result = {}
        for t in range(horizon):
            state, _, z = wm.step_z(state, data["a"][:, t], data["a_prev"][:, t],
                                    normalize_xi(data["xi"][:, t]), z,
                                    data["scene"][:, t] if "scene" in data else None)
            truth = unflatten_state(data["s1"][:, t], object_state=wm.object_state)
            valid = data["mask"][:, t] > 0
            if t + 1 in (4, 8, 16, 32) or t + 1 == horizon:
                count = int(valid.sum())
                metrics = {"valid_windows": count}
                for name, index in (("root_error_m", 0), ("joint_rmse_rad", 4), ("box_error_m", 6)):
                    if index >= len(state):
                        continue
                    error = (state[index] - truth[index]).square()
                    error = error.mean(-1) if index == 4 else error.sum(-1)
                    metrics[name] = (torch.where(valid, error, 0.).sum() / max(count, 1)).sqrt().item() if count else None
                if kin is not None and len(kin.sole_idx) == 2:
                    ph, vh = kin.sole_points(*kin.forward(*state[:6]))
                    pt, vt = kin.sole_points(*kin.forward(*truth[:6]))
                    for name, error in (("sole_height_mae_m", (ph[..., 2]-pt[..., 2]).abs().mean((-1, -2))),
                                        ("sole_velocity_rmse_m_s", (vh-vt).square().mean((-1, -2, -3)))):
                        value = torch.where(valid, error, 0.).sum() / max(count, 1)
                        metrics[name] = (value.sqrt() if "rmse" in name else value).item() if count else None
                result[str(t + 1)] = metrics
            # A reset cannot seed the next open-loop state from the previous episode.
            state = tuple(torch.where(valid.reshape((-1,) + (1,) * (s.ndim - 1)), s, actual)
                          for s, actual in zip(state, truth))
        return result
