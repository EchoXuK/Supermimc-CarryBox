"""Reference timing, short-clip gradients and real-transition budget regressions."""

import ast
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from blinkmimic.motion import MotionLib, GOAL_Q_OFFSET, goal_frame_offsets
from blinkmimic.networks import Policy
from blinkmimic.objective import actor_objective
from blinkmimic.training_control import TransitionBudget, evaluation_rng, training_summaries

OFFSETS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
ROOT = Path(__file__).resolve().parents[1]
GPU = "--gpu" in sys.argv
if GPU:
    sys.argv.remove("--gpu")


def motion_fixture():
    m = MotionLib.__new__(MotionLib)
    m.device, m.goal_offsets = "cpu", None
    m.clip_id = torch.cat((torch.zeros(33), torch.ones(50))).long()
    m.clip_end = torch.tensor([32, 82])
    m.root_p = torch.zeros(83, 3)
    m.root_R = torch.eye(3).expand(83, -1, -1)
    m.root_v_local = m.root_w_local = torch.zeros(83, 3)
    m.q = torch.arange(83.).unsqueeze(-1).expand(-1, 2) * .001
    m.qd = torch.zeros(83, 2)
    return m


class ReferenceWindowTests(unittest.TestCase):
    def test_exact_offsets_and_clip_boundaries(self):
        m = motion_fixture()
        m.configure_goal_sampling(dict(goal_frames=10, goal_frame_offsets=OFFSETS))
        f = torch.tensor([0, 31, 33])
        future = m.future_frames(f, 10)
        torch.testing.assert_close(future[0], torch.tensor(OFFSETS).clamp(max=32))
        self.assertTrue(bool((future[1] == 32).all()))
        torch.testing.assert_close(future[2], 33 + torch.tensor(OFFSETS))
        self.assertEqual(future[2, -1].item() - 33, 45)
        self.assertEqual(m.goal(f, 10).shape[-1], m.goal_dim(10))

    def test_legacy_and_new_preview_keep_the_same_pd_skip(self):
        m = motion_fixture()
        f = torch.tensor([0, 33])
        old = m.goal(f, 10)
        m.configure_goal_sampling(dict(goal_frames=10, goal_frame_offsets=OFFSETS))
        new = m.goal(f, 10)
        policy = Policy(3, m.goal_dim(10), 2, hist=2, enc_hidden=(8,), dec_hidden=(8,),
                            ref_skip=(GOAL_Q_OFFSET, [0., 0.], [.1, .1]))
        torch.testing.assert_close(policy.reference_action(old), policy.reference_action(new), atol=0, rtol=0)
        torch.testing.assert_close(policy.reference_action(new), m.q[f + 1] / .1)
        self.assertEqual(goal_frame_offsets({}), tuple(range(1, 11)))

    def test_reject_offsets_that_break_next_frame_reference_or_dimensions(self):
        for offsets in ([0] + OFFSETS[1:], OFFSETS[:-1], [1] * 10):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                goal_frame_offsets(dict(goal_frames=10, goal_frame_offsets=offsets))

    def test_32_step_gradient_and_terminal_tail(self):
        m = motion_fixture()
        m.configure_goal_sampling(dict(goal_frames=10, goal_frame_offsets=OFFSETS))
        theta = torch.nn.Parameter(torch.tensor(.1))
        pi = lambda obs, goal, visual=None: theta.expand(len(obs), 2)

        def step_z(state, action, previous, xi, z, scene):
            delta = torch.cat((action, action.new_zeros(len(action), 1)), -1)
            return (state[0] + delta,) + state[1:], None, z

        ns = dict(torch=torch, pi=pi, wm=SimpleNamespace(step_z=step_z), motion=m, K=10,
                  mdp=SimpleNamespace(obs=lambda s, a, xi, f: s[0], add_noise=lambda x: x))
        tree = ast.parse((ROOT / "src/blinkmimic/train.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_imag_step_visual")
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "imagination", "exec"), ns)
        for horizon, start, alive_steps in ((32, 0, 32), (32, 30, 2), (48, 0, 32), (48, 30, 2), (48, 33, 48)):
            theta.grad = None
            state = (torch.zeros(1, 3),) * 6
            history, action, f, z = torch.zeros(1, 2, 3), torch.zeros(1, 2), torch.tensor([start]), torch.zeros(1, 2)
            goal, rewards, dones = m.goal(f, 10), [], []
            for _ in range(horizon):
                state, action, history, f, goal, z = ns["_imag_step_visual"](state, history, action, f, None, None, goal, z, None)
                rewards.append(state[0][:, 0]); dones.append((f >= m.clip_end[m.clip_id[f]]).float())
            bootstrap = theta.expand(1) * (0 if start == 33 else 1000)
            objective, _, survival = actor_objective(torch.stack(rewards), torch.stack(dones), bootstrap, .99)
            objective.sum().backward()
            expected = sum((t + 1) * .99 ** t for t in range(alive_steps))
            self.assertAlmostEqual(theta.grad.item(), expected, delta=1e-3)
            self.assertEqual(f.item(), min(start + horizon, m.clip_end[m.clip_id[start]].item()))
            self.assertEqual(survival.item(), float(start == 33))
            self.assertAlmostEqual(state[0][0, 0].item(), .1 * alive_steps, places=5)

    def test_long_imagination_keeps_short_clip_and_excludes_terminal_critic_duplicates(self):
        m = motion_fixture()
        tree = ast.parse((ROOT / 'src/blinkmimic/train.py').read_text())
        sample = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_valid_starts')
        f = torch.tensor([0, 33, 0, 33])
        buffer = SimpleNamespace(sample_starts=lambda b, h: ({'f': f}, {'f': f}))
        ns = dict(torch=torch, args=SimpleNamespace(no_sync=True), buffer=buffer,
                  MIN_IMAGINATION=32, HZ=48, Hh=10, motion=m)
        exec(compile(ast.Module(body=[sample], type_ignores=[]), 'sampling', 'exec'), ns)
        s, _ = ns['sample_valid_starts'](2)
        self.assertEqual(s['f'].tolist(), [0, 33])
        imagine = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'imagine_actor')
        mask = next(n for n in imagine.body if isinstance(n, ast.If) and
                    ast.unparse(n.test) == 'args.min_imagination_steps is not None')
        ns.update(s=s, dev='cpu', args=SimpleNamespace(min_imagination_steps=32))
        exec(compile(ast.Module(body=[mask], type_ignores=[]), 'critic_mask', 'exec'), ns)
        valid = ns['valid']
        self.assertEqual(valid.sum(0).tolist(), [33, 49])  # initial + terminal states, each once
        self.assertEqual(valid[1:].sum(0).tolist(), [32, 48])
        # Exercise the actual critic-selection code with deliberately corrupt tail targets.
        y = torch.ones(48, 2); y[32:, 0] = float('nan')
        theta = torch.tensor(0., requires_grad=True)
        ns.update(critic=lambda x: theta.expand(len(x)))
        loss_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_critic_loss')
        exec(compile(ast.Module(body=[loss_fn], type_ignores=[]), 'critic_loss', 'exec'), ns)
        loss = ns['_critic_loss'](torch.zeros(96, 3), y.flatten(), valid[1:].flatten())
        self.assertAlmostEqual(loss.item(), 1.)
        self.assertAlmostEqual(torch.autograd.grad(loss, theta)[0].item(), -2.)


class TrainingBudgetTests(unittest.TestCase):

    def test_world_model_probe_reports_multi_step_error_and_stops_at_resets(self):
        from blinkmimic.mdp import flatten_state
        from blinkmimic.training_control import world_model_probe
        zero, rotation, joints = torch.zeros(2, 3), torch.eye(3).repeat(2, 1, 1), torch.zeros(2, 2)
        state = (zero, rotation, zero, zero, joints, joints)
        initial = flatten_state(state).unsqueeze(1).expand(-1, 32, -1)
        targets = torch.stack([flatten_state((zero + t + 1,) + state[1:]) for t in range(32)], 1)
        mask = torch.ones(2, 32); mask[1, 4:] = 0
        data = dict(s=initial, s1=targets, mask=mask, a=torch.zeros(2, 32, 2),
                    a_prev=torch.zeros(2, 32, 2), xi=torch.zeros(2, 32, 3))
        wm = SimpleNamespace(object_state=False,
                             step_z=lambda s, a, ap, xi, z, scene: ((s[0] + .5,) + s[1:], None, None))
        metrics = world_model_probe(wm, None, SimpleNamespace(sample_windows=lambda b, h: data),
                                     lambda x: x, "cpu", batch=2)
        self.assertEqual(metrics["4"]["valid_windows"], 2)
        self.assertEqual(metrics["32"]["valid_windows"], 1)
        self.assertAlmostEqual(metrics["32"]["root_error_m"], 16 * 3 ** .5, places=4)

    def test_heartbeat_flushes_final_state(self):
        import json
        from blinkmimic.training_control import TrainingHeartbeat
        with TemporaryDirectory() as directory:
            heartbeat = TrainingHeartbeat(directory)
            heartbeat.mark("finished", 5, 100)
            heartbeat.close()
            self.assertEqual(json.loads((Path(directory) / "progress.json").read_text())["phase"], "finished")

    def test_foot_quality_counts_only_extra_lifts_during_reference_stance(self):
        from blinkmimic.quality import foot_quality
        reference = np.zeros((50, 2, 3))
        actual = reference.copy()
        actual[30:, 0, 2] = .06
        metrics = foot_quality(actual, reference, np.zeros_like(reference))
        self.assertEqual(metrics["extra_lift_count"], 1)
        self.assertGreater(metrics["raised_stance_fraction"], 0)
        velocity = np.zeros_like(reference); velocity[..., 0] = 1.
        self.assertEqual(foot_quality(actual, reference, velocity)["extra_lift_count"], 0)

    def test_budget_is_independent_of_env_count_and_collection_chunking(self):
        for n, chunk in ((512, 2), (8192, 16), (16384, 3)):
            budget, steps = TransitionBudget(4096, 4194304), 0
            while steps < 4194304:
                steps = min(4194304, steps + n * chunk)
                budget.updates += budget.due(steps, 99)
            self.assertEqual(budget.updates, 1024)
            self.assertTrue(budget.ready(steps, 0, 999))
            restored = TransitionBudget(4096, 4194304, budget.updates)
            self.assertEqual(restored.due(steps + 8192, 99), 2)
        self.assertEqual(TransitionBudget().due(1, 5), 5)
        self.assertFalse(TransitionBudget().ready(100000, 2, 3))

    def test_validation_restores_python_numpy_torch_rng_and_precision(self):
        import blinkmimic.networks as networks
        random.seed(9); np.random.seed(9); torch.manual_seed(9)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        old_amp = networks.AMP_DTYPE
        with evaluation_rng("cpu"):
            random.random(); np.random.rand(); torch.rand(4)
            networks.AMP_DTYPE = torch.bfloat16
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, atol=0, rtol=0)
        self.assertEqual(networks.AMP_DTYPE, old_amp)


class TrainingSummaryTests(unittest.TestCase):
    def test_complete_state_archive_survives_later_publication(self):
        tree = ast.parse((ROOT / "src/blinkmimic/train.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_retain_full_checkpoint")
        ns = dict(os=os)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "checkpoint", "exec"), ns)
        with TemporaryDirectory() as directory:
            last, archive = Path(directory) / "last.pt", Path(directory) / "full_000001.pt"
            last.write_bytes(b"complete optimizer state 1")
            ns["_retain_full_checkpoint"](str(last), str(archive))
            replacement = Path(directory) / "new.pt"
            replacement.write_bytes(b"complete optimizer state 2")
            os.replace(replacement, last)
            self.assertEqual(archive.read_bytes(), b"complete optimizer state 1")
            self.assertEqual(last.read_bytes(), b"complete optimizer state 2")
            # A retried publication can replace a prior archive and stale temp.
            Path(str(archive) + ".tmp").write_bytes(b"interrupted save")
            ns["_retain_full_checkpoint"](str(last), str(archive))
            self.assertEqual(archive.read_bytes(), b"complete optimizer state 2")

    def test_masked_episode_nan_and_termination_counts(self):
        aggregates = {key:[torch.tensor(1.), torch.tensor(3.)]
                      for key in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")}
        stats = dict(r=[torch.tensor(1.), torch.tensor(3.)],
                     ep_len=[torch.tensor(float("nan")), torch.tensor(5.), torch.tensor(7.)],
                     ep_ret=[torch.tensor(float("nan"))], term=[torch.tensor(1), torch.tensor(4)],
                     done=[torch.tensor(2), torch.tensor(8)])
        self.assertEqual(training_summaries(aggregates, stats, "cpu"), [2.] * 8 + [6., 0., 5., 10.])

    def test_nonfinite_losses_are_retained_for_failure_detection(self):
        aggregates = {key:[] for key in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")}
        stats = {key:[] for key in ("r", "ep_len", "ep_ret", "term", "done")}
        for bad in (float("nan"), float("inf")):
            aggregates["wm"] = [torch.tensor(bad)]
            self.assertFalse(np.isfinite(training_summaries(aggregates, stats, "cpu")[0]))


@unittest.skipUnless(GPU, "pass --gpu for isolated real validation")
class GPUProbeTests(unittest.TestCase):
    def test_real_probe_preserves_policy_normalization_rng_and_mode(self):
        from blinkmimic.box_carry import robot_spec
        from blinkmimic.dr import XiLayout
        from blinkmimic.kinematics import Kinematics
        from blinkmimic.networks import make_policy
        from blinkmimic.robot import get_robot
        from blinkmimic.training_control import ValidationProbe
        robot = robot_spec(get_robot("g1"))
        kin, layout = Kinematics(robot, "cuda:0"), XiLayout(robot, "cuda:0")
        args = dict(task="box_carry", vision="latent", visual_dim=4, goal_frames=10,
                    hist=2, goal_frame_offsets=OFFSETS, enc_hidden="32,16", dec_hidden="32,16", pi_ref_skip=True)
        probe = ValidationProbe(robot, kin, "cuda:0", layout.dim, args)
        policy = make_policy(probe.mdp.obs_dim, probe.motion.goal_dim(10), robot.num_joints, robot, args).cuda()
        policy.train()
        state = {k:v.detach().clone() for k,v in policy.state_dict().items()}
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        result = probe.run(policy)
        torch.testing.assert_close(policy.state_dict(), state, atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
        self.assertTrue(policy.training)
        self.assertEqual(len(result["per_clip"]), 30)
        self.assertEqual([v["clips"] for v in result["per_stage"].values()], [10, 10, 10])


if __name__ == "__main__":
    unittest.main()
