"""Behavioral checks for carry smoothness incentives and episode-local quality metrics."""

import unittest

import numpy as np
import torch

from blinkmimic.box_carry import CarryMDP, build_task, robot_spec
from blinkmimic.buffer import ReplayBuffer
from blinkmimic.dr import XiLayout
from blinkmimic.kinematics import Kinematics
from blinkmimic.quality import episode_quality
from blinkmimic.robot import get_robot


class CarrySmoothTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = robot_spec(get_robot("g1"))
        cls.kin = Kinematics(cls.robot, "cpu")
        cls.layout = XiLayout(cls.robot, "cpu")
        cls.motion, cls.legacy = build_task("box_carry", cls.robot, None, cls.kin, "cpu", cls.layout.dim, max_clips=3)
        cls.smooth = CarryMDP(cls.kin, cls.motion, cls.legacy.default_q, cls.layout.dim, cls.robot.sensing,
                              action_scale=cls.robot.action_scale_list(), residual_rate_weight=.6, joint_vel_weight=.2)
        cls.acc = CarryMDP(cls.kin, cls.motion, cls.legacy.default_q, cls.layout.dim, cls.robot.sensing,
                          action_scale=cls.robot.action_scale_list(), residual_acc_weight=.2)
        cls.f = cls.motion.clip_start + torch.tensor([20, 100, 40])
        cls.alpha = torch.tensor(cls.robot.action_scale_list())

    def reference_actions(self, f):
        a = (self.motion.q[f] - self.legacy.default_q) / self.alpha
        prev = torch.maximum(f - 1, self.motion.clip_start[self.motion.clip_id[f]])
        ap = (self.motion.q[prev] - self.legacy.default_q) / self.alpha
        return a, ap

    def test_demonstrated_motion_and_constant_correction_are_not_penalized(self):
        S = self.motion.ref_state(self.f)
        a, ap = self.reference_actions(self.f)
        for offset in (0., .7):
            old = self.legacy.reward(S, self.f, a + offset, ap + offset)
            new, terms = self.smooth.reward(S, self.f, a + offset, ap + offset, terms=True)
            torch.testing.assert_close(new, old)
            torch.testing.assert_close(terms["smooth_penalty"], torch.zeros(3), atol=1e-8, rtol=0)
        # A clip's first frame cannot inherit a reference-rate jump from the preceding clip.
        f = self.motion.clip_start
        a, ap = self.reference_actions(f)
        _, terms = self.smooth.reward(self.motion.ref_state(f), f, a, ap, terms=True)
        torch.testing.assert_close(terms["smooth_penalty"], torch.zeros(3), atol=1e-8, rtol=0)

    def test_oscillation_and_wrong_velocity_have_finite_corrective_gradients(self):
        S = self.motion.ref_state(self.f)
        qd = (S[5] + .8).detach().requires_grad_()
        a, ap = self.reference_actions(self.f)
        delta = torch.full_like(a, .3, requires_grad=True)
        _, terms = self.smooth.reward(S[:5] + (qd,) + S[6:], self.f, a + delta, ap - delta, terms=True)
        loss = terms["smooth_penalty"].sum()
        gd, gv = torch.autograd.grad(loss, (delta, qd))
        self.assertTrue(torch.isfinite(gd).all() and torch.isfinite(gv).all())
        self.assertTrue((gd > 0).all() and (gv > 0).all())
        self.assertGreater(loss.item(), .01)

    def test_standing_still_is_not_a_smoothness_shortcut(self):
        S = self.motion.ref_state(self.f)
        a, ap = self.reference_actions(self.f)
        _, moving = self.smooth.reward(S, self.f, a, ap, terms=True)
        stopped = S[:5] + (torch.zeros_like(S[5]),) + S[6:]
        _, still = self.smooth.reward(stopped, self.f, a, ap, terms=True)
        self.assertTrue((still["smooth_penalty"] > moving["smooth_penalty"]).all())
        torch.testing.assert_close(moving["task_reward"], self.legacy.reward(S, self.f, a, ap))

    def test_reset_transition_does_not_charge_an_unobserved_residual_history(self):
        # Random resets may be mid-clip; testing only the first motion frame misses this bias.
        S = self.motion.ref_state(self.f)
        a, ap = self.reference_actions(self.f)
        a = a + 1.
        active = torch.tensor([False, True, True])
        _, terms = self.smooth.reward(S, self.f, a, ap, terms=True, smooth_valid=active)
        self.assertEqual(terms["smooth_penalty"][0].item(), 0.)
        self.assertTrue((terms["smooth_penalty"][1:] > .1).all())
        # qd tracking still applies on the reset transition.
        moving_wrong = S[:5] + (S[5] + 1.,) + S[6:]
        _, terms = self.smooth.reward(moving_wrong, self.f, a, ap, terms=True, smooth_valid=active)
        self.assertGreater(terms["smooth_penalty"][0].item(), 0.)

    def test_quality_detects_added_jitter_at_equal_motion_progress(self):
        t = np.arange(501) * .02
        reference = np.sin(2 * np.pi * t)[:, None] * np.ones((1, 29))
        actual = reference + .03 * np.sin(2 * np.pi * 10 * t)[:, None]
        quality = episode_quality(actual, reference, actual[1:], np.ones(29), np.zeros(29))
        self.assertGreater(quality["actual"]["acc_rms"], quality["reference"]["acc_rms"] * 2)
        self.assertGreater(quality["actual"]["hf6_fraction"], quality["reference"]["hf6_fraction"] + .01)
        self.assertLess(quality["reference"]["hf6_fraction"], 1e-4)
        self.assertIsNone(episode_quality(reference[:3], reference[:3], reference[1:3], np.ones(29), np.zeros(29)))

    def test_second_difference_preserves_reference_and_linear_corrections(self):
        S = self.motion.ref_state(self.f)
        a, ap = self.reference_actions(self.f)
        prev2 = torch.maximum(self.f-2, self.motion.clip_start[self.motion.clip_id[self.f]])
        app = (self.motion.q[prev2]-self.legacy.default_q)/self.alpha
        # Constant and linearly changing corrections carry no new acceleration cost.
        for x, y, z in [(0., 0., 0.), (.4, .4, .4), (.3, .2, .1)]:
            reward, terms = self.acc.reward(S, self.f, a+x, ap+y, terms=True, a_prev2=app+z)
            torch.testing.assert_close(terms['smooth_penalty'], torch.zeros(3), atol=1e-7, rtol=0)
            torch.testing.assert_close(reward, self.legacy.reward(S, self.f, a+x, ap+y))
        f = self.motion.clip_start
        a, ap = self.reference_actions(f)
        _, terms = self.acc.reward(self.motion.ref_state(f), f, a, ap, terms=True, a_prev2=ap)
        torch.testing.assert_close(terms['smooth_penalty'], torch.zeros(3), atol=1e-7, rtol=0)

    def test_second_difference_masks_missing_history_and_has_correct_gradient(self):
        S = self.motion.ref_state(self.f)
        a, ap = self.reference_actions(self.f)
        app = (self.motion.q[self.f-2]-self.legacy.default_q)/self.alpha
        delta = torch.full_like(a, .2, requires_grad=True)
        valid = torch.tensor([False, False, True])
        _, terms = self.acc.reward(S, self.f, a+delta, ap-delta, terms=True,
                                    a_prev2=app+delta, smooth_acc_valid=valid)
        gradient, = torch.autograd.grad(terms['smooth_penalty'].sum(), (delta,))
        self.assertTrue(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient[:2], torch.zeros_like(gradient[:2]))
        self.assertTrue((gradient[2] > 0).all())
        self.assertTrue((terms['residual_acc'][:2] == 0).all())
        with self.assertRaisesRegex(ValueError, 'two previous actions'):
            self.acc.reward(S, self.f, a, ap)

    def test_replay_preserves_reset_age_after_ring_wrap(self):
        buffer = ReplayBuffer(4, 2, "cpu", 29, self.layout.dim)
        for t in range(6):
            transition = {key: torch.zeros((2,) + tuple(value.shape[2:]), dtype=value.dtype)
                          for key, value in buffer.data.items()}
            transition["ep_step"] = torch.tensor([t % 3, 100 + t], dtype=torch.int32)
            transition["s"][:] = t
            transition['a_prev'][:] = t
            buffer.add(transition)
        start, history = buffer.starts_at(torch.tensor([0, 1, 3]), torch.tensor([0, 0, 1]), 3)
        torch.testing.assert_close(start["ep_step"], torch.tensor([2, 0, 105], dtype=torch.int32))
        torch.testing.assert_close(start['a_prev2_valid'], torch.tensor([False, False, True]))
        torch.testing.assert_close(start['a_prev2'][:, 0], torch.tensor([2., 3., 4.]))
        # The reset sample repeats its own state instead of borrowing the preceding episode.
        torch.testing.assert_close(history["s"][1], torch.full_like(history["s"][1], 3.))

    def test_body_smoothing_preserves_grasp_corrections_and_task_reward(self):
        _, body = build_task("box_carry", self.robot, None, self.kin, "cpu", self.layout.dim, max_clips=3,
                             reward_options={"carry_residual_acc": .1, "carry_residual_acc_arms": 0.})
        a, ap = self.reference_actions(self.f)
        app = (self.motion.q[self.f-2]-self.legacy.default_q)/self.alpha
        delta = torch.full_like(a, .2, requires_grad=True)
        S = self.motion.ref_state(self.f)
        _, terms = body.reward(S, self.f, a+delta, ap-delta, terms=True, a_prev2=app+delta)
        gradient, = torch.autograd.grad(terms['smooth_penalty'].sum(), (delta,))
        arms = torch.tensor([any(part in n for part in ('shoulder', 'elbow', 'wrist'))
                             for n in self.robot.joint_names])
        self.assertEqual(int(arms.sum()), 14)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient[:, ~arms] > 0).all())
        torch.testing.assert_close(gradient[:, arms], torch.zeros_like(gradient[:, arms]))
        torch.testing.assert_close(terms['task_reward'], self.legacy.reward(S, self.f, a+delta, ap-delta))

    def test_weak_arm_smoothing_interpolates_full_and_body_only_gradients(self):
        weights = [0.2 if any(part in n for part in ('shoulder', 'elbow', 'wrist')) else 1.
                   for n in self.robot.joint_names]
        weak = CarryMDP(self.kin, self.motion, self.legacy.default_q, self.layout.dim, self.robot.sensing,
                        action_scale=self.robot.action_scale_list(), residual_acc_weight=.2,
                        residual_acc_joint_weights=weights)
        a, ap = self.reference_actions(self.f)
        app = (self.motion.q[self.f-2]-self.legacy.default_q)/self.alpha
        delta = torch.full_like(a, .2, requires_grad=True)
        S = self.motion.ref_state(self.f)
        _, terms = weak.reward(S, self.f, a+delta, ap-delta, terms=True, a_prev2=app+delta)
        _, full = self.acc.reward(S, self.f, a+delta, ap-delta, terms=True, a_prev2=app+delta)
        grad, = torch.autograd.grad(terms['smooth_penalty'].sum(), delta, retain_graph=True)
        full_grad, = torch.autograd.grad(full['smooth_penalty'].sum(), delta)
        self.assertTrue(torch.isfinite(grad).all() and (grad > 0).all())
        torch.testing.assert_close(grad, full_grad*torch.tensor(weights), atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(terms['task_reward'], full['task_reward'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
