"""AMP data, reward, gradient and checkpoint contracts. Add --gpu inside trainreg for GPU checks."""

import copy
import io
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from blinkmimic.amp import AMPFeatures, DiscriminatorUpdates, ExpertPairs, MotionPrior, attach_motion_prior
from blinkmimic.box_carry import CarryMDP, SCENE_DIM, build_task, robot_spec
from blinkmimic.buffer import ReplayBuffer
from blinkmimic.dr import XiLayout
from blinkmimic.evaluation_metrics import CarryEvalMetrics, evaluation_group_key
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP, flatten_state, unflatten_state
from blinkmimic.motion import CONTROL_DT
from blinkmimic.networks import Policy, WorldModel, check_checkpoint_config
from blinkmimic.objective import actor_objective
from blinkmimic.reward_audit import RewardAudit
from blinkmimic.robot import get_robot
from blinkmimic.rotations import mv3, so3_exp, yaw_mat

GPU = "--gpu" in sys.argv
if GPU:
    sys.argv.remove("--gpu")


class AMPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(71)
        cls.robot = robot_spec(get_robot("g1"))
        cls.kin = Kinematics(cls.robot, "cpu")
        cls.layout = XiLayout(cls.robot, "cpu")
        cls.motion, cls.legacy = build_task("box_carry", cls.robot, None, cls.kin, "cpu", cls.layout.dim, max_clips=3)
        cls.f = cls.motion.clip_start + torch.tensor([20, 100, 40])

    def mdp(self, **kwargs):
        arms = [0. if any(s in n for s in ("shoulder", "elbow", "wrist")) else 1. for n in self.robot.joint_names]
        mdp = CarryMDP(self.kin, self.motion, self.legacy.default_q, self.layout.dim, self.robot.sensing,
                       imitation_mode="amp", action_acc_weight=.1, action_acc_joint_weights=arms, **kwargs)
        mdp.motion_prior = MotionPrior(self.kin)
        return mdp

    def replay(self):
        buffer = ReplayBuffer(3, 3, "cpu", 29, self.layout.dim, object_state=True)
        for i in range(5):
            f = self.f + i
            tr = {key: torch.zeros_like(value[0]) for key, value in buffer.data.items()}
            tr.update(s=flatten_state(self.motion.ref_state(f)), s1=flatten_state(self.motion.ref_state(f + 1)),
                      f=f, scene=self.motion.context(f), ep_step=torch.full((3,), i, dtype=torch.int32),
                      done=torch.ones(3, dtype=torch.bool), cut=torch.ones(3, dtype=torch.bool))
            buffer.add(tr)
        return buffer

    def test_reference_window_and_clip_tails(self):
        self.assertEqual(self.motion.goal_dim(45), 4056)
        self.assertAlmostEqual(45 * CONTROL_DT, .9)
        self.assertAlmostEqual(32 * CONTROL_DT, .64)
        future = self.motion.future_frames(self.f, 45)
        torch.testing.assert_close(future[:, 0], self.f + 1)
        torch.testing.assert_close(future[:, -1], self.f + 45)
        tails = self.motion.future_frames(self.motion.clip_end - 2, 45)
        self.assertTrue((self.motion.clip_id[tails] == torch.arange(3)[:, None]).all())
        torch.testing.assert_close(tails[:, -1], self.motion.clip_end)

    def test_experts_are_train_only_adjacent_and_stage_matched(self):
        sampler = ExpertPairs(self.motion)
        stages = torch.arange(3).repeat(400)
        f = sampler.sample_frames(stages)
        torch.testing.assert_close(self.motion.stages[self.motion.clip_id[f]], stages)
        torch.testing.assert_close(self.motion.clip_id[f], self.motion.clip_id[f + 1])
        self.assertTrue((f < self.motion.clip_end[self.motion.clip_id[f]]).all())
        val = copy.copy(self.motion)
        val.rows = [dict(row, split="val") for row in self.motion.rows]
        with self.assertRaisesRegex(ValueError, "train split"):
            ExpertPairs(val)

    def test_robot_goal_exposes_translation_error_without_changing_other_features(self):
        mdp = self.mdp(goal_anchor="robot")
        S = self.motion.ref_state(self.f)
        original = self.motion.goal(self.f, 45)
        torch.testing.assert_close(mdp.goal(S, self.f, 45), original)
        shift = torch.tensor([[2., -.5, .1], [-1., 3., -.1], [.2, -.3, .4]])
        p = (S[0] + shift).detach().requires_grad_()
        moved = (p,) + S[1:]
        goal = mdp.goal(moved, self.f, 45)
        delta = mv3(yaw_mat(S[1]).transpose(-1, -2), -shift)
        expected = original.clone()
        body_dim = 15 + 2 * self.robot.num_joints
        expected[:, :45 * body_dim].view(3, 45, body_dim)[..., :3] += delta[:, None]
        expected[:, 45 * body_dim:-6].view(3, 45, 17)[..., :3] += delta[:, None]
        torch.testing.assert_close(goal, expected)
        torch.testing.assert_close(self.legacy.goal(moved, self.f, 45), original)
        gradient = torch.autograd.grad(goal.sum(), p)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient.norm(dim=-1) > 0).all())
        self.assertEqual(goal.shape, (3, 4056))
        self.assertNotEqual(evaluation_group_key({}), evaluation_group_key({"carry_goal_anchor": "robot"}))

    def test_exported_goal_matches_live_goal_and_legacy_layout(self):
        from tools.export_onnx import PolicyWithMotion

        class GoalPolicy(torch.nn.Module):
            def forward(self, history, goal, depth=None):
                return goal

        S = self.motion.ref_state(self.f)
        p = S[0] + torch.tensor([1.2, -.7, .2])
        model = PolicyWithMotion(GoalPolicy(), self.motion, 45, "robot")
        result = model(torch.zeros(3, 2, self.legacy.obs_dim), self.f[:, None].float(), root_pos_w=p)
        torch.testing.assert_close(result[0], self.motion.goal(self.f, 45, p), atol=2e-6, rtol=2e-6)
        with self.assertRaisesRegex(ValueError, "root_pos_w"):
            model(torch.zeros(3, 2, self.legacy.obs_dim), self.f[:, None].float())
        old = PolicyWithMotion(GoalPolicy(), self.motion, 45)
        torch.testing.assert_close(old(torch.zeros(3, 2, self.legacy.obs_dim), self.f[:, None].float())[0],
                                   self.motion.goal(self.f, 45))

    def test_expert_and_live_features_share_projected_fk(self):
        features = AMPFeatures(self.kin)
        self.assertEqual(features.dim, 195)
        self.assertIn("left_palm_link", features.body_names)
        self.assertNotIn("left_wrist_yaw_link", features.body_names)
        cached = features.from_fk(self.motion.body_p[self.f], self.motion.body_R[self.f],
                                  self.motion.body_v[self.f], self.motion.body_w[self.f])
        torch.testing.assert_close(features(self.motion.ref_state(self.f)), cached, atol=2e-5, rtol=2e-5)

    def test_features_ignore_world_translation_and_heading(self):
        features = AMPFeatures(self.kin)
        S = self.motion.ref_state(self.f)
        yaw = so3_exp(torch.tensor([[0., 0., .7]]).expand(3, -1))
        shifted = (mv3(yaw, S[0]) + torch.tensor([1., -2., 0.]), yaw @ S[1], mv3(yaw, S[2]), mv3(yaw, S[3])) + S[4:]
        torch.testing.assert_close(features(S), features(shifted), atol=2e-5, rtol=2e-5)

    def test_replay_pairs_survive_terminal_resets_and_ring_wrap(self):
        w = self.replay().sample_transitions(200)
        expected = flatten_state(self.motion.ref_state(w["f"] + 1))
        torch.testing.assert_close(w["s1"], expected)
        self.assertTrue((w["f"] >= self.motion.clip_start[self.motion.clip_id[w["f"]]] + 22).all())

    def test_horizon_starts_never_use_an_incomplete_clip_tail(self):
        short = copy.copy(self.motion)
        short.clip_end = self.f + 34
        start, _ = self.replay().sample_horizon_starts(100, 2, short, 32)
        self.assertTrue((start["f"] + 32 <= short.clip_end[short.clip_id[start["f"]]]).all())
        torch.testing.assert_close(start["f"], self.f[short.clip_id[start["f"]]] + 2)

    def test_horizon_sampling_waits_when_only_tails_are_available(self):
        replay = self.replay()
        short = copy.copy(self.motion)
        short.clip_end = self.f + 31
        self.assertFalse(replay.horizon_start_weights(short, 32).any())
        self.assertEqual(replay.sample_horizon_starts(8, 2, short, 32), (None, None))
        empty = ReplayBuffer(3, 3, "cpu", 29, self.layout.dim, object_state=True)
        self.assertEqual(empty.sample_horizon_starts(8, 2, self.motion, 32), (None, None))
        # Collection can make a full horizon available without changing the sampler.
        short.clip_end = self.f + 34
        weights = replay.horizon_start_weights(short, 32)
        start, _ = replay.sample_horizon_starts(8, 2, short, 32, weights=weights)
        self.assertTrue((start["f"] + 32 <= short.clip_end[short.clip_id[start["f"]]]).all())

    def test_reward_matches_between_collection_and_imagination(self):
        mdp = self.mdp()
        f = self.f + 1
        S0, S1 = self.motion.ref_state(self.f), self.motion.ref_state(f)
        a = torch.zeros(3, 29)
        kw = dict(previous_state=S0, a_prev2=a, smooth_acc_valid=torch.ones(3, dtype=torch.bool))
        real, detail = mdp.reward(S1, f, a, a, terms=True, **kw)
        imaginary = mdp.reward(S1, f, a, a, **kw)
        torch.testing.assert_close(real, imaginary)
        torch.testing.assert_close(real, detail["amp_reward"] + detail["carry_task_reward"] - detail["control_penalty"])
        torch.testing.assert_close(detail["track"], torch.zeros(3))
        with patch.object(MDP, "reward", side_effect=AssertionError("tracking reward entered actor graph")):
            mdp.reward(S1, f, a, a, **kw)

    def test_far_approach_progress_is_signed_bounded_and_stops_before_grasp(self):
        mdp = self.mdp(approach_progress_weight=1.)
        S0 = list(self.motion.ref_state(self.f))
        S0[0] = S0[0].clone()
        S0[0][:, :2] = S0[6][:, :2] + torch.tensor([3., 0.])
        S0 = tuple(S0)
        a = torch.zeros(3, 29)
        def evaluate(dx, distance=3., move_box=False):
            p0 = S0[0].clone()
            p0[:, :2] = S0[6][:, :2] + torch.tensor([distance, 0.])
            previous = (p0,) + S0[1:]
            p1 = p0.clone()
            p1[:, 0] += dx
            current = (p1.requires_grad_(),) + S0[1:]
            if move_box:
                current = current[:6] + (current[6] + torch.tensor([dx, 0., 0.]),) + current[7:]
            reward, terms = mdp.reward(current, self.f, a, a, terms=True, previous_state=previous, a_prev2=a)
            torch.testing.assert_close(reward, sum(v for k, v in terms.items() if k.startswith("reward_")))
            torch.testing.assert_close(reward, terms["amp_reward"] + terms["train_carry_task_reward"] - terms["control_penalty"])
            return current, terms
        for dx, expected in [(0., 0.), (-.003, .01), (.003, -.01), (-.1, .02), (.1, -.02)]:
            _, terms = evaluate(dx)
            torch.testing.assert_close(terms["reward_approach_progress"], torch.tensor([expected, 0., 0.]), atol=1e-6, rtol=1e-5)
        for distance in [0., .3, .5]:
            current, terms = evaluate(0., distance=distance)
            torch.testing.assert_close(terms["reward_approach_progress"], torch.zeros(3))
            gp = torch.autograd.grad(terms["reward_approach_progress"].sum(), current[0])[0]
            self.assertTrue(torch.isfinite(gp).all())
            torch.testing.assert_close(gp, torch.zeros_like(gp))
        current, terms = evaluate(-.003)
        gp = torch.autograd.grad(terms["reward_approach_progress"].sum(), current[0])[0]
        self.assertTrue(torch.isfinite(gp).all())
        self.assertLess(gp[0, 0], 0.)
        torch.testing.assert_close(gp[1:], torch.zeros(2, 3))
        _, old = self.mdp().reward(S0, self.f, a, a, terms=True, previous_state=S0, a_prev2=a)
        _, stationary = evaluate(0.)
        torch.testing.assert_close(old["carry_task_reward"], stationary["carry_task_reward"])
        # Same pre-transition target: moving the box alone earns no progress.
        moved = S0[:6] + (S0[6] + torch.tensor([.1, 0., 0.]),) + S0[7:]
        _, terms = mdp.reward(moved, self.f, a, a, terms=True, previous_state=S0, a_prev2=a)
        torch.testing.assert_close(terms["reward_approach_progress"], torch.zeros(3))
        f = self.motion.clip_end[0:1]
        S = self.motion.ref_state(f)
        _, terms = mdp.reward(S, f, a[:1], a[:1], terms=True, previous_state=S, a_prev2=a[:1])
        self.assertEqual(terms["reward_approach_progress"].item(), 0.)

    def test_progress_gradient_reaches_actor_through_frozen_world_model(self):
        torch.manual_seed(73)
        mdp = self.mdp(goal_anchor="robot", approach_progress_weight=1.)
        pi = Policy(mdp.obs_dim, self.motion.goal_dim(45), 29, hist=2, enc_hidden=(16,), dec_hidden=(16,))
        wm = WorldModel(29, self.layout.dim, hidden=(16,), structured=True, object_state=True, scene_dim=SCENE_DIM)
        with torch.no_grad():
            wm.net[-1].weight.normal_(std=1e-5)
        wm.requires_grad_(False)
        f = self.f[:1]
        S = list(x[:1].clone() for x in self.motion.ref_state(self.f))
        S[0][:, :2] = S[6][:, :2] + torch.tensor([3., 0.])
        S[2].zero_()
        S = tuple(S)
        zero = torch.zeros(1, 29)
        action = pi(torch.zeros(1, 2, mdp.obs_dim), mdp.goal(S, f, 45))
        S1, _, _ = wm.step_z(S, action, zero, torch.zeros(1, self.layout.dim), scene=self.motion.context(f))
        _, terms = mdp.reward(S1, f + 1, action, zero, terms=True, previous_state=S, a_prev2=zero)
        (-terms["reward_approach_progress"].sum()).backward()
        gradients = [p.grad for p in pi.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
        self.assertTrue(all(p.grad is None for p in wm.parameters()))
        self.assertTrue(all(p.grad is None for p in mdp.motion_prior.parameters()))

    def test_physical_acceleration_masks_history_and_preserves_arms(self):
        mdp = self.mdp()
        S = self.motion.ref_state(self.f)
        a = torch.ones(3, 29, requires_grad=True)
        zeros = torch.zeros_like(a)
        kw = dict(previous_state=S, a_prev2=zeros)
        _, detail = mdp.reward(S, self.f, a, zeros, terms=True, smooth_acc_valid=torch.ones(3, dtype=torch.bool), **kw)
        grad = torch.autograd.grad(detail["smooth_penalty"].sum(), a)[0]
        body = mdp.action_acc_joint_weights.bool()
        self.assertTrue((grad[:, body] > 0).all())
        torch.testing.assert_close(grad[:, ~body], torch.zeros_like(grad[:, ~body]))
        _, masked = mdp.reward(S, self.f, a, zeros, terms=True, smooth_acc_valid=torch.zeros(3, dtype=torch.bool), **kw)
        torch.testing.assert_close(masked["smooth_penalty"], torch.zeros(3))

    def test_compiled_reward_preserves_state_gradient(self):
        mdp = self.mdp()
        S0 = self.motion.ref_state(self.f)
        S1 = self.motion.ref_state(self.f + 1)
        q = S1[4].detach().requires_grad_()
        state = S1[:4] + (q,) + S1[5:]
        a = torch.zeros(3, 29)
        kw = dict(previous_state=S0, a_prev2=a)
        eager = mdp.reward(state, self.f + 1, a, a, **kw)
        eager_grad = torch.autograd.grad(eager.sum(), q)[0]
        compiled = torch.compile(mdp.reward, backend="aot_eager")
        actual = compiled(state, self.f + 1, a, a, **kw)
        grad = torch.autograd.grad(actual.sum(), q)[0]
        torch.testing.assert_close(actual, eager)
        torch.testing.assert_close(grad, eager_grad, atol=2e-6, rtol=2e-5)

    def test_release_ignores_reference_palms_and_requires_stable_box_and_standing(self):
        mdp = self.mdp()
        f = self.motion.clip_end[2:3]
        S = list(self.motion.ref_state(f))
        S[8], S[9] = torch.zeros_like(S[8]), torch.zeros_like(S[9])
        a = torch.zeros(1, 29)
        def reward(state):
            return mdp.reward(tuple(state), f, a, a, terms=True, previous_state=tuple(state), a_prev2=a)
        base, detail = reward(S)
        self.assertEqual(detail["released"].item(), 1.)
        self.assertGreater(detail["release_recovery"].item(), .1)
        old = self.motion.body_p[f].clone()
        try:
            self.motion.body_p[f] += .3
            shifted, _ = reward(S)
            torch.testing.assert_close(base, shifted)
        finally:
            self.motion.body_p[f] = old
        moving = list(S)
        moving[9] = torch.ones_like(S[9])
        _, moving_detail = reward(moving)
        self.assertLess(moving_detail["release_recovery"].item(), detail["release_recovery"].item() * .1)
        crouched = list(S)
        crouched[4] = S[4].clone()
        for side in ("left", "right"):
            for joint, value in (("hip_pitch", -.9), ("knee", 1.8), ("ankle_pitch", -.9)):
                crouched[4][:, self.robot.joint_names.index(f"{side}_{joint}_joint")] = value
        _, crouch_detail = reward(crouched)
        self.assertLess(crouch_detail["release_recovery"].item(), detail["release_recovery"].item())

    def test_release_target_does_not_count_a_hand_still_on_the_box(self):
        mdp = self.mdp()
        f = self.motion.clip_end[2:3]
        S = list(self.motion.ref_state(f))
        S[8], S[9] = torch.zeros_like(S[8]), torch.zeros_like(S[9])
        a = torch.zeros(1, 29)
        def terms(state):
            return mdp.reward(tuple(state), f, a, a, terms=True, previous_state=tuple(state), a_prev2=a)[1]
        clear = terms(S)
        self.assertEqual(clear["release_success"].item(), 1.)
        fk = self.kin.forward(*S[:6])
        S[0] = S[0] + S[6] - fk[0][:, mdp.hands[0]]
        held = terms(S)
        self.assertEqual(held["released"].item(), 1.)  # target remains active in evaluation's denominator
        self.assertEqual(held["release_success"].item(), 0.)
        self.assertLess(held["hand_clear"].item(), .01)
        self.assertLess(held["release_recovery"].item(), clear["release_recovery"].item() * .02)

    def test_slip_gradient_reduces_speed_without_lifting_the_foot(self):
        mdp = self.mdp(amp_reward_weight=0.)
        f = self.f[1:2]  # carry: no release/standing bonus
        S = self.motion.ref_state(f)
        positions, orientation, velocity, angular = self.kin.forward(*S[:6])
        positions, velocity = positions.clone(), velocity.clone()
        positions[:, mdp.feet, 2] = .04
        velocity[:, mdp.feet, :2] = .25
        positions.requires_grad_(True)
        velocity.requires_grad_(True)
        a = torch.zeros(1, 29)
        with patch.object(self.kin, "forward", return_value=(positions, orientation, velocity, angular)):
            reward, detail = mdp.reward(S, f, a, a, terms=True, previous_state=S, a_prev2=a)
        gp, gv = torch.autograd.grad(reward.sum(), (positions, velocity))
        self.assertGreater(detail["foot_slip"].item(), 0.)
        self.assertTrue(torch.isfinite(gp).all() and torch.isfinite(gv).all())
        torch.testing.assert_close(gp[:, mdp.feet, 2], torch.zeros(1, 2))
        self.assertTrue((gv[:, mdp.feet, :2] < 0).all())

    def test_reward_audit_contributions_match_value_and_gradient(self):
        mdp = self.mdp()
        S = tuple(x.detach().clone().requires_grad_() for x in self.motion.ref_state(self.f))
        a = torch.full((3, 29), .1, requires_grad=True)
        previous = torch.full((3, 29), -.1)
        reward, detail = mdp.reward(S, self.f, a, previous, terms=True, previous_state=S,
                                    a_prev2=torch.full_like(a, .2))
        contributions = [value for name, value in detail.items() if name.startswith("reward_")]
        self.assertEqual(len(contributions), 12)
        combined = torch.stack(contributions).sum(0)
        torch.testing.assert_close(combined, reward)
        leaves = (*S, a)
        for value in contributions:
            gradients = (torch.autograd.grad(value.sum(), leaves, retain_graph=True, allow_unused=True)
                         if value.requires_grad else (None,) * len(leaves))
            self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gradients))
        actual = torch.autograd.grad(reward.sum(), leaves, retain_graph=True, allow_unused=True)
        audited = torch.autograd.grad(combined.sum(), leaves, allow_unused=True)
        for x, y in zip(actual, audited):
            if x is not None:
                torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)

    def test_reward_audit_uses_live_transitions_and_recomputes_gradients(self):
        mdp = self.mdp()
        S, S1 = self.motion.ref_state(self.f), self.motion.ref_state(self.f + 1)
        a = torch.zeros(3, 29)
        reward, terms = mdp.reward(S1, self.f + 1, a, a, terms=True, previous_state=S, a_prev2=a,
                                   smooth_acc_valid=torch.zeros(3, dtype=torch.bool))
        for value in terms.values():
            value[1] = float("nan")  # a finished world must not affect the audit
        audit = RewardAudit(torch.arange(3))
        audit.update(dict(s=flatten_state(S), s1=flatten_state(S1), f=self.f, a=a, a_prev=a,
                          ep_step=torch.zeros(3, dtype=torch.int32), r=reward, terms=terms),
                     torch.tensor([True, False, True]), a)
        result = audit.finish(self.robot, None, "train", None, 3,
                              {"args": {"imitation_mode": "amp", "carry_action_acc": .1},
                               "motion_prior": mdp.motion_prior.checkpoint(cpu=True)})
        self.assertEqual(result["gradient_samples"], 2)
        self.assertTrue(result["all_gradients_finite"])
        for key in ("value_decomposition_max_error", "live_state_recompute_max_error", "gradient_decomposition_max_error"):
            self.assertLess(result[key], 1e-5)

    def test_approach_palms_keep_a_far_gradient_and_common_metrics(self):
        old = self.mdp(amp_reward_weight=0., foot_slip_weight=0.)
        new = self.mdp(amp_reward_weight=0., foot_slip_weight=0., palm_far_weight=.25, palm_far_std=.5,
                       palm_far_kernel="inverse_quadratic")
        S = list(self.motion.ref_state(self.f))
        S[0] = (S[0] + torch.tensor([.5, 0., 0.])).requires_grad_()
        a = torch.zeros(3, 29)
        options = dict(terms=True, previous_state=tuple(S), a_prev2=a)
        ro, before = old.reward(tuple(S), self.f, a, a, **options)
        rn, after = new.reward(tuple(S), self.f, a, a, **options)
        go = torch.autograd.grad(before["reward_palms"].sum(), S[0], retain_graph=True)[0]
        gn = torch.autograd.grad(after["reward_palms"].sum(), S[0], retain_graph=True)[0]
        self.assertGreater(gn[0].norm().item(), 1e6 * go[0].norm().item())
        self.assertLess(gn[0, 0].item(), 0.)
        self.assertTrue(torch.isfinite(gn).all())
        torch.testing.assert_close(rn[1:], ro[1:])
        torch.testing.assert_close(after["carry_task_reward"], before["carry_task_reward"])
        torch.testing.assert_close(after["train_carry_task_reward"] - after["carry_task_reward"], after["palms_long_range_reward"])
        self.assertTrue((after["reward_palms"] <= .02).all())
        torch.testing.assert_close(rn, sum(value for key, value in after.items() if key.startswith("reward_")))
        _, at_goal = new.reward(self.motion.ref_state(self.f), self.f, a, a, **options)
        self.assertAlmostEqual(at_goal["reward_palms"][0].item(), .02, places=6)
        f = self.motion.clip_end[2:3]
        S = self.motion.ref_state(f)
        _, released = new.reward(S, f, a[:1], a[:1], terms=True, previous_state=S, a_prev2=a[:1])
        self.assertEqual(released["reward_palms"].item(), 0.)

    def test_far_approach_gradient_survives_metre_scale_errors(self):
        distances = torch.tensor([.5, 1., 2., 2.85, 4.])
        S = [x[:1].expand(len(distances), *x.shape[1:]).clone() for x in self.motion.ref_state(self.f)]
        S[0][:, 0] += distances
        S[0].requires_grad_()
        f, a = self.f[:1].expand(len(distances)), torch.zeros(len(distances), 29)
        gaussian = self.mdp(palm_far_weight=.25, palm_far_std=.5)
        self.assertEqual(gaussian.palm_far_kernel, "gaussian")  # legacy checkpoint default
        inverse = self.mdp(palm_far_weight=.25, palm_far_std=.5, palm_far_kernel="inverse_quadratic")
        gradients = []
        for mdp in (gaussian, inverse):
            _, detail = mdp.reward(tuple(S), f, a, a, terms=True, previous_state=tuple(S), a_prev2=a)
            gradients.append(torch.autograd.grad(detail["reward_palms"].sum(), S[0])[0])
        old, new = gradients
        self.assertTrue(torch.isfinite(new).all())
        self.assertTrue((new[:, 0] < -1e-6).all())
        self.assertGreater(new[3].norm().item(), 1e8 * old[3].norm().item())

    def test_evaluators_share_task_metrics_and_ignore_finished_worlds(self):
        mdp = self.mdp()
        S = self.motion.ref_state(self.f)
        a = torch.zeros(3, 29)
        _, terms = mdp.reward(S, self.f, a, a, terms=True, previous_state=S, a_prev2=a)
        metrics = CarryEvalMetrics(mdp, torch.arange(3))
        metrics.update({"terms": terms}, torch.ones(3, dtype=torch.bool))
        later = {key: value.clone() for key, value in terms.items()}
        for value in later.values():
            value[[0, 2]] = float("nan")
        metrics.update({"terms": later}, torch.tensor([False, True, False]))
        result = metrics.per_world()
        self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))
        torch.testing.assert_close(result["mean_amp_style"], terms["amp_style"])
        torch.testing.assert_close(result["mean_control_penalty"], terms["control_penalty"])
        torch.testing.assert_close(result["carry_task_norm_return"],
                                   terms["carry_task_reward"] * torch.tensor([1., 2., 1.]) /
                                   ((self.motion.clip_len - 1) * mdp.max_task_reward))

    def test_evaluation_groups_separate_modes_shapes_and_reward_settings(self):
        tracking = dict(task="box_carry", imitation_mode="tracking", goal_frames=10)
        amp = dict(tracking, imitation_mode="amp", goal_frames=45)
        self.assertNotEqual(evaluation_group_key(tracking), evaluation_group_key(amp))
        for changed in (dict(amp, amp_reward_weight=3.), dict(amp, carry_action_acc=.2), dict(amp, carry_palm_far_weight=.25),
                        dict(amp, carry_palm_far_kernel="inverse_quadratic"),
                        dict(amp, hist=3), dict(amp, amp="bf16")):
            self.assertNotEqual(evaluation_group_key(amp), evaluation_group_key(changed))
        self.assertEqual(evaluation_group_key(amp), evaluation_group_key(dict(amp, log_dir="another_run")))

    def test_style_only_gradient_crosses_32_world_model_steps(self):
        torch.manual_seed(13)
        prior = MotionPrior(self.kin)
        mdp = self.mdp(goal_anchor="robot")
        pi = Policy(self.legacy.obs_dim, self.motion.goal_dim(45), 29, hist=2,
                    enc_hidden=(16,), dec_hidden=(16,), visual_dim=32)
        wm = WorldModel(29, self.layout.dim, hidden=(16,), structured=True, object_state=True,
                        scene_dim=SCENE_DIM, z_dim=32)
        # A fitted model must depend on actions; an untouched WM deliberately has a zero final layer.
        with torch.no_grad():
            wm.net[-1].weight.normal_(std=1e-4)
        wm.requires_grad_(False)
        pi.depth_enc.requires_grad_(False)
        S = tuple(x[:1] for x in self.motion.ref_state(self.f))
        f = self.f[:1].clone()
        a_prev = torch.zeros(1, 29)
        xi = torch.zeros(1, self.layout.dim)
        history = torch.zeros(1, 2, self.legacy.obs_dim)
        z = torch.zeros(1, 32)
        rewards, actions = [], []
        wm_before = {k: v.clone() for k, v in wm.state_dict().items()}
        prior_before = {k: v.clone() for k, v in prior.state_dict().items()}
        for _ in range(32):
            a = pi(history, mdp.goal(S, f, 45), visual=z)
            S1, _, z = wm.step_z(S, a, a_prev, xi, z=z, scene=self.motion.context(f))
            style, _ = prior(prior.features(S), prior.features(S1), self.motion.stages[self.motion.clip_id[f]])
            rewards.append(style * .1)
            actions.append(a)
            f = f + 1
            o = self.legacy.obs(S1, a, xi, f)
            history = torch.cat((history[:, 1:], o[:, None]), 1)
            S, a_prev = S1, a
        J, _, _ = actor_objective(torch.stack(rewards), torch.zeros(32, 1), None, .99)
        first_action_grad = torch.autograd.grad(rewards[-1].sum(), actions[0], retain_graph=True)[0]
        self.assertTrue(torch.isfinite(first_action_grad).all())
        self.assertGreater(first_action_grad.abs().sum().item(), 0.)
        (-J.mean()).backward()
        gradients = [p.grad for p in pi.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
        for module, before in ((wm, wm_before), (prior, prior_before)):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
            for key, value in module.state_dict().items():torch.testing.assert_close(value, before[key])

    def test_discriminator_update_and_checkpoint_resume(self):
        torch.manual_seed(29)
        mdp = self.mdp()
        prior = mdp.motion_prior
        trainer = DiscriminatorUpdates(prior, self.motion, batch=12)
        replay = self.replay()
        before = {k: v.clone() for k, v in prior.discriminator.state_dict().items()}
        metrics = trainer.update(replay)
        self.assertTrue(all(torch.isfinite(v) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in prior.discriminator.state_dict().items()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in prior.parameters()))
        self.assertEqual(trainer.updates, 1)
        ck = {"motion_prior": prior.checkpoint(), **trainer.checkpoint()}
        stream = io.BytesIO()
        torch.save(ck, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=True)
        other = self.mdp()
        attach_motion_prior(other, restored)
        resumed = DiscriminatorUpdates(other.motion_prior, self.motion, batch=12)
        resumed.load_checkpoint(restored)
        rng = torch.get_rng_state()
        trainer.update(replay)
        torch.set_rng_state(rng)
        resumed.update(replay)
        self.assertEqual(resumed.updates, trainer.updates)
        for key, value in prior.state_dict().items():torch.testing.assert_close(value, other.motion_prior.state_dict()[key])
        # Periodic checkpoints require no optimizer to evaluate; old tracking policies require no prior.
        attach_motion_prior(other, {"motion_prior": prior.checkpoint()})
        attach_motion_prior(self.legacy, {})

    def test_batched_evaluation_uses_each_checkpoints_discriminator(self):
        mdp = self.mdp()
        a, b = MotionPrior(self.kin), MotionPrior(self.kin)
        with torch.no_grad():
            for p in a.discriminator.parameters():p.zero_()
            for p in b.discriminator.parameters():p.zero_()
            b.discriminator[-1].bias.fill_(1.)
        attach_motion_prior(mdp, [{"motion_prior": a.checkpoint()}, {"motion_prior": b.checkpoint()}])
        x = torch.zeros(6, 195)
        reward, score = mdp.motion_prior(x, x, torch.arange(3).repeat(2))
        torch.testing.assert_close(score[:3], torch.zeros(3))
        torch.testing.assert_close(score[3:], torch.ones(3))
        torch.testing.assert_close(reward[:3], torch.full((3,), .75))
        torch.testing.assert_close(reward[3:], torch.ones(3))

    def test_incompatible_initialization_and_reference_rewards_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "goal_frames"):
            check_checkpoint_config({"goal_frames": 10}, {"goal_frames": 45, "imitation_mode": "amp"}, "--init_from")
        with self.assertRaisesRegex(ValueError, "imitation_mode"):
            check_checkpoint_config({"goal_frames": 45}, {"goal_frames": 45, "imitation_mode": "amp"}, "--resume")
        check_checkpoint_config({}, {}, "legacy")
        with self.assertRaisesRegex(ValueError, "carry_goal_anchor"):
            check_checkpoint_config({}, {"carry_goal_anchor": "robot"}, "--resume")
        with self.assertRaisesRegex(ValueError, "carry_approach_progress_weight"):
            check_checkpoint_config({"imitation_mode": "amp"}, {"imitation_mode": "amp", "carry_approach_progress_weight": 1.}, "--resume")
        with self.assertRaisesRegex(ValueError, "reference-relative"):
            self.mdp(residual_acc_weight=.1)
        result = subprocess.run([sys.executable, "-m", "blinkmimic.train", "--pi_ref_skip"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AMP requires no", result.stderr)


@unittest.skipUnless(GPU, "GPU checks run with --gpu through trainreg")
class AMPGPUContracts(unittest.TestCase):
    def test_horizon_sampling_checks_readiness_outside_cuda_graph(self):
        replay = ReplayBuffer(3, 3, "cuda", 29, 1)
        replay.size = replay.T
        replay.data["f"].fill_(98)
        replay.use_device_ptr(device_size=True)
        motion = SimpleNamespace(clip_id=torch.zeros(100, dtype=torch.long, device="cuda"),
                                 clip_end=torch.tensor([99], device="cuda"))
        self.assertEqual(replay.sample_horizon_starts(6, 2, motion, 32), (None, None))
        replay.data["f"][0] = 0
        weights = replay.horizon_start_weights(motion, 32)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            replay.sample_horizon_starts(6, 2, motion, 32, weights=weights)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            start, _ = replay.sample_horizon_starts(6, 2, motion, 32, weights=weights)
        replay.data["f"][0] = 98
        replay.data["f"][1] = 10
        weights.copy_(replay.horizon_start_weights(motion, 32))
        graph.replay()
        torch.testing.assert_close(start["f"], torch.full_like(start["f"], 10))

    def test_terminal_pairs_and_compiled_reward_gradient(self):
        from blinkmimic.env import TrackingSim, make_sim
        robot = robot_spec(get_robot("g1"))
        kin, layout = Kinematics(robot, "cuda:0"), XiLayout(robot, "cuda:0")
        m, mdp = build_task("box_carry", robot, None, kin, "cuda:0", layout.dim, max_clips=3,
                            reward_options={"imitation_mode": "amp", "carry_action_acc": .1, "carry_palm_far_weight": .25,
                                            "carry_palm_far_kernel": "inverse_quadratic", "carry_goal_anchor": "robot",
                                            "carry_approach_progress_weight": 1.})
        mdp.motion_prior = MotionPrior(kin).cuda()
        env = TrackingSim(make_sim(robot), 3, robot, m, mdp, "cuda:0", vision=True)
        env.reset(torch.arange(3, device="cuda"), m.clip_end - 1)
        env.observe()
        S0, a0 = env.state(), env.a_prev.clone()
        goal_compiled = torch.compile(mdp.goal, options={"max_fusion_size": 8})
        torch.testing.assert_close(goal_compiled(S0, env.f, 45), mdp.goal(S0, env.f, 45), atol=2e-5, rtol=2e-5)
        shifted_p = (S0[0] + .3).detach().requires_grad_()
        shifted_S = (shifted_p,) + S0[1:]
        goal_gradient = torch.autograd.grad(goal_compiled(shifted_S, env.f, 45).sum(), shifted_p)[0]
        self.assertTrue(torch.isfinite(goal_gradient).all())
        self.assertTrue((goal_gradient.norm(dim=-1) > 0).all())
        a = (m.q[env.f + 1] - env.default_q) / env.alpha
        out = env.step(a)
        self.assertTrue(out["done"].all())
        S1 = unflatten_state(out["s1"], object_state=True)
        reward = mdp.reward(S1, out["f"] + 1, a, a0, previous_state=S0,
                            a_prev2=a0, smooth_acc_valid=torch.zeros(3, dtype=torch.bool, device="cuda"))
        torch.testing.assert_close(reward, out["r"])
        torch.testing.assert_close(out["depth1_clean"], env.camera.render(S1, out["scene"]))
        args = (S1, out["f"] + 1, a, a0)
        kw = dict(previous_state=S0, a_prev2=a0, smooth_acc_valid=torch.zeros(3, dtype=torch.bool, device="cuda"))
        eager = mdp.reward(*args, **kw)
        compiled = torch.compile(mdp.reward, options={"max_fusion_size": 8})
        torch.testing.assert_close(compiled(*args, **kw), eager, atol=2e-5, rtol=2e-5)
        q = S1[4].detach().requires_grad_()
        state = S1[:4] + (q,) + S1[5:]
        compiled_reward = compiled(state, *args[1:], **kw)
        grad = torch.autograd.grad(compiled_reward.sum(), q)[0]
        self.assertTrue(torch.isfinite(grad).all())
        f = m.clip_start + 20
        previous = list(m.ref_state(f - 1))
        previous[0] = previous[0].clone()
        previous[0][:, :2] = previous[6][:, :2] + torch.tensor([3., 0.], device="cuda")
        current = list(m.ref_state(f))
        current[0] = (previous[0] + torch.tensor([-.003, 0., 0.], device="cuda")).requires_grad_()
        zero = torch.zeros(3, 29, device="cuda")
        options = dict(previous_state=tuple(previous), a_prev2=zero)
        eager = mdp.reward(tuple(current), f, zero, zero, **options)
        result = compiled(tuple(current), f, zero, zero, **options)
        torch.testing.assert_close(result, eager, atol=2e-5, rtol=2e-5)
        gp = torch.autograd.grad(result.sum(), current[0])[0]
        self.assertTrue(torch.isfinite(gp).all())
        self.assertLess(gp[0, 0], -3.)


if __name__ == "__main__":
    unittest.main()
