"""Carry reward, policy gradient and simulator regressions.

Run with PYTHONPATH=src .venv/bin/python -m unittest tools.test_carry_rewards.
Pass --gpu for the registered simulator/compiler integration.
"""
import sys
import unittest
import torch
from blinkmimic.box_carry import build_task, robot_spec
from blinkmimic.dr import XiLayout
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import unflatten_state
from blinkmimic.networks import Critic, Policy, WorldModel, make_policy
from blinkmimic.objective import actor_objective
from blinkmimic.robot import get_robot
GPU = "--gpu" in sys.argv
if GPU:
    sys.argv.remove("--gpu")


class PolicyGradientTests(unittest.TestCase):
    def test_actor_keeps_visual_encoder_out_of_policy_gradient(self):
        from blinkmimic.vision import DEPTH_SHAPE
        model = Policy(3, 4, 2, hist=2, enc_hidden=(16, 8), dec_hidden=(16, 8), visual_dim=4)
        depth = torch.randn(5, *DEPTH_SHAPE)
        actions = model(torch.randn(5, 2, 3), torch.randn(5, 4), depth=depth)
        actions.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.depth_enc.parameters()))
        self.assertGreater(model.dec[-1].weight.grad.abs().sum().item(), 0.)


    def test_push_keeps_box_state_and_does_not_end_the_episode(self):
        from types import SimpleNamespace
        from blinkmimic.env import TrackingSim

        written = []
        state = tuple(torch.zeros(2, 3) for _ in range(10))
        env = SimpleNamespace(push_timer=torch.tensor([0., 0.]), device="cpu", state=lambda: state,
                              spec=SimpleNamespace(dr=SimpleNamespace(push_velocity=(.1,) * 6, push_interval_s=(1., 2.))),
                              robot=SimpleNamespace(write_root_link_velocity_to_sim=lambda vel, env_ids: written.append((vel, env_ids))))
        done = torch.tensor([False, True])
        pushed = TrackingSim._push(env, done)
        torch.testing.assert_close(pushed, torch.tensor([True, False]))
        torch.testing.assert_close(written[0][1], torch.tensor([0]))
        self.assertEqual(written[0][0].shape, (1, 6))
        torch.testing.assert_close(done, torch.tensor([False, True]))


class CarryRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = robot_spec(get_robot("g1"))
        cls.kin = Kinematics(cls.robot, "cpu")
        cls.layout = XiLayout(cls.robot, "cpu")
        cls.motion, cls.legacy = build_task("box_carry", cls.robot, None, cls.kin, "cpu", cls.layout.dim, max_clips=3)
        _, cls.shaped = build_task("box_carry", cls.robot, None, cls.kin, "cpu", cls.layout.dim, max_clips=3,
                                  reward_options={"carry_reward_profile": "grasp_place"})
        frames = torch.arange(cls.motion.num_frames)
        stage = cls.motion.stages[cls.motion.clip_id]
        contact = cls.motion.hand_contact.amax(-1) > .5
        cls.frames = torch.stack((frames[(stage == 0) & ~contact][0], frames[stage == 1][20],
                                  frames[(stage == 2) & ~contact][0]))

    def reward(self, state, mdp=None):
        f = self.frames
        action = (self.motion.q[f] - self.legacy.default_q) / torch.tensor(self.robot.action_scale_list())
        return (mdp or self.shaped).reward(state, f, action, action, terms=True)

    def test_stage_gates_and_fixed_original_task_score(self):
        S = self.motion.ref_state(self.frames)
        old, detail_old = self.reward(S, self.legacy)
        new, detail = self.reward(S)
        torch.testing.assert_close(old - new, detail["grasp_place_penalty"], atol=1e-8, rtol=1e-5)
        torch.testing.assert_close(detail_old["grasp_place_penalty"], detail["grasp_place_penalty"], atol=0, rtol=0)
        torch.testing.assert_close(detail["task_reward"], detail_old["task_reward"], atol=0, rtol=0)
        self.assertEqual(detail["grasp_penalty"][0].item(), 0.)
        self.assertEqual(detail["grasp_penalty"][2].item(), 0.)
        torch.testing.assert_close(detail["place_penalty"][:2], torch.zeros(2))

    def test_drop_penalty_is_bounded_and_disabled_after_release(self):
        S = self.motion.ref_state(self.frames)
        position = S[6].clone()
        position[:, 2] -= .06
        _, detail = self.reward(S[:6] + (position,) + S[7:])
        torch.testing.assert_close(detail["drop_penalty"], torch.tensor([0., .0025, 0.]), atol=2e-8, rtol=1e-5)
        position[:, 2] -= .5
        _, detail = self.reward(S[:6] + (position,) + S[7:])
        torch.testing.assert_close(detail["drop_penalty"], torch.tensor([0., .01, 0.]), atol=1e-8, rtol=0)

    def test_release_penalty_requires_position_and_speed_stability(self):
        S = self.motion.ref_state(self.frames)
        S = S[:8] + (torch.zeros_like(S[8]), S[9])
        _, stable = self.reward(S)
        self.assertEqual(stable["place_penalty"][2].item(), 0.)
        position = S[6].clone()
        position[2, 0] += .1
        _, misplaced = self.reward(S[:6] + (position,) + S[7:])
        expected = .005 * (1 - torch.exp(torch.tensor(-1.)))
        torch.testing.assert_close(misplaced["place_penalty"][2], expected, atol=1e-8, rtol=1e-5)
        velocity = S[8].clone()
        velocity[2, 0] = .15
        _, moving = self.reward(S[:8] + (velocity, S[9]))
        torch.testing.assert_close(moving["place_penalty"][2], expected, atol=1e-8, rtol=1e-5)

    def test_bptt_crosses_the_real_world_model_to_the_actor(self):
        torch.manual_seed(3)
        f = self.frames
        args = dict(hist=2, enc_hidden="16,8", dec_hidden="16,8", pi_ref_skip=True)
        pi = make_policy(self.shaped.obs_dim, self.motion.goal_dim(2), self.robot.num_joints, self.robot, args)
        wm = WorldModel(self.robot.num_joints, self.layout.dim, hidden=(32, 32), structured=True,
                        object_state=True, scene_dim=10)
        critic_dim = self.shaped.critic_features(self.motion.ref_state(f), f, torch.zeros(3, self.layout.dim)).shape[-1]
        critic = Critic(critic_dim + self.motion.goal_dim(2))
        for model in (wm, critic):
            model.requires_grad_(False)
        S = self.motion.ref_state(f)
        xi = torch.zeros(3, self.layout.dim)
        previous = (self.motion.q[f] - self.legacy.default_q) / torch.tensor(self.robot.action_scale_list())
        obs = self.shaped.obs(S, previous, xi, f).unsqueeze(1).repeat(1, 2, 1)
        action = pi(obs, self.motion.goal(f, 2))
        S1 = wm.step_z(S, action, previous, xi, scene=self.motion.context(f))[0]
        reward = self.shaped.reward(S1, f + 1, action, previous)
        boot = critic(torch.cat((self.shaped.critic_features(S1, f + 1, xi), self.motion.goal(f + 1, 2)), -1))
        objective, _, _ = actor_objective(reward.unsqueeze(0), torch.zeros(1, 3), boot, .99)
        (-objective.mean()).backward()
        self.assertTrue(all(p.grad is None for model in (wm, critic) for p in model.parameters()))
        gradients = [p.grad for p in pi.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)


@unittest.skipUnless(GPU, "pass --gpu for the actual carry/policy/compiled reward integration")
class CarrySimTests(unittest.TestCase):
    def test_real_timeout_reward_snapshot_and_compiled_policy(self):
        from blinkmimic.env import TrackingSim, make_sim
        robot = robot_spec(get_robot("g1"))
        kin, layout = Kinematics(robot, "cuda:0"), XiLayout(robot, "cuda:0")
        motion, mdp = build_task("box_carry", robot, None, kin, "cuda:0", layout.dim, max_clips=3,
                                 reward_options={"carry_reward_profile": "grasp_place"})
        mdp.feet = [robot.tracked_bodies.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        motion.set_ref_contact(mdp.feet)
        env = TrackingSim(make_sim(robot), 3, robot, motion, mdp, "cuda:0", randomize=False,
                          max_episode_steps=1, reset_noise=False, vision=True, training_contact_reward=.25)
        frames = motion.clip_start + 10
        env.reset(torch.arange(3, device="cuda:0"), frames)
        previous = env.a_prev.clone()
        obs = env.observe().unsqueeze(1).repeat(1, 2, 1)
        args = dict(hist=2, enc_hidden="32,16", dec_hidden="32,16", pi_ref_skip=True,
                    vision="latent", visual_dim=4)
        pi = make_policy(mdp.obs_dim, motion.goal_dim(2), robot.num_joints, robot, args).to("cuda:0")
        goal = motion.goal(frames, 2)
        with torch.no_grad():
            visual = pi.depth_enc(env.depth)
        action = (motion.q[frames + 1] - env.default_q) / env.alpha
        out = env.step(action)
        S1 = unflatten_state(out["s1"], object_state=True)
        expected = mdp.reward(S1, frames + 1, action, previous, w_contact=.25)
        torch.testing.assert_close(out["r"], expected, atol=2e-7, rtol=2e-5)
        self.assertTrue(out["done"].all())
        self.assertTrue((env.ep_step == 0).all())
        self.assertFalse(out["term"].any())
        self.assertAlmostEqual(env.max_step_reward, .20)
        eager = pi(obs, goal, visual=visual)
        compiled = torch.compile(pi)(obs, goal, visual=visual)
        torch.testing.assert_close(compiled, eager, atol=2e-4, rtol=2e-4)
        compiled.square().mean().backward()
        self.assertTrue(torch.isfinite(pi.dec[0].weight.grad).all())
        self.assertGreater(pi.dec[0].weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in pi.depth_enc.parameters()))
        # The new differentiable reward must also survive the production compiler.
        compiled_reward = torch.compile(mdp.reward, options={"max_fusion_size": 8})
        torch.testing.assert_close(compiled_reward(S1, frames + 1, action, previous, w_contact=.25), expected,
                                   atol=2e-7, rtol=2e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
