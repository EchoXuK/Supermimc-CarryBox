"""Reference gait incentives, sole kinematics and low-amplitude recontact events."""
import unittest

import numpy as np
import torch

from blinkmimic.box_carry import build_task, build_mdp, robot_spec
from blinkmimic.dr import XiLayout
from blinkmimic.kinematics import Kinematics
from blinkmimic.quality import contact_events, gait_quality, match_events
from blinkmimic.robot import get_robot


class GaitEventTests(unittest.TestCase):
    def test_gradient_attribution_bypasses_compiled_methods_and_restores_them(self):
        import ast
        from pathlib import Path
        from types import SimpleNamespace
        parameter = torch.nn.Parameter(torch.tensor([.2, -.3]))
        parameter.grad = torch.ones_like(parameter)
        saved_grad = parameter.grad
        def original():
            return parameter.square().sum()
        def compiled():
            raise AssertionError('Retained diagnostic graph entered compiled method')
        compiled._torchdynamo_orig_callable = original
        mdp = SimpleNamespace(obs=compiled, reward=compiled)
        motion = SimpleNamespace(goal=compiled)
        def imagine(**kwargs):
            a, b, c = mdp.obs(), mdp.reward(), motion.goal()
            return None, {'gait': a, 'task_and_value': b + c, 'total': a + b + c}
        modules = [torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)]
        ns = dict(torch=torch, wm_modules=modules[0], critic=modules[1], actor_params=[parameter],
                  imagine_actor=imagine, mdp=mdp, motion=motion, dev='cpu', args=SimpleNamespace(pi_batch=16))
        source = Path(__file__).resolve().parents[1]/'src/blinkmimic/train.py'
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'reward_gradient_probe')
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'probe_compile_boundary', 'exec'), ns)
        result = ns['reward_gradient_probe']()
        self.assertGreater(result['total']['gradnorm'], 0)
        self.assertTrue(result['total']['finite'])
        self.assertIs(mdp.obs, compiled)
        self.assertIs(mdp.reward, compiled)
        self.assertIs(motion.goal, compiled)
        self.assertIs(parameter.grad, saved_grad)
        def failure(**kwargs):
            self.assertIs(mdp.obs, original)
            raise RuntimeError('probe failed')
        ns['imagine_actor'] = failure
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            ns['reward_gradient_probe']()
        self.assertIs(mdp.obs, compiled)
        self.assertIs(mdp.reward, compiled)
        self.assertIs(motion.goal, compiled)
        self.assertIs(parameter.grad, saved_grad)

    def test_imagination_uses_the_executed_action_boundary(self):
        import ast
        from pathlib import Path
        from types import SimpleNamespace
        from tools.test_training_windows import motion_fixture
        motion = motion_fixture()
        seen = []
        def step(state, action, previous, xi, z, scene):
            seen.append(action)
            return state, None, z
        ns = dict(torch=torch, pi=lambda obs, goal, visual=None: torch.full((1, 2), 100.),
                  wm=SimpleNamespace(step_z=step), motion=motion, K=10,
                  mdp=SimpleNamespace(obs=lambda s, a, xi, f: s[0], add_noise=lambda x: x))
        source = Path(__file__).resolve().parents[1]/'src/blinkmimic/train.py'
        fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name=='_imag_step_visual')
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'action_boundary', 'exec'), ns)
        state = (torch.zeros(1, 3),)*6
        out = ns['_imag_step_visual'](state, torch.zeros(1, 2, 3), torch.zeros(1, 2), torch.zeros(1, dtype=torch.long),
                                      None, None, motion.goal(torch.zeros(1, dtype=torch.long), 10), torch.zeros(1, 2), None)
        torch.testing.assert_close(seen[0], torch.full((1, 2), 20.))
        torch.testing.assert_close(out[1], seen[0])

    def test_heel_tap_below_four_cm_is_detected_without_full_foot_liftoff(self):
        ref = np.zeros((101, 2, 4, 3))
        actual = ref.copy()
        actual[35:43, 0, :2, 2] = .012
        result = gait_quality(actual, ref)
        self.assertEqual(result['heel_extra_touchdowns'], 1)
        self.assertEqual(result['whole_extra_touchdowns'], 0)
        self.assertEqual(result['extra_recontacts'], 1)
        self.assertEqual(gait_quality(ref, ref)['extra_recontacts'], 0)

    def test_one_tick_contact_noise_is_ignored(self):
        h = np.zeros(101)
        h[30] = .03
        _, touchdown, liftoff = contact_events(h)
        self.assertEqual(len(touchdown) + len(liftoff), 0)

    def test_heel_recontact_during_toe_support_is_not_a_swing_interruption(self):
        ref = np.zeros((101, 2, 4, 3))
        ref[25:75, 0, :2, 2] = .025
        actual = ref.copy()
        actual[40:50, 0, :2, 2] = 0.
        result = gait_quality(actual, ref)
        self.assertEqual(result['extra_recontacts'], 1)
        self.assertEqual(result['swing_interruptions'], 0)

    def test_normal_and_expert_repeated_steps_are_preserved(self):
        ref = np.zeros((201, 2, 4, 3))
        for start, foot in [(35, 0), (65, 0), (105, 1)]:
            ref[start:start + 10, foot, :, 2] = .03
        result = gait_quality(ref, ref)
        self.assertEqual(result['extra_recontacts'], 0)
        self.assertEqual(result['missed_steps'], 0)
        self.assertEqual(result['repeated_same_foot'], 1)
        self.assertEqual(result['reference_repeated_same_foot'], 1)

    def test_event_matching_maximizes_count_before_timing(self):
        self.assertEqual(match_events([10, 13], [5, 12], 6), [(0, 0), (1, 1)])

    def test_failure_prefix_and_reset_do_not_create_landings(self):
        ref = np.zeros((201, 2, 4, 3))
        self.assertEqual(gait_quality(ref[:51], ref)['extra_recontacts'], 0)
        self.assertEqual(len(contact_events(np.zeros(20))[1]), 0)


class GaitRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.robot = robot_spec(get_robot('g1'))
        cls.kin = Kinematics(cls.robot, 'cpu')
        cls.dim = XiLayout(cls.robot, 'cpu').dim
        cls.motion, cls.old = build_task('box_carry', cls.robot, None, cls.kin, 'cpu', cls.dim,
                                         max_clips=6, reward_options={'carry_reward_profile': 'grasp_place'})
        cls.new = build_mdp('box_carry', cls.robot, cls.kin, cls.motion, 'cpu', cls.dim,
                            {'carry_reward_profile': 'grasp_place', 'carry_gait_profile': 'reference_phase'})
        cls.f = cls.motion.clip_start + torch.minimum(cls.motion.clip_len - 1, torch.tensor(30))
        cls.alpha = torch.tensor(cls.robot.action_scale_list())

    def inputs(self, f=None):
        f = self.f if f is None else f
        S = self.motion.ref_state(f)
        apf = torch.maximum(f - 1, self.motion.clip_start[self.motion.clip_id[f]])
        a = (self.motion.q[f] - self.new.default_q) / self.alpha
        ap = (self.motion.q[apf] - self.new.default_q) / self.alpha
        return S, a, ap

    def test_expert_has_zero_cost_and_original_task_score_is_unchanged(self):
        S, a, ap = self.inputs()
        old, od = self.old.reward(S, self.f, a, ap, terms=True)
        new, nd = self.new.reward(S, self.f, a, ap, terms=True, w_contact=.25)
        torch.testing.assert_close(nd['gait_penalty'], torch.zeros_like(new), atol=1e-9, rtol=0)
        torch.testing.assert_close(new, old)
        torch.testing.assert_close(nd['task_reward'], od['task_reward'])
        torch.testing.assert_close(nd['foot_contact_reward'], torch.zeros_like(new))

    def test_raised_foot_and_wrong_leg_velocity_have_corrective_gradients(self):
        S, a, ap = self.inputs()
        qd = (S[5] + .2).detach().requires_grad_()
        state = S[:5] + (qd,) + S[6:]
        bodies = list(self.kin.forward(*state[:6]))
        lift = torch.tensor(.008, requires_grad=True)
        offset = torch.zeros_like(bodies[0])
        offset[..., self.kin.sole_idx, 2] = 1.
        bodies[0] = bodies[0] + lift * offset
        costs = self.new.gait_costs(state, self.f, bodies)
        g_lift, g_qd = torch.autograd.grad(sum(v.sum() for v in costs), (lift, qd))
        self.assertTrue(torch.isfinite(g_lift) and torch.isfinite(g_qd).all())
        self.assertGreater(g_lift.item(), 0)
        # Isolate the direct velocity term to avoid intentional FK cross-coupling.
        qd2 = (S[5] + .2).detach().requires_grad_()
        lv = self.new.gait_costs(S[:5] + (qd2,) + S[6:], self.f, self.kin.forward(*S[:6]))[2]
        g = torch.autograd.grad(lv.sum(), qd2)[0]
        self.assertTrue((g[:, self.new.leg_ids] > 0).all())

    def test_v2_height_correction_survives_large_foot_velocity_error(self):
        separate = build_mdp('box_carry', self.robot, self.kin, self.motion, 'cpu', self.dim,
                             {'carry_reward_profile': 'grasp_place',
                              'carry_gait_profile': 'reference_phase_v2'})
        S, a, ap = self.inputs()
        exact, detail = separate.reward(S, self.f, a, ap, terms=True)
        old, old_detail = self.new.reward(S, self.f, a, ap, terms=True)
        torch.testing.assert_close(detail['gait_penalty'], torch.zeros_like(exact), atol=1e-9, rtol=0)
        torch.testing.assert_close(exact, old)
        torch.testing.assert_close(detail['task_reward'], old_detail['task_reward'])
        bodies = self.kin.forward(*S[:6])

        def height_gradient(mdp, speed_error):
            lift = torch.tensor(.008, requires_grad=True)
            position = torch.zeros_like(bodies[0])
            position[..., self.kin.sole_idx, 2] = 1.
            velocity = torch.zeros_like(bodies[2])
            velocity[..., self.kin.sole_idx, 2] = speed_error
            altered = (bodies[0] + lift * position, bodies[1],
                       bodies[2] + velocity, bodies[3])
            motion_loss = mdp.gait_costs(S, self.f, altered)[1].sum()
            return torch.autograd.grad(motion_loss, lift)[0]

        slow = height_gradient(separate, 0.)
        fast = height_gradient(separate, .9)
        self.assertTrue(torch.isfinite(fast))
        self.assertGreater(fast.item(), 0.)
        torch.testing.assert_close(fast, slow, atol=1e-5, rtol=1e-5)
        old_slow = height_gradient(self.new, 0.)
        old_fast = height_gradient(self.new, .9)
        self.assertLess(old_fast.item(), old_slow.item() / 1000.)

    def test_v2_velocity_correction_survives_height_error(self):
        separate = build_mdp('box_carry', self.robot, self.kin, self.motion, 'cpu', self.dim,
                             {'carry_gait_profile': 'reference_phase_v2'})
        S, _, _ = self.inputs()
        bodies = self.kin.forward(*S[:6])

        def velocity_gradient(lift):
            speed = torch.tensor(.15, requires_grad=True)
            position = torch.zeros_like(bodies[0])
            position[..., self.kin.sole_idx, 2] = lift
            direction = torch.zeros_like(bodies[2])
            direction[..., self.kin.sole_idx, 2] = 1.
            altered = (bodies[0] + position, bodies[1],
                       bodies[2] + speed * direction, bodies[3])
            return torch.autograd.grad(separate.gait_costs(S, self.f, altered)[1].sum(), speed)[0]

        near, far = velocity_gradient(0.), velocity_gradient(.06)
        self.assertTrue(torch.isfinite(far))
        self.assertGreater(far.item(), 0.)
        torch.testing.assert_close(near, far, atol=1e-5, rtol=1e-5)

    def test_freezing_or_shortening_a_reference_swing_is_penalized(self):
        m = self.motion
        f = m.sole_v.square().sum((-1, -2, -3)).argmax().view(1)
        S, _, _ = self.inputs(f)
        bodies = self.kin.forward(*S[:6])
        expert = self.new.gait_costs(S, f, bodies)
        stopped = (bodies[0], bodies[1], torch.zeros_like(bodies[2]), torch.zeros_like(bodies[3]))
        shortened = (bodies[0], bodies[1], bodies[2] * .5, bodies[3] * .5)
        self.assertGreater(self.new.gait_costs(S, f, stopped)[1].item(), expert[1].item())
        self.assertGreater(self.new.gait_costs(S, f, shortened)[1].item(), expert[1].item())

    def test_v3_large_errors_retain_height_speed_and_leg_corrections(self):
        mdp = build_mdp('box_carry', self.robot, self.kin, self.motion, 'cpu', self.dim,
                        {'carry_gait_profile': 'reference_phase_v3'})
        S, a, ap = self.inputs()
        _, exact = mdp.reward(S, self.f, a, ap, terms=True)
        torch.testing.assert_close(exact['gait_penalty'], torch.zeros_like(a[:, 0]), atol=1e-9, rtol=0)
        bodies = self.kin.forward(*S[:6])
        height = torch.tensor(.10, requires_grad=True)
        speed = torch.tensor(1.5, requires_grad=True)
        leg = torch.tensor(4., requires_grad=True)
        direction = torch.zeros_like(bodies[0]); direction[..., self.kin.sole_idx, 2] = 1.
        altered = (bodies[0] + height * direction, bodies[1], bodies[2] + speed * direction, bodies[3])
        qd = S[5].clone(); qd[:, mdp.leg_ids] = qd[:, mdp.leg_ids] + leg
        _, motion_cost, leg_cost = mdp.gait_costs(S[:5] + (qd,) + S[6:], self.f, altered)
        gradients = torch.autograd.grad((motion_cost + leg_cost).sum(), (height, speed, leg))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient))
            self.assertGreater(gradient.item(), 1e-5)
        self.assertTrue(bool((motion_cost <= 2).all() and (leg_cost <= 1).all()))
        # The rational tail must correct errors on either side of the reference.
        e = torch.tensor([-4., -.1, .1, 4.], requires_grad=True)
        derivative = torch.autograd.grad(mdp.phase_cost(e.square()).sum(), e)[0]
        self.assertTrue(bool((derivative * e > 0).all()))

    def test_reach_profile_follows_reference_and_only_shapes_lift(self):
        mdp = build_mdp('box_carry', self.robot, self.kin, self.motion, 'cpu', self.dim,
                        {'carry_reward_profile': 'grasp_place_reach'})
        S, a, ap = self.inputs()
        reward, detail = mdp.reward(S, self.f, a, ap, terms=True)
        legacy_reward, legacy_detail = self.old.reward(S, self.f, a, ap, terms=True)
        torch.testing.assert_close(detail['reach_penalty'], torch.zeros_like(reward), atol=1e-9, rtol=0)
        torch.testing.assert_close(reward, legacy_reward, atol=1e-9, rtol=0)
        # A 60 cm approach error still has an inward gradient; other stages retain
        # their original shaping, and the original task score remains comparable.
        offset = torch.tensor(.6, requires_grad=True)
        direction = torch.zeros_like(S[0]); direction[..., 0] = 1
        altered = (S[0] + offset * direction,) + S[1:]
        reward, detail = mdp.reward(altered, self.f, a, ap, terms=True)
        old_reward, old_detail = self.old.reward(altered, self.f, a, ap, terms=True)
        is_lift = self.motion.stages[self.motion.clip_id[self.f]] == 0
        self.assertTrue(bool((detail['reach_penalty'][is_lift] > 0).all()))
        self.assertTrue(bool((detail['reach_penalty'][~is_lift] == 0).all()))
        for name in ('reach_root_penalty', 'reach_hand_penalty'):
            gradient = torch.autograd.grad(detail[name].sum(), offset, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient))
            self.assertGreater(gradient.item(), 1e-5)
        torch.testing.assert_close(reward, old_reward - detail['reach_penalty'])
        torch.testing.assert_close(detail['task_reward'], old_detail['task_reward'])
        self.assertTrue(bool((detail['reach_penalty'] <= .02).all()))

    def test_sole_velocity_matches_position_finite_difference(self):
        from blinkmimic.rotations import so3_exp
        S, _, _ = self.inputs()
        p, R, v, w, q, qd = S[:6]
        dt = 1e-4
        P, V = self.kin.sole_points(*self.kin.forward(p, R, v, w, q, qd))
        P1, _ = self.kin.sole_points(*self.kin.forward(p + dt*v, so3_exp(dt*w) @ R, v, w, q + dt*qd, qd))
        torch.testing.assert_close((P1-P)/dt, V, atol=.003, rtol=.03)

    def test_perfect_terminal_reference_and_batch_shapes(self):
        f = self.motion.clip_end
        S, a, ap = self.inputs(f)
        _, terms = self.new.reward(S, f, a, ap, terms=True)
        self.assertEqual(terms['gait_penalty'].shape, f.shape)
        torch.testing.assert_close(terms['gait_penalty'], torch.zeros_like(f, dtype=torch.float), atol=1e-9, rtol=0)

    def test_sole_geometry_and_velocity_respect_mirroring(self):
        from blinkmimic.mirror import Mirror
        layout = XiLayout(self.robot, 'cpu')
        mirror = Mirror(self.robot, layout.slices, 'cpu')
        S, _, _ = self.inputs()
        left = self.kin.sole_points(*self.kin.forward(*S[:6]))
        right = self.kin.sole_points(*self.kin.forward(*mirror.state(S)[:6]))
        reflection = torch.tensor([1., -1., 1.])
        for original, reflected in zip(left, right):
            expected = original[:, [1, 0]][:, :, [1, 0, 3, 2]] * reflection
            torch.testing.assert_close(reflected, expected, atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
