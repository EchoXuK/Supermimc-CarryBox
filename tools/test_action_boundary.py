"""PD control and stored transitions share the environment action boundary."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys

import torch

GPU = '--gpu' in sys.argv
if GPU:
    sys.argv.remove('--gpu')


class ActionBoundaryTests(unittest.TestCase):
    def test_all_step_consumers_receive_the_executed_action(self):
        source = Path(__file__).resolve().parents[1] / 'src/blinkmimic/env.py'
        cls = next(n for n in ast.parse(source.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'TrackingSim')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step_core')
        ns = dict(torch=torch, DECIMATION=1, PHYSICS_DT=.002,
                  flatten_state=lambda state: state[0])
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'step_action_boundary', 'exec'), ns)
        seen = {}
        def reward(state, frame, action, previous, **kwargs):
            seen['reward_action'] = action.clone()
            return action.sum(-1), {}
        state = (torch.zeros(1, 3),)
        env = SimpleNamespace(
            pipe=SimpleNamespace(stateful=False), camera=None, _obs=None, _obs_state=None,
            state=lambda: state, f=torch.zeros(1, dtype=torch.long), xi=torch.tensor([[0., .1, .2, .3]]),
            a_prev=torch.zeros(1, 3), carry=False, ep_step=torch.zeros(1, dtype=torch.long),
            default_q=torch.ones(3), J=3, alpha=.1,
            robot=SimpleNamespace(set_joint_position_target=lambda x: seen.update(target=x.clone())),
            scene=SimpleNamespace(write_data_to_sim=lambda: None, update=lambda dt: None),
            sim=SimpleNamespace(step=lambda: None),
            mdp=SimpleNamespace(reward=reward, termination=lambda s, f: torch.zeros(1, dtype=torch.bool)),
            training_contact_reward=.25,
            motion=SimpleNamespace(clip_end=torch.tensor([100]), clip_id=torch.tensor([0])),
            max_episode_steps=100, push_enabled=False, gamma=.99,
            ep_disc_ret=torch.zeros(1), ep_ret=torch.zeros(1))
        action = torch.tensor([[-30., 2., 40.]], requires_grad=True)
        original = action.detach().clone()
        out = ns['step_core'](env, action)
        executed = torch.tensor([[-20., 2., 20.]])
        for value in (out['a'], env.a_prev, seen['reward_action']):
            torch.testing.assert_close(value, executed)
        torch.testing.assert_close(seen['target'], 1 + env.xi[:, 1:] + .1 * executed)
        torch.testing.assert_close(action.detach(), original)
        out['a'].sum().backward()
        torch.testing.assert_close(action.grad, torch.tensor([[0., 1., 0.]]))


@unittest.skipUnless(GPU, 'Pass --gpu for the actual PD interface')
class SimActionBoundaryTests(unittest.TestCase):
    def test_eval_step_records_the_same_bounded_pd_target(self):
        from blinkmimic.box_carry import build_task, robot_spec
        from blinkmimic.dr import XiLayout
        from blinkmimic.env import TrackingSim, make_sim
        from blinkmimic.kinematics import Kinematics
        from blinkmimic.robot import get_robot
        robot = robot_spec(get_robot('g1'))
        kin = Kinematics(robot, 'cuda:0')
        layout = XiLayout(robot, 'cuda:0')
        motion, mdp = build_task('box_carry', robot, None, kin, 'cuda:0', layout.dim, max_clips=3)
        env = TrackingSim(make_sim(robot), 3, robot, motion, mdp, 'cuda:0',
                          eval_mode=True, randomize=False, vision=True)
        env.reset(torch.arange(3, device='cuda:0'), motion.clip_start)
        action = (motion.q[env.f + 1] - env.default_q) / env.alpha
        action[:, 0], action[:, 1] = -21., 21.
        expected = action.clamp(-20, 20)
        targets = []
        install = env.robot.set_joint_position_target
        def capture(target):
            targets.append(target.clone())
            install(target)
        env.robot.set_joint_position_target = capture
        out = env.step(action)
        torch.testing.assert_close(targets[0], env.default_q + env.xi[:, 1:1 + robot.num_joints] + env.alpha * expected)
        torch.testing.assert_close(out['a'], expected)
        torch.testing.assert_close(env.a_prev, expected)
        self.assertTrue(torch.isfinite(out['r']).all())


if __name__ == '__main__':
    unittest.main()
