"""CPU regressions for checkpoint startup, reward isolation and evaluation caches.

CLI bodies are extracted with AST so their orchestration can run with a CPU
simulator fixture without importing the CUDA-only training/evaluation startup.
"""

import ast
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from blinkmimic.box_carry import build_mdp, build_task, robot_spec
from blinkmimic.dr import XiLayout
from blinkmimic.eval_protocol import EvaluationCache, evaluation_protocol, result_path
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import flatten_state, unflatten_state
from blinkmimic.motion import goal_frame_offsets
from blinkmimic.networks import check_pruned, make_policy, initialize_policy, Policy
from blinkmimic.robot import get_robot

ROOT = Path(__file__).resolve().parents[1]


def cli_tree(path):
    return ast.parse((ROOT / path).read_text())


def cli_functions(path, names, namespace):
    nodes = [node for node in cli_tree(path).body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)


def small_state(n):
    zero = torch.zeros(n, 3)
    rotation = torch.eye(3).repeat(n, 1, 1)
    return (zero.clone(), rotation.clone(), zero.clone(), zero.clone(), torch.zeros(n, 2), torch.zeros(n, 2),
            zero.clone(), rotation.clone(), zero.clone(), zero.clone())




class ResumeMetadataTests(unittest.TestCase):
    def test_removed_cli_options_are_rejected(self):
        nodes = [node for node in cli_tree("src/blinkmimic/train.py").body
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and ast.unparse(node.value.func) == "parser.add_argument"]
        parser = argparse.ArgumentParser()
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "train_options", "exec"),
             dict(parser=parser, REPO=str(ROOT), os=os))
        for option in ("--algorithm", "--fpo_epochs"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args([option, "1"])

    def startup(self, args, checkpoint, fail_load=False):
        class Model:
            def load_state_dict(self, state):
                if fail_load:
                    raise RuntimeError("state shape mismatch")

        ns = dict(args=args, os=os, json=json, dev="cpu", pi_avg=None,
                  torch=SimpleNamespace(load=lambda *a, **kw: checkpoint), check_pruned=check_pruned,
                  wm=Model(), pi=Model(), critic=Model(), VISION=False, load_opt_states=lambda ck: None,
                  goal_frame_offsets=goal_frame_offsets)
        # Execute the actual startup branches and metadata writes in source order.
        nodes = [node for node in cli_tree("src/blinkmimic/train.py").body
                 if (isinstance(node, ast.If) and ast.unparse(node.test) in ("args.init_from", "args.resume"))
                 or (isinstance(node, ast.With) and "args.log_dir" in ast.unparse(node)
                     and any(name in ast.unparse(node) for name in ("args.json", "train_pid.json")))
                 or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and (ast.unparse(node.value.func) == "os.makedirs" or
                          (ast.unparse(node.value.func) == "json.dump" and "args.log_dir" in ast.unparse(node))))]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "train_startup", "exec"), ns)

    def test_failed_validation_and_state_load_preserve_existing_metadata(self):
        with TemporaryDirectory() as directory:
            options = dict(log_dir=directory, resume="last.pt", init_from=None, algorithm="bptt",
                           carry_reward_profile="grasp_place", task="box_carry", vision="none", split="train",
                           stages=None, max_clips=None,
                           carry_residual_rate=0., carry_joint_vel=0., carry_residual_acc=0.,
                           carry_residual_acc_arms=1., contact_reward=0., gamma=.99, lam=.95, horizon=32,
                           wm_update_interval=0, policy_warmup_steps=None, min_reset_steps=1,
                           min_imagination_steps=None, constant_lr=False,
                           real_contact_reward=False, dataset_fingerprint=None)
            options.update(carry_gait_profile="legacy", carry_gait_scale=1.)
            paths = [Path(directory) / name for name in ("args.json", "train_pid.json")]
            originals = [json.dumps(options).encode(), b'{"pid": 12345}']
            for path, value in zip(paths, originals):
                path.write_bytes(value)
            checkpoint = dict(args=dict(options), opt_pi={}, wm={}, pi={}, critic={}, iteration=3, wall_time=2., sim_steps=100)
            bad = dict(options, carry_reward_profile="legacy")
            with self.assertRaisesRegex(ValueError, "carry_reward_profile"):
                self.startup(SimpleNamespace(**bad), checkpoint)
            self.assertEqual([path.read_bytes() for path in paths], originals)
            with self.assertRaisesRegex(RuntimeError, "state shape mismatch"):
                self.startup(SimpleNamespace(**options), checkpoint, fail_load=True)
            self.assertEqual([path.read_bytes() for path in paths], originals)
            for algorithm in ("bptt",):
                checkpoint["args"] = dict(options, algorithm=algorithm)
                for key, value in (("contact_reward", .25), ("gamma", .9), ("lam", .8),
                                   ("carry_joint_vel", .1), ("carry_residual_rate", .1),
                                   ("carry_residual_acc", .1), ("carry_residual_acc_arms", 0.),
                                   ("carry_gait_profile", "reference_phase"), ("carry_gait_scale", .5),
                                   ("min_reset_steps", 32), ("min_imagination_steps", 16)):
                    changed = dict(checkpoint["args"], **{key:value})
                    with self.subTest(algorithm=algorithm, key=key), self.assertRaisesRegex(ValueError, key):
                        self.startup(SimpleNamespace(**changed), checkpoint)
                    self.assertEqual([path.read_bytes() for path in paths], originals)
            checkpoint["args"] = dict(options)
            self.startup(SimpleNamespace(**options), checkpoint)
            self.assertEqual(json.loads(paths[0].read_text()), options)
            self.assertEqual(json.loads(paths[1].read_text())["pid"], os.getpid())

    def test_removed_algorithm_is_rejected_before_weights_or_metadata(self):
        bad = {"algorithm": "fpo_hybrid"}
        for nets_only in (False, True):
            with self.assertRaisesRegex(ValueError, "only bptt"):
                check_pruned(bad, nets_only=nets_only)
        policy = Policy(3, 4, 2, hist=2, enc_hidden=(8,), dec_hidden=(8,))
        original = {k: v.clone() for k, v in policy.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "only bptt"):
            initialize_policy(policy, {}, bad)
        torch.testing.assert_close(policy.state_dict(), original, atol=0, rtol=0)
        for args in ({}, {"algorithm": "bptt"}):
            check_pruned(args)
        with self.assertRaisesRegex(ValueError, "only bptt"):
            make_policy(3, 4, 2, None, bad)
        with TemporaryDirectory() as directory:
            for name in ("args.json", "train_pid.json"):
                (Path(directory) / name).write_text("original")
            opts = SimpleNamespace(log_dir=directory, init_from=None, resume="last.pt")
            with self.assertRaisesRegex(ValueError, "only bptt"):
                self.startup(opts, {"args": bad, "opt_pi": {}})
            for name in ("args.json", "train_pid.json"):
                self.assertEqual((Path(directory) / name).read_text(), "original")


class CheckpointRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = robot_spec(get_robot("g1"))
        cls.kin = Kinematics(cls.robot, "cpu")
        cls.xi_dim = XiLayout(cls.robot, "cpu").dim
        cls.motion, cls.mdp = build_task("box_carry", cls.robot, None, cls.kin, "cpu", cls.xi_dim, max_clips=3)
        frame = torch.arange(cls.motion.num_frames)
        stage, contact = cls.motion.stages[cls.motion.clip_id], cls.motion.hand_contact.amax(-1) > .5
        cls.frames = torch.stack((frame[(stage == 0) & ~contact][10], frame[stage == 1][20],
                                  frame[(stage == 2) & ~contact][0]))

    def scores(self, configs):
        motion, mdp, frames, joints = self.motion, self.mdp, self.frames, self.robot.num_joints

        class Env:
            def __init__(self, sim, n, *a, **kw):
                self.n, self.done_eval, self.failed_eval = n, torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
                self.a_prev2 = torch.zeros(n, joints)

            def reset(self, ids, starts):
                self.f = frames.repeat(len(ids) // 3) - 1

            def observe(self):
                return torch.zeros(self.n, mdp.obs_dim)

            def state(self):
                return self.s

            def step(self, actions):
                self.f += 1
                state = list(motion.ref_state(self.f))
                state[6] = state[6].clone()
                state[6][:, 0] += .1
                state[5] = state[5] + .3
                self.s = tuple(state)
                previous = torch.zeros_like(actions)
                ep_step = torch.full((self.n,), 3)
                reward, detail = mdp.reward(self.s, self.f, actions, previous, terms=True,
                                             a_prev2=previous, smooth_valid=ep_step > 0, smooth_acc_valid=ep_step > 1)
                self.done_eval[:] = True
                return dict(r=reward, terms=detail, a=actions, a_prev=previous, ep_step=ep_step)

        ns = dict(torch=torch, C=3, TASK="box_carry", VISION=False, dev="cpu", robot=self.robot, kin=self.kin,
                  motion=motion, mdp=mdp, build_mdp=build_mdp, XiLayout=XiLayout, TrackingSim=Env,
                  make_sim=lambda robot: None, load_policy=lambda ck: lambda obs, goal, depth: torch.zeros(len(obs), joints),
                  networks=SimpleNamespace(AMP_DTYPE=None), FEET=[self.robot.tracked_bodies.index(name) for name in
                      ("left_ankle_roll_link", "right_ankle_roll_link")], CONTACT_Z=.045, FLAT_Z=.033)
        cli_functions("tools/eval_metrics.py", {"run"}, ns)
        return ns["run"]([{"args": dict(task="box_carry", vision="none", hist=2, goal_frames=2, **cfg)} for cfg in configs])

    def test_mixed_profile_order_does_not_change_each_checkpoints_score(self):
        configs = [{"carry_reward_profile": "legacy"}, {"carry_reward_profile": "grasp_place"}]
        forward, reverse = self.scores(configs), self.scores(configs[::-1])
        keys = ("norm_return", "task_norm_return", "carry_reward_profile", "mean_grasp_place_penalty")
        for left, right in zip(forward, reverse[::-1]):
            self.assertEqual({key: left[key] for key in keys}, {key: right[key] for key in keys})
        self.assertEqual(forward[0]["task_norm_return"], forward[1]["task_norm_return"])
        self.assertGreater(forward[0]["norm_return"], forward[1]["norm_return"])
        self.assertGreater(sum(forward[1]["mean_grasp_place_penalty"].values()), 0.)

    def test_each_checkpoint_also_uses_its_own_smoothing_weights(self):
        scores = self.scores([{"carry_joint_vel": 0.}, {"carry_joint_vel": 1.}])
        self.assertEqual(scores[0]["task_norm_return"], scores[1]["task_norm_return"])
        self.assertGreater(scores[0]["norm_return"], scores[1]["norm_return"])


class EvaluationCacheTests(unittest.TestCase):
    def test_previous_protocol_cannot_reuse_version_six_cache(self):
        with TemporaryDirectory() as directory:
            protocol = self.protocol()
            self.assertEqual(protocol["version"], 6)
            previous = copy.deepcopy(protocol)
            previous["version"] = 5
            old = EvaluationCache(directory, previous)
            self.save(old, directory)
            current = EvaluationCache(directory, protocol)
            self.assertNotEqual(current.csv_path, old.csv_path)
            self.assertEqual(current.completed(), set())

    def test_finished_nan_worlds_cannot_poison_carry_diagnostics(self):
        tree = cli_tree("src/blinkmimic/evaluation.py")
        evaluate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_policy")
        nodes = [node for node in ast.walk(evaluate) if isinstance(node, ast.AugAssign)
                 and ast.unparse(node.target) in ("shaping_sum", "release_speed", "release_error")]
        ns = dict(torch=torch, alive=torch.tensor([True, False]), released=torch.tensor([True, False]),
                  shaping_sum=torch.zeros(2, 3), release_speed=torch.zeros(2), release_error=torch.zeros(2),
                  out={"terms": {key: torch.tensor([.1, float("nan")]) for key in
                      ("grasp_penalty", "drop_penalty", "place_penalty", "object_pos")}},
                  S1=[None] * 8 + [torch.tensor([[.15, 0., 0.], [float("nan"), 0., 0.]])])
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "carry_eval_diagnostics", "exec"), ns)
        torch.testing.assert_close(ns["shaping_sum"], torch.tensor([[.1, .1, .1], [0., 0., 0.]]))
        torch.testing.assert_close(ns["release_speed"], torch.tensor([.15, 0.]))
        torch.testing.assert_close(ns["release_error"], torch.tensor([.1, 0.]))

    def protocol(self, **changes):
        args = SimpleNamespace(robot="g1", split="val", max_clips=3, stages=None, repeats=1,
                               seed=0, dr_from=None, dr_parts="all", group=2)
        run_args = dict(task="box_carry", algorithm="bptt", carry_reward_profile="legacy")
        motion = SimpleNamespace(names=["approach", "carry", "place"], clip_len=torch.tensor([8, 9, 10]), num_frames=27)
        for key, value in changes.items():
            (run_args if key in ("carry_reward_profile", "carry_joint_vel") else vars(args))[key] = value
        return evaluation_protocol(args, run_args, motion, evaluator="batch", motion_dir="box_carry_three_stage_v2")

    def save(self, cache, directory):
        cache.record(str(Path(directory) / "ckpt_000001.pt"), {"wall_time": 36., "saved_at": "saved"},
                     {"per_clip": {name: .8 for name in cache.protocol["motion"]["names"]},
                      "norm_return": .8, "norm_return_framew": .8, "n_failed": 0, "failed": []})

    def test_legacy_partial_cache_is_preserved_and_not_reused(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "eval.csv"
            path.write_text(EvaluationCache.header + "1,0.01,saved,evaluated,0.8,0.8,0,\n")
            old = path.read_bytes()
            (Path(directory) / "ckpt_000001_eval.json").write_text('{"per_clip":{"approach":0.8}}')
            protocol = self.protocol(max_clips=None)
            protocol["motion"]["names"] = [f"val_{i}" for i in range(146)]
            protocol["motion"]["lengths"] = [8] * 146
            protocol["motion"]["frames"] = 8 * 146
            cache = EvaluationCache(directory, protocol)
            self.assertNotEqual(cache.csv_path, path)
            self.assertEqual(cache.completed(), set())
            self.assertEqual(path.read_bytes(), old)

    def test_protocol_changes_get_independent_caches_and_single_result_paths(self):
        with TemporaryDirectory() as directory:
            original = self.protocol()
            cache = EvaluationCache(directory, original)
            self.save(cache, directory)
            self.assertEqual(EvaluationCache(directory, self.protocol()).completed(), {1})
            variations = [dict(split="test"), dict(max_clips=None), dict(stages="03_lower_place"), dict(repeats=3),
                          dict(seed=1), dict(dr_from="self"), dict(dr_parts="sensing"), dict(group=3),
                          dict(carry_reward_profile="grasp_place"), dict(carry_joint_vel=1.)]
            for change in variations:
                with self.subTest(change=change):
                    protocol = self.protocol(**change)
                    other = EvaluationCache(directory, protocol)
                    self.assertNotEqual(cache.csv_path, other.csv_path)
                    self.assertEqual(other.completed(), set())
                    self.assertNotEqual(result_path("ckpt_000001.pt", original), result_path("ckpt_000001.pt", protocol))
            protocol = self.protocol()
            protocol["motion"]["names"][-1] = "another_clip"
            self.assertEqual(EvaluationCache(directory, protocol).completed(), set())
            self.assertEqual(cache.completed(), {1})

    def test_result_requires_matching_protocol_and_actual_clip_set(self):
        with TemporaryDirectory() as directory:
            cache = EvaluationCache(directory, self.protocol())
            self.save(cache, directory)
            path = cache.result_path(str(Path(directory) / "ckpt_000001.pt"))
            result = json.loads(path.read_text())
            del result["per_clip"]["place"]
            path.write_text(json.dumps(result))
            self.assertEqual(cache.completed(), set())
            self.save(cache, directory)
            result = json.loads(path.read_text())
            result["evaluation_protocol"]["split"] = "test"
            path.write_text(json.dumps(result))
            self.assertEqual(cache.completed(), set())
            path.write_text("{")
            self.assertEqual(cache.completed(), set())
            self.save(cache, directory)
            self.assertEqual(cache.completed(), {1})
            self.assertEqual(len(cache.csv_path.read_text().splitlines()), 2)

    def test_interrupted_result_and_csv_update_is_not_reused(self):
        with TemporaryDirectory() as directory:
            cache = EvaluationCache(directory, self.protocol())
            self.save(cache, directory)
            path = cache.result_path(str(Path(directory) / "ckpt_000001.pt"))
            result = json.loads(path.read_text())
            result["norm_return"] = .5
            path.write_text(json.dumps(result))
            self.assertEqual(cache.completed(), set())
            cache.record(str(Path(directory) / "ckpt_000001.pt"), {"wall_time": 36., "saved_at": "saved"}, result)
            self.assertEqual(cache.completed(), {1})
            self.assertEqual(len(cache.csv_path.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
