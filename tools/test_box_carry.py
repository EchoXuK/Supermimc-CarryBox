"""Task contract checks: NPZ retarget, independent MuJoCo depth rays, scene mirroring and WM integration.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python tools/test_box_carry.py
"""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import sys

import mujoco
import numpy as np
import torch

from blinkmimic.box_carry import DEFAULT_MOTIONS, NPZ_JOINT_NAMES, SCENE_DIM, CarryMotionLib, build_task, robot_spec
from blinkmimic.dr import XiLayout
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import flatten_state, unflatten_state
from blinkmimic.mirror import Mirror
from blinkmimic.networks import Policy, WorldModel
from blinkmimic.robot import get_robot
from blinkmimic.rotations import mat_to_quat
from blinkmimic.vision import DEPTH_FAR, DEPTH_NEAR, DepthCamera, LatentDecoder, latent_prediction_loss, reconstruction_indices

GPU = "--gpu" in sys.argv
if GPU:
    sys.argv.remove("--gpu")
FULL = "--full" in sys.argv
if FULL:
    sys.argv.remove("--full")


class CarryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = robot_spec(get_robot("g1"))
        cls.kin = Kinematics(cls.robot, "cpu")
        cls.layout = XiLayout(cls.robot, "cpu")
        cls.motion, cls.mdp = build_task("box_carry", cls.robot, None, cls.kin, "cpu", cls.layout.dim, max_clips=3)
        cls.f = cls.motion.clip_start + torch.tensor([20, 100, 40])

    def test_retarget_and_split(self):
        # Dataset bodies 25/26 and 37/38 are ankle and palm frames in the Isaac breadth-first order.
        for c, row in enumerate(self.motion.rows):
            f = int(self.f[c])
            local = f - int(self.motion.clip_start[c])
            with np.load(DEFAULT_MOTIONS / row["clip_npz"]) as x:
                for name, idx in (("left_ankle_roll_link", 25), ("right_ankle_roll_link", 26),
                                  ("left_palm_link", 37), ("right_palm_link", 38)):
                    tracked = self.robot.tracked_bodies.index(name)
                    np.testing.assert_allclose(self.motion.body_p[f, tracked], x["body_pos_w"][local, idx], atol=2e-5)
                np.testing.assert_array_equal(self.motion.hand_contact[f], x["contact_info"][local, 2:4, 0])
                np.testing.assert_allclose(mat_to_quat(self.motion.object_R[f]).abs(),
                                           np.abs(x["object_quat_w"][local]), atol=1e-6)
        with (DEFAULT_MOTIONS / "manifest.csv").open() as f:
            rows = list(csv.DictReader(f))
        sets = [{r["capture_id"] for r in rows if r["split"] == s} for s in ("train", "val", "test")]
        for i in range(3):
            for j in range(i):
                self.assertFalse(sets[i] & sets[j])
        self.assertEqual([sum(r["split"] == s for r in rows) for s in ("train", "val", "test")], [748, 146, 106])
        weights = torch.zeros(3).index_add_(0, self.motion.stages[self.motion.bin_clip], self.motion.bin_prior)
        torch.testing.assert_close(weights, torch.full((3,), 1 / 3))

    def test_state_and_structured_dynamics(self):
        S, S1 = self.motion.ref_state(self.f), self.motion.ref_state(self.f + 1)
        restored = unflatten_state(flatten_state(S), object_state=True)
        for x, y in zip(S, restored):
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
        for structured in (False, True):
            wm = WorldModel(29, self.layout.dim, hidden=(32, 32), object_state=True, structured=structured,
                            scene_dim=SCENE_DIM, z_dim=32)
            out = wm.integrate(S, wm.targets(S, S1))
            for x, y in zip(out, S1):
                torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
        # An object drop must terminate even if the robot remains on reference.
        dropped = S[:6] + (S[6] + torch.tensor([0., 0., -1.]),) + S[7:]
        self.assertTrue(self.mdp.termination(dropped, self.f).all())
        self.assertTrue((self.mdp.soft_termination(dropped, self.f) > .99).all())
        a = torch.zeros(3, 29)
        self.assertTrue((self.mdp.reward(S, self.f, a, a) > self.mdp.reward(dropped, self.f, a, a)).all())

    def test_scene_mirror(self):
        S, ctx = self.motion.ref_state(self.f), self.motion.context(self.f)
        mirror = Mirror(self.robot, self.layout.slices, "cpu")
        camera = DepthCamera(self.kin, "cpu")
        depth = camera.render(S, ctx)
        mirrored = camera.render(mirror.state(S), mirror.scene(ctx))
        torch.testing.assert_close(mirrored, depth.flip(-1), atol=2e-5, rtol=2e-5)
        for x, y in zip(S, mirror.state(mirror.state(S))):
            torch.testing.assert_close(x, y)
        # Depth must change when the box moves, even with identical robot states and reference goals.
        shifted = S[:6] + (S[6] + torch.tensor([1., 1., 0.]),) + S[7:]
        self.assertGreater((camera.render(shifted, ctx) - depth).abs().max().item(), .01)

    def test_depth_against_mujoco(self):
        S, ctx = self.motion.ref_state(self.f), self.motion.context(self.f)
        camera = DepthCamera(self.kin, "cpu")
        depths = camera.render(S, ctx).numpy().reshape(3, -1)
        bp, bR, _, _ = self.kin.forward(*S[:6], all_bodies=True)
        for i in range(3):
            spec = mujoco.MjSpec.from_string('<mujoco><worldbody><geom type="plane" size="0 0 .1"/></worldbody></mujoco>')
            pose = mat_to_quat(S[7][i]).numpy()
            body = spec.worldbody.add_body(pos=S[6][i].numpy(), quat=pose)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=ctx[i, :3].numpy())
            for j in range(2):
                body = spec.worldbody.add_body(pos=ctx[i, 3 + 3 * j:6 + 3 * j].numpy())
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.2, .2, .01])
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            R = bR[i, camera.torso]
            origin = (bp[i, camera.torso] + R @ camera.offset).double().numpy()
            rays = ((R @ camera.mount) @ camera.rays.T).T.double().numpy()
            for pixel in range(0, len(rays), 7):
                ray = rays[pixel]
                norm = np.linalg.norm(ray)
                distance = mujoco.mj_ray(model, data, origin, ray / norm, None, 1, -1, np.zeros(1, dtype=np.int32))
                optical = distance / norm if distance >= 0 else DEPTH_FAR
                expected = (np.clip(optical, DEPTH_NEAR, DEPTH_FAR) - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR)
                self.assertAlmostEqual(float(depths[i, pixel]), float(expected), places=5)

    def test_robot_table_collision_filter(self):
        # This reference has an 11 cm knee/slab overlap. OmniContact's slabs support the box only.
        from mjlab.scene import Scene
        from blinkmimic.env import make_scene_cfg
        scene = Scene(make_scene_cfg(self.robot, 1, carry=True), device="cpu")
        model = scene.compile()
        data = mujoco.MjData(model)
        path = DEFAULT_MOTIONS / "02_carry_move/train/carry_box_20-30-30_261_5806747_001_02_000.npz"
        with np.load(path) as x:
            f = 91
            root = model.body("robot/pelvis").id
            qa = int(model.jnt_qposadr[model.body_jntadr[root]])
            data.qpos[qa:qa + 3], data.qpos[qa + 3:qa + 7] = x["base_pos_w"][f], x["base_quat_w"][f]
            for i, name in enumerate(NPZ_JOINT_NAMES):
                j = model.joint("robot/" + name).id
                data.qpos[model.jnt_qposadr[j]] = np.clip(x["joint_pos"][f, i], *model.jnt_range[j])
            for i in (1, 2):
                mc = model.body_mocapid[model.body(f"table{i}/table").id]
                data.mocap_pos[mc] = x[f"table{i}_pos_w"][0]
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            bodies = [model.body(int(model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
            self.assertFalse(any(b.startswith("table") for b in bodies) and any(b.startswith("robot/") for b in bodies))
        # The repaired filter must still allow real box/slab support contacts.
        box = model.body("box/box").id
        qa = int(model.jnt_qposadr[model.body_jntadr[box]])
        data.qpos[qa:qa + 3] = data.mocap_pos[model.body_mocapid[model.body("table2/table").id]] + [0., 0., .159]
        data.qpos[qa + 3:qa + 7] = [1., 0., 0., 0.]
        mujoco.mj_forward(model, data)
        self.assertTrue(any({model.body(int(model.geom_bodyid[g])).name for g in (c.geom1, c.geom2)}
                            == {"box/box", "table2/table"} for c in data.contact))

    def test_floor_drop_and_valid_release(self):
        # This entire 266-frame carry used to accept a box left on the floor at its initial xy.
        path = DEFAULT_MOTIONS / "02_carry_move/train/carry_box_18-35-22_197_5791428_003_02_000.npz"
        m = CarryMotionLib.from_file(path, self.robot, self.kin, "cpu")
        from blinkmimic.box_carry import CarryMDP
        mdp = CarryMDP(self.kin, m, self.mdp.default_q, self.layout.dim, self.robot.sensing)
        f = torch.arange(m.num_frames)
        S = m.ref_state(f)
        half = m.context(f)[:, :3]
        op = S[6][:1].expand_as(S[6]).clone()
        op[:, 2] = (S[7][0, 2].abs() * half).sum(-1)
        dropped = S[:6] + (op, S[7], torch.zeros_like(S[8]), torch.zeros_like(S[9]))
        self.assertTrue(mdp.termination(dropped, f).all())
        self.assertTrue((mdp.soft_termination(dropped, f) > .99).all())
        # The demonstrated floor/table placement and release must remain valid.
        self.assertFalse(self.mdp.termination(self.motion.ref_state(self.f), self.f).any())
        cid = 2
        end = self.motion.clip_end[cid]
        release_f = torch.arange(self.motion.clip_start[cid], end + 1)
        released = (self.motion.hand_contact[release_f] == 0).all(-1)
        self.assertTrue(released.any())
        release_f = release_f[released]
        self.assertFalse(self.mdp.termination(self.motion.ref_state(release_f), release_f).any())

    def test_joint_projection_updates_reference(self):
        path = DEFAULT_MOTIONS / "01_approach_lift/train/carry_box_20-20-20_282_5810480_000_01_000.npz"
        m = CarryMotionLib.from_file(path, self.robot, self.kin, "cpu")
        # The cleaned data removed the old out-of-limit prefix. Inject a known
        # violation into a private fixture instead of depending on that prefix.
        order = [NPZ_JOINT_NAMES.index(n) for n in self.robot.joint_names]
        with np.load(path) as x:
            payload = {k: x[k].copy() for k in x.files}
        payload["joint_pos"][10, order[0]] = float(self.kin.q_hi[0]) + .3
        with TemporaryDirectory() as directory:
            np.savez(Path(directory) / "projection.npz", **payload)
            m = CarryMotionLib(Path(directory), [dict(m.rows[0], clip_npz="projection.npz")],
                               self.kin, "cpu", self.robot.joint_names)
        raw = torch.from_numpy(payload["joint_pos"][:, order])
        expected = raw.clamp(self.kin.q_lo, self.kin.q_hi)
        self.assertGreater((raw - expected).abs().max().item(), .2)
        torch.testing.assert_close(m.q, expected, atol=2e-6, rtol=2e-5)
        f = torch.tensor([min(105, m.num_frames - 1)])
        from blinkmimic.motion import GOAL_Q_OFFSET, _gradient
        torch.testing.assert_close(m.goal(f - 1, 1)[:, GOAL_Q_OFFSET:GOAL_Q_OFFSET + 29], expected[f], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(m.qd, _gradient(expected, .02), atol=1e-4, rtol=2e-5)
        bp = self.kin.forward(*m.ref_state(f)[:6])[0]
        torch.testing.assert_close(m.body_p[f], bp)

    def test_actor_latent_gradient_boundary(self):
        pi = Policy(self.mdp.obs_dim, self.motion.goal_dim(2), 29, hist=2, enc_hidden=(16,), dec_hidden=(16,), visual_dim=32)
        z = torch.randn(3, 32, requires_grad=True)
        actions = pi(torch.zeros(3, 2, self.mdp.obs_dim), self.motion.goal(self.f, 2), visual=z)
        dz = torch.autograd.grad(actions.square().sum(), z, allow_unused=True, retain_graph=True)[0]
        self.assertIsNone(dz)
        actions.square().sum().backward()
        self.assertGreater(pi.dec[-1].weight.grad.abs().sum().item(), 0.)
        # Stop only the actor's input; the autoregressive WM latent keeps its dynamics gradient.
        wm = WorldModel(29, self.layout.dim, hidden=(16,), object_state=True, scene_dim=SCENE_DIM, z_dim=32)
        _, _, z1 = wm.step_z(self.motion.ref_state(self.f), actions.detach(), torch.zeros(3, 29),
                              torch.zeros(3, self.layout.dim), z, self.motion.context(self.f))
        torch.testing.assert_close(torch.autograd.grad(z1.sum(), z)[0], torch.ones_like(z))

    def test_encoder_trains_only_from_reconstruction(self):
        pi = Policy(self.mdp.obs_dim, self.motion.goal_dim(2), 29, hist=2, enc_hidden=(16,), dec_hidden=(16,), visual_dim=32)
        depth = torch.rand(3, 1, 18, 32)
        actions = pi(torch.zeros(3, 2, self.mdp.obs_dim), self.motion.goal(self.f, 2), depth=depth)
        actions.square().sum().backward()
        self.assertTrue(all(p.grad is None for p in pi.depth_enc.parameters()))
        decoder = LatentDecoder(32)
        loss = (decoder(pi.depth_enc(depth)) - depth).abs().mean()
        loss.backward()
        for module in (pi.depth_enc, decoder):
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in module.parameters()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0.)

    def test_latent_loss_uses_valid_current_variance(self):
        torch.manual_seed(17)
        target = torch.randn(4, 3, 32) * .01
        prediction = (target + torch.randn_like(target) * .005).requires_grad_(True)
        mask = torch.tensor([[1., 1., 1.], [1., 0., 0.], [1., 1., 0.], [1., 1., 1.]])
        # A reset's targets must not change the variance used to scale the preceding episode.
        target[mask == 0] = 1000.
        scale = target[mask.bool()].var(0, unbiased=False).mean()
        expected = (((prediction - target).square().mean(-1) / scale) * mask).sum() / len(target)
        loss = latent_prediction_loss(prediction, target, mask)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(latent_prediction_loss(prediction * 7, target * 7, mask), loss)
        loss.backward()
        self.assertTrue(prediction.grad.isfinite().all())
        self.assertEqual(prediction.grad[mask == 0].abs().sum().item(), 0.)

    def test_reconstruction_covers_original_and_mirror(self):
        torch.manual_seed(11)
        mask = torch.ones(256, 4)
        mask[::3, 2:] = 0
        ids = reconstruction_indices(mask, 64)
        self.assertEqual(len(ids), 64)
        self.assertEqual(len(ids.unique()), 64)
        self.assertTrue((mask.flatten()[ids] == 1).all())
        mirrored = ids < 128 * 4
        self.assertTrue(mirrored.any() and (~mirrored).any())
        # Short windows retain a fixed size, with any invalid extras weighted out by the caller.
        short = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
        ids = reconstruction_indices(short, 64)
        self.assertEqual(len(ids), short.numel())
        self.assertEqual(short.flatten()[ids].sum().item(), 2.)


@unittest.skipUnless(FULL, "pass --full for every train/val/test reference frame")
class FullDatasetTests(unittest.TestCase):
    def test_all_augmented_references(self):
        robot = robot_spec(get_robot("g1"))
        kin = Kinematics(robot, "cpu")
        layout = XiLayout(robot, "cpu")
        for split, count in (("train", 748), ("val", 146), ("test", 106)):
            with self.subTest(split=split):
                m, mdp = build_task("box_carry", robot, None, kin, "cpu", layout.dim, split=split)
                self.assertEqual(len(m.names), count)
                self.assertTrue((m.q >= kin.q_lo - 2e-6).all() and (m.q <= kin.q_hi + 2e-6).all())
                detected = 0
                for start in range(0, m.num_frames, 8192):
                    f = torch.arange(start, min(start + 8192, m.num_frames))
                    S = m.ref_state(f)
                    self.assertFalse(mdp.termination(S, f).any())
                    self.assertTrue((mdp.soft_termination(S, f) < .001).all())
                    zero = torch.zeros(len(f), robot.num_joints)
                    self.assertTrue(mdp.reward(S, f, zero, zero).isfinite().all())
                    self.assertTrue(mdp.critic_features(S, f, torch.zeros(len(f), layout.dim)).isfinite().all())
                    self.assertTrue(m.goal(f, 10).isfinite().all())
                    # Every demonstrated carry at least 11 cm above the floor must detect a floor drop.
                    op = S[6].clone()
                    op[:, 2] = (S[7][:, 2].abs() * m.context(f)[:, :3]).sum(-1)
                    held = (m.stages[m.clip_id[f]] == 1) & (S[6][:, 2] - op[:, 2] > .11)
                    dropped = S[:6] + (op, S[7], torch.zeros_like(S[8]), torch.zeros_like(S[9]))
                    self.assertTrue(mdp.termination(dropped, f)[held].all())
                    detected += int(held.sum())
                print(f"full {split}: {count} clips / {m.num_frames} frames; projected={m.projected_joint_frames}; floor-drop checks={detected}", flush=True)


@unittest.skipUnless(GPU, "pass --gpu for the MuJoCo-Warp reset/transition check")
class CarrySimTests(unittest.TestCase):
    def test_reset_previous_action_matches_installed_target(self):
        from blinkmimic.env import TrackingSim, make_sim
        robot = robot_spec(get_robot("g1"))
        kin = Kinematics(robot, "cuda:0")
        layout = XiLayout(robot, "cuda:0")
        m, mdp = build_task("box_carry", robot, None, kin, "cuda:0", layout.dim, max_clips=3)
        env = TrackingSim(make_sim(robot), 3, robot, m, mdp, "cuda:0", reset_noise=False,
                          randomize=True, max_episode_steps=1, vision=True)
        f = m.clip_start + torch.tensor([20, 100, 40], device="cuda:0")
        env.reset(torch.arange(3, device="cuda:0"), f)
        # reset installs q as the PD target; the remembered action must describe that same target.
        target = env.default_q + env.xi[:, 1:1 + robot.num_joints] + env.alpha * env.a_prev
        torch.testing.assert_close(target, env.state()[4], atol=2e-6, rtol=2e-5)
        previous = env.a_prev.clone()
        action = (m.q[f + 1] - env.default_q - env.xi[:, 1:1 + robot.num_joints]) / env.alpha
        tr = env.step(action)
        torch.testing.assert_close(tr["a_prev"], previous)
        # Reference playback pays for the true small inter-frame target change at the first step too.
        actual_rate = (tr["a"] - tr["a_prev"]).square().sum(-1)
        reference_rate = ((m.q[f + 1] - m.q[f]) / env.alpha).square().sum(-1)
        torch.testing.assert_close(actual_rate, reference_rate, atol=2e-5, rtol=2e-4)

    def test_physical_reset_and_terminal_depth(self):
        from blinkmimic.env import TrackingSim, make_sim
        robot = robot_spec(get_robot("g1"))
        kin = Kinematics(robot, "cuda:0")
        layout = XiLayout(robot, "cuda:0")
        m, mdp = build_task("box_carry", robot, None, kin, "cuda:0", layout.dim, max_clips=3)
        env = TrackingSim(make_sim(robot), 3, robot, m, mdp, "cuda:0", reset_noise=False,
                          randomize=False, max_episode_steps=1, bin_fix=True, vision=True)
        ids, f = torch.arange(3, device="cuda:0"), m.clip_start.clone()
        env.reset(ids, f)
        for actual, ref in zip(env.state(), m.ref_state(f)):
            torch.testing.assert_close(actual, ref, atol=2e-6, rtol=2e-5)
        ctx = m.context(f)
        torch.testing.assert_close(env.sim.model.geom_size[:, env.box_geom], ctx[:, :3])
        expected_inertia = ctx[:, 9:10] / 3 * (ctx[:, :3].square().sum(-1, keepdim=True) - ctx[:, :3].square())
        torch.testing.assert_close(env.sim.model.body_inertia[:, env.box_body], expected_inertia)
        for i, mc in enumerate(env.table_mocap):
            torch.testing.assert_close(env.sim.data.mocap_pos[:, mc], ctx[:, 3 + 3 * i:6 + 3 * i])
        old = tuple(x[1:].clone() for x in env.state())
        env.reset(ids[:1], f[:1] + 10)
        for actual, ref in zip(env.state(), old):
            torch.testing.assert_close(actual[1:], ref, atol=0, rtol=0)
        env.observe()
        depth0 = env.depth.clone()
        action = (m.q[env.f + 1] - env.default_q) / env.alpha
        tr = env.step(action)
        self.assertTrue(tr["done"].all())
        torch.testing.assert_close(tr["depth"], depth0)
        # s1/depth1 are read BEFORE auto-reset changes box size, tables and reference frame.
        S1 = unflatten_state(tr["s1"], object_state=True)
        torch.testing.assert_close(tr["depth1_clean"], env.camera.render(S1, tr["scene"]))
        self.assertTrue(torch.isfinite(tr["s1"]).all())
        self.assertTrue((env.ep_step == 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
