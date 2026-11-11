"""Augmented OmniContact three-stage carrying task (50 Hz, wxyz, Isaac joint order)."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .mdp import MDP, MAX_STEP_REWARD
from .motion import CONTROL_DT, MotionLib
from .rotations import mat_to_6d, mm3, mv3, quat_to_mat, rot_angle_sq, yaw_mat

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTIONS = ROOT / "box_carry_three_stage_v2" / "augmented"
ASSET = ROOT / "data" / "g1_carry" / "g1_29dof.xml"
STAGES = ("01_approach_lift", "02_carry_move", "03_lower_place")
TABLE_HALF_SIZE = (0.2, 0.2, 0.01)  # table*_pos_w is the slab centre, as in OmniContact
SCENE_DIM = 10  # box half size (3), two table centres (6), box mass (1)
STAND_TILT = 20. * np.pi / 180.
NPZ_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)


def load_npz(path: str, joint_names=NPZ_JOINT_NAMES):
    with np.load(path, allow_pickle=False) as x:
        order = [NPZ_JOINT_NAMES.index(n) for n in joint_names]
        return x["base_pos_w"], x["base_quat_w"], x["joint_pos"][:, order], float(x["fps"].item())


def _robot_cfg():
    from .g1.g1_constants import DEFAULT_JOINT_POS, _actuators

    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(ASSET))
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        for key in list(spec.keys):
            spec.delete(key)
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER  # SimulationCfg owns the integrator
        # Robot/ground/box/table bits: 2/1/4/8. OmniContact slabs support the box only.
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.contype, geom.conaffinity = 2, 1 | 4
        return spec

    return EntityCfg(
        spec_fn=spec_fn,
        init_state=EntityCfg.InitialStateCfg(pos=(0., 0., .76), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.}),
        articulation=EntityArticulationInfoCfg(actuators=_actuators(), soft_joint_pos_limit_factor=.9),
    )


def scene_entities():
    def box():
        return mujoco.MjSpec.from_string('''<mujoco><worldbody><body name="box" pos="0 0 .3">
          <freejoint name="box_joint"/><geom name="box_geom" type="box" size=".15 .15 .15" mass="2"
          contype="4" conaffinity="11" friction="1 .005 .0001" condim="4" rgba=".8 .55 .25 1"/>
          </body></worldbody></mujoco>''')

    def table():
        return mujoco.MjSpec.from_string('''<mujoco><worldbody><body name="table" mocap="true" pos="0 0 -5">
          <geom name="slab" type="box" size=".2 .2 .01" contype="8" conaffinity="4"
          friction="1 .005 .0001" rgba=".3 .5 .6 1"/></body></worldbody></mujoco>''')

    return {"box": EntityCfg(spec_fn=box), "table1": EntityCfg(spec_fn=table), "table2": EntityCfg(spec_fn=table)}


def robot_spec(robot):
    if robot.name != "g1":
        raise ValueError("box_carry uses the G1 29-DoF retarget and rubber-hand collision asset")
    if not ASSET.is_file():
        raise FileNotFoundError(f"Missing {ASSET}; run .venv/bin/python tools/prepare_box_carry_assets.py")
    tracked = tuple(n.replace("wrist_yaw", "palm") for n in robot.tracked_bodies)
    # Small reset perturbations preserve the demonstrated grasp at mid-carry resets.
    dr = replace(robot.dr, reset_pos=(.01, .01, .005), reset_rot=(.01, .01, .02),
                 reset_lin_vel=(.05, .05, .05), reset_ang_vel=(.05, .05, .05), reset_joint_pos=.02)
    return replace(robot, mjcf_path=str(ASSET), robot_cfg=_robot_cfg, tracked_bodies=tracked,
                   motion_dir=str(DEFAULT_MOTIONS), motion_glob="*/*/*.npz", dr=dr, njmax=1000, nconmax=256)


class CarryMotionLib(MotionLib):
    object_state = True

    @classmethod
    def from_file(cls, file, robot, kin, device):
        file = Path(file).absolute()
        for root in file.parents:
            manifest = root / "manifest.csv"
            if manifest.is_file():
                with manifest.open() as f:
                    rows = [r for r in csv.DictReader(f) if r["variant"] == "augmented"
                            and (root / r["clip_npz"]).resolve() == file.resolve()]
                if rows:
                    return cls(root, rows, kin, device, robot.joint_names)
        raise ValueError(f"No augmented manifest entry for {file}")

    @classmethod
    def from_dir(cls, path, robot, kin, device, split="train", stages=None, max_clips=None):
        root = Path(path)
        with (root / "manifest.csv").open() as f:
            rows = [r for r in csv.DictReader(f) if r["variant"] == "augmented" and r["split"] == split
                    and (stages is None or r["stage"] in stages)]
        rows.sort(key=lambda r: (r["stage"], r["clip_npz"]))
        if max_clips is not None:
            # A small smoke run still exercises each selected stage.
            groups = [[r for r in rows if r["stage"] == s] for s in STAGES]
            rows = [g[i] for i in range(max(len(g) for g in groups)) for g in groups if i < len(g)][:max_clips] if rows else []
        if not rows:
            raise ValueError(f"No augmented {split} clips in {root}")
        return cls(root, rows, kin, device, robot.joint_names)

    def __init__(self, root, rows, kin, device, joint_names):
        files = [str(root / r["clip_npz"]) for r in rows]
        lo, hi = kin.q_lo.cpu().numpy(), kin.q_hi.cpu().numpy()
        self.projected_joint_frames = 0
        self.max_joint_correction = 0.

        def load_reference(path):
            p, quat, q, fps = load_npz(path, joint_names)
            projected = np.clip(q, lo, hi)
            correction = np.abs(projected - q)
            self.projected_joint_frames += int((correction > 1e-6).any(-1).sum())
            self.max_joint_correction = max(self.max_joint_correction, float(correction.max()))
            return p, quat, projected, fps

        # Project once before velocities, FK, goals and the action skip are constructed.
        super().__init__(files, kin, device, load_reference)
        self.names = [f"{r['stage']}/{Path(r['clip_npz']).stem}" for r in rows]
        self.rows = rows
        objects, contacts, scenes, stages = [], [], [], []
        for row, file, n in zip(rows, files, self.clip_len.cpu().tolist()):
            with np.load(file, allow_pickle=False) as x:
                if abs(float(x["fps"].item()) - 50.) > 1e-5 or x["joint_pos"].shape[0] != n:
                    raise ValueError(f"Expected the curated 50 Hz NPZ format: {file}")
                # object_quat_w already includes the XML geometry rotation. Do not apply it again.
                objects.append(tuple(torch.from_numpy(x[k].copy()).float() for k in
                                     ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w")))
                contacts.append(torch.from_numpy(x["contact_info"][:, 2:4, 0].copy()).float())
                half = x["carry_box_selected_size_cm"].astype(np.float32) / 200.
                tables = np.stack((x["table1_pos_w"][0], x["table2_pos_w"][0])).copy()
                if row.get("support_surface_z_m"):
                    tables[1, 2] = float(row["support_surface_z_m"])
                scenes.append(np.concatenate((half, tables.ravel(), [2.])).astype(np.float32))
                stages.append(STAGES.index(row["stage"]))
        self.object_p, quat, self.object_v, self.object_w = (torch.cat([o[i] for o in objects]).to(device) for i in range(4))
        self.object_R = quat_to_mat(quat)
        self.hand_contact = torch.cat(contacts).to(device)
        self.scenes = torch.tensor(np.stack(scenes), device=device)
        self.stages = torch.tensor(stages, dtype=torch.long, device=device)
        self.stage_onehot = torch.nn.functional.one_hot(self.stages, len(STAGES)).float()
        self.standing_height_threshold = torch.tensor([
            float(row["standing_height_threshold_m"]) if stage == 2 else 0.
            for row, stage in zip(rows, stages)], device=device)
        # Equal stage prior, then equal clips within a stage, then uniform time in that clip.
        counts = torch.bincount(self.bin_clip, minlength=len(rows)).float()
        stage_count = torch.bincount(self.stages, minlength=3).clamp(min=1).float()
        self.bin_prior = (1 / counts[self.bin_clip] / stage_count[self.stages[self.bin_clip]])
        self.bin_prior /= self.bin_prior.sum()

    def context(self, f):
        return self.scenes[self.clip_id[f]]

    def ref_state(self, f):
        return super().ref_state(f) + (self.object_p[f], self.object_R[f], self.object_v[f], self.object_w[f])

    def goal_dim(self, K):
        return super().goal_dim(K) + K * 17 + 6

    def goal(self, f, K, root_p=None):
        base = super().goal(f, K, root_p)
        fut = self.future_frames(f, K)
        Ht = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(1)
        origin = self.root_p[f] if root_p is None else root_p
        obj = torch.cat((mv3(Ht, self.object_p[fut] - origin.unsqueeze(1)),
                         mat_to_6d(mm3(Ht, self.object_R[fut])), mv3(Ht, self.object_v[fut]),
                         mv3(Ht, self.object_w[fut]), self.hand_contact[fut]), -1)
        cid = self.clip_id[f]
        return torch.cat((base, obj.flatten(-2), self.scenes[cid, :3], self.stage_onehot[cid]), -1)


def scene_features(S, scene):
    tables = scene[..., 3:9].unflatten(-1, (2, 3))
    local = mv3(yaw_mat(S[1]).transpose(-1, -2).unsqueeze(-3), tables - S[0].unsqueeze(-2))
    return torch.cat((scene[..., :3], local.flatten(-2), scene[..., 9:10]), -1)


class CarryMDP(MDP):
    max_step_reward = 9.5 * CONTROL_DT

    def __init__(self, *args, action_scale=None, residual_rate_weight=0., joint_vel_weight=0., residual_acc_weight=0.,
                 residual_acc_joint_weights=None, imitation_mode="tracking", amp_reward_weight=5.,
                 action_acc_weight=0., action_acc_joint_weights=None, foot_slip_weight=.25,
                 palm_far_weight=0., palm_far_std=.5, palm_far_kernel="gaussian", goal_anchor="reference",
                 approach_progress_weight=0., **kwargs):
        super().__init__(*args, **kwargs)
        if imitation_mode not in ("tracking", "amp"):
            raise ValueError("imitation_mode must be tracking or amp")
        self.imitation_mode = imitation_mode
        if goal_anchor not in ("reference", "robot"):
            raise ValueError("goal_anchor must be reference or robot")
        self.goal_anchor = goal_anchor
        self.approach_progress_weight = float(approach_progress_weight)
        self.motion_prior = None
        self.amp_reward_weight = float(amp_reward_weight)
        self.action_acc_weight = float(action_acc_weight)
        self.foot_slip_weight = float(foot_slip_weight)
        self.palm_far_weight, self.palm_far_std = float(palm_far_weight), float(palm_far_std)
        if palm_far_kernel not in ("gaussian", "inverse_quadratic"):
            raise ValueError("palm_far_kernel must be gaussian or inverse_quadratic")
        self.palm_far_kernel = palm_far_kernel
        if not np.isfinite(self.palm_far_weight) or not 0 <= self.palm_far_weight <= 1:
            raise ValueError("palm_far_weight must be in [0, 1]")
        if not np.isfinite(self.palm_far_std) or self.palm_far_std <= 0:
            raise ValueError("palm_far_std must be positive")
        if imitation_mode != "amp" and self.palm_far_weight:
            raise ValueError("Far palm shaping requires AMP")
        self.residual_rate_weight = float(residual_rate_weight)
        self.joint_vel_weight = float(joint_vel_weight)
        self.residual_acc_weight = float(residual_acc_weight)
        if not all(np.isfinite(w) and w >= 0 for w in (self.residual_rate_weight, self.joint_vel_weight, self.residual_acc_weight,
                                                       self.action_acc_weight, self.amp_reward_weight, self.foot_slip_weight,
                                                       self.approach_progress_weight)):
            raise ValueError("Carry smoothing weights must be finite and nonnegative")
        if imitation_mode == "amp" and (residual_rate_weight or joint_vel_weight or residual_acc_weight):
            raise ValueError("AMP cannot use reference-relative smoothing or joint-velocity tracking")
        if imitation_mode != "amp" and self.approach_progress_weight:
            raise ValueError("Approach progress shaping requires AMP")
        if (self.residual_rate_weight or self.residual_acc_weight) and action_scale is None:
            raise ValueError("Residual smoothness reward needs the robot's action scales")
        self.action_scale = None if action_scale is None else self.default_q.new_tensor(action_scale)
        self.residual_acc_joint_weights = (torch.ones_like(self.default_q) if residual_acc_joint_weights is None
                                           else self.default_q.new_tensor(residual_acc_joint_weights))
        if (self.residual_acc_joint_weights.shape != self.default_q.shape
                or not torch.isfinite(self.residual_acc_joint_weights).all()
                or (self.residual_acc_joint_weights < 0).any()):
            raise ValueError("Residual acceleration joint weights must be finite, nonnegative and match the joints")
        self.hands = [self.kin.tracked_idx.index(self.kin.body_names.index(n)) for n in ("left_palm_link", "right_palm_link")]
        self.feet = [self.kin.tracked_idx.index(self.kin.body_names.index(n)) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.torso = self.kin.tracked_idx.index(self.kin.body_names.index("torso_link"))
        self.action_acc_joint_weights = (torch.ones_like(self.default_q) if action_acc_joint_weights is None
                                        else self.default_q.new_tensor(action_acc_joint_weights))
        if (self.action_acc_joint_weights.shape != self.default_q.shape
                or not torch.isfinite(self.action_acc_joint_weights).all()
                or (self.action_acc_joint_weights < 0).any()):
            raise ValueError("Action acceleration weights must be finite, nonnegative and match the joints")
        self.max_task_reward = 4.5 * CONTROL_DT
        if imitation_mode == "amp":
            self.max_step_reward = (4.5 + self.amp_reward_weight + self.approach_progress_weight) * CONTROL_DT
        self.critic_extra += 40  # object state, reference errors, scene context
        self.object_term_scale = self.default_q.new_tensor([.04, .01, .02])

    def obs(self, S, a_prev, xi, f):
        return super().obs(S[:6], a_prev, xi, f)

    def goal(self, S, f, K):
        return self.motion.goal(f, K, S[0] if self.goal_anchor == "robot" else None)

    def reward(self, S, f, a, a_prev, terms=False, w_contact=0., smooth_valid=None,
               a_prev2=None, smooth_acc_valid=None, previous_state=None):
        amp_mode = self.imitation_mode == "amp"
        if amp_mode:
            if previous_state is None or self.motion_prior is None:
                raise ValueError("AMP reward requires previous_state and a restored/trained motion_prior")
            r = torch.zeros_like(a[..., 0])
            detail = {}
            if terms:
                # Diagnostics never contribute a robot tracking gradient.
                with torch.no_grad():
                    _, detail = super().reward(S[:6], f, a, a_prev, terms=True)
                detail["legacy_track"] = detail["track"]
                detail["track"] = torch.zeros_like(r)
        else:
            r, detail = super().reward(S[:6], f, a, a_prev, terms=True, w_contact=w_contact)
        m = self.motion
        op, oR, ov, ow = S[6:]
        ep = (op - m.object_p[f]).square().sum(-1)
        er = rot_angle_sq(oR, m.object_R[f])
        ev = (ov - m.object_v[f]).square().sum(-1)
        fk = self.kin.forward(*S[:6])
        bp = fk[0][..., self.hands, :]
        ref = m.body_p[f][..., self.hands, :] - m.object_p[f].unsqueeze(-2)
        rel = bp - op.unsqueeze(-2)
        eh = (rel - ref).square().sum(-1)
        contact = m.hand_contact[f]
        # Palm proximity is a differentiable contact proxy; labels are left/right HANDS, never the all-zero foot columns.
        local = mv3(oR.transpose(-1, -2).unsqueeze(-3), rel)
        outside = torch.relu(local.abs() - m.context(f)[..., None, :3]).norm(dim=-1)
        touch = torch.sigmoid((.055 - outside) / .01)
        hand_clear = (1 - touch).prod(-1)
        grasp = (1 - (touch - contact).square()).mean(-1)
        hand = torch.exp(-eh / .1**2).mean(-1)
        released = (1 - contact).prod(-1) * (m.stages[m.clip_id[f]] == 2).float()
        recovery = released * torch.exp(-ov.square().sum(-1) / .15**2)
        obj = 1.5 * torch.exp(-ep / .15**2) + .5 * torch.exp(-er / .4**2) + .5 * torch.exp(-ev / .5**2)
        # Common task metrics apply to both old tracking and new AMP policies.
        # After each hand's release it must no longer be pulled toward its reference palm.
        hand_active = (m.stages[m.clip_id[f]] != 2).unsqueeze(-1) | (contact > .5)
        hand_near = torch.exp(-eh / .1**2)
        common_task_hand = (hand_near * hand_active).mean(-1)
        hand_score = hand_near
        if amp_mode and self.palm_far_weight:
            # Meter-scale approach errors saturate even a wider Gaussian. The
            # inverse-quadratic tail preserves navigation gradients and the maximum.
            scaled_error = eh / self.palm_far_std**2
            hand_far = (1 / (1 + scaled_error) if self.palm_far_kernel == "inverse_quadratic"
                        else torch.exp(-scaled_error))
            blend = (1 - self.palm_far_weight) * hand_near + self.palm_far_weight * hand_far
            hand_score = torch.where((m.stages[m.clip_id[f]] == 0).unsqueeze(-1), blend, hand_near)
        task_hand = (hand_score * hand_active).mean(-1)
        height = S[0][..., 2] - fk[0][..., self.feet, 2].amax(-1)
        torso_z = fk[1][..., self.torso, :, 2]
        tilt = torch.atan2(torso_z[..., :2].norm(dim=-1), torso_z[..., 2])
        threshold = m.standing_height_threshold[m.clip_id[f]]
        upright = torch.sigmoid((height - threshold) / .02) * torch.sigmoid((STAND_TILT - tilt) / .05)
        stable = torch.exp(-ep / .15**2 - er / .4**2 - ov.square().sum(-1) / .15**2 - ow.square().sum(-1) / .5**2)
        # A reference release label is only a target. A hand still on the box
        # must not earn the placed-and-released recovery bonus or count as success.
        task_recovery = released * hand_clear * stable * upright
        carry_task = (obj + task_hand + .5 * grasp + .5 * task_recovery) * CONTROL_DT
        common_carry_task = (obj + common_task_hand + .5 * grasp + .5 * task_recovery) * CONTROL_DT
        approach_progress = torch.zeros_like(r)
        if amp_mode and self.approach_progress_weight:
            # Reward actual robot closing, measured against ONE fixed box position
            # per transition; moving the box alone cannot earn progress. Stop at
            # 0.6 m, outside the training set's 90th-percentile first-grasp distance
            # (0.54 m). The 0.3 m/s cap matches its median approach speed (0.29 m/s).
            target = previous_state[6][..., :2]
            before = (previous_state[0][..., :2] - target).norm(dim=-1).clamp_min(.6)
            after = (S[0][..., :2] - target).norm(dim=-1).clamp_min(.6)
            closing = ((before - after) / (CONTROL_DT * .3)).clamp(-1., 1.)
            navigating = (m.stages[m.clip_id[f]] == 0) & (contact.amax(-1) < .5)
            approach_progress = self.approach_progress_weight * torch.where(navigating, closing, 0.) * CONTROL_DT
            carry_task = carry_task + approach_progress
        amp_style, amp_score, foot_slip = torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r)
        if amp_mode:
            prior = self.motion_prior
            amp_style, amp_score = prior(prior.features(previous_state), prior.features.from_fk(*fk), m.stages[m.clip_id[f]])
            rate = (a - a_prev).square().sum(-1)
            limit = (torch.relu(S[4] - self.soft_hi) + torch.relu(self.soft_lo - S[4])).sum(-1)
            # Contact selects where sliding is penalized. Its height derivative
            # would reward lifting a slipping foot and oppose standing recovery.
            c_feet = torch.sigmoid((.045 - fk[0][..., self.feet, 2]) / .005).detach()
            foot_slip = (c_feet * fk[2][..., self.feet, :2].square().sum(-1) / .3**2).mean(-1)
            control = (-self.cfg.w_action_rate * rate - self.cfg.w_joint_limit * limit + self.foot_slip_weight * foot_slip) * CONTROL_DT
            r = self.amp_reward_weight * amp_style * CONTROL_DT + carry_task - control
        else:
            r = r + (obj + hand + .5 * grasp + .5 * recovery) * CONTROL_DT
            control = (-self.cfg.w_action_rate * detail["action_rate"] - self.cfg.w_joint_limit * detail["joint_limit"]) * CONTROL_DT
        task_reward = r
        residual_rate = torch.zeros_like(r)
        residual_acc = torch.zeros_like(r)
        joint_vel_error = torch.zeros_like(r)
        action_acc = torch.zeros_like(r)
        if self.residual_rate_weight or self.residual_acc_weight:
            start = m.clip_start[m.clip_id[f]]
            prev_f = torch.maximum(f - 1, start)
        if self.residual_rate_weight:
            # Penalize changes of the learned correction, not the demonstrated movement.
            # f is the transition's END frame; never subtract a frame from another clip.
            ref_rate = (m.q[f] - m.q[prev_f]) / self.action_scale
            residual_rate = ((a - a_prev - ref_rate) ** 2).sum(-1)
            if smooth_valid is not None:
                residual_rate = residual_rate * smooth_valid
        if self.residual_acc_weight:
            if a_prev2 is None:
                raise ValueError("Residual acceleration needs two previous actions")
            prev2_f = torch.maximum(f - 2, start)
            ref_acc = (m.q[f] - 2 * m.q[prev_f] + m.q[prev2_f]) / self.action_scale
            residual_acc = ((a - 2 * a_prev + a_prev2 - ref_acc).square() * self.residual_acc_joint_weights).sum(-1)
            if smooth_acc_valid is not None:
                residual_acc = residual_acc * smooth_acc_valid
        if self.joint_vel_weight:
            # Reference-relative: standing still is not the minimum for a moving demonstration.
            joint_vel_error = (S[5] - m.qd[f]).square().mean(-1)
        if self.action_acc_weight:
            if a_prev2 is None:
                raise ValueError("Action acceleration needs two previous actions")
            action_acc = ((a - 2 * a_prev + a_prev2).square() * self.action_acc_joint_weights).sum(-1)
            if smooth_acc_valid is not None:
                action_acc = action_acc * smooth_acc_valid
        smooth_penalty = (self.residual_rate_weight * residual_rate + self.joint_vel_weight * joint_vel_error
                          + self.residual_acc_weight * residual_acc + self.action_acc_weight * action_acc) * CONTROL_DT
        r = r - smooth_penalty
        detail.update(object_pos=ep.sqrt(), object_ori=er.sqrt(), hand_error=eh.mean(-1).sqrt(),
                      object_track=obj * CONTROL_DT, hand_contact=grasp, release_stability=recovery,
                      task_reward=task_reward, residual_rate=residual_rate, joint_vel_error=joint_vel_error, residual_acc=residual_acc,
                       smooth_penalty=smooth_penalty, action_acc=action_acc, foot_slip=foot_slip,
                       carry_task_reward=common_carry_task, train_carry_task_reward=carry_task,
                       palms_long_range_reward=(task_hand - common_task_hand) * CONTROL_DT,
                       approach_progress_reward=approach_progress,
                       amp_reward=amp_style * self.amp_reward_weight * CONTROL_DT,
                       amp_score=amp_score, amp_style=amp_style, control_penalty=control + smooth_penalty,
                       grasp_success=((touch > .5) == (contact > .5)).all(-1).float(),
                       released=released, hand_clear=hand_clear,
                       stand_success=((height >= threshold) & (tilt <= STAND_TILT)).float(),
                       release_success=(released.bool() & (touch < .5).all(-1) & (ep < .15**2) & (er < .4**2)
                                        & (ov.square().sum(-1) < .15**2) & (ow.square().sum(-1) < .5**2)).float(),
                       release_recovery=task_recovery)
        if amp_mode and terms:
            # Signed contributions for reward audits; their sum is exactly r.
            detail.update(
                reward_amp=amp_style * self.amp_reward_weight * CONTROL_DT,
                reward_box_position=1.5 * torch.exp(-ep / .15**2) * CONTROL_DT,
                reward_box_orientation=.5 * torch.exp(-er / .4**2) * CONTROL_DT,
                reward_box_velocity=.5 * torch.exp(-ev / .5**2) * CONTROL_DT,
                reward_palms=task_hand * CONTROL_DT,
                reward_contact=.5 * grasp * CONTROL_DT,
                reward_standing=.5 * task_recovery * CONTROL_DT,
                reward_approach_progress=approach_progress,
                reward_action_rate=self.cfg.w_action_rate * rate * CONTROL_DT,
                reward_joint_limit=self.cfg.w_joint_limit * limit * CONTROL_DT,
                reward_foot_slip=-self.foot_slip_weight * foot_slip * CONTROL_DT,
                reward_action_acc=-self.action_acc_weight * action_acc * CONTROL_DT,
            )
        return (r, detail) if terms else r

    def _object_failure_margins(self, S, f):
        m, op = self.motion, S[6]
        held = (m.stages[m.clip_id[f]] == 1) | (m.hand_contact[f].amax(-1) > .5)
        # A dropped box rests ABOVE the floor. Compare to the demonstrated height;
        # allow 10 cm while carrying/grasping and 15 cm before grasp or after release.
        height_tol = torch.where(held, .10, .15)
        return torch.stack(((op - m.object_p[f]).norm(dim=-1) - .75,
                            m.object_p[f][..., 2] - op[..., 2] - height_tol, -.05 - op[..., 2]), -1)

    def termination(self, S, f):
        return super().termination(S[:6], f) | (self._object_failure_margins(S, f) > 0).any(-1)

    def soft_termination(self, S, f, tz=.02, tt=.04):
        robot = super().soft_termination(S[:6], f, tz, tt)
        failure = torch.sigmoid(self._object_failure_margins(S, f) / self.object_term_scale)
        return 1 - (1 - robot) * (1 - failure).prod(-1)

    def critic_features(self, S, f, xi):
        base = super().critic_features(S[:6], f, xi)
        op, oR, ov, ow = S[6:]
        m, Rt = self.motion, S[1].transpose(-1, -2)
        obj = torch.cat((mv3(Rt, op - S[0]), mat_to_6d(mm3(Rt, oR)), mv3(Rt, ov), mv3(Rt, ow),
                         mv3(Rt, m.object_p[f] - op), mat_to_6d(mm3(oR.transpose(-1, -2), m.object_R[f])),
                         mv3(Rt, m.object_v[f] - ov), mv3(Rt, m.object_w[f] - ow), scene_features(S, m.context(f))), -1)
        return torch.cat((base, obj), -1)


def build_task(task, robot, motion_dir, kin, device, xi_dim, split="train", stages=None, max_clips=None, reward_options=None):
    if task == "box_carry":
        motion = CarryMotionLib.from_dir(motion_dir or robot.motion_dir, robot, kin, device, split, stages, max_clips)
        cls = CarryMDP
    else:
        motion, cls = MotionLib.from_dir(motion_dir or robot.motion_dir, robot, kin, device), MDP
    options = reward_options or {}
    extra = (dict(action_scale=robot.action_scale_list(), residual_rate_weight=options.get("carry_residual_rate", 0.),
                  joint_vel_weight=options.get("carry_joint_vel", 0.),
                  residual_acc_weight=options.get("carry_residual_acc", 0.),
                  residual_acc_joint_weights=[options.get("carry_residual_acc_arms", 1.)
                      if any(part in name for part in ("shoulder", "elbow", "wrist")) else 1.
                      for name in robot.joint_names],
                  imitation_mode=options.get("imitation_mode", "tracking"),
                  amp_reward_weight=options.get("amp_reward_weight", 5.),
                  action_acc_weight=options.get("carry_action_acc", 0.),
                  action_acc_joint_weights=[options.get("carry_action_acc_arms", 0.)
                      if any(part in name for part in ("shoulder", "elbow", "wrist")) else 1.
                      for name in robot.joint_names],
                  foot_slip_weight=options.get("amp_foot_slip_weight", .25),
                  palm_far_weight=options.get("carry_palm_far_weight", 0.),
                  palm_far_std=options.get("carry_palm_far_std", .5),
                  palm_far_kernel=options.get("carry_palm_far_kernel", "gaussian"),
                  goal_anchor=options.get("carry_goal_anchor", "reference"),
                  approach_progress_weight=options.get("carry_approach_progress_weight", 0.)) if task == "box_carry" else {})
    mdp = cls(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=device), xi_dim, robot.sensing, **extra)
    if task != "box_carry":
        mdp.max_step_reward = MAX_STEP_REWARD
    return motion, mdp
