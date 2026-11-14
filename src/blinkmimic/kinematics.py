"""Differentiable forward kinematics of the robot in torch, built from its MJCF.

The same FK computes tracked-body poses and velocities for the reference motion, for logged
simulator states, and for world-model predictions, so the analytic reward head matches the
reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import RobotSpec
from .rotations import axis_angle_to_mat, mm3, mv3, quat_to_mat


class Kinematics:
    def __init__(self, robot: RobotSpec, device: str):
        m = mujoco.MjModel.from_xml_path(robot.mjcf_path)
        self.device = device
        tracked = robot.tracked_bodies
        self.joint_names = robot.joint_names
        jname_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
        # moving bodies in tree order (mujoco orders parents before children)
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            fixed = m.body_jntnum[b] == 0
            jid = m.body_jntadr[b] if not fixed else -1
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if not fixed and m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                pb = m.body_parentid[b]
                while pb not in mj_to_local:  # skip fixed parents
                    raise RuntimeError(f"fixed parent for {name}")
                parent.append(mj_to_local[pb])
                joint_idx.append(-1 if fixed else jname_to_idx[m.joint(jid).name])
            axes.append(np.zeros(3) if fixed else m.jnt_axis[jid].copy())
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # joint limits (in joint_names order)
        lo, hi = np.zeros(robot.num_joints), np.zeros(robot.num_joints)
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)
        # G1's four spherical sole pads, in heel-left/right, toe-left/right order.
        self.sole_idx, offsets, radii = [], [], []
        for side in ("left", "right"):
            name = side + "_ankle_roll_link"
            if name not in tracked:
                break
            bid = m.body(name).id
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid
                  and (m.geom_contype[g] or m.geom_conaffinity[g])
                  and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
            gs.sort(key=lambda g: (m.geom_pos[g, 0], -m.geom_pos[g, 1]))
            if len(gs) != 4:
                break
            self.sole_idx.append(tracked.index(name))
            offsets.append(m.geom_pos[gs].copy())
            radii.append(m.geom_size[gs, 0].copy())
        self.sole_offsets = torch.tensor(np.array(offsets), dtype=torch.float32, device=device)
        self.sole_radii = torch.tensor(np.array(radii), dtype=torch.float32, device=device)

    def sole_points(self, bp, bR, bv, bw):
        """Sphere-bottom positions and centre velocities, shape (..., foot=2, pad=4, xyz).

        The world-down radius offset is constant, so position derivatives equal centre
        velocities. These geometric probes are shared by rewards and evaluation.
        """
        if len(self.sole_idx) != 2:
            raise ValueError("Reference gait needs four spherical sole pads on each foot")
        r = mv3(bR[..., self.sole_idx, :, :].unsqueeze(-3), self.sole_offsets)
        pos = bp[..., self.sole_idx, :].unsqueeze(-2) + r
        bottom = torch.stack((pos[..., 0], pos[..., 1], pos[..., 2] - self.sole_radii), -1)
        vel = bv[..., self.sole_idx, :].unsqueeze(-2) + torch.cross(
            bw[..., self.sole_idx, :].unsqueeze(-2).expand_as(r), r, dim=-1)
        return bottom, vel

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,J) in joint_names order.
        """
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + mv3(Rp, self.off_p[i])
            R_off = mm3(Rp, self.off_R[i])
            if j < 0:
                Ri, wi = R_off, W[par]
            else:
                Ri = mm3(R_off, axis_angle_to_mat(self.axes[i], q[..., j]))
                axis_w = mv3(R_off, self.axes[i])
                wi = W[par] + axis_w * qd[..., j : j + 1]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )
