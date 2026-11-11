"""Run an exported policy (tools/export_onnx.py) in the simulator the way a robot controller would.

The observation is assembled from raw robot state and the ONNX outputs only (no blinkmimic.mdp
code), which is the loop a deployment controller implements. A robot with a sensing chain
(RobotSpec.sensing: IMU estimate, joint-velocity filter) takes the simulator's chain instead, which
stands in for the robot's own estimator. The motion plays once from its first frame with no
randomization or noise, as in blinkmimic.evaluate.

Usage: python tools/play_onnx.py logs/<run>/dance1_subject2.onnx --robot g1 \
           --motion data/lafan1_g1/dance1_subject2.csv
"""

import argparse

import numpy as np
import onnxruntime as ort
import torch

from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MAX_STEP_REWARD, MDP
from blinkmimic.motion import MotionLib
from blinkmimic.robot import get_robot
from blinkmimic.box_carry import CarryMotionLib, CarryMDP, robot_spec


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--robot", type=str, default="g1")
    ap.add_argument("--motion", type=str, required=True, help="the motion baked into the ONNX")
    a = ap.parse_args()

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    vec = lambda k: np.array(meta[k].split(","), dtype=np.float32)  # noqa: E731
    q0, H = vec("default_joint_pos"), int(meta["observation_history"])

    dev = "cuda:0"
    robot = get_robot(a.robot)
    task = meta.get("task", "tracking")
    visual = meta.get("vision", "none") == "latent"
    if task == "box_carry":
        robot = robot_spec(robot)
    assert ",".join(robot.joint_names) == meta["joint_names"]
    kin = Kinematics(robot, dev)
    motion = (CarryMotionLib.from_file(a.motion, robot, kin, dev) if task == "box_carry" else
              MotionLib([a.motion], kin, dev, robot.load_motion))
    mdp_cls = CarryMDP if task == "box_carry" else MDP
    mdp = mdp_cls(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
    env = TrackingSim(make_sim(robot), 1, robot, motion, mdp, dev, eval_mode=True, vision=visual)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(torch.arange(1, device=dev), motion.clip_start.clone())

    ref_quat = vec("anchor_quat_w_init")
    a_prev = env.a_prev[0].cpu().numpy().copy()
    hist, ret, obs_err = None, 0.0, 0.0
    while not bool(env.done_eval.all()):
        sim_o = env.observe()[0].cpu().numpy()
        if robot.sensing.stateful:
            o = sim_o  # filtered/estimated signals: the robot's estimator makes them, the simulator's chain stands in
        else:
            # sensors: joint encoders, root attitude and gyro (root frame)
            _, R, _, w, q, qd = (x[0].cpu().numpy() for x in env.state()[:6])
            R_rel = R.T @ quat_to_mat(ref_quat)
            o = np.concatenate((q - q0, qd, R.T @ w, -R[2], a_prev, R_rel[:, :2].reshape(6))).astype(np.float32)
            obs_err = max(obs_err, float(np.abs(o - sim_o).max()))
        hist = np.repeat(o[None], H, 0) if hist is None else np.concatenate((hist[1:], o[None]))
        feed = {"obs_history": hist[None], "time_step": np.array([[float(env.f.item())]], dtype=np.float32)}
        if visual:
            feed["depth"] = env.depth.cpu().numpy()
        if meta.get("carry_goal_anchor", "reference") == "robot":
            feed["root_pos_w"] = env.state()[0].cpu().numpy()
        out = dict(zip([x.name for x in sess.get_outputs()], sess.run(None, feed)))
        act = out["actions"][0]
        alive = not bool(env.done_eval.item())
        r = env.step(torch.from_numpy(act).to(dev).unsqueeze(0))["r"].item()
        ret += r if alive else 0.0
        env.f = torch.minimum(env.f, motion.clip_end - 1)
        a_prev, ref_quat = act, out["next_anchor_quat_w"][0]
    norm = ret / ((motion.clip_len[0].item() - 1) * getattr(mdp, "max_step_reward", MAX_STEP_REWARD))
    print(f"{meta['motion']}: norm return {norm:.4f}, failed {bool(env.failed_eval.item())}"
          + ("" if robot.sensing.stateful else f", max |obs - simulator obs| {obs_err:.2e}"))  # fmt: skip


if __name__ == "__main__":
    main()
