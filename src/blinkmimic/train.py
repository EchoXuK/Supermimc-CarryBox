"""SuperMimic training (SuperMimic paper, Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
from concurrent.futures import ThreadPoolExecutor
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--carry_gait_profile", choices=["legacy", "reference_phase", "reference_phase_v2", "reference_phase_v3"], default="legacy")
parser.add_argument("--carry_gait_scale", type=float, default=1.)
parser.add_argument("--carry_reward_profile", choices=["legacy", "grasp_place", "grasp_place_reach"], default="legacy")
parser.add_argument("--task", choices=["box_carry", "tracking"], default="box_carry")
parser.add_argument("--split", choices=["train", "val", "test"], default="train")
parser.add_argument("--stages", type=str, default=None, help="comma-separated stage names; default all three")
parser.add_argument("--max_clips", type=int, default=None, help="limit clips for a smoke run, interleaved across stages")
parser.add_argument("--vision", choices=["latent", "none"], default=None, help="default latent for box_carry, none for tracking")
parser.add_argument("--visual_dim", type=int, default=32)
parser.add_argument("--latent_weight", type=float, default=1.)
parser.add_argument("--latent_recon_weight", type=float, default=1.)
parser.add_argument("--latent_recon_images", type=int, default=64)
parser.add_argument("--no_compile", action="store_true", help="eager torch operations for small smoke runs")
parser.add_argument("--profile", action="store_true", help="log phase timings, WM/critic gradients and carry reward terms")
parser.add_argument("--robot", type=str, default="g1", help="robot package under blinkmimic/ (g1)")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--min_imagination_steps", type=int, default=None,
                    help="box_carry: minimum remaining reference transitions at actor starts; default --horizon")
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--goal_frame_offsets", type=str, default=None,
                    help="comma-separated reference offsets at 50 Hz; default 1..goal_frames; first must be 1")
parser.add_argument("--wm_update_interval", type=int, default=0,
                    help="one WM minibatch per this many real transitions; 0 preserves --wm_batches")
parser.add_argument("--policy_warmup_steps", type=int, default=None,
                    help="real transitions before actor updates; overrides --wm_only_iters")
parser.add_argument("--min_reset_steps", type=int, default=1,
                    help="minimum remaining reference transitions at sampled training resets; 1 preserves legacy sampling")
parser.add_argument("--probe_minutes", type=float, default=0., help="isolated 30-clip validation interval")
parser.add_argument("--probe_initial", action="store_true", help="also evaluate the random policy before collection")
parser.add_argument("--constant_lr", action="store_true")
parser.add_argument("--real_contact_reward", action="store_true", help="use the configured foot bonus in real BPTT collection too")
parser.add_argument("--dataset_fingerprint", type=str, default=None)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
# ---- InstantTrack options (defaults reproduce SuperMimic) ----
parser.add_argument("--wm_hidden", type=str, default="2048,2048,2048", help="world-model MLP widths")
parser.add_argument("--enc_hidden", type=str, default="2048,1024,512,256", help="motion-encoder MLP widths")
parser.add_argument("--dec_hidden", type=str, default="2048,1024,512,256", help="shared-decoder MLP widths")
parser.add_argument("--no_sync", action="store_true", help="remove host-device synchronizations from the update loop")
parser.add_argument("--fail_reset", type=float, default=0.0,
                    help="share of resets sampled one bin before recent termination points (anti-forgetting of hard clips)")
parser.add_argument("--fused_adam", action="store_true", help="fused Adam kernels and a compiled critic update")
parser.add_argument("--pi_beta1", type=float, default=0.9, help="Adam beta1 of the policy (SHAC uses 0.7)")
parser.add_argument("--pi_clip_value", type=float, default=0.1, help="element-wise policy gradient clip after the norm clip (<= 0: off)")
parser.add_argument("--fused_rot", action="store_true", help="elementwise 3x3 products (fused by torch.compile) instead of cuBLAS bmm")
parser.add_argument("--compile_more", action="store_true",
                    help="also compile the per-sim-step MDP heads (reward, termination, obs, goal), the WM normalizer inputs, and the critic bootstrap")  # fmt: skip
parser.add_argument("--pi_ema", type=float, default=0.0,
                    help="Polyak average of the policy weights (decay per update); checkpoints store the average as 'pi'")  # fmt: skip
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--contact_reward", type=float, default=0.0,
                    help="weight of the contact-matching / anti-slip reward term in imagination (evaluation reward unchanged)")
parser.add_argument("--carry_residual_rate", type=float, default=0., help="carry: weight of reference-relative action-change penalty")
parser.add_argument("--carry_joint_vel", type=float, default=0., help="carry: weight of mean squared joint-velocity error to reference")
parser.add_argument("--carry_residual_acc", type=float, default=0., help="carry: weight of reference-relative action second differences")
parser.add_argument("--carry_residual_acc_arms", type=float, default=1., help="carry: arm multiplier for residual acceleration; 0 preserves grasp corrections")
# ---- method changes of this release (defaults = SuperMimic baseline; scripts/train.sh sets the released recipe) ----
parser.add_argument("--wm_structured", action="store_true",
                    help="semi-implicit world model: velocity increments + position corrections on top of the trapezoidal drift")
parser.add_argument("--wm_mirror", type=float, default=0.0,
                    help="share of each world-model batch replaced by its left-right mirror image (physical symmetry; 0 = off)")
parser.add_argument("--pi_opt", type=str, default="adam", choices=["adam", "muon"],
                    help="policy optimizer: adam, or muon = Muon (Jordan et al. 2024) on the hidden weight matrices of the encoder and "
                         "decoder at Adam's update RMS (same lr), Adam on the first / last layers and vectors")
parser.add_argument("--wm_opt", type=str, default="adam", choices=["adam", "muon"],
                    help="world-model optimizer: adam, or muon = Muon on the hidden weight matrices of the world-model MLP (same "
                         "scheme as --pi_opt muon), Adam elsewhere")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--bin_fix", action="store_true",
                    help="correct per-bin reset statistics when several episodes of one step share a bin (env.py _bin_ema)")
parser.add_argument("--pi_lr_mult", type=float, default=1.0,
                    help="policy learning rate = --lr x this (world model and critic keep --lr)")
parser.add_argument("--lr_decay_ret", type=float, default=0.0,
                    help="progress-triggered decay of --pi_lr_mult (0 = off): once the logged training return (ep_ret_norm) "
                         "reaches this, the policy LR multiplier ramps linearly to 1 over --lr_decay_len iterations")
parser.add_argument("--lr_decay_len", type=int, default=300)
parser.add_argument("--lr_decay_by", type=int, default=0,
                    help="with --lr_decay_ret: start the decay at this iteration at the latest, also if the training return "
                         "has not reached --lr_decay_ret (0 = no deadline)")
parser.add_argument("--ref_contact_fix", action="store_true",
                    help="reference foot contact of the training contact reward from a per-clip calibrated floor "
                         "(height < 5th-percentile + 1.5 cm and horizontal speed < 0.5 m/s) instead of a fixed 4.5 cm")
parser.add_argument("--pi_ref_skip", action="store_true",
                    help="reference-action skip connection, a = (q_ref(f+1) - q0) / alpha + network output")
parser.add_argument("--eprobe_share", type=float, default=0.0,
                    help="evaluation-guided allocation (0 = off): this share of the resets goes to the --eprobe_win bins before "
                         "places where the deployed policy (EMA, deterministic, no randomization or noise) fails in a sweep over "
                         "all clips run every --eprobe_every iterations in a separate evaluation-mode simulator: one world per "
                         "--eprobe_tile frames of each clip, started a random lead in [--eprobe_lead_min, --eprobe_lead_max] "
                         "frames earlier on the reference")
parser.add_argument("--eprobe_tile", type=int, default=300)
parser.add_argument("--eprobe_lead_min", type=int, default=200)
parser.add_argument("--eprobe_lead_max", type=int, default=450)
parser.add_argument("--eprobe_every", type=int, default=250)
parser.add_argument("--eprobe_start", type=int, default=1250, help="first sweep (before it, nearly every probe fails and the focus is ~uniform)")
parser.add_argument("--eprobe_win", type=int, default=10, help="bins (1 s each) before a failure that receive the focus resets")
parser.add_argument("--eprobe_decay", type=float, default=0.5, help="per-sweep decay of the failure counts")
parser.add_argument("--eprobe_graph", action="store_true",
                    help="engineering: one sweep step (observation, policy, simulator step incl. MuJoCo Warp's solver loop, "
                         "bookkeeping) is replayed as a CUDA graph, without the per-step host check for all probes done "
                         "(the sweep always runs lead + tile steps, which every probe needs at most). Same computation; the "
                         "simulator is not bitwise reproducible either way")
parser.add_argument("--eprobe_async", type=int, default=0,
                    help="with --eprobe_graph: run each sweep on a side CUDA stream alongside training, this many sweep steps "
                         "launched per iteration, with a snapshot of the EMA policy taken when the sweep starts; its failures "
                         "update the focus distribution once the last step has run (ceil(steps / this) + 1 iterations later "
                         "instead of at once). 0 = off")
# ---- systems (wall-clock only; the same updates up to floating-point rounding) ----
parser.add_argument("--collect_graph", action="store_true",
                    help="engineering, with --graph_early: one collection step up to the episode ends (observation, policy "
                         "and action noise, simulator step incl. MuJoCo Warp's solver loop, reward, buffer write, normalizer "
                         "updates) is replayed as a CUDA graph; bin statistics and resets of ended episodes stay eager. Same "
                         "computation, different random stream")
parser.add_argument("--reset_masked", action="store_true",
                    help="engineering, with --collect_graph: the episode ends too (bin statistics, resets) as fixed-size masked "
                         "updates inside the collection graph, so a collection step has no host sync; the reset worlds get "
                         "the same values as with ids, random draws are made for all worlds")
parser.add_argument("--graph_early", action="store_true",
                    help="engineering, with --cuda_graph: the replay buffer keeps its fill level on the device and the updates "
                         "are captured at once instead of after the buffer is full (~375 iterations at --utd 0.4). Sampling "
                         "draws uniform floats scaled by the fill level instead of randint: same distribution, different "
                         "random stream")
parser.add_argument("--cuda_graph", action="store_true",
                    help="systems: capture one world-model update and one policy + critic update (forward, backward, optimizer, "
                         "EMA) as CUDA graphs once the replay buffer is full, then replay them (needs --no_sync --fused_adam)")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--ckpt_light", action="store_true", help="periodic checkpoints hold only the deployed policy (what the evaluators read), are written by a background thread, and last.pt is written only at the end (checkpoint I/O off the training clock)")
parser.add_argument("--ckpt_minutes", type=float, default=3.0, help="also checkpoint every n minutes of training wall-clock")
parser.add_argument("--max_hours", type=float, default=None, help="stop after this much training wall-clock")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--motion_dir", type=str, default=None, help="defaults to the robot's motion directory")
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--init_from", type=str, default=None,
                    help="initialize networks from a full checkpoint, with fresh optimizers, iteration and wall clock")
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
args = parser.parse_args()
if args.task != "box_carry" and args.carry_reward_profile != "legacy":
    parser.error("carry reward profiles require --task box_carry")
if args.task != "box_carry" and args.carry_gait_profile != "legacy":
    parser.error("carry gait profiles require --task box_carry")
if not math.isfinite(args.carry_gait_scale) or args.carry_gait_scale < 0:
    parser.error("carry_gait_scale must be finite and nonnegative")
if args.resume and args.init_from:
    parser.error("choose --resume or --init_from")
if args.task != "box_carry" and (args.carry_residual_rate or args.carry_joint_vel or args.carry_residual_acc):
    parser.error("carry smoothing rewards require --task box_carry")
args.vision = args.vision or ("latent" if args.task == "box_carry" else "none")
if args.vision == "latent" and args.latent_recon_weight <= 0:
    parser.error("--vision latent requires --latent_recon_weight > 0 to train the depth encoder")
if args.warmup_steps < args.wm_window or args.buffer_steps < args.warmup_steps:
    parser.error("require wm_window <= warmup_steps <= buffer_steps")
if args.task == "box_carry" and args.collect_graph:
    parser.error("box_carry uses eager collection for per-world box geometry/inertia resets; omit --collect_graph")

import torch
import torch.nn as nn
from functools import partial

from blinkmimic.buffer import ReplayBuffer
from blinkmimic.dr import XiLayout
from blinkmimic.env import PHYSICS_DT, TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP, inv_dim, unflatten_state
from blinkmimic.motion import CONTROL_DT, MotionLib
from blinkmimic.motion import goal_frame_offsets
from blinkmimic.training_control import TransitionBudget, ValidationProbe, TrainingHeartbeat, ParameterUpdates, world_model_probe, training_summaries
import blinkmimic.networks as networks
import blinkmimic.rotations as rotations
from blinkmimic.networks import Critic, WorldModel, check_pruned, initialize_policy, make_policy
from blinkmimic.mirror import Mirror
from blinkmimic.objective import actor_objective
from blinkmimic.mdp import flatten_state
from blinkmimic.robot import get_robot
from blinkmimic.sensing import ObsPipeline
from blinkmimic.box_carry import SCENE_DIM, build_task, robot_spec
from blinkmimic.vision import LatentDecoder, latent_prediction_loss, reconstruction_indices

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
goal_frame_offsets(vars(args))
if args.wm_update_interval < 0 or (args.policy_warmup_steps is not None and args.policy_warmup_steps < 0):
    parser.error("training step budgets must be nonnegative")
if args.min_reset_steps < 1:
    parser.error("min_reset_steps must be positive")
if args.horizon < 1:
    parser.error("horizon must be positive")
if args.min_imagination_steps is not None and (args.task != "box_carry" or
        not 1 <= args.min_imagination_steps <= args.horizon or args.cuda_graph):
    parser.error("min_imagination_steps requires box_carry, 1..horizon, and no CUDA graph")
if args.probe_minutes < 0 or ((args.probe_minutes or args.probe_initial) and args.task != "box_carry"):
    parser.error("validation probes need box_carry and a nonnegative interval")
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
robot = get_robot(args.robot)
if args.task == "box_carry":
    robot = robot_spec(robot)
OBJECT = args.task == "box_carry"
VISION = args.vision == "latent"
unflatten_state = partial(unflatten_state, object_state=OBJECT)
J = robot.num_joints
kin = Kinematics(robot, dev)
layout = XiLayout(robot, dev)
normalize_xi = layout.normalize
motion, mdp = build_task(args.task, robot, args.motion_dir, kin, dev, layout.dim, args.split,
                         args.stages.split(",") if args.stages else None, args.max_clips, reward_options=vars(args))
REAL_CONTACT = args.contact_reward if args.real_contact_reward else 0.
MAX_REWARD = mdp.max_step_reward + (2 * REAL_CONTACT * CONTROL_DT if args.carry_gait_profile == "legacy" else 0.)
MIN_IMAGINATION = args.min_imagination_steps or args.horizon
if not bool((motion.clip_len > MIN_IMAGINATION).any()):
    raise ValueError("No reference clip supplies the minimum imagination transitions")
print(f"task={args.task} vision={args.vision} split={args.split}: {len(motion.names)} clips, {motion.num_frames} frames", flush=True)
if OBJECT:
    print(f"joint-limit projection: {motion.projected_joint_frames} frames, max correction {motion.max_joint_correction:.4f} rad", flush=True)
pipe = ObsPipeline(robot, mdp, layout, dev, PHYSICS_DT)
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = motion.goal_dim(K)
CRITIC_IN = inv_dim(J) + mdp.critic_extra + GOAL_DIM

widths = lambda s: tuple(int(x) for x in s.split(","))  # noqa: E731
networks.RunningNorm.no_sync = args.no_sync
rotations.ELEMENTWISE = args.fused_rot

wm = WorldModel(J, layout.dim, use_prev_action=robot.has_action_delay, hidden=widths(args.wm_hidden),
                structured=args.wm_structured, dt=CONTROL_DT, object_state=OBJECT,
                z_dim=args.visual_dim if VISION else 0, scene_dim=SCENE_DIM if OBJECT else 0).to(dev)  # fmt: skip
pi = make_policy(mdp.obs_dim, GOAL_DIM, J, robot, vars(args)).to(dev)
critic = Critic(CRITIC_IN).to(dev)
latent_dec = LatentDecoder(args.visual_dim).to(dev) if VISION else None
wm_modules = nn.ModuleList([wm, pi.depth_enc, latent_dec]) if VISION else wm
actor_params = [p for n, p in pi.named_parameters() if not n.startswith("depth_enc.")]
actor_modules = nn.ModuleList([pi.enc, pi.dec])
VISION_STATS = torch.zeros(2, device=dev)
HEALTH_STATS = torch.zeros(2, device=dev)
PROFILE_TERMS = ("object_pos", "object_ori", "hand_error", "track", "object_track", "hand_contact", "release_stability",
                 "task_reward", "residual_rate", "joint_vel_error", "residual_acc", "smooth_penalty")
if args.task == "box_carry":
    PROFILE_TERMS += ("body_pos_reward", "body_ori_reward", "body_linvel_reward", "body_angvel_reward",
                      "root_pos_reward", "root_ori_reward", "action_rate_reward", "joint_limit_reward",
                      "foot_contact_reward", "hand_track_reward", "grasp_reward", "release_reward",
                      "gait_contact_penalty", "gait_motion_penalty", "gait_velocity_penalty", "gait_penalty",
                      "action_saturation_fraction")
if args.carry_reward_profile != "legacy":
    PROFILE_TERMS += ("grasp_penalty", "drop_penalty", "place_penalty")
if args.carry_reward_profile == "grasp_place_reach":
    PROFILE_TERMS += ("reach_root_penalty", "reach_hand_penalty", "reach_penalty")
PROFILE_COLUMNS = PROFILE_TERMS + tuple(k + "_env_std" for k in PROFILE_TERMS)
profile_terms, profile_times = [], []
GRAPH = args.cuda_graph
if GRAPH:
    assert args.no_sync and args.fused_adam, "--cuda_graph needs --no_sync --fused_adam"
    assert not pipe.stateful, "--cuda_graph: not with stateful sensing"
assert not args.eprobe_async or args.eprobe_share > 0, "--eprobe_async needs --eprobe_share"
assert not args.lr_decay_by or args.lr_decay_ret > 0, "--lr_decay_by needs --lr_decay_ret"
fused = {"fused": True} if args.fused_adam else {}
# --cuda_graph: the learning rate is a device tensor (filled every iteration) and Adam keeps its step on the GPU, so the
# captured optimizer steps follow the schedule
adam_kw = lambda: {"fused": True, "capturable": True, "lr": torch.tensor(args.lr, device=dev)} if GRAPH else {"lr": args.lr, **fused}  # noqa: E731
if args.wm_opt == "muon" or args.pi_opt == "muon":
    from blinkmimic.optim import MuonAdam
if args.wm_opt == "muon":
    opt_wm = MuonAdam(wm_modules, [wm.net], lr=args.lr, betas=(0.9, 0.999), fused=args.fused_adam, capturable=GRAPH)
else:
    opt_wm = torch.optim.Adam(wm_modules.parameters(), **adam_kw())
if args.pi_opt == "muon":
    opt_pi = MuonAdam(actor_modules, [pi.enc, pi.dec], lr=args.lr, betas=(args.pi_beta1, 0.95), fused=args.fused_adam, capturable=GRAPH)
else:
    opt_pi = torch.optim.Adam(actor_params, betas=(args.pi_beta1, 0.95), **adam_kw())
NONFINITE = torch.zeros((), dtype=torch.long, device=dev)  # --no_sync: skipped non-finite policy updates
pi_avg = torch.optim.swa_utils.AveragedModel(pi, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.pi_ema),
                                             use_buffers=True) if args.pi_ema > 0 else None  # fmt: skip
opt_v = torch.optim.Adam(critic.parameters(), **adam_kw())

start_it, wall_offset, sim_steps = 0, 0.0, 0


def load_opt_states(ck: dict):
    """Optimizer states of a checkpoint; with --cuda_graph the device learning-rate tensors are kept (filled)."""
    lr_t = {id(o): [g["lr"] for g in o.param_groups] for o in (opt_wm, opt_pi, opt_v)}
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    for o in (opt_wm, opt_pi, opt_v):
        for g, lr0 in zip(o.param_groups, lr_t[id(o)]):
            if torch.is_tensor(lr0) and g["lr"] is not lr0:
                lr0.fill_(float(g["lr"]))
                g["lr"] = lr0


INIT_BINS = None  # --resume: reset-sampler statistics, copied into the simulator once it exists
RESUME_CTRL = None  # --resume: the control state of last.pt, applied once the objects exist
if args.init_from:
    ck = torch.load(args.init_from, map_location=dev, weights_only=False)
    check_pruned(ck.get("args", {}), what=args.init_from)
    for key in ("task", "vision"):
        if ck["args"].get(key, {"task": "tracking", "vision": "none"}[key]) != getattr(args, key):
            raise ValueError(f"--init_from requires the checkpoint's {key}")
    wm.load_state_dict(ck["wm"])
    initialize_policy(pi, ck["pi"], ck["args"])
    critic.load_state_dict(ck["critic"])
    if VISION:
        latent_dec.load_state_dict(ck["latent_dec"])
    if pi_avg is not None:
        pi_avg.module.load_state_dict(pi.state_dict())
        pi_avg.n_averaged.fill_(1)
    print(f"initialized networks from {args.init_from} at iteration {ck['iteration']}; fresh optimizers and 0-hour clock", flush=True)
    del ck
if args.resume:
    # needs the full training state: last.pt. "pi" is the Polyak average with --pi_ema (what evaluation uses), so the
    # live policy comes from "pi_live" and the average from "pi_avg"
    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    check_pruned(ck.get("args", {}), what=args.resume)
    assert "opt_pi" in ck, f"--resume {args.resume}: no optimizer state (use last.pt)"
    for key in ("task", "vision", "split", "stages", "max_clips"):
        expected = {"task": "tracking", "vision": "none", "split": "train"}.get(key)
        if ck["args"].get(key, expected) != getattr(args, key):
            raise ValueError(f"--resume requires the checkpoint's {key}={ck['args'].get(key, expected)!r}")
    for key, default in (("carry_reward_profile", "legacy"),
                         ("carry_gait_profile", "legacy"), ("carry_gait_scale", 1.)):
        if ck["args"].get(key, default) != getattr(args, key):
            raise ValueError(f"--resume requires the checkpoint's {key}; use --init_from for a new experiment")
    if goal_frame_offsets(ck["args"]) != goal_frame_offsets(vars(args)):
        raise ValueError("--resume requires the checkpoint's goal_frame_offsets")
    for key, default in (("horizon", 32), ("min_imagination_steps", None),
                         ("wm_update_interval", 0), ("policy_warmup_steps", None), ("min_reset_steps", 1),
                         ("constant_lr", False), ("real_contact_reward", False), ("dataset_fingerprint", None)):
        if ck["args"].get(key, default) != getattr(args, key):
            raise ValueError(f"--resume requires the checkpoint's {key}")
    for key, default in (("carry_residual_rate", 0.), ("carry_joint_vel", 0.), ("carry_residual_acc", 0.),
                         ("carry_residual_acc_arms", 1.), ("contact_reward", 0.), ("gamma", .99), ("lam", .95)):
        if ck["args"].get(key, default) != getattr(args, key):
            raise ValueError(f"--resume requires the checkpoint's {key}; use --init_from to change the objective")
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck.get("pi_live", ck["pi"])), critic.load_state_dict(ck["critic"])
    if VISION:
        latent_dec.load_state_dict(ck["latent_dec"])
    load_opt_states(ck)
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
    if pi_avg is not None:
        if ck.get("pi_avg") is not None:
            pi_avg.load_state_dict(ck["pi_avg"])
        else:  # checkpoint of the code before this fix: "pi" holds the average; continue the EMA from it
            pi_avg.module.load_state_dict(ck["pi"])
            pi_avg.n_averaged.fill_(1)
            print(f"--resume: no pi_avg in {args.resume}; EMA continued from the saved averaged policy 'pi'", flush=True)
    RESUME_CTRL = ck.get("ctrl")
    if RESUME_CTRL is not None:
        INIT_BINS = {k: v.to(dev) for k, v in RESUME_CTRL["env_bins"].items()}
    else:
        print(f"--resume: no control state in {args.resume}; LR trigger, probe failures and reset statistics restart", flush=True)
    # a warm restart either way: the replay buffer, simulator state and random streams are not saved
    del ck


# Publish run metadata only after checkpoint validation and all state loads succeed.
os.makedirs(args.log_dir, exist_ok=True)
with open(os.path.join(args.log_dir, "train_pid.json"), "w") as stream:
    json.dump({"pid": os.getpid()}, stream)
with open(os.path.join(args.log_dir, "args.json"), "w") as stream:
    json.dump(vars(args), stream, indent=2)


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((J,), 3e-3), torch.full((J,), 5e-2))).to(dev)  # fmt: skip
if OBJECT:
    ERR_FLOOR = torch.cat((ERR_FLOOR, torch.tensor([.003] * 3 + [.005] * 3 + [.02] * 3 + [.05] * 3, device=dev)))


def _wm_rollout(s, a, a_prev, xi, z=None, scene=None):
    S = unflatten_state(s[:, 0])
    preds, zs = [], []
    for j in range(args.wm_window):
        S, _, z = wm.step_z(S, a[:, j], a_prev[:, j], normalize_xi(xi[:, j]), z,
                            scene[:, j] if scene is not None else None)
        preds.append(S)
        if VISION:
            zs.append(z)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(len(S))), torch.stack(zs, 1) if VISION else None


FEET = [robot.tracked_bodies.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")] if robot.name == "g1" else []
mdp.feet = FEET
assert args.contact_reward == 0 or FEET, "--contact_reward needs foot bodies (g1)"
if args.ref_contact_fix:
    motion.set_ref_contact(FEET)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.inc_norm.std(), ERR_FLOOR)
    S1 = unflatten_state(s1)
    e = (wm.state_error(S_hat, S1) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
    a = (pi(o_hist, g)).clamp(-20., 20.)
    S1 = wm.step(S, a, a_prev, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K)


def _imag_step_visual(S, o_hist, a_prev, f, xi, xi_n, g, z, scene):
    a = (pi(o_hist, g, visual=z)).clamp(-20., 20.)
    S1, _, z1 = wm.step_z(S, a, a_prev, xi_n, z, scene)
    alive = f < motion.clip_end[motion.clip_id[f]]
    S1 = tuple(torch.where(alive.reshape((-1,) + (1,) * (x.ndim - 1)), y, x) for x, y in zip(S, S1))
    a = torch.where(alive[:, None], a, a_prev)
    z1 = torch.where(alive[:, None], z1, z)
    # Clamp tails for short clips and starts close to the end; all heads stay within the same clip.
    f1 = torch.minimum(f + 1, motion.clip_end[motion.clip_id[f]])
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.where(alive[:, None, None], torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1), o_hist)
    return S1, a, o_hist, f1, motion.goal(f1, K), z1


def _imag_step_object(S, o_hist, a_prev, f, xi, xi_n, g, scene):
    a = (pi(o_hist, g)).clamp(-20., 20.)
    S1, _, _ = wm.step_z(S, a, a_prev, xi_n, scene=scene)
    alive = f < motion.clip_end[motion.clip_id[f]]
    S1 = tuple(torch.where(alive.reshape((-1,) + (1,) * (x.ndim - 1)), y, x) for x, y in zip(S, S1))
    a = torch.where(alive[:, None], a, a_prev)
    f1 = torch.minimum(f + 1, motion.clip_end[motion.clip_id[f]])
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    history = torch.where(alive[:, None, None], torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1), o_hist)
    return S1, a, history, f1, motion.goal(f1, K)


def _imag_step_sensed(S, o_hist, a_prev, f, xi, xi_n, g, st):
    """Imagination step with the robot's stateful signal chain (filters, delay) at the control rate."""
    a = (pi(o_hist, g)).clamp(-20., 20.)
    S1 = wm.step(S, a, a_prev, xi_n)
    f1 = f + 1
    gyro, accel = pipe.imu_from_states(S, S1, CONTROL_DT) if robot.sensing.imu is not None else (None, None)
    st = pipe.filter_update(st, S1[5], gyro, accel, xi, CONTROL_DT, noise=True, ctrl_rate=True)
    o1, st = pipe.observe(st, pipe.raw_obs(st, S1, a, xi, f1), noise=True, delay=True)
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K), st


def _reward_term(S, f, a, a_prev, ep_step, a_prev2, acc_valid):
    d = mdp.soft_termination(S, f)
    if OBJECT or VISION:
        d = torch.where(f >= motion.clip_end[motion.clip_id[f]], torch.ones_like(d), d)
    reward_options = {"smooth_valid": ep_step > 0, "a_prev2": a_prev2,
                      "smooth_acc_valid": acc_valid} if OBJECT else {}
    return mdp.reward(S, f, a, a_prev, w_contact=args.contact_reward, **reward_options), d


def _gae_targets(V, rews, dones):
    """GAE(lambda) returns over the imagined transitions; V is (H+1, B), bootstrapped from V[-1]."""
    R = V[-1]
    targets = []
    for t in range(HZ - 1, -1, -1):
        R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
        targets.append(R)
    return torch.stack(targets[::-1])


def _critic_loss(X, Y, valid=None):
    error = critic(X) - Y
    if valid is None:
        return error.square().mean()
    # Keep minibatch shapes fixed for torch.compile while excluding terminal
    # duplicates from both the loss and its gradient.
    return torch.where(valid, error, 0.).square().sum() / valid.sum().clamp(min=1)


def _wm_norm_data(S0, S1, a, a_prev, xi, scene=None):
    return wm.inputs(S0, a, a_prev, normalize_xi(xi), scene), wm.targets(S0, S1), WorldModel.increments(S0, S1)


def _critic_input(S, f, xi_n, g):
    return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


compile_fn = (lambda fn: fn) if args.no_compile else torch.compile
# The carry FK plus the smoothing reductions exceed Triton's fusion limit on this GPU.
# Bound fusion only for reward heads; keep the WM/actor compilation unchanged.
reward_compile = (partial(torch.compile, options={"max_fusion_size": 8})
                  if OBJECT and (args.carry_residual_rate or args.carry_joint_vel or args.carry_residual_acc
                                 or args.carry_reward_profile != "legacy")
                  and not args.no_compile else compile_fn)
wm_rollout = compile_fn(_wm_rollout)
wm_err = compile_fn(_wm_err)
imag_step = compile_fn(_imag_step)
imag_step_visual = compile_fn(_imag_step_visual)
imag_step_object = compile_fn(_imag_step_object)
imag_step_sensed = compile_fn(_imag_step_sensed)
reward_term = reward_compile(_reward_term)
critic_input_c = compile_fn(_critic_input)
wm_norm_data_c = compile_fn(_wm_norm_data)
gae_targets_c = compile_fn(_gae_targets)
critic_loss_c = compile_fn(_critic_loss)


def lr_at(it: int) -> float:
    if args.constant_lr:
        return args.lr
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
mirror = Mirror(robot, layout.slices, dev) if args.wm_mirror > 0 else None


def _mirror_windows(w: dict, n: int) -> dict:
    """Replace the first n windows by their mirror images (states, actions, sim context)."""
    out = dict(w)
    ms = lambda x: flatten_state(mirror.state(unflatten_state(x)))  # noqa: E731
    for k, fn in (("s", ms), ("s1", ms), ("a", mirror.joints), ("a_prev", mirror.joints), ("xi", mirror.xi)):
        out[k] = torch.cat((fn(w[k][:n]), w[k][n:]))
    for k in ("depth", "depth1", "depth1_clean", "scene"):
        if k in w:
            fn = mirror.scene if k == "scene" else mirror.depth
            out[k] = torch.cat((fn(w[k][:n]), w[k][n:]))
    return out


def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    if mirror is not None:
        w = _mirror_windows(w, int(args.wm_batch * args.wm_mirror))
    z0 = None
    if VISION:
        with torch.no_grad():
            z0 = pi.depth_enc(w["depth"][:, 0].float())
            target = pi.depth_enc(w["depth1"].flatten(0, 1).float()).view(args.wm_batch, args.wm_window, -1)
            dz = target - torch.cat((z0.unsqueeze(1), target[:, :-1]), 1)
            wm.update_z_stats(target, dz, w["mask"])
    S_hat, Z_hat = wm_rollout(w["s"], w["a"], w["a_prev"], w["xi"], z0, w.get("scene"))
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    if VISION:
        latent_loss = latent_prediction_loss(Z_hat, target, w["mask"])
        idx = reconstruction_indices(w["mask"], args.latent_recon_images)
        rec = latent_dec(pi.depth_enc(w["depth1"].flatten(0, 1)[idx].float()))
        rec_error = (rec - w["depth1_clean"].flatten(0, 1)[idx].float()).abs().flatten(1).mean(-1)
        valid = w["mask"].flatten()[idx]
        recon = (rec_error * valid).sum() / valid.sum().clamp(min=1)
        loss = loss + args.latent_weight * latent_loss + args.latent_recon_weight * recon
        VISION_STATS.copy_(torch.stack((latent_loss.detach() / args.wm_window, recon.detach())))
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    wm_gn = nn.utils.clip_grad_norm_(wm_modules.parameters(), 1.0)
    if args.profile:
        HEALTH_STATS[0].copy_(wm_gn.detach())
    opt_wm.step()
    if VISION and pi_avg is not None:
        with torch.no_grad():
            torch._foreach_copy_(list(pi_avg.module.depth_enc.parameters()), list(pi.depth_enc.parameters()))
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


def sample_valid_starts(B: int):
    if args.no_sync:
        # Prefer full-length starts without a host count. If there are too few,
        # carry imagination clamps to this clip and ends its objective at the tail.
        s, h = buffer.sample_starts(2 * B, Hh)
        f = s["f"]
        ok = f + MIN_IMAGINATION <= motion.clip_end[motion.clip_id[f]]
        idx = torch.sort((~ok).to(torch.uint8), stable=True).indices[:B]
        return {k: v[idx] for k, v in s.items()}, {k: v[idx] for k, v in h.items()}
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buffer.sample_starts(2 * B, Hh)
        f = s["f"]
        ok = f + MIN_IMAGINATION <= motion.clip_end[motion.clip_id[f]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


@torch.no_grad()
def ema_update():
    """The EMA step of ``AveragedModel.update_parameters`` (same lerp) without its host-side first-call test; the
    host-side counter n_averaged is advanced by the training loop."""
    avg = list(pi_avg.module.parameters()) + list(pi_avg.module.buffers())
    live = [t.detach() for t in list(pi.parameters()) + list(pi.buffers())]
    assert all(t.is_floating_point() for t in avg)
    torch._foreach_lerp_(avg, live, 1 - args.pi_ema)


class Graphed:
    """Run ``fn`` eagerly until ``ready()``; then eagerly on a side stream for ``warmup`` calls, capture it as a CUDA
    graph on the next call and replay it from then on. Each call is one update; outputs are cloned out of the graph pool."""

    def __init__(self, fn, ready, warmup: int = 3):
        self.fn, self.ready, self.warmup, self.calls, self.graph, self.out = fn, ready, warmup, 0, None, None

    def __call__(self):
        if self.graph is None and not self.ready():
            return self.fn()
        self.calls += 1
        if self.calls <= self.warmup:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                out = self.fn()
            torch.cuda.current_stream().wait_stream(side)
            return out
        if self.graph is None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # return the eager phase's cached blocks before the graph takes its own pool
            t0 = time.time()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.fn()
            torch.cuda.synchronize()
            print(f"cuda_graph: captured {self.fn.__name__} in {time.time() - t0:.1f} s, "
                  f"{torch.cuda.memory_reserved() / 2**30:.1f} GB reserved", flush=True)  # fmt: skip
        self.graph.replay()
        return tuple(x.clone() for x in self.out)


def imagine_actor(critic_data=True, diagnostics=False):
    B = min(16, args.pi_batch) if diagnostics else args.pi_batch
    s, h = sample_valid_starts(B)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    a_prev2 = s["a_prev2"]
    xi_n = normalize_xi(xi)
    z = None
    if VISION:
        with torch.no_grad():
            z = pi.depth_enc(s["depth"].float())
    if pipe.stateful:  # the observations the policy actually saw, and the sensing state after the last one
        o_hist, st = h["o"], pipe.unpack(s["obs_state"])
    else:
        with torch.no_grad():
            o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        o_hist = mdp.add_noise(o_hist)
    set_requires_grad(wm_modules, False)
    set_requires_grad(critic, False)
    g0 = motion.goal(f, K)
    g = g0
    S0 = S
    Ss, As, Aps, Ap2s, Fs, Gs = [], [], [], [], [], []
    for t in range(HZ):
        Aps.append(a_prev)
        Ap2s.append(a_prev2)
        a_prev2 = a_prev
        if VISION:
            step = _imag_step_visual if diagnostics else imag_step_visual
            S, a_prev, o_hist, f, g, z = step(S, o_hist, a_prev, f, xi, xi_n, g, z, s.get("scene"))
        elif OBJECT:
            step = _imag_step_object if diagnostics else imag_step_object
            S, a_prev, o_hist, f, g = step(S, o_hist, a_prev, f, xi, xi_n, g, s["scene"])
        elif pipe.stateful:
            S, a_prev, o_hist, f, g, st = imag_step_sensed(S, o_hist, a_prev, f, xi, xi_n, g, st)
        else:
            S, a_prev, o_hist, f, g = imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(len(S)))
    f_all = flat(Fs)
    episode_steps = (s["ep_step"].unsqueeze(0) + torch.arange(HZ, device=dev).unsqueeze(1)).flatten()
    history_valid = ((torch.arange(HZ, device=dev).unsqueeze(1) > 0) | s["a_prev2_valid"].unsqueeze(0)).flatten()
    reward_fn = _reward_term if diagnostics else reward_term
    r, d = reward_fn(S_all, f_all, flat(As), flat(Aps), episode_steps,
                       flat(Ap2s), (episode_steps > 1) & history_valid)
    r, d = r.view(HZ, B), d.view(HZ, B)
    v_boot = critic((critic_input_c if args.compile_more and not diagnostics else _critic_input)(S, f, xi_n, g))
    # --cuda_graph: survival products as a loop (cumprod's backward reads the host)
    J, c_t, c = actor_objective(r, d, v_boot, args.gamma, loop_cumprod=GRAPH)
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    loss = -J.mean()
    X = None
    if critic_data:
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(len(S)))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
    valid = None
    if args.min_imagination_steps is not None:
        remaining = motion.clip_end[motion.clip_id[s["f"]]] - s["f"]
        valid = torch.arange(HZ + 1, device=dev)[:, None] <= remaining[None, :]
    output = loss, r_sum, c, X, rews, dones, valid
    if diagnostics:
        _, detail = mdp.reward(S_all, f_all, flat(As), flat(Aps), terms=True, w_contact=args.contact_reward,
                              smooth_valid=episode_steps > 0, a_prev2=flat(Ap2s),
                              smooth_acc_valid=(episode_steps > 1) & history_valid)
        discount = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        weighted = lambda key: (discount * c_t * detail[key].view(HZ, B)).sum(0).mean()
        names = ("body_pos_reward", "body_ori_reward", "body_linvel_reward", "body_angvel_reward",
                 "root_pos_reward", "root_ori_reward", "action_rate_reward", "joint_limit_reward",
                 "object_track", "hand_track_reward", "grasp_reward", "release_reward",
                 "grasp_penalty", "drop_penalty", "place_penalty", "reach_root_penalty", "reach_hand_penalty",
                 "gait_contact_penalty", "gait_motion_penalty", "gait_velocity_penalty")
        losses = {key: weighted(key) * (1 if "penalty" in key else -1) for key in names}
        losses["gait"] = weighted("gait_penalty")
        losses["task_and_value"] = loss - losses["gait"]
        losses["total"] = loss
        return (*output, losses)
    return output


def reward_gradient_probe():
    """Read-only attribution on 16 real starts; preserve RNG, gradients and module flags."""
    from blinkmimic.training_control import evaluation_rng
    params = list(wm_modules.parameters()) + list(critic.parameters())
    flags = [p.requires_grad for p in params]
    saved_grads = [p.grad for p in actor_params]
    # --compile_more also wraps these shared bound methods.  The diagnostic
    # batch has a different shape and requests reward details, so calling the
    # wrappers here would compile new forward/backward graphs during a probe.
    callables = [(obj, name, getattr(obj, name)) for obj, name in
                 ((mdp, "obs"), (mdp, "reward"), (motion, "goal"))]
    try:
        for obj, name, fn in callables:
            setattr(obj, name, getattr(fn, "_torchdynamo_orig_callable", fn))
        with evaluation_rng(dev), torch.enable_grad():
            *_, losses = imagine_actor(critic_data=False, diagnostics=True)
            output, vectors = {}, {}
            for key, value in losses.items():
                grads = torch.autograd.grad(value, actor_params, retain_graph=True, allow_unused=True)
                vector = torch.cat([g.detach().flatten() if g is not None else torch.zeros_like(p).flatten()
                                    for g, p in zip(grads, actor_params)])
                output[key] = dict(loss=float(value.detach()), gradnorm=float(vector.norm()),
                                   finite=bool(torch.isfinite(vector).all()))
                if key in ("gait", "task_and_value", "total"):
                    vectors[key] = vector
            a, b = vectors["gait"], vectors["task_and_value"]
            output["task_gait_cosine"] = float(torch.dot(a, b) / (a.norm()*b.norm()).clamp(min=1e-20))
            output["combined_fraction"] = float(vectors["total"].norm() / (a.norm()+b.norm()).clamp(min=1e-20))
            output["batch"] = min(16, args.pi_batch)
            return output
    finally:
        for obj, name, fn in callables:
            setattr(obj, name, fn)
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)
        for p, grad in zip(actor_params, saved_grads):
            p.grad = grad


def step_actor():
    gn = nn.utils.clip_grad_norm_(actor_params, 1.0)
    if args.pi_clip_value > 0:
        nn.utils.clip_grad_value_(actor_params, args.pi_clip_value)
    if args.no_sync:
        # skip a non-finite update without a host read: zero the gradients on the GPU and gate the optimizer step (parameters
        # and all optimizer state unchanged; the plain Adam path without --fused_adam only zeroes the gradients, so its
        # momenta still move the parameters); counted in the log column nonfinite
        finite = torch.isfinite(gn)
        NONFINITE.add_((~finite).long())
        for p in actor_params:
            if p.grad is not None:
                p.grad.copy_(torch.where(finite, p.grad, 0.0))
        if isinstance(opt_pi, torch.optim.Optimizer) and not opt_pi.defaults.get("fused"):
            opt_pi.step()
        elif isinstance(opt_pi, torch.optim.Optimizer):
            opt_pi.found_inf = (~finite).float()  # the fused kernel skips the step and undoes the step count
            try:
                opt_pi.step()
            finally:
                opt_pi.found_inf = None
        else:
            opt_pi.step(ok=finite)
    elif torch.isfinite(gn):
        opt_pi.step()
    else:
        NONFINITE.add_(1)
    return gn


def update_policy_average():
    if pi_avg is not None:
        if GRAPH and int(pi_avg.n_averaged) > 0:
            ema_update()
        else:
            pi_avg.update_parameters(pi)
        if VISION:
            with torch.no_grad():
                torch._foreach_copy_(list(pi_avg.module.depth_enc.parameters()), list(pi.depth_enc.parameters()))


def policy_update():
    loss, r_sum, c, X, rews, dones, valid = imagine_actor()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    gn = step_actor()
    update_policy_average()
    set_requires_grad(wm_modules, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = torch.zeros(())
    critic.norm.update(X if valid is None else X[valid])
    with torch.no_grad():
        targets = (gae_targets_c if args.fused_adam else _gae_targets)(critic(X), rews, dones)  # (H, B)
    Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
    # A transition starting at the terminal frame does not exist. Include the
    # terminal state in normalization, but never fit repeated-tail GAE.
    transitions = None if valid is None else valid[1:].reshape(-1)
    n = Xf.shape[0]
    mb = n // args.critic_minibatches
    for _ in range(args.critic_epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(args.critic_minibatches):
            idx = perm[i * mb : (i + 1) * mb]
            loss_fn = critic_loss_c if args.fused_adam else _critic_loss
            v_loss = loss_fn(Xf[idx], Yf[idx], None if transitions is None else transitions[idx])
            opt_v.zero_grad(set_to_none=True)
            v_loss.backward()
            v_gn = nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            if args.profile:
                HEALTH_STATS[1].copy_(v_gn.detach())
            opt_v.step()
    return loss.detach(), v_loss.detach(), (r_sum / HZ / MAX_REWARD).mean(), gn.detach(), c.mean().detach()


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(robot)
if args.compile_more:
    # the simulator wrapper calls these once per control step in eager mode otherwise (FK + rotations: thousands of kernels)
    mdp.reward, mdp.termination, mdp.obs = reward_compile(mdp.reward), compile_fn(mdp.termination), compile_fn(mdp.obs)
    motion.goal = compile_fn(motion.goal)
env = TrackingSim(sim_cfg, args.num_envs, robot, motion, mdp, dev, gamma=args.gamma, fail_reset=args.fail_reset,
                  bin_fix=args.bin_fix, vision=VISION, training_contact_reward=REAL_CONTACT,
                  min_reset_steps=args.min_reset_steps)
if INIT_BINS is not None:  # --resume: the checkpoint's reset-sampler statistics (episodes already running keep their start)
    env.bin_G.copy_(INIT_BINS["bin_G"]), env.bin_fail.copy_(INIT_BINS["bin_fail"]), env.bin_w_avg.copy_(INIT_BINS["bin_w_avg"])
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, J, layout.dim,
                      mdp.obs_dim if pipe.stateful else None, pipe.state_dim if pipe.stateful else None,
                      object_state=OBJECT, vision=VISION)  # fmt: skip
obs_hist = torch.zeros(args.num_envs, Hh, mdp.obs_dim, device=dev)


CG = {"calls": 0}  # --collect_graph: static tensors and the graph


@torch.no_grad()
def _collect_core():
    """One collection step up to the episode ends, on the static tensors of CG (the --collect_graph body)."""
    o = env.observe()
    fresh = env.ep_step == 0
    oh = CG["obs_hist"]
    oh_new = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(oh), torch.cat((oh[:, 1:], o.unsqueeze(1)), 1))
    g = motion.goal(env.f, K)
    pi.obs_norm.update(o)
    pi.goal_norm.update(g)
    a = pi(oh_new, g, depth=env.depth)
    a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
    out = env.step_core(a)
    buffer.add_device(out)
    S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
    wm_in, wm_out, wm_inc = (wm_norm_data_c if args.compile_more else _wm_norm_data)(S0, S1, out["a"], out["a_prev"], out["xi"], out.get("scene"))
    wm.in_norm.update(wm_in)
    wm.out_norm.update(wm_out)
    wm.inc_norm.update(wm_inc)
    oh.copy_(oh_new)
    # the simulator wrapper rebinds these per step: copy them back into the static tensors the graph reads
    CG["f"].copy_(env.f), CG["ep_step"].copy_(env.ep_step), CG["a_prev"].copy_(env.a_prev)
    env.f, env.ep_step, env.a_prev = CG["f"], CG["ep_step"], CG["a_prev"]
    core = {"done": out["done"], "term": out["term"], "f": out["f"], "r_mean": out["r"].mean() / MAX_REWARD}
    if args.reset_masked:  # the episode ends in the graph too (reset_masked writes f, ep_step, a_prev in place)
        out = env.step_tail_masked(out)
        n = out["n_done"]
        nan = torch.full((), float("nan"), device=dev)
        core.update(ep_len=torch.where(n > 0, out["ep_len_sum"] / n.clamp(min=1), nan),
                    ep_ret=torch.where(n > 0, out["ep_ret_norm_sum"] / n.clamp(min=1), nan),
                    term_sum=out["term"].float().sum(), done_sum=n)  # fmt: skip
    return core


@torch.no_grad()
def collect_graphed_step():
    import warp as wp
    from mjlab.sim.sim_data import TorchArray

    global obs_hist
    if CG["calls"] == 0:
        CG.update(obs_hist=obs_hist.clone(), f=env.f.clone(), ep_step=env.ep_step.clone(), a_prev=env.a_prev.clone())
        env.f, env.ep_step, env.a_prev = CG["f"], CG["ep_step"], CG["a_prev"]
    CG["calls"] += 1
    if CG["calls"] <= 3:  # warmup on the default stream (Warp's own stream orders with it implicitly)
        out = _collect_core()
    elif "graph" not in CG:
        torch.cuda.synchronize()
        t0 = time.time()
        CG["graph"] = torch.cuda.CUDAGraph()
        setitem, env.sim.use_cuda_graph = TorchArray.__setitem__, False  # as eprobe_sweep_graphed
        TorchArray.__setitem__ = lambda self, idx, value: self._tensor.__setitem__(idx, value)
        try:
            with torch.cuda.graph(CG["graph"], capture_error_mode="thread_local"):
                ws = wp.stream_from_torch(torch.cuda.current_stream())
                with wp.ScopedStream(ws, sync_enter=False, sync_exit=False):
                    wp.capture_begin(stream=ws, external=True)
                    try:
                        CG["out"] = _collect_core()
                    finally:
                        CG["wp_graph"] = wp.capture_end(stream=ws)  # keep: it holds the captured kernels' modules
        finally:
            TorchArray.__setitem__, env.sim.use_cuda_graph = setitem, True
        torch.cuda.synchronize()
        print(f"--collect_graph: captured one collection step in {time.time() - t0:.1f} s", flush=True)
        CG["graph"].replay()  # the capture itself does not run the step
        out = CG["out"]
    else:
        CG["graph"].replay()
        out = CG["out"]
    buffer.advance_host()
    obs_hist = CG["obs_hist"]
    return out


@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    if args.collect_graph:
        for _ in range(n_steps):
            core = collect_graphed_step()
            sim_steps += args.num_envs
            stats["r"].append(core["r_mean"].clone())
            if args.reset_masked:  # every step appends; steps without an ended episode carry NaN means (skipped in the log)
                for k_s, k_c in (("ep_len", "ep_len"), ("ep_ret", "ep_ret"), ("term", "term_sum"), ("done", "done_sum")):
                    stats[k_s].append(core[k_c].clone())
                continue
            out = env.step_tail({"done": core["done"], "term": core["term"], "f": core["f"]})
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].mean())
                stats["ep_ret"].append(out["ep_ret_norm"].mean())
                stats["term"].append(core["term"].float().sum())
                stats["done"].append(core["done"].float().sum())
        return
    for _ in range(n_steps):
        o = env.observe()
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        g = motion.goal(env.f, K)
        pi.obs_norm.update(o)
        pi.goal_norm.update(g)
        a = pi(obs_hist, g, depth=env.depth)
        a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
        out = env.step(a)
        buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        wm_in, wm_out, wm_inc = (wm_norm_data_c if args.compile_more else _wm_norm_data)(S0, S1, out["a"], out["a_prev"], out["xi"], out.get("scene"))
        wm.in_norm.update(wm_in)
        wm.out_norm.update(wm_out)
        wm.inc_norm.update(wm_inc)
        sim_steps += args.num_envs
        stats["r"].append(out["r"].mean() / MAX_REWARD)
        if args.profile and OBJECT:
            out["terms"]["action_saturation_fraction"] = (out["a"].abs() >= 20.).float().mean(-1)
            values = torch.stack([out["terms"][k] for k in PROFILE_TERMS])
            profile_terms.append(torch.cat((values.mean(-1), values.std(-1, unbiased=False))))
        if "ep_len" in out:
            stats["ep_len"].append(out["ep_len"].mean())
            stats["ep_ret"].append(out["ep_ret_norm"].mean())
            stats["term"].append(out["term"].float().sum())
            stats["done"].append(out["done"].float().sum())

if GRAPH:
    assert not args.reset_masked or args.collect_graph, "--reset_masked needs --collect_graph"
    if args.reset_masked:
        env.use_static_focus()
    assert not args.collect_graph or (args.graph_early and not env.push_enabled and not pipe.stateful), \
        "--collect_graph: needs --graph_early; no pushes, sensing state"
    buffer.use_device_ptr(device_size=args.graph_early)
    # captured sampling has static bounds (buffer.size): capture once the buffer is full (--graph_early: device-side
    # bounds, capture at once)
    full = (lambda: True) if args.graph_early else (lambda: buffer.size == buffer.T)
    wm_update_g = Graphed(wm_update, full)
    policy_update_g = Graphed(policy_update, lambda: full() and int(pi_avg.n_averaged if pi_avg is not None else 1) > 0)
else:
    wm_update_g, policy_update_g = wm_update, policy_update


EPROBE = args.eprobe_share > 0  # --eprobe_share; off: code path and random stream unchanged
if EPROBE:
    EP_TILE = torch.cat([torch.arange(int(motion.clip_start[c]), int(motion.clip_end[c]) - 1, args.eprobe_tile)
                         for c in range(len(motion.names))]).to(dev)  # fmt: skip
    P_EP = len(EP_TILE)
    assert pi_avg is not None, "--eprobe_share sweeps the EMA policy: needs --pi_ema"
    assert not args.eprobe_async or args.eprobe_graph, "--eprobe_async needs --eprobe_graph"
    EP_PI = copy.deepcopy(pi_avg.module) if args.eprobe_async else pi_avg.module  # sweep policy (async: a snapshot)
    EP_SIDE, EP_ASYNC, ITER_NOW = (torch.cuda.Stream() if args.eprobe_async else None), {}, [0]
    penv = TrackingSim(make_sim(robot), P_EP, robot, motion, mdp, dev, eval_mode=True, randomize=False, reset_noise=False, vision=VISION)
    EP_FAIL = torch.zeros(motion.num_bins, device=dev)  # decayed counts of first failures per bin
    # bin b -> the next bin of the same clip (or itself at the clip end): walking it --eprobe_win times spreads a failure
    # weight back over the window before it
    _nxt = torch.arange(motion.num_bins, device=dev) + 1
    EP_NEXT = torch.where((_nxt < motion.num_bins) & (motion.bin_clip[_nxt.clamp(max=motion.num_bins - 1)] == motion.bin_clip), _nxt, _nxt - 1)
    print(f"--eprobe: {len(EP_TILE)} probe worlds (tiles of {args.eprobe_tile} frames)", flush=True)


@torch.no_grad()
def eprobe_sweep_eager(n_steps, stop, fcap):
    hist = None
    fail_f = torch.full((P_EP,), -1, dtype=torch.long, device=dev)
    for _ in range(n_steps):
        o = penv.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi_avg.module(hist, motion.goal(penv.f, K), depth=penv.depth)
        alive = ~penv.done_eval.clone()
        f_now = penv.f.clone()
        out = penv.step(a)
        fail_f = torch.where(alive & (out["term"] | ~torch.isfinite(a).all(-1)), f_now, fail_f)  # a non-finite world counts as failed
        penv.done_eval |= penv.f >= stop
        penv.f = torch.minimum(penv.f, fcap)
        if bool(penv.done_eval.all()):
            break
    return fail_f


EPG = {}  # --eprobe_graph: static tensors and the graph


def _eprobe_step():
    o = penv.observe()
    h = torch.cat((EPG["hist"][:, 1:], o.unsqueeze(1)), 1)
    a = EP_PI(h, motion.goal(penv.f, K), depth=penv.depth)
    alive = ~penv.done_eval.clone()
    f_now = penv.f.clone()
    out = penv.step(a)
    ff = torch.where(alive & (out["term"] | ~torch.isfinite(a).all(-1)), f_now, EPG["fail_f"])  # non-finite world: failed
    penv.done_eval |= penv.f >= EPG["stop"]
    f1 = torch.minimum(penv.f, EPG["fcap"])
    # the simulator wrapper rebinds these per step: copy them back into the static tensors the graph reads
    EPG["hist"].copy_(h), EPG["fail_f"].copy_(ff), EPG["f"].copy_(f1)
    EPG["ep_step"].copy_(penv.ep_step), EPG["a_prev"].copy_(penv.a_prev)
    penv.f, penv.ep_step, penv.a_prev = EPG["f"], EPG["ep_step"], EPG["a_prev"]


@torch.no_grad()
def eprobe_sweep_graphed(n_steps, stop, fcap):
    for _ in range(eprobe_graph_setup(n_steps, stop, fcap)):
        EPG["graph"].replay()
    return EPG["fail_f"].clone()


@torch.no_grad()
def eprobe_graph_setup(n_steps, stop, fcap):
    """eprobe_sweep_eager as replays of one captured step. The first history (H copies of o_0) is set eagerly; the
    first step shifts in o_0 again (the state has not moved), so the histories match the eager loop."""
    import warp as wp
    from mjlab.sim.sim_data import TorchArray

    if not EPG:
        EPG.update(hist=torch.zeros(P_EP, Hh, mdp.obs_dim, device=dev), fail_f=torch.zeros(P_EP, dtype=torch.long, device=dev),
                   stop=torch.zeros(P_EP, dtype=torch.long, device=dev), fcap=torch.zeros(P_EP, dtype=torch.long, device=dev),
                   f=penv.f.clone(), ep_step=penv.ep_step.clone(),
                   a_prev=penv.a_prev.clone())  # fmt: skip
    EPG["stop"].copy_(stop), EPG["fcap"].copy_(fcap), EPG["fail_f"].fill_(-1)
    EPG["f"].copy_(penv.f), EPG["ep_step"].copy_(penv.ep_step), EPG["a_prev"].copy_(penv.a_prev)
    penv.f, penv.ep_step, penv.a_prev = EPG["f"], EPG["ep_step"], EPG["a_prev"]
    EPG["hist"].copy_(penv.observe().unsqueeze(1).expand(-1, Hh, -1))
    penv._obs = None
    if "graph" not in EPG:
        for _ in range(3):  # warmup on the default stream (Warp's own stream orders with it implicitly)
            _eprobe_step()
        n_steps -= 3  # the warmup steps are sweep steps; the capture below records its step without running it
        torch.cuda.synchronize()
        t0 = time.time()
        EPG["graph"] = torch.cuda.CUDAGraph()
        # mjlab writes simulator arrays on Warp's stream and launches Warp's own step graph, neither of which may enter a
        # torch capture: write on the capture stream, launch the step's kernels (Warp's capture_while solver loop becomes a
        # conditional node of this graph)
        setitem, penv.sim.use_cuda_graph = TorchArray.__setitem__, False
        TorchArray.__setitem__ = lambda self, idx, value: self._tensor.__setitem__(idx, value)
        try:
            # own capture stream: cuBLAS workspaces are keyed by stream, and graphs captured on torch's shared default capture
            # stream would share one workspace with the training graphs that this graph runs concurrently with (--eprobe_async)
            with torch.cuda.graph(EPG["graph"], stream=torch.cuda.Stream(), capture_error_mode="thread_local"):
                ws = wp.stream_from_torch(torch.cuda.current_stream())
                with wp.ScopedStream(ws, sync_enter=False, sync_exit=False):
                    wp.capture_begin(stream=ws, external=True)
                    try:
                        _eprobe_step()
                    finally:
                        EPG["wp_graph"] = wp.capture_end(stream=ws)  # keep: it holds the captured kernels' modules
        finally:
            TorchArray.__setitem__, penv.sim.use_cuda_graph = setitem, True
        torch.cuda.synchronize()
        print(f"--eprobe_graph: captured one sweep step in {time.time() - t0:.1f} s", flush=True)
    return n_steps  # steps left to replay


@torch.no_grad()
def eprobe_burst(it):
    """--eprobe_share: one sweep of deployed-policy rollouts (evaluation conditions, EMA policy) over all clips; first
    failures update EP_FAIL; the focus reset distribution of the training simulator becomes the failure weights spread over
    the --eprobe_win bins before each failure."""
    m, P = motion, P_EP
    t0 = time.time()
    cid = m.clip_id[EP_TILE]
    lead = int(torch.randint(args.eprobe_lead_min, args.eprobe_lead_max + 1, (1,)))
    f0 = torch.maximum(EP_TILE - lead, m.clip_start[cid])
    # a probe is done at its tile end or at the clip's last frame (the full evaluation also checks the last transition);
    # finished probes keep stepping with the frame capped so that f + 1 stays in the clip
    stop = torch.minimum(EP_TILE + args.eprobe_tile, m.clip_end[cid])
    fcap = torch.minimum(stop, m.clip_end[cid] - 1)
    ids = torch.arange(P, device=dev)
    penv.done_eval[:] = False
    penv.failed_eval[:] = False
    penv.reset(ids, f0)
    if args.eprobe_async:
        avg = list(pi_avg.module.parameters()) + list(pi_avg.module.buffers())
        torch._foreach_copy_(list(EP_PI.parameters()) + list(EP_PI.buffers()), avg)
        EP_ASYNC.update(it=it, lead=lead, t0=t0, left=eprobe_graph_setup(lead + args.eprobe_tile, stop, fcap), first=True)
        return
    if args.eprobe_graph:
        fail_f = eprobe_sweep_graphed(lead + args.eprobe_tile, stop, fcap)
    else:
        fail_f = eprobe_sweep_eager(lead + args.eprobe_tile, stop, fcap)
    eprobe_finish(it, lead, t0, fail_f)


@torch.no_grad()
def eprobe_async_tick(launch):
    """--eprobe_async, per iteration: launch=False: if the running sweep's last steps were launched in an earlier iteration,
    wait for them and apply its failures; launch=True: launch the next steps of the running sweep on the side stream."""
    if not EP_ASYNC:
        return
    if not launch:
        if EP_ASYNC["left"] == 0:
            torch.cuda.current_stream().wait_event(EP_ASYNC["done"])
            eprobe_finish(EP_ASYNC["it"], EP_ASYNC["lead"], EP_ASYNC["t0"], EPG["fail_f"].clone())
            EP_ASYNC.clear()
        return
    if EP_ASYNC["left"] == 0:
        return
    if EP_ASYNC.pop("first", False):
        EP_SIDE.wait_stream(torch.cuda.current_stream())  # the reset, static inputs and policy snapshot
    k = min(args.eprobe_async, EP_ASYNC["left"])
    with torch.cuda.stream(EP_SIDE):
        for _ in range(k):
            EPG["graph"].replay()
    EP_ASYNC["left"] -= k
    if EP_ASYNC["left"] == 0:
        EP_ASYNC["done"] = torch.cuda.Event()
        EP_ASYNC["done"].record(EP_SIDE)


def eprobe_finish(it, lead, t0, fail_f):
    m, P = motion, P_EP
    EP_FAIL.mul_(args.eprobe_decay)
    fb = m.frame_bin[fail_f[fail_f >= 0]]
    EP_FAIL.index_add_(0, fb, torch.ones_like(fb, dtype=EP_FAIL.dtype))
    if EP_FAIL.sum() > 0:
        w, q = EP_FAIL.clone(), EP_FAIL.clone()
        for _ in range(args.eprobe_win):  # q[b] = sum of w over b .. b + win (same clip): the window before each failure
            w = w[EP_NEXT] * (EP_NEXT != torch.arange(m.num_bins, device=dev)).float()
            q += w
        env.set_focus(q / q.sum(), args.eprobe_share)
    nf = int((fail_f >= 0).sum())
    fl = sorted({(m.names[int(m.clip_id[f])].replace("_subject", "_s"), int(f - m.clip_start[m.clip_id[f]])) for f in fail_f[fail_f >= 0].tolist()})
    lag = f" | applied at it {ITER_NOW[0]}" if args.eprobe_async else ""
    print(f"--eprobe it {it}: lead {lead} | {nf}/{P} failures {fl[:8]} | {time.time() - t0:.1f} s{lag}", flush=True)


stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
probe = ValidationProbe(robot, kin, dev, layout.dim, vars(args)) if args.probe_minutes or args.probe_initial else None
parameter_groups = {"actor": actor_params, "world_model": wm.parameters(), "critic": critic.parameters()}
if VISION:
    parameter_groups.update(depth_encoder=pi.depth_enc.parameters(), latent_decoder=latent_dec.parameters())
parameter_updates = ParameterUpdates(parameter_groups) if probe is not None else None
if args.probe_initial:
    t_probe = time.monotonic()
    initial = probe.run(pi_avg.module if pi_avg is not None and int(pi_avg.n_averaged) else pi)
    initial.update(iteration=start_it, sim_steps=sim_steps, training_seconds=wall_offset,
                   evaluation_seconds=time.monotonic() - t_probe)
    initial["parameter_updates"] = parameter_updates.sample()
    json.dump(initial, open(os.path.join(args.log_dir, "probe_initial.json"), "w"), indent=2)
# The formal budget starts at the first real collection, including replay warmup.
t_start = time.monotonic() - wall_offset
training_started_epoch = time.time() - wall_offset
heartbeat = TrainingHeartbeat(args.log_dir) if args.profile else None
if heartbeat:
    heartbeat.mark("warmup", start_it, sim_steps)
collect(args.warmup_steps, stats)  # the replay buffer is not checkpointed, so a resumed run refills it too
# Keep the warm-up reward terms until the first log: stats["r"] includes
# these same transitions. Clearing only the terms would mix two windows.
log_path = os.path.join(args.log_dir, "train_log.csv")
new_log = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
log_f = open(log_path, "a")
if new_log:
    log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr"
                ",pi_lr,pi_mult,nonfinite" + (",latent_loss,depth_recon" if VISION else "")
                + ("") + ",wm_updates,actor_updates\n")  # fmt: skip
    log_f.flush()
with open(log_path) as stream:
    log_columns = stream.readline().strip().split(",")
LOG_EXTRA = "pi_lr" in log_columns  # lr = base LR; pi_lr = policy LR (x pi_mult)
BUDGET_LOG = "wm_updates" in log_columns
# --ckpt_light: one background writer; a failed write is re-raised in the training loop (or at the end), at most 2 queued
CKPT_POOL, CKPT_PENDING = (ThreadPoolExecutor(max_workers=1) if args.ckpt_light else None), []


def ckpt_wait(drain=False):
    while CKPT_PENDING and (drain or CKPT_PENDING[0].done() or len(CKPT_PENDING) > 2):
        CKPT_PENDING.pop(0).result()


def _save_atomic(obj, path):
    write_started = time.monotonic()
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)
    with open(os.path.join(args.log_dir, "checkpoint_timings.jsonl"), "a") as stream:
        stream.write(json.dumps({"iteration": obj["iteration"], "kind": "background_write", "path": path,
                                 "duration_s": time.monotonic() - write_started}) + "\n")


def _retain_full_checkpoint(last, path):
    """Keep the published full state when a later save replaces last.pt."""
    temporary = path + ".tmp"
    if os.path.exists(temporary):
        os.remove(temporary)
    os.link(last, temporary)
    os.replace(temporary, path)


STOP_REQUESTED = False


def request_checkpoint_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGUSR1, request_checkpoint_stop)
if args.max_hours is not None:
    deadline = training_started_epoch + args.max_hours * 3600
    json.dump({"started_at": training_started_epoch, "deadline": deadline, "max_hours": args.max_hours,
               "start_iteration": start_it}, open(os.path.join(args.log_dir, "run_timing.json"), "w"), indent=2)
    print(f"training deadline: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline))}", flush=True)
next_ckpt_s = wall_offset + args.ckpt_minutes * 60
t_log = time.time()
profile_steps = sim_steps
profile_path = os.path.join(args.log_dir, "diagnostics.csv")
if args.profile:
    expected_header = ("iteration,env_steps_per_s,collect_ms,wm_ms,policy_ms,wm_gradnorm,critic_gradnorm,peak_allocated_gb,device_used_gb" +
                       ("," + ",".join(PROFILE_COLUMNS) if OBJECT else "") + ",logging_ms,memory_query_ms\n")
    if os.path.exists(profile_path):
        with open(profile_path) as stream:
            if stream.readline() != expected_header:
                profile_path = os.path.join(args.log_dir, "diagnostics_gait_v1.csv")
    profile_log = open(profile_path, "a")
    profile_timing_columns = True
    if profile_log.tell() == 0:
        profile_log.write(expected_header)
    else:
        with open(profile_path) as stream:
            profile_timing_columns = "logging_ms" in stream.readline().strip().split(",")
    profile_window_start = time.monotonic()
    memory_sample_time, device_used_gb = -float("inf"), 0.
step_acc = 0.0
budget = TransitionBudget(args.wm_update_interval, args.policy_warmup_steps,
                          (RESUME_CTRL or {}).get("wm_updates", 0))
actor_updates = (RESUME_CTRL or {}).get("actor_updates", 0)
next_probe_s = wall_offset + args.probe_minutes * 60
last_wm_loss, last_wm_error = torch.zeros((), device=dev), torch.zeros((), device=dev)
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
# first iteration with a policy update (after --wm_only_iters); a resumed run is past it
POLICY_START = args.wm_only_iters if (start_it > 0 or args.wm_only_iters == 0) else None
DECAY_AT = [None]  # --lr_decay_ret: iteration of the trigger
if RESUME_CTRL is not None:
    DECAY_AT[0] = RESUME_CTRL["decay_at"]
    if EPROBE and RESUME_CTRL.get("ep_fail") is not None:
        EP_FAIL.copy_(RESUME_CTRL["ep_fail"].to(dev))
    if RESUME_CTRL.get("focus_q") is not None:
        env.set_focus(RESUME_CTRL["focus_q"].to(dev), RESUME_CTRL["focus_share"])
    print(f"--resume: LR trigger at {DECAY_AT[0]}, focus {'set' if RESUME_CTRL.get('focus_q') is not None else 'none'}", flush=True)


for it in range(start_it, args.iterations):
    actor_ready = budget.ready(sim_steps, it, args.wm_only_iters)
    lr = lr_at(it)
    pi_mult = args.pi_lr_mult  # --lr_decay_ret: ramps to 1 after the trigger
    if DECAY_AT[0] is not None:
        pi_mult = args.pi_lr_mult + (1.0 - args.pi_lr_mult) * min(1.0, (it - DECAY_AT[0]) / max(args.lr_decay_len, 1))
    for o in (opt_wm, opt_pi, opt_v):
        for gparam in o.param_groups:
            if torch.is_tensor(gparam["lr"]):
                gparam["lr"].fill_(lr * (pi_mult if o is opt_pi else 1.0))
            else:
                gparam["lr"] = lr * (pi_mult if o is opt_pi else 1.0)
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    if args.eprobe_async:
        ITER_NOW[0] = it
        eprobe_async_tick(launch=False)
        assert not (EP_ASYNC and it >= args.eprobe_start and it % args.eprobe_every == 0), "--eprobe_async too small"
    if EPROBE and it >= args.eprobe_start and it % args.eprobe_every == 0:
        eprobe_burst(it)
    if args.eprobe_async:
        eprobe_async_tick(launch=True)
    if args.profile:
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
    if heartbeat:
        heartbeat.mark("collect", it + 1, sim_steps)
    collect(n, stats)
    if args.profile:
        events[1].record()
    if heartbeat:
        heartbeat.mark("world_model", it + 1, sim_steps)
    for _ in range(budget.due(sim_steps, args.wm_batches)):
        last_wm_loss, last_wm_error = wm_update_g()
        budget.updates += 1
    agg["wm"].append(last_wm_loss), agg["wm8"].append(last_wm_error)
    if args.profile:
        events[2].record()
    if POLICY_START is None and actor_ready:
        POLICY_START = it
        print(f"policy updates start at iteration {POLICY_START}", flush=True)
        open(os.path.join(args.log_dir, "policy_start.txt"), "w").write(f"{POLICY_START}\n")
    if actor_ready:
        if heartbeat:
            heartbeat.mark("policy_critic", it + 1, sim_steps)
        n_avg0 = int(pi_avg.n_averaged) if pi_avg is not None else 0
        pl, vl, ir, gn, surv = policy_update_g()
        actor_updates += 1
        if GRAPH and pi_avg is not None and int(pi_avg.n_averaged) == n_avg0:
            pi_avg.n_averaged += 1  # ema_update (and a graph replay) does not advance the host-side counter
        agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
    if args.profile:
        events[3].record()
        profile_times.append(events)
    if heartbeat:
        heartbeat.mark("logging", it + 1, sim_steps)
    elapsed = time.monotonic() - t_start
    probe_due = (probe is not None and args.probe_minutes > 0 and elapsed >= next_probe_s
                 and (args.max_hours is None or elapsed < args.max_hours * 3600))
    if probe_due:
        if heartbeat:
            heartbeat.mark("validation", it + 1, sim_steps)
        t_probe = time.monotonic()
        result = probe.run(pi_avg.module if pi_avg is not None and int(pi_avg.n_averaged) else pi)
        if buffer.size >= HZ:
            result["world_model"] = world_model_probe(wm, pi.depth_enc if VISION else None,
                                                     buffer, normalize_xi, dev, kin=kin, horizon=HZ)
        if OBJECT and args.carry_gait_profile != "legacy" and actor_updates:
            result["reward_gradients"] = reward_gradient_probe()
        result["parameter_updates"] = parameter_updates.sample()
        result.update(iteration=it + 1, sim_steps=sim_steps, training_seconds=time.monotonic() - t_start,
                      evaluation_seconds=time.monotonic() - t_probe)
        json.dump(result, open(os.path.join(args.log_dir, f"probe_{it + 1:06d}.json"), "w"), indent=2)
        print(f"[probe] it {it + 1} success={result['success_rate']:.3f} task={result['task_norm_return']:.3f} "
              f"drops={result['failure_causes']['box_drop']} wall={result['evaluation_seconds']:.1f}s", flush=True)
        next_probe_s = (int((time.monotonic() - t_start) // (args.probe_minutes * 60)) + 1) * args.probe_minutes * 60
    if (it + 1) % args.log_every == 0:
        logging_started = time.monotonic()
        memory_query_ms = 0.
        ips = args.log_every / (time.time() - t_log)
        if args.profile:
            torch.cuda.synchronize()
            timings = [sum(es[j].elapsed_time(es[j + 1]) for es in profile_times) / len(profile_times)
                       for j in ((0, 1, 2))]
            diagnostic = [it + 1, 0., *timings, *HEALTH_STATS.tolist(),
                          torch.cuda.max_memory_allocated() / 2**30]
            # cudaMemGetInfo can wait on allocations by other GPU contexts.
            # Sample at a wall-clock cadence, independent of algorithm iteration length.
            if time.monotonic() - memory_sample_time >= 30:
                query_started = time.monotonic()
                free_memory, total_memory = torch.cuda.mem_get_info()
                memory_query_ms = (time.monotonic() - query_started) * 1000
                device_used_gb = (total_memory - free_memory) / 2**30
                memory_sample_time = time.monotonic()
            diagnostic.append(device_used_gb)
            if OBJECT:
                diagnostic += torch.stack(profile_terms).mean(0).tolist()
            profile_terms.clear()
            profile_times.clear()
        t_log = time.time()
        wall_h = (time.monotonic() - t_start) / 3600
        summaries = training_summaries(agg, stats, dev)
        term_frac = summaries[10] / max(summaries[11], 1)
        row = [it + 1, wall_h, sim_steps, *summaries[:10], term_frac, ips, lr]
        if (int(NONFINITE) or not all(math.isfinite(row[i]) for i in (3, 4, 5, 6, 7, 8, 9, 10))
                or (args.profile and not bool(torch.isfinite(HEALTH_STATS).all()))):
            raise FloatingPointError("Non-finite training loss/gradient; stop this attempt before further updates")
        extra = [lr * pi_mult, float(pi_mult), int(NONFINITE)] if LOG_EXTRA else []
        if VISION:
            extra += VISION_STATS.tolist()
        if BUDGET_LOG:
            extra += [budget.updates, actor_updates]
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row + extra) + "\n")
        deadline = args.lr_decay_by > 0 and it + 1 >= args.lr_decay_by
        if args.lr_decay_ret > 0 and DECAY_AT[0] is None and (row[12] >= args.lr_decay_ret or deadline):
            DECAY_AT[0] = it + 1
            why = f"training return {row[12]:.3f} >= {args.lr_decay_ret}" if row[12] >= args.lr_decay_ret else f"deadline --lr_decay_by {args.lr_decay_by}"
            print(f"--lr_decay_ret: {why} at it {it + 1}: policy LR x{args.pi_lr_mult} -> x1 over {args.lr_decay_len} its", flush=True)
        log_f.flush()
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
              f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s", flush=True)  # fmt: skip
        if args.profile:
            profile_now = time.monotonic()
            diagnostic[1] = (sim_steps - profile_steps) / (profile_now - profile_window_start)
            timing_values = [(profile_now - logging_started) * 1000, memory_query_ms]
            if profile_timing_columns:
                diagnostic += timing_values
            else:  # keep a pre-existing CSV schema intact when resuming an older run
                with open(os.path.join(args.log_dir, "logging_timings.jsonl"), "a") as stream:
                    stream.write(json.dumps({"iteration": it + 1, "logging_ms": timing_values[0],
                                             "memory_query_ms": timing_values[1]}) + "\n")
            profile_window_start, profile_steps = profile_now, sim_steps
            profile_log.write(",".join(str(x) for x in diagnostic) + "\n")
            profile_log.flush()
        stats = {k: [] for k in stats}
        agg = {k: [] for k in agg}
    wall = time.monotonic() - t_start
    out_of_time = args.max_hours is not None and wall >= args.max_hours * 3600
    timed = (args.ckpt_minutes > 0 and wall >= next_ckpt_s) or probe_due
    if timed:
        next_ckpt_s += args.ckpt_minutes * 60
    final_ck = it + 1 == args.iterations or out_of_time or STOP_REQUESTED
    if heartbeat:
        heartbeat.mark("checkpoint", it + 1, sim_steps)
    checkpoint_started = time.monotonic()
    checkpoint_kind = None
    if args.ckpt_light and not final_ck and not timed and (it + 1) % args.ckpt_every == 0:
        sd = pi_avg.module.state_dict() if pi_avg is not None else pi.state_dict()
        small = {"pi": {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}, "iteration": it + 1,
                 "wall_time": time.monotonic() - t_start, "sim_steps": sim_steps, "args": vars(args),
                 "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        ckpt_wait()
        CKPT_PENDING.append(CKPT_POOL.submit(_save_atomic, small, path))
        checkpoint_kind = "light_submit"
    elif (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations or timed or out_of_time or STOP_REQUESTED:
        ck = {"wm": wm.state_dict(), "pi": pi_avg.module.state_dict() if pi_avg is not None else pi.state_dict(), "pi_live": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
              "wall_time": time.monotonic() - t_start, "sim_steps": sim_steps, "args": vars(args),
              "pi_avg": pi_avg.state_dict() if pi_avg is not None else None,
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        ck["ctrl"] = {"decay_at": DECAY_AT[0], "wm_updates": budget.updates, "actor_updates": actor_updates,
                      "ep_fail": EP_FAIL.cpu() if EPROBE else None,
                      "focus_q": env.focus_q.cpu() if env.focus_q is not None else None,
                      "focus_share": float(env.focus_share_t) if env.focus_static else env.focus_share,
                      "env_bins": {"bin_G": env.bin_G.cpu(), "bin_fail": env.bin_fail.cpu(), "bin_w_avg": env.bin_w_avg.cpu()}}  # fmt: skip
        if VISION:
            ck["latent_dec"] = latent_dec.state_dict()
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        _retain_full_checkpoint(last, os.path.join(args.log_dir, f"full_{it + 1:06d}.pt"))
        small = {k: ck[k] for k in ("pi", "pi_live", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)
        checkpoint_kind = "full"
    if checkpoint_kind is not None:
        with open(os.path.join(args.log_dir, "checkpoint_timings.jsonl"), "a") as stream:
            stream.write(json.dumps({"iteration": it + 1, "kind": checkpoint_kind,
                                     "duration_s": time.monotonic() - checkpoint_started,
                                     "training_seconds": time.monotonic() - t_start}) + "\n")
    if out_of_time:
        print(f"time budget of {args.max_hours} h reached at it {it + 1}", flush=True)
        break
    if STOP_REQUESTED:
        print(f"requested checkpoint stop at it {it + 1}", flush=True)
        break
if CKPT_POOL is not None:  # every background checkpoint written (a failed write raises here: non-zero exit)
    ckpt_wait(drain=True)
    CKPT_POOL.shutdown()
if heartbeat:
    heartbeat.mark("finished", it + 1, sim_steps)
    heartbeat.close()
