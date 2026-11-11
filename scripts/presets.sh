# Training presets (flags of `python -m blinkmimic.train`). The tracking presets preserve the original LAFAN1 recipes; every
# change is a flag, so all methods share simulator, MDP and evaluator.
declare -A PRESET=(
  # SuperMimic (paper hyperparameters, mjlab reimplementation)
  [supermimic]="--task tracking --vision none"
  # InstantTrack (main configuration of InstantTrack v1.0, same code base)
  [instanttrack]="--task tracking --vision none --pi_ema 0.995 --no_sync --fused_adam --fused_rot --compile_more --lr 4e-4 --wm_batches 5 --wm_hidden 1024,1024,1024 --fail_reset 0.3 --contact_reward 0.25"
  # this release: InstantTrack + structured world model, mirror augmentation, Muon on policy and world model,
  # reference-action skip, evaluation-guided reset allocation (--eprobe_*), policy batch 2048 with a 4x policy learning
  # rate decayed to 1x over 300 iterations (once the training return reaches 0.75, at iteration 1400 at the latest),
  # CUDA graphs for the updates, the collection step and the probe sweep; 3500 iterations, a policy checkpoint every 100
  [main]="--task tracking --vision none --pi_ema 0.995 --no_sync --fused_adam --fused_rot --compile_more --lr 4e-4 --wm_batches 5 --wm_hidden 1024,1024,1024 --fail_reset 0.3 --contact_reward 0.25 --wm_structured --wm_mirror 0.5 --pi_opt muon --wm_opt muon --pi_ref_skip --wm_only_iters 0 --bin_fix --ref_contact_fix --ckpt_minutes 0 --cuda_graph --eprobe_share 0.15 --graph_early --collect_graph --reset_masked --pi_batch 2048 --pi_lr_mult 4 --lr_decay_ret 0.75 --lr_decay_len 300 --iterations 3500 --ckpt_every 100 --ckpt_light --eprobe_graph --eprobe_async 20 --eprobe_start 1000 --eprobe_every 125 --eprobe_decay 0.7 --lr_decay_by 1400"
  # Augmented box carrying: real depth -> shared encoder -> autoregressive WM latent -> actor.
  # Collection stays eager because clip resets update per-world geometry and inertia.
  [box_carry]="--task box_carry --imitation_mode amp --vision latent --goal_frames 45 --horizon 32 --carry_action_acc 0.1 --carry_action_acc_arms 0 --carry_palm_far_weight 0.25 --carry_palm_far_std 0.5 --carry_palm_far_kernel inverse_quadratic --carry_goal_anchor robot --carry_approach_progress_weight 1 --amp_reward_weight 5 --amp_disc_lr 1e-5 --amp_disc_batch 256 --amp_grad_penalty 10 --amp_foot_slip_weight 0.25 --num_envs 512 --buffer_steps 256 --warmup_steps 32 --wm_batch 256 --pi_batch 256 --wm_window 4 --wm_batches 5 --wm_hidden 1024,1024,1024 --enc_hidden 512,256,128 --dec_hidden 512,256,128 --wm_structured --wm_mirror 0.5 --pi_opt muon --wm_opt muon --pi_ema 0.995 --wm_only_iters 200 --bin_fix --fail_reset 0.2 --no_sync --fused_adam --fused_rot --compile_more --lr 4e-4 --iterations 20000 --ckpt_every 500 --ckpt_minutes 0 --log_every 25"
  # Original carry recipe, for evaluating/continuing the retained 10-frame tracking model.
  [box_carry_tracking]="--task box_carry --imitation_mode tracking --vision latent --goal_frames 10 --horizon 16 --carry_residual_acc 0.1 --carry_residual_acc_arms 0 --num_envs 512 --buffer_steps 256 --warmup_steps 32 --wm_batch 256 --pi_batch 256 --wm_window 4 --wm_batches 5 --wm_hidden 1024,1024,1024 --enc_hidden 512,256,128 --dec_hidden 512,256,128 --wm_structured --wm_mirror 0.5 --pi_ref_skip --pi_opt muon --wm_opt muon --pi_ema 0.995 --wm_only_iters 200 --bin_fix --ref_contact_fix --contact_reward 0.25 --fail_reset 0.2 --no_sync --fused_adam --fused_rot --compile_more --lr 4e-4 --iterations 20000 --ckpt_every 500 --ckpt_minutes 0 --log_every 25"
)
