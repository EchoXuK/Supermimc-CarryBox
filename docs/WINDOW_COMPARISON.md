# 搬箱参考窗口与纯 BPTT 训练

当前训练路径为：真实采集 → 环形回放 → 世界模型更新 → BPTT Actor 和想象轨迹 Critic 更新。
Actor 保留软终止和末端价值；Critic 使用想象轨迹的 λ-return。

## 时间与数据

搬箱默认十个参考偏移为 `1,5,10,15,20,25,30,35,40,45`，控制周期 0.02 秒，最远参考 0.9 秒。
首个偏移仍为下一控制帧，PD 跳连和普通 Actor 的残差动作单位保持一致；机器人和箱体采用相同采样。
默认 BPTT 32 步（0.64 秒）。现存六小时模型使用 **48 步**，不能把它解释为 32 步模型。
远期参考在各自片段末帧截停，想象轨迹在片段末端停止累计奖励与 bootstrap，不跨片段。

`--goal_frame_offsets` 写入检查点、评估身份和 ONNX；旧模型缺少该字段时按 `1..goal_frames` 读取。
续训校验 horizon 和采样配置；批量评估按各检查点配置分组。新评估协议为 v6，独立于历史缓存。
当前裁剪数据为 748 条训练、146 条验证、106 条测试，共 297400 帧。
当前数据及训练来源见 `logs/lift_reach_6h_20261112_145613/dataset_fingerprint.json`。

## 预算、奖励与恢复

`--wm_update_interval 4096` 表示每累计 4096 条真实转移执行一个 WM minibatch。
`--policy_warmup_steps 4194304` 以真实转移数指定 Actor 预热；缺省时保持 `wm_batches` / `wm_only_iters` 行为。
完整检查点保存预算、迭代、环境步数、计时、EMA、归一化和优化器状态。恢复后重新采集环形回放。
`--init_from` 只复制兼容的普通 Actor、WM、Critic 和视觉网络，使用全新优化器与训练时钟。
`--resume` 校验任务、奖励、目标窗口和预算配置；缺少算法元数据以及旧 `algorithm=bptt` 均可读取。

`--real_contact_reward` 让真实与想象轨迹使用相同的训练脚部权重，评估单独报告固定原任务分数。
抓放、抬箱前伸和参考相对步态配置见 [奖励说明](CARRY_GAIT_REWARD.md)。额外动作平滑默认关闭。
六小时训练使用 `grasp_place_reach`、`reference_phase_v3`、Adam Actor、Muon WM、恒定 `4e-4` 学习率；
这些实验覆盖参数没有改动默认配方的 Actor 优化器。

## 训练与验证

训练、基准和 GPU 验证均通过本机 trainreg 登记。以下是交给登记入口的训练命令：

```bash
PRESET_NAME=box_carry bash scripts/train.sh logs/carry_run \
  --motion_dir box_carry_three_stage_v2/augmented \
  --horizon 48 --min_imagination_steps 32 --min_reset_steps 32 \
  --goal_frame_offsets 1,5,10,15,20,25,30,35,40,45 \
  --pi_opt adam --wm_opt muon --lr 4e-4 --constant_lr \
  --carry_reward_profile grasp_place_reach --carry_gait_profile reference_phase_v3 \
  --contact_reward 0.25 --real_contact_reward \
  --wm_update_interval 4096 --policy_warmup_steps 4194304 \
  --profile --probe_minutes 20 --ckpt_every 200 --ckpt_minutes 20
```

CPU 回归：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
.venv/bin/python -m unittest tools.test_carry_rewards tools.test_checkpoint_evaluation \
 tools.test_training_windows tools.test_box_carry tools.test_carry_gait \
 tools.test_carry_smooth tools.test_action_boundary tools.test_reset_sampling \
 tools.test_trim_box_carry_init tools.test_clean_box_carry_terminal_init
```

GPU 检查通过 `--gpu` 启用各测试中的真实模拟与编译用例。原有 `tools/test_env.py` 和 `test_reset.py`
是独立 GPU 脚本，不能作为 CPU unittest 模块收集。

## 监控与现存结果

`train_log.csv` 记录策略梯度、非有限计数和 WM/Actor 更新数；`diagnostics.csv` 记录各阶段耗时、
WM/Critic 梯度、逐项奖励和动作饱和。`progress.json` 每五秒报告阶段。
独立验证探针冻结归一化，恢复随机状态和网络模式，不写入训练回放；另报告 4/8/16/32 步 WM 误差。
`checkpoint_timings.jsonl` 分别记录保存提交和写入耗时；完整保存保留硬链接快照。

现存训练与评估位于 `logs/lift_reach_6h_20261112_145613`。历史模型的训练成绩与原视频保留，
历史评估协议继续保留原版本；清理后的新验证单独位于 `logs/pure_bptt_cleanup_20261114`。
原训练 `source_v1` 及清单已删除；新的纯版源码快照只描述清理后验证，不表示原训练源码。
