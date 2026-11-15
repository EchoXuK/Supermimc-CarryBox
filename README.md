# Supermimc-CarryBox · teacher

基于 [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) 的 **纯 BPTT 三阶段搬箱 teacher**，包含当前核心代码、G1 搬箱资产和第 **25845** 轮最终权重。该模型同时为六小时终点和验证选出的最佳模型。

[推理权重](checkpoints/teacher_025845/policy.pt) · [完整训练状态](checkpoints/teacher_025845/full_training.pt) · [模型卡](checkpoints/teacher_025845/MODEL_CARD.json) · [发布与验证说明](docs/TEACHER_RELEASE.md)

## 模型与数据

- 普通 Policy + 世界模型 + BPTT Actor + 想象轨迹 λ-return Critic，使用深度 latent 和下一帧参考 PD 跳连。
- 控制周期 0.02 秒；历史模型使用 **48 步 BPTT（0.96 秒）**、最远 0.9 秒未来参考。十个参考偏移：`1,5,10,15,20,25,30,35,40,45`。
- 最新裁剪数据：748 train / 146 val / 106 test，共 297400 帧；NPZ 已通过 Git LFS 随分支提供，[清单与格式](box_carry_three_stage_v2/augmented/README.md) 已提供。
- 奖励为 `grasp_place_reach` 与 `reference_phase_v3`，额外动作平滑关闭。相位和箱体目标仍由专家参考提供，此模型需要参考轨迹条件。

历史验证集三阶段平均成功率 **91.53%**，测试集 **94.50%**；测试原任务回报 **0.8427**。这些是独立片段结果，步态抖动和同脚重复点地仍可能出现。分阶段数值见 [验证](results/teacher_025845/validation_historical.json) 和 [测试](results/teacher_025845/test_historical.json)。

## 安装与权重校验

```bash
# 先安装系统 git-lfs；克隆后在仓库根目录执行
git lfs install
git lfs pull
(cd box_carry_three_stage_v2/augmented && sha256sum -c SHA256SUMS)

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-tested.txt
.venv/bin/python -m pip install -e . --no-deps
(cd checkpoints/teacher_025845 && sha256sum -c SHA256SUMS)
```

核心依赖范围在 pyproject.toml，上述命令固定本机验证版本（含可选 ONNX 依赖），版本见 [requirements-tested.txt](requirements-tested.txt)。需 NVIDIA GPU 运行 MuJoCo-Warp 仿真；G1 XML 和网格已在 `data/g1_carry`，无需依赖其他本机项目复制资产。

## 评估与导出

以下 GPU 命令在需要 trainreg 的服务器上应交给本机登记入口执行。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -m blinkmimic.evaluate \
  --run_dir checkpoints/teacher_025845 \
  --ckpt checkpoints/teacher_025845/policy.pt \
  --motion_dir box_carry_three_stage_v2/augmented --split test --repeats 3 \
  --output results/teacher_025845/new_test_eval.json

PYTHONPATH=src .venv/bin/python tools/export_onnx.py \
  --ckpt checkpoints/teacher_025845/policy.pt \
  --motion "$(find box_carry_three_stage_v2/augmented -name '*.npz' -print -quit)" --out teacher.onnx
```

`--max_clips 3` 可用于三阶段各一条的冒烟评估。评估与 ONNX 均从权重读取参考偏移，避免将历史模型误按默认采样读取。

## 训练与验证

从零训练相同算法/奖励配置的示例（环境数需按本机测量）：

```bash
CUDA_VISIBLE_DEVICES=0 PRESET_NAME=box_carry bash scripts/train.sh logs/teacher_new \
  --motion_dir box_carry_three_stage_v2/augmented --seed 0 \
  --horizon 48 --min_imagination_steps 32 --min_reset_steps 32 \
  --pi_opt adam --wm_opt muon --lr 4e-4 --constant_lr \
  --carry_reward_profile grasp_place_reach --carry_gait_profile reference_phase_v3 \
  --real_contact_reward --wm_update_interval 4096 --policy_warmup_steps 4194304 \
  --max_hours 6 --iterations 10000000
```

完整训练状态可用 `--resume` 恢复，需要一致的奖励、采样、预算和网络配置；参数见 `checkpoints/teacher_025845/training_args.json`。延长训练时调整累计 `--max_hours` 上限与 `--iterations`。`--init_from` 保留普通网络初始化并创建新优化器。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
.venv/bin/python -m unittest tools.test_action_boundary tools.test_box_carry \
  tools.test_carry_smooth tools.test_carry_gait tools.test_training_windows \
  tools.test_reset_sampling tools.test_trim_box_carry_init tools.test_clean_box_carry_terminal_init \
  tools.test_carry_rewards tools.test_checkpoint_evaluation
```

数据与 GPU 用例需要实际 NPZ/GPU；本机发布前共 **100 项通过（95 CPU、5 GPU）**，另完成编译 BPTT 3 次更新、完整保存和恢复后 2 次更新。CPU 部署动作/奖励与清理前完全一致，ONNX 最大动作误差 6.08e-6。证明摘要见 [release_validation.json](results/teacher_025845/release_validation.json)。

上游代码和资产许可见 [LICENSE](LICENSE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [G1 资产许可](data/g1_carry/LICENSE)。
