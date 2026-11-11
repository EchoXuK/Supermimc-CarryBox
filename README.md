# Supermimc-CarryBox

基于 [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) 的 Unitree G1 三阶段搬箱框架，使用世界模型中的显式 BPTT 和可微 AMP 风格奖励。

- 控制周期 0.02 s；连续 45 帧未来参考（0.90 s），32 步 BPTT（0.64 s）。
- 保留接近举起、搬运移动、放下起身、箱体目标和深度 latent。
- actor 直接输出关节目标；运动风格由阶段条件判别器学习。
- 专家仅使用 748 条 train，验证集 146 条、测试集 106 条独立划分。

## 最优权重与结果

[最优策略](checkpoints/amp_seed0_2h/best_policy.pt) · [模型说明](checkpoints/amp_seed0_2h/README.md) · [完整审查报告](docs/AMP_TRAINING_REPORT.md) · [学习曲线](results/amp_seed0_2h/learning_curves.png)

单卡、seed 0 从零训练 7200.13 秒，6144 环境，20,869 次更新。最优策略按全部验证片段的共同任务回报选出第 17,465 轮。

| 指标 | 最优策略 / val | 最优策略 / test |
|---|---:|---:|
| 共同任务回报 | 0.6736 | 0.6497 |
| 独立片段平均完成比例 | 97.65% | 98.17% |
| 未提前失败片段 | 132/146 | 97/106 |
| 箱体平均误差 | 0.0861 m | 0.0975 m |

这些是独立阶段片段指标。长接近验证片段仅 3/12 条未提前失败，尚未稳定收敛。共同任务回报低于原 tracking 的 0.6920，实际动作加速度和放置后起身表现有所改善。

## 安装

已验证环境为 Linux、Python 3.12、PyTorch 2.10.0+cu128、mjlab 1.6.0；具体版本在 [requirements-tested.txt](requirements-tested.txt)。CUDA 仿真需要 NVIDIA GPU。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-tested.txt
.venv/bin/python -m pip install -e . --no-deps
```

G1 搬箱 MJCF 与网格位于 [data/g1_carry](data/g1_carry)。准备与 [清单](box_carry_three_stage_v2/augmented/manifest.csv) 对应的 NPZ 数据；格式说明见 [数据入口](box_carry_three_stage_v2/augmented/README.md)。也可通过参数指向现有 augmented 目录。

## 评估

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m blinkmimic.evaluate \
  --run_dir checkpoints/amp_seed0_2h \
  --ckpt checkpoints/amp_seed0_2h/best_policy.pt \
  --motion_dir /path/to/box_carry_three_stage_v2/augmented --split test
```

添加 --max_clips 3 可各阶段取一条进行冒烟验证。在要求登记的服务器上，请将命令放在本机 trainreg run -- env 后执行。

## 从零训练

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh logs/carry_amp_new \
  --motion_dir /path/to/box_carry_three_stage_v2/augmented \
  --seed 0 --max_hours 2 --iterations 10000000
```

默认 512 环境；此次测得最快的候选配置为 6144 环境，使用 --num_envs 6144 --cuda_graph --graph_early，稳态设备占用约 36.48 GiB，吞吐中位数约 46,000 控制 step/s。8192 环境在该卡上 OOM。批量选择需按可用显存调整。

--imitation_mode amp 为搬箱默认模式；--amp bf16 仍表示混合精度。原 tracking 配方通过 PRESET_NAME=box_carry_tracking 选择。

## 核心入口

| 文件 | 作用 |
|---|---|
| [amp.py](src/blinkmimic/amp.py) | 运动判别器、归一化、专家采样和可微奖励 |
| [box_carry.py](src/blinkmimic/box_carry.py) | 三阶段数据、场景、奖励、终止与目标反馈 |
| [train.py](src/blinkmimic/train.py) | WM、判别器、32 步 actor BPTT 与 critic 更新 |
| [evaluate.py](src/blinkmimic/evaluate.py) | 按保存参数评估与逐项奖励/梯度审查 |
| [tools](tools) | CPU/GPU 检查、吞吐测量、ONNX 导出及监测 |

AMP 机制参考 [OpenLegged/supermimic_AMP 固定版本 596dfe3](https://github.com/OpenLegged/supermimic_AMP/tree/596dfe37e0f75769b635386cd589960cce7faaa2)。上游代码与资产归属见 [LICENSE](LICENSE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
