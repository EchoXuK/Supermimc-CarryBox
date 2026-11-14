# Teacher 025845

纯 BlinkMimic/BPTT 搬箱六小时训练的验证选中模型，同时为六小时终点，seed=0。

- `policy.pt`：EMA 推理 Actor、live Actor、WM 与参数，推理时使用 `pi`。
- `full_training.pt`：完整训练状态，含 Critic、视觉、EMA、优化器与计数器；使用 `--resume` 恢复，回放重新采集。
- `training_args.json`：真实训练参数；含历史本机路径，换机器时显式覆盖数据/输出路径。
- `SHA256SUMS`：两份权重的 SHA-256，文件与本机已验证最终权重完全相同。

48 步 BPTT（0.96 秒），未来参考最远 0.9 秒，观测历史 10 帧；参考偏移为 1,5,10,15,20,25,30,35,40,45。普通 Policy 保留下一帧参考 PD 跳连、深度 latent 和动作边界 [-20,20]。

模型及验证/测试数字见 MODEL_CARD.json；结果是三阶段独立片段评估，不等于连续完成搬箱任务。原训练源码快照已删除，此分支提供清理后、动作/奖励数值等价的纯版源码。

校验权重文件：

```bash
(cd checkpoints/teacher_025845 && sha256sum -c SHA256SUMS)
```
