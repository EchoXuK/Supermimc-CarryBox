# 纯 BlinkMimic 清理记录（2026-11-14）

按用户要求直接删除 FPO++ 实现和相关历史源码，不另行归档。当前唯一训练路径为真实采集 → 环形回放 → WM → BPTT Actor → 想象轨迹 λ-return Critic。

## 删除与保留

删除 Flow Actor、时间编码、流积分、CFM、逐样本比值、ASPO、混合梯度诊断、独立真实 rollout/GAE、延迟归一化和算法转换；删除算法 CLI、专属配方、实现文件、专属测试及文档、第三方专属声明和旧源码快照。

保留搬箱物理环境、最新裁剪数据、视觉重建、抓放和步态奖励、动作边界、重置采样、参考窗口、转移预算、EMA、监控、评估与 ONNX。BPTT 的采集顺序、探索噪声、归一化更新时机、软终止、末端价值和 λ-return 未改变。

六小时实验仍为 `horizon=48`、最远参考 0.9 秒；默认搬箱配方仍为 32 步，默认优化器及其他参数保持原值。所有纯模型、优化器状态、实际训练成绩、奖励分析和三段视频保留。

## 接口与测试迁移

`make_policy` 统一构造普通 Policy。检查点缺少算法字段或旧字段为 bptt 均可加载，其他值在加载网络权重、写入运行元数据或缓存前拒绝。`--init_from` 保留普通网络初始化，`--resume` 保留完整状态恢复；不再支持架构转换。新评估协议为 v6，历史评估保持原版本并标为历史结果。

原专属测试中的抓放奖励、视觉冻结、WM→Actor 梯度、外力扰动、真实动作和编译奖励迁入 `tools/test_carry_rewards.py`；续训元数据保护、奖励隔离和缓存测试迁入 `tools/test_checkpoint_evaluation.py`。窗口、短片段、训练预算和 GPU 验证探针已改为普通 Actor。

## 历史来源与元数据

`logs/lift_reach_6h_20261112_145613/source_v1` 及其清单已经删除，不能继续声称可用原训练源码复现实验。新的 `source_pure_20261114` 是清理后验证快照，不是原训练源码；历史命令和登记记录保留真实原路径，后续入口另存 `reproduce_command.json`。

对纯检查点按物理 inode 去重，移除冗余算法参数及参考提交字段。临时序列化后，逐张量按字节比对（包括 NaN 位模式和有符号零），再原子替换；所有非配置字段、预算、迭代、计时、模型、归一化、EMA 和优化器状态均保留，原硬链接组和软链接保持可用。

迁移的初版字典复制漏保留 state_dict 的 `_metadata` 属性，在正式完成前已纠正：保留属性复制并验证；已处理条目用同实验未迁移完整检查点的同构模块版本信息补齐，且重新核对原张量指纹。修复过程和逐文件记录保存在验证目录。

## 验证证据

全部新验证独立位于 `logs/pure_bptt_cleanup_20261114`，历史成绩不作为本轮验证。

- `behavior_equivalence.json`：固定保留模型和输入，CPU 动作及各项奖励完全一致，原视频 SHA-256 一致。
- `cpu_regression_complete.log`、`full_dataset_reference.log`：95 项 CPU 回归通过（含全部 297400 帧参考扫描），另有 5 项登记 GPU 用例通过，总计 100 项。
- `onnx_equivalence.json`、`onnx_validation.log`：普通 Actor 与 ONNX Runtime 的动作和参考输出比较，容差 `atol=rtol=1e-4`。
- `checkpoint_migration.jsonl`、`checkpoint_migration_fixed.log`：物理文件去重、逐张量核对、属性修复及原子替换记录。
- `gpu_checks`、`gpu_scratch`、`gpu_resume`：登记 GPU 集成、编译 BPTT 更新、完整保存及恢复验证。

`tools/test_env.py`、`test_reset.py` 是独立 GPU 脚本，不属于 CPU unittest 收集；初次全目录收集的入口错误保留在日志中。CPU 正式命令为 [窗口说明](WINDOW_COMPARISON.md) 中的显式测试列表，并设置 `PYTHONPATH=src:tools`。

## 复现入口

从项目根目录按本机 trainreg 流程登记后执行 [窗口说明](WINDOW_COMPARISON.md) 的训练命令。GPU 短验证的完整参数保存在对应目录的 `command.json`，包含首次至少 3 次 Actor 更新、完整保存和恢复后至少 2 次更新。数据、源码、删除清单及最终审查结果分别见验证目录中的 JSON 和报告。
