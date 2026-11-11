# Carry AMP 最优策略

- best_policy.pt：第 17,465 轮周期检查点，包含 EMA actor、视觉编码器、观测归一化、AMP 判别器及其归一化。
- 策略按全部 146 条 val 的共同任务回报选出；训练 seed 0，总时长 7200.128 秒，最终训练更新为 20,869。
- Actor 编码/解码隐藏层均为 512–256–128，latent 32；加载时按保存的 enc_hidden/dec_hidden 配置构建网络。
- 输入：10 步观测历史、45 帧/4056 维目标、深度图；根部/箱体目标相对实际根部位置，保留参考 yaw 坐标。
- 训练使用 32 步 BPTT、0.02 秒控制周期。actor 不使用参考关节角动作跳连。
- 本文件是周期策略检查点，不含完整 WM/优化器恢复状态。完整训练状态保留在原训练服务器。
- 保留了原检查点字节与 SHA256；嵌入的 motion_dir 是训练时临时路径，评估请显式传 --motion_dir。
- args.json 只将 motion_dir/log_dir 改为仓库相对路径，训练超参数与原运行一致。

representative_policy.onnx 不依赖判别器，烘焙的是验证集中一条 70 帧搬运参考，不是完整轨迹库。它需要 obs_history、time_step、depth、root_pos_w；根部位置必须与烘焙参考使用同一世界坐标。64 个随机参考帧/实际根部位置上的 PyTorch 对照最大误差为 6.68e-6。

模型仍未稳定解决长距离接近；完整指标见 ../../docs/AMP_TRAINING_REPORT.md。
文件校验值见 SHA256SUMS。
