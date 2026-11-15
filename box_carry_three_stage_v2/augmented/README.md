# 三阶段搬箱数据

本目录包含 teacher_025845 使用的裁剪后 NPZ 数据，使用 Git LFS 存储。748 条 train、146 条 val、106 条 test，共 1000 条片段、297400 帧，总计 629969550 字节。目录为 `<stage>/<split>/*.npz`，上传的是实际文件，保留原有数据划分。

在仓库根目录下载和校验：

```bash
git lfs install
git lfs pull
(cd box_carry_three_stage_v2/augmented && sha256sum -c SHA256SUMS)
```

若 NPZ 开头为 `version https://git-lfs.github.com/spec/v1`，说明尚未下载数据本体，请完成 `git lfs pull`。训练和评估使用 `--motion_dir box_carry_three_stage_v2/augmented`。

每个 NPZ 均与最终模型的冻结训练快照逐文件 SHA-256 一致。`dataset_fingerprint.json` 提供原训练指纹、分组、帧数、大小和逐文件哈希；`manifest.csv`、`manifest.json` 保留原裁剪和阶段标注。不要混用裁剪前帧号。

NPZ 格式由 [CarryMotionLib](../../src/blinkmimic/box_carry.py) 读取，包含机器人、箱体、手部接触、阶段及支撑表面信息，50 Hz。模型和数据指纹见 [模型卡](../../checkpoints/teacher_025845/MODEL_CARD.json)，机器人资产见 `data/g1_carry`。
