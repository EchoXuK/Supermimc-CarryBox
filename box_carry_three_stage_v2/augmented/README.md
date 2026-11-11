# 搬箱数据入口

清单共 1000 条 50 Hz 增强片段：748 train、146 val、106 test。
阶段为 01_approach_lift、02_carry_move、03_lower_place。

manifest.csv 的 clip_npz 相对本目录解析。准备对应 NPZ 文件并保留：

- 01_approach_lift/train|val|test/*.npz
- 02_carry_move/train|val|test/*.npz
- 03_lower_place/train|val|test/*.npz

NPZ 数据需从原数据集取得；本仓库提供清单和代码。
加载、关节映射、FK、接触、箱体/桌面以及站立高度字段见 ../../src/blinkmimic/box_carry.py。
复制原 augmented 入口时，需解析 NPZ 符号链接或保留其指向的原目录。

数据保留 2026-11-11 开局 init 清理结果；改变片段内容或帧起点会改变报告中的评估结果。
