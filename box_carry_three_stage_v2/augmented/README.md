# 三阶段搬箱数据入口

本分支只提供最新裁剪数据的清单，NPZ 原始轨迹需另行准备。按 manifest 中相对路径放到本目录，或用 `--motion_dir /path/to/augmented` 指定现有数据。

748 条 train、146 条 val、106 条 test，共 1000 条片段、297400 帧。清单对应 teacher_025845 使用的数据。观测、控制与参考均以 50 Hz 处理；不得混用裁剪前帧号。

数据文件与清单指纹见 `../../checkpoints/teacher_025845/MODEL_CARD.json`。NPZ 格式由 `src/blinkmimic/box_carry.py` 的 `CarryMotionLib` 读取；包含机器人、箱体、手部接触、阶段及支撑表面信息。机器人资产已提供在 `data/g1_carry`。
