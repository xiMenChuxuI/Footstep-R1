# R1 平地随机落脚点任务实现报告

## 1. 目标

在不修改原 `r1_foothold_tracker` 平地任务和 `model_1000.pt` 的前提下，
新增 `r1_foothold_tracker_random`。新任务把速度指令换算为名义落脚点，
再在当前站立足坐标系内加入独立 XY 随机偏移，使策略学习响应变化的落脚目标。

## 2. 实际数据流

```text
世界坐标系速度指令
  -> 各环境 step_period * dt
  -> 站立足坐标系名义落脚点
  -> 相位切换时采样 XY 偏移
  -> 可达区域及左右脚防交叉裁剪
  -> [x, y, z, sin(yaw), cos(yaw)] * 2
  -> 54 维 Actor 观测
  -> 12 维关节位置目标
  -> Footstep 原有 PD 控制器
  -> Unitree R1
```

当前平地配置中，速度为 `0.4 m/s`，单步时间为 `30 * 0.01 = 0.30 s`，
因此名义前向步长为 `0.12 m`。左右脚名义横向位置分别为
`+0.20 m` 和 `-0.20 m`。

## 3. 随机范围与约束

Stage 1 训练范围为：

```text
X: [-0.04, +0.04] m
Y: [-0.05, +0.05] m
```

Stage 2 在 Stage 1 tag 之后扩大为：

```text
X: [-0.06, +0.06] m
Y: [-0.08, +0.08] m
```

`play.py` 默认展示范围保持为：

```text
X: [-0.08, +0.08] m
Y: [-0.10, +0.10] m
```

随机值以批量 tensor 方式按环境采样，只在 gait phase switch 时更新，
摆动期内保持不变。最终目标经过 `reach_x`、`reach_y` 和
`min_lateral_separation` 裁剪，防止左右腿目标交叉。

框架的 reset 顺序是先执行 `_reset_system()`，再重采样命令。新任务在
`_resample_commands()` 完成后重新锁存首个目标，确保 episode 第一目标
使用刚生成的速度指令和步周期。

## 4. 兼容性

- 原任务继续使用 `RandomFootholdGoalSampler`，行为不变。
- 新任务使用 `PerturbedFootholdGoalSampler`。
- Actor/Critic 维度保持 `54/71`。
- 策略仍输出 12 维关节位置目标，继续使用原 PD 控制器。
- 新旧任务使用相同 experiment 目录，因此新任务可加载原平地 checkpoint。
- 新训练使用独立 run name，不覆盖原 `model_1000.pt`。

## 5. 可视化

新任务在原有左右脚目标框基础上增加：

- 灰白色名义落脚框；
- 蓝色或红色的随机后有效目标框；
- 黄色名义目标到随机目标连线；
- 橙色摆动脚到有效目标误差连线；
- 白色实际首次触地点及 touchdown error。

## 6. 运行命令

两轮冒烟测试：

```bash
python gym/scripts/train.py \
  --task=r1_foothold_tracker_random \
  --resume \
  --load_run=Jul15_12-06-24_r1_stage3_latched_event_trajectory_1000 \
  --checkpoint=1000 \
  --num_envs=64 \
  --max_iterations=2 \
  --run_name=r1_random_foothold_smoke \
  --headless \
  --disable_wandb
```

正式热启动训练：

```bash
python gym/scripts/train.py \
  --task=r1_foothold_tracker_random \
  --resume \
  --load_run=Jul15_12-06-24_r1_stage3_latched_event_trajectory_1000 \
  --checkpoint=1000 \
  --num_envs=4096 \
  --max_iterations=1000 \
  --run_name=r1_random_foothold_xy_stage1_1000 \
  --headless \
  --disable_wandb
```

演示：

```bash
python gym/scripts/play.py \
  --task=r1_foothold_tracker_random \
  --load_run=Jul23_14-57-45_r1_random_foothold_xy_stage1_1000 \
  --checkpoint=2000
```

回退到训练随机范围演示：

```bash
python gym/scripts/play.py \
  --task=r1_foothold_tracker_random \
  --load_run=Jul23_14-57-45_r1_random_foothold_xy_stage1_1000 \
  --checkpoint=2000 \
  --foothold_jitter_x=0.04 \
  --foothold_jitter_y=0.05
```

策略审计与指标评估：

```bash
python gym/scripts/audit_foothold_checkpoint.py \
  --task=r1_foothold_tracker_random \
  --load_run=Jul23_14-57-45_r1_random_foothold_xy_stage1_1000 \
  --checkpoint=2000

python gym/scripts/diagnose_r1_policy.py \
  --task=r1_foothold_tracker_random \
  --load_run=Jul23_14-57-45_r1_random_foothold_xy_stage1_1000 \
  --checkpoint=2000
```

## 7. 测试与训练结果

实现阶段结果：

- `13` 个 foothold 单元测试全部通过。
- 新任务成功加载原平地 `model_1000.pt`。
- GPU PhysX 和 GPU pipeline 正常。
- Actor/Critic/Action 维度为 `54/71/12`。
- 64 环境两轮训练无 NaN、CUDA 越界或 tensor shape 错误。
- 摆动阶段目标最大变化为 `0.00000000`。
- phase switch 时目标平均变化为 `0.223884`。
- 同一状态屏蔽 10 维目标后，动作 RMS 差异为 `0.740000 rad`。
- 预训练 checkpoint 下，短步组实际触地 X 为 `0.1207 m`，
  长步组为 `0.1421 m`，证明策略输出会响应目标变化。

正式训练结果：

```text
run:
Jul23_14-57-45_r1_random_foothold_xy_stage1_1000

checkpoint:
model_2000.pt

新增训练轮数: 1000
新增仿真步数: 98,304,000
训练用时: 1181.24 s
最终训练 episode length: 1001.00
最终训练 mean reward: 267.69
```

训练范围固定诊断：

| 指标 | 结果 |
| --- | ---: |
| 世界前进速度 | `0.4219 m/s` |
| stance contact ratio | `0.9975` |
| flight ratio | `0.0005` |
| touchdown XY error | `0.1350 m` |
| 接触足滑移速度 | `0.0288 m/s` |
| 200 步重置次数 | `1` |
| X/Y 响应斜率 | `0.2847 / 1.1351` |
| X/Y Pearson 相关系数 | `0.0599 / 0.1587` |

放大 play 范围固定诊断：

| 指标 | 结果 |
| --- | ---: |
| 世界前进速度 | `0.4161 m/s` |
| stance contact ratio | `0.9969` |
| touchdown XY error | `0.1667 m` |
| 接触足滑移速度 | `0.0281 m/s` |
| 200 步重置次数 | `1` |
| X/Y 响应斜率 | `0.1457 / 0.5646` |
| X/Y Pearson 相关系数 | `0.0467 / 0.1308` |

目标消融审计：

- 正常目标组速度为 `0.4139 m/s`，重置 `2` 次。
- goal 清零组速度为 `-0.1694 m/s`，重置 `32` 次。
- 同一状态清零目标造成 `0.803225 rad` 的动作 RMS 差异。
- 短步/长步组实际触地 X 分别为 `0.1224/0.1353 m`。
- 窄步/宽步组实际横向距离分别为 `0.1817/0.2111 m`。

这些结果证明 10 维 foothold goal 已经实际影响策略动作和落脚位置，
新 checkpoint 不是仅依靠原速度跟踪前进。

## 8. 当前结论与限制

本阶段的代码、训练、checkpoint 加载、相位锁存、可视化和目标消融均已完成。
机器人在训练范围和放大 play 范围内都能保持稳定前进。

但是，随机目标到实际 touchdown 的逐样本相关性仍然偏弱，尤其是 X 方向。
这表示策略已经会使用 foothold goal，但还不是高精度的任意落脚点跟踪器。
放大 play 随机范围主要用于视觉展示，不能把它的精度等同于训练范围。
下一阶段若以精确随机落脚为目标，应优先调整 touchdown/stance 事件奖励和
随机课程，而不是继续单纯增加训练轮数。

## 9. Stage 2 扩大随机偏移

Stage 2 从以下 checkpoint 热启动：

```text
Jul23_14-57-45_r1_random_foothold_xy_stage1_1000/model_2000.pt
```

训练随机范围扩大到 `X±0.06 m / Y±0.08 m`，`play.py` 继续使用
`X±0.08 m / Y±0.10 m`。因此展示配置只比训练范围多 `2 cm`，减少
训练与展示之间的分布差异。

训练命令：

```bash
python gym/scripts/train.py \
  --task=r1_foothold_tracker_random \
  --resume \
  --load_run=Jul23_14-57-45_r1_random_foothold_xy_stage1_1000 \
  --checkpoint=2000 \
  --num_envs=4096 \
  --max_iterations=1000 \
  --run_name=r1_random_foothold_xy_stage2_expanded_1000 \
  --headless \
  --disable_wandb
```

正式训练结果：

```text
run:
Jul24_08-29-34_r1_random_foothold_xy_stage2_expanded_1000

checkpoint:
model_3000.pt

新增训练轮数: 1000
新增仿真步数: 98,304,000
训练用时: 1170.18 s
最终训练 episode length: 1001.00
最终训练 mean reward: 268.69
最终 swing-foot XY tracking reward: 3.6333
最终 touchdown accuracy reward: 1.0373
最终 termination reward: 0.0000
```

训练范围固定诊断：

| 指标 | 结果 |
| --- | ---: |
| 世界前进速度 | `0.4165 m/s` |
| stance contact ratio | `0.9987` |
| flight ratio | `0.0010` |
| touchdown XY error | `0.0887 m` |
| 接触足滑移速度 | `0.0329 m/s` |
| 200 步重置次数 | `1` |
| X/Y 响应斜率 | `0.1892 / 0.4562` |
| X/Y Pearson 相关系数 | `0.1009 / 0.1378` |

放大 play 范围固定诊断：

| 指标 | 结果 |
| --- | ---: |
| 世界前进速度 | `0.4220 m/s` |
| stance contact ratio | `0.9991` |
| flight ratio | `0.0009` |
| touchdown XY error | `0.1058 m` |
| 接触足滑移速度 | `0.0317 m/s` |
| 200 步重置次数 | `1` |
| X/Y 响应斜率 | `0.2400 / 0.5217` |
| X/Y Pearson 相关系数 | `0.1096 / 0.1682` |

目标消融审计：

- Actor/Critic 维度保持 `54/71`，foothold goal 位于 Actor 观测
  `[44:54]`。
- 摆动阶段目标最大变化为 `0.00000000`，phase switch 时平均变化为
  `0.228840`。
- 同一状态屏蔽 10 维目标后，动作 RMS 差异为 `0.680203 rad`。
- 正常目标组速度为 `0.4311 m/s`、重置 `1` 次；goal 清零组速度降为
  `0.1018 m/s`、重置增至 `24` 次。
- 短步/长步组实际触地 X 分别为 `0.1117/0.1412 m`。
- 窄步/宽步组实际横向距离分别为 `0.1869/0.2172 m`。

Stage 2 达到了本次“训练偏移更明显、play 略大于训练”的目标。与 Stage 1
相比，play 范围和训练范围的差距从 X/Y 各 `4/5 cm` 缩小为各 `2 cm`；
在 `X±8 cm / Y±10 cm` 的展示范围内仍保持稳定行走，并且横向响应斜率
提高到 `0.5217`。不过逐样本相关性仍然偏弱，因此当前模型适合展示
“随机目标会改变实际落脚”，尚不能描述为高精度逐框踩踏。

## 10. Stage 3 平衡精度课程

### 10.1 配置与奖励

Stage 3 将训练随机范围调整为 `X±0.05 m / Y±0.06 m`，Play 默认范围调整为
`X±0.06 m / Y±0.08 m`。新增两个不改变策略观测维度的奖励：

- `late_swing_foot_xy_tracking`：摆动后半程平滑生效，
  `sigma=0.0064`；
- `touchdown_accuracy_fine`：仅在首次有效触地事件生效，只计算 XY，
  `sigma=0.0049`。

Actor/Critic/Action 维度继续保持 `54/71/12`，原 `model_3000.pt` 可以直接
加载。增加 `--foothold_precision_profile` 参数，支持：

```text
range_only
endpoint_precision
endpoint_stance
```

### 10.2 三组短对照

三组实验均固定 seed `42`、使用 1024 环境、从同一个 Stage 2
`model_3000.pt` 独立续训 150 轮。随后分别在训练和 Play 范围使用
seeds `11/22/33` 诊断。

| Profile | 训练误差 | Play误差 | 训练接触率 | Play接触率 | Play Y响应斜率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| range_only | `13.23 cm` | `13.06 cm` | `99.84%` | `99.83%` | `0.275` |
| endpoint_precision | `9.89 cm` | `9.34 cm` | `99.67%` | `99.88%` | `0.453` |
| endpoint_stance | `14.24 cm` | `13.73 cm` | `99.87%` | `99.84%` | `0.655` |

`endpoint_precision` 在稳定性门槛内误差最低。将
`stance_accuracy_latched` 从 `3.0` 提高到 `4.5` 没有继续改善触地精度，
所以正式配置采用：

```text
late_swing_foot_xy_tracking = 2.0
touchdown_accuracy_fine = 20.0
stance_accuracy_latched = 3.0
```

### 10.3 正式 500 轮训练

正式训练从 Stage 2 `model_3000.pt` 重新开始，不继承短实验模型：

```bash
python gym/scripts/train.py \
  --task=r1_foothold_tracker_random \
  --resume \
  --load_run=Jul24_08-29-34_r1_random_foothold_xy_stage2_expanded_1000 \
  --checkpoint=3000 \
  --foothold_precision_profile=endpoint_precision \
  --seed=42 \
  --num_envs=4096 \
  --max_iterations=500 \
  --run_name=r1_random_foothold_stage3_precision_balanced_500 \
  --headless \
  --disable_wandb
```

训练输出：

```text
run:
Jul24_09-29-19_r1_random_foothold_stage3_precision_balanced_500

最终 checkpoint:
model_3500.pt

新增仿真步数: 49,152,000
训练用时: 598.41 s
最终 mean reward: 279.65
最终 episode length: 1001.00
最终 termination reward: 0.0000
```

训练奖励持续提高，但固定多 seed 诊断发现 `model_3500.pt` 的真实 touchdown
误差回升到训练 `13.22 cm`、Play `13.09 cm`。这属于奖励代理指标过拟合，
因此 `model_3500.pt` 保留用于复盘，但不作为推荐演示模型。

### 10.4 Checkpoint 选择与最终结论

使用同一诊断条件比较关键 checkpoint：

| Checkpoint | 训练误差 | Play误差 | Play接触率 | Play最大重置 |
| --- | ---: | ---: | ---: | ---: |
| Stage 2 `model_3000` | `8.35 cm` | `10.45 cm` | `99.85%` | `1` |
| 短实验 `model_3150` | `9.89 cm` | `9.34 cm` | `99.88%` | `1` |
| 正式 `model_3200` | `9.22 cm` | `8.46 cm` | `99.85%` | `2` |
| 正式 `model_3300` | `9.41 cm` | `10.72 cm` | `99.80%` | `1` |
| 正式 `model_3500` | `13.22 cm` | `13.09 cm` | `99.83%` | `1` |

当前推荐 Play checkpoint 为正式 run 的 `model_3200.pt`。它相对 Stage 2
将 Play 平均 touchdown 误差从 `10.45 cm` 降到 `8.46 cm`，改善约
`1.99 cm`；但一个固定 seed 出现 2 次重置，因此没有完全通过每个 seed
重置 `≤1` 的严格门槛。

`model_3200.pt` 的目标消融结果：

- 摆动期目标最大变化：`0.00000000`；
- 同状态 goal 清零动作 RMS 差异：`0.837491 rad`；
- 正常目标速度/重置：`0.4273 m/s / 1`；
- goal 清零速度/重置：`0.2761 m/s / 32`；
- 窄/宽目标实际横向落脚：`0.1844/0.2210 m`，相差 `3.66 cm`。

因此 Stage 3 确实增强了策略对随机目标的响应，并改善了 Play 平均精度，
但没有达到训练误差 `≤7.5 cm` 的目标，也没有在所有 seed 上满足重置
`≤1`。本阶段应描述为“部分通过”：功能、目标响应和 Play 精度改善已完成，
高精度和全 seed 稳定性仍需后续重新设计奖励或采用早停课程。

推荐演示命令：

```bash
python gym/scripts/play.py \
  --task=r1_foothold_tracker_random \
  --load_run=Jul24_09-29-19_r1_random_foothold_stage3_precision_balanced_500 \
  --checkpoint=3200
```
