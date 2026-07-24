# R1 平滑随机路线落脚与朝向跟踪实现报告

## 1. 任务目标

新增 `r1_foothold_tracker_route_random`，保留现有平地直线任务和随机步宽任务。
新任务不再要求机器人始终朝世界 X 正方向，而是由平滑随机路线统一生成：

```text
路线切线朝向
  -> 世界坐标速度向量
  -> stance-foot-frame 落脚目标
  -> 双脚目标 yaw
  -> 路线朝向奖励
  -> PPO 关节位置目标
  -> 原 PD 控制器
```

## 2. 路线生成逻辑

每个环境维护 `route_heading`、`route_heading_rate` 和
`initial_route_heading`。路线只在左脚重新成为摆动脚时更新，即每个完整
左右步周期更新一次：

```text
heading_rate = 0.8 * heading_rate + random_noise
route_heading = route_heading + heading_rate
```

转向速度和相对初始朝向均被裁剪。到达朝向边界后，转向速度反向并衰减为
原来的 `0.5`，防止路线长期停在边界。

世界速度由路线切线直接生成：

```text
vx = 0.4 * cos(route_heading)
vy = 0.4 * sin(route_heading)
```

落脚目标继续使用每个环境的实际 `step_period * dt`。双脚 yaw 目标改为
`route_heading + yaw_offset`，因此位置、速度、脚朝向和身体朝向属于同一
条随机路线。

## 3. 奖励变化

新任务将原 `base_heading` 权重设为 `0`，新增：

```text
route_heading_tracking = 4.0
```

该奖励比较机器人实际 base heading 与随机路线切线。世界速度跟踪、
foothold trajectory、touchdown、stance、接触、滑移和关节正则奖励继续保留。

## 4. 三阶段课程

每轮 PPO 使用 `24` 个仿真步，因此在 `12000` 和 `24000` 个仿真步切换阶段：

| 阶段 | 训练轮次 | 朝向范围 | 随机增量 | 最大周期转角 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 500 | `±10°` | `±2°` | `4°` |
| 2 | 500 | `±30°` | `±4°` | `8°` |
| 3 | 500 | `±45°` | `±6°` | `12°` |

`play.py` 默认使用 `±60° / ±8° / 15°`，并保留命令行覆盖参数。

## 5. 兼容性

- Actor/Critic/Action 维度保持 `54/71/12`。
- 从随机落脚任务 `model_2000.pt` 热启动。
- 原 `r1_foothold_tracker` 和 `r1_foothold_tracker_random` 行为不变。
- 策略仍输出 12 维关节位置目标，继续使用原 Footstep PD 控制器。
- 新任务使用独立 run name，不覆盖已有 checkpoint。

## 6. 可视化

在原有名义目标、随机目标、实际脚位置和误差线基础上新增：

- 绿色路线切线箭头；
- 最近 24 个完整步态周期的实际 base 轨迹。

## 7. 运行命令

正式训练：

```bash
python gym/scripts/train.py \
  --task=r1_foothold_tracker_route_random \
  --resume \
  --load_run=Jul23_14-57-45_r1_random_foothold_xy_stage1_1000 \
  --checkpoint=2000 \
  --num_envs=4096 \
  --max_iterations=1500 \
  --run_name=r1_random_route_curriculum_1500 \
  --headless \
  --disable_wandb
```

最终演示：

```bash
python gym/scripts/play.py \
  --task=r1_foothold_tracker_route_random \
  --load_run=Jul23_16-25-19_r1_random_route_curriculum_1500 \
  --checkpoint=3500
```

该命令默认使用 `±60° / ±8° / 15°` 路线参数和
`X±8 cm / Y±10 cm` 落脚偏移。回退到训练最终阶段：

```bash
python gym/scripts/play.py \
  --task=r1_foothold_tracker_route_random \
  --load_run=Jul23_16-25-19_r1_random_route_curriculum_1500 \
  --checkpoint=3500 \
  --route_heading_bound=45 \
  --route_noise=6 \
  --route_rate_limit=12 \
  --foothold_jitter_x=0.04 \
  --foothold_jitter_y=0.05
```

## 8. 训练前基线

使用尚未进行路线课程训练的随机落脚 `model_2000.pt`，在第一阶段
`±10° / ±2° / 4°` 配置下诊断：

| 指标 | 结果 |
| --- | ---: |
| 重置次数 | `0` |
| 路线朝向平均误差 | `0.0238 rad` |
| 最终平均横向位移 | `0.2481 m` |
| touchdown XY error | `0.0351 m` |
| stance contact ratio | `0.9999` |
| X/Y 目标响应相关系数 | `0.3817 / 0.8775` |

该结果说明路线方向、世界速度、落脚位置和脚 yaw 已经形成闭环。正式训练结果、
各阶段诊断、最终 checkpoint 和限制如下。

## 9. 正式训练结果

```text
run:
Jul23_16-25-19_r1_random_route_curriculum_1500

checkpoint:
model_3500.pt

新增训练轮数: 1500
新增仿真步数: 147,456,000
训练用时: 1812.16 s
最终训练 episode length: 1001.00
最终训练 mean reward: 272.76
最终 route heading reward: 3.9489 / 4.0
```

训练全程没有 NaN、CUDA 越界或 tensor shape 错误。Actor/Critic/Action
维度保持 `54/71/12`。

## 10. 分阶段固定诊断

所有阶段使用 `64` 个环境、`600` 个采样步，即每组 `38400` 个样本：

| 指标 | 10° / model_2500 | 30° / model_3000 | 45° / model_3500 | 60° play |
| --- | ---: | ---: | ---: | ---: |
| 重置次数 | `1` | `0` | `0` | `0` |
| stance contact ratio | `0.9994` | `1.0000` | `1.0000` | `1.0000` |
| flight ratio | `0.0002` | `0.0000` | `0.0000` | `0.0000` |
| 世界 X 速度 | `0.4111` | `0.4318` | `0.4064` | `0.4034 m/s` |
| 路线朝向误差 | `0.0555` | `0.0467` | `0.0487` | `0.0546 rad` |
| 最终平均横向位移 | `0.1626` | `0.3541` | `0.5112` | `0.5432 m` |
| touchdown XY error | `0.0356` | `0.0342` | `0.0323` | `0.0525 m` |
| touchdown yaw error | `0.0176` | `0.0098` | `0.0124` | `0.0176 rad` |
| 接触足滑移速度 | `0.0296` | `0.0253` | `0.0192` | `0.0204 m/s` |
| X 目标相关系数 | `0.3346` | `0.6376` | `0.4467` | `0.5839` |
| Y 目标相关系数 | `0.8048` | `0.6706` | `0.8489` | `0.9111` |

横向位移随路线范围扩大而明显增加，证明当前行为已经从随机步宽变为累积的
随机路线。60° play 范围仍能稳定运行，但 touchdown 误差比 45° 训练范围大。

## 11. 目标消融

最终 `model_3500.pt` 的审计结果：

- foothold goal 位于 Actor observation `[44:54]`。
- 摆动期目标最大变化为 `0.00000000`。
- phase switch 平均目标变化为 `0.224527`。
- 同一状态清零 goal 后，动作 RMS 差异为 `0.764272 rad`。
- 正常组实际步宽约 `0.1984 m`，goal 清零后缩到 `0.0611 m`。
- 短步/长步实际 X 分别为 `0.1112/0.1462 m`。
- 窄步/宽步实际横向距离分别为 `0.1862/0.2110 m`。

这些结果证明随机路线 checkpoint 会实际使用 10 维 stance-foot-frame 目标。

## 12. 当前结论与限制

当前版本已经实现：

- 随机路线横向位移累积；
- 身体朝向跟随路线切线；
- 双脚位置和 yaw 与路线一致；
- 旧世界 X 固定朝向奖励被新路线朝向奖励替代；
- 10°、30°、45° 三阶段课程和 60° play 展示；
- 绿色路线方向箭头和最近 24 个步态周期轨迹。

第一版仍限制为相对初始朝向 `±45°` 训练，不支持掉头和连续 360° 绕行。
60° 是超出训练分布的展示配置，虽然固定诊断稳定，但应优先使用 45° 配置
做定量实验。

## 13. Play 流程验证

最终 checkpoint 已通过 `play.py` 的 2×2 录像流程，生成：

```text
logs/Unitree_R1_Foothold_Tracker/videos/
20260724_081329_2x2_r1_random_route_curriculum_1500.mp4
```

视频规格为 `1280×720`、`180` 帧、`3.6 s`。headless 录像不会执行 Isaac
Gym viewer 的 debug line 绘制，因此录像中不包含落脚框和绿色路线轨迹。
正常启动非 headless `play.py` 时会显示完整调试可视化。
