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
各阶段诊断、最终 checkpoint 和限制将在训练完成后补充。
