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

训练范围为：

```text
X: [-0.04, +0.04] m
Y: [-0.05, +0.05] m
```

`play.py` 默认展示范围为：

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
  --load_run=r1_random_foothold_xy_stage1_1000 \
  --checkpoint=-1
```

回退到训练随机范围演示：

```bash
python gym/scripts/play.py \
  --task=r1_foothold_tracker_random \
  --load_run=r1_random_foothold_xy_stage1_1000 \
  --checkpoint=-1 \
  --foothold_jitter_x=0.04 \
  --foothold_jitter_y=0.05
```

策略审计与指标评估：

```bash
python gym/scripts/audit_foothold_checkpoint.py \
  --task=r1_foothold_tracker_random \
  --load_run=r1_random_foothold_xy_stage1_1000 \
  --checkpoint=-1

python gym/scripts/diagnose_r1_policy.py \
  --task=r1_foothold_tracker_random \
  --load_run=r1_random_foothold_xy_stage1_1000 \
  --checkpoint=-1
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

1000 轮正式训练及训练后诊断结果将在训练完成后继续补充。
