# R1 高站姿训练与 MuJoCo Sim2Sim 报告

## 1. 问题定位

旧平地模型能够稳定前进，但视觉上长期屈膝、机身高度偏低。代码审查与 Isaac Gym 确定性诊断结果表明，主要问题是腿部弯曲，而不是躯干前倾：

| 指标 | 旧 `model_1000.pt` |
|---|---:|
| 平均基座高度 | 0.6839 m |
| 平均站立腿膝角 | 0.8942 rad |
| 平均绝对基座 pitch | 0.0112 rad |
| 平均前进速度 | 0.4203 m/s |

原配置的 `base_height_target=0.68`，允许机器人保持低姿态。更重要的是，继承的高度奖励使用约 1 m 的指数特征尺度，数厘米高度误差的奖励变化很小，只提高目标数字不能形成足够清晰的学习信号。

## 2. 实现方式

新增独立任务 `r1_foothold_tracker_tall`，保留原 `r1_foothold_tracker`：

- 目标基座高度：`0.72 m`；
- 满奖励容差：`±0.01 m`；
- 高度误差平方尺度：`0.0004 m²`，对应 2 cm 特征尺度；
- 高度奖励权重：`2.0`；
- 站立腿膝角目标：`0.50 rad`；
- 膝角奖励权重：`0.5`；
- 直立姿态奖励：`2.0 -> 3.0`；
- 初始基座高度：`0.715 -> 0.725 m`。

高度奖励为：

```text
e_h = max(|h - 0.72| - 0.01, 0)
r_h = exp(-e_h² / 0.0004)
```

膝关节奖励只作用于当前站立腿，避免把摆动腿强行锁直：

```text
r_knee = exp(-(q_stance_knee - 0.50)² / 0.04)
```

Actor/Critic 维度保持 `54/71`，默认关节角、动作尺度、PD 参数和落脚目标格式均不改变，因此可以从旧平地 checkpoint 热启动，也可以沿用同一套 Sim2Sim 适配器。

主要实现文件：

- `gym/envs/humanoid/r1_tall_foothold_tracker.py`：厘米尺度高度奖励与站立腿伸展奖励；
- `gym/envs/humanoid/r1_tall_foothold_tracker_config.py`：高站姿任务和训练参数；
- `gym/envs/__init__.py`：注册 `r1_foothold_tracker_tall`；
- `gym/scripts/diagnose_r1_policy.py`：增加基座高度、pitch 和站立膝角诊断；
- `sim2sim/mujoco/config/r1_tall_model_1500.yaml`：新 checkpoint 的 MuJoCo 配置；
- `sim2sim/tests/test_r1_mujoco_adapter.py`：高站姿策略 MuJoCo 自动冒烟测试。

## 3. 训练

基础 checkpoint：

```text
Jul15_12-06-24_r1_stage3_latched_event_trajectory_1000/model_1000.pt
```

正式训练命令：

```bash
conda activate footstep
cd /root/work/Footstep-sim2sim

PYTHONPATH=/root/work/Footstep-sim2sim python gym/scripts/train.py \
  --task=r1_foothold_tracker_tall \
  --resume \
  --load_run=Jul15_12-06-24_r1_stage3_latched_event_trajectory_1000 \
  --checkpoint=1000 \
  --num_envs=4096 \
  --max_iterations=500 \
  --run_name=r1_tall_posture_stage1_500 \
  --headless \
  --disable_wandb
```

输出 checkpoint：

```text
logs/Unitree_R1_Foothold_Tracker/
Sep03_13-28-24_r1_tall_posture_stage1_500/model_1500.pt
```

训练完成 49152000 个采样步，最终 episode length 为 1001，无 NaN、CUDA 越界或观测维度错误。

## 4. Isaac Gym 新旧对比

| 指标 | 旧模型 | 高站姿模型 | 变化 |
|---|---:|---:|---:|
| 平均基座高度 | 0.6839 m | 0.7114 m | +2.75 cm |
| 平均站立腿膝角 | 0.8942 rad | 0.6409 rad | -0.2533 rad（约 14.5°） |
| 平均绝对基座 pitch | 0.0112 rad | 0.0063 rad | 降低 43.8% |
| 平均前进速度 | 0.4203 m/s | 0.4136 m/s | 基本保持 |
| 站立足接触率 | 99.21% | 99.87% | +0.66 个百分点 |
| 平均垂直速度 | 0.1167 m/s | 0.0635 m/s | 降低 45.6% |
| touchdown XY 误差 | 0.1307 m | 0.1445 m | +1.38 cm |
| 接触脚滑移速度 | 0.0249 m/s | 0.0311 m/s | +0.0062 m/s |
| 200 步诊断重置 | 1 | 1 | 不变 |

结论：机器人站高与支撑腿伸展得到明显改善，速度和稳定性保持；代价是落脚精度及滑移略有退化，后续可以在不降低高度的前提下做小权重精度微调。

## 5. MuJoCo Sim2Sim

新 checkpoint 导出误差：

```text
NumPy 与原 PyTorch Actor 最大绝对误差：4.77e-6
checkpoint SHA256：099a8c16f404fe8b75c14a040267e8e8f7196ac9a3525cd8d4f9e928d8389452
```

官方 R1 MuJoCo 60 秒测试：

| 指标 | 旧模型 | 高站姿模型 |
|---|---:|---:|
| 连续运行 | 60 s | 60 s |
| 前进距离 | 28.23 m | 27.89 m |
| 平均 X 速度 | 0.470 m/s | 0.465 m/s |
| 最终基座高度 | 0.6827 m | 0.7073 m |
| 跌倒 | 否 | 否 |

高站姿效果在 MuJoCo 中保留，最终高度提高约 2.46 cm，且 60 秒内没有失稳。

演示产物：

```text
artifacts/sim2sim/r1_tall_model_1500/r1_tall_mujoco_walk_10s.mp4
artifacts/sim2sim/r1_tall_model_1500/baseline_vs_tall_mujoco_10s.mp4
```

第二个视频将旧模型与高站姿模型并排播放，可直接用于观察屈膝程度和机身高度差异。

打开新模型 Viewer：

```bash
conda activate r1_mujoco
cd /root/work/Footstep-sim2sim

python -m sim2sim.mujoco.play_r1_flat \
  --config sim2sim/mujoco/config/r1_tall_model_1500.yaml \
  --policy artifacts/sim2sim/r1_tall_model_1500/policy_weights.npz \
  --duration 60 \
  --realtime
```

录制视频：

```bash
python -m sim2sim.mujoco.play_r1_flat \
  --config sim2sim/mujoco/config/r1_tall_model_1500.yaml \
  --policy artifacts/sim2sim/r1_tall_model_1500/policy_weights.npz \
  --headless --duration 30 \
  --record artifacts/sim2sim/r1_tall_model_1500/r1_tall_mujoco_walk.mp4
```

## 6. 测试结果

- Isaac Gym 落脚目标与高站姿单元测试：11/11 通过；
- MuJoCo 适配器、官方模型与高站姿策略测试：6/6 通过；
- 64 环境、2 轮 GPU 训练冒烟：通过；
- 4096 环境、500 轮正式训练：通过；
- MuJoCo 5 秒自动测试与 60 秒长测试：均通过；
- Python 编译检查和 `git diff --check`：通过。

## 7. 当前限制

1. 新模型更高、更直，但仍保留必要的膝关节弯曲，避免接近运动学奇异位形。
2. Touchdown XY 误差和接触滑移有小幅退化，不能宣称所有指标同时改善。
3. 本次只验证固定方向平地策略，尚未迁移到随机落脚、随机路线和复杂地形版本。
4. MuJoCo 使用高刚度上半身保持控制近似 Isaac Gym 中折叠的固定关节，这仍属于明确记录的仿真差异。
