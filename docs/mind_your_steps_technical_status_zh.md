# Mind Your Steps 落脚点跟踪任务阶段技术报告

## 1. 报告结论

当前项目已完成 **第一阶段：Unitree R1 平地版本的工程实现与短程运行验收**。

这意味着新任务的代码结构、目标生成、相位切换、观测、奖励、任务注册和
GPU 训练链路均已打通，但尚不能说明 PPO 策略已经收敛，也不能说明机器人
已经学会稳定、准确地连续踩点。

下一步应进入 **第二阶段：平地策略收敛训练与奖励调参**。在平地二维落脚点
跟踪达到稳定验收指标之前，不建议直接加入移动踏石或复杂三维地形，否则很难
判断失败来自策略、奖励还是地形。后续所有 foothold policy 训练均以 R1 为本体。

## 2. 当前运行环境

| 项目 | 当前状态 |
| --- | --- |
| 服务器 GPU | NVIDIA GeForce RTX 4090，24 GB |
| NVIDIA 驱动 | 580.126.09 |
| Conda 环境 | `footstep` |
| Python | 3.8 |
| PyTorch | 1.10.0+cu113 |
| PyTorch CUDA Runtime | 11.3 |
| 仿真后端 | Isaac Gym Preview 4 / GPU PhysX |
| Tensor Pipeline | GPU，已验证可用 |

PyTorch 显示的 CUDA 11.3 是 wheel 携带的运行时版本，不等同于服务器驱动
支持的最高 CUDA 版本。当前 RTX 4090 已成功完成 GPU PhysX 和 PPO 迭代，
因此本阶段没有升级 CUDA、PyTorch 或 Isaac Gym 依赖的必要。

## 3. 第一阶段已完成内容

### 3.1 仓库与控制链路分析

已完成原 `humanoid_controller`、配置系统、任务注册、Actor/Critic 观测、奖励
注册、步态相位以及 LIPM/IPC3D guidance 接口分析。分析记录见
`docs/mind_your_steps_integration.md`。

新任务保持原有控制架构：

```text
Goal Generator
  -> stance-foot-frame foothold goal
  -> PPO foothold tracking policy
  -> joint position targets
  -> existing PD controller
  -> Isaac Gym GPU PhysX
```

没有替换为 LocoMuJoCo、MuJoCo、MJX 或 JAX，也没有删除原有 humanoid、LIPM、
IPC3D 功能。

### 3.2 新任务与目标生成器

已注册新任务：

```text
r1_foothold_tracker
```

旧名称 `humanoid_foothold_tracker` 仅作为兼容别名保留，也会加载完全相同的 R1
类和配置。后续训练命令统一使用 `r1_foothold_tracker`。

已实现以下核心行为：

- 左右脚目标各使用 `[x, y, z, sin(yaw), cos(yaw)]`，总维度为 10；
- 所有 Actor 目标均锁存在当前站立足坐标系；
- 目标只在 episode reset 或半周期 phase switch 时更新；
- 摆动期间目标保持不变，不随移动中的 base frame 重新计算；
- phase `[0, 0.5)` 为左脚摆动、右脚站立；
- phase `[0.5, 1.0)` 为右脚摆动、左脚站立；
- 相位切换不依赖真实接触状态；
- 保存 swing、stance、previous swing 和 phase switch 批量 tensor；
- 支持步长、方向、偏航、高度与 hold-still 采样；
- 实现左右脚可达范围限制和防交叉约束；
- 采样过程使用批量 Torch 运算，没有逐环境 Python 循环。

### 3.3 Actor 与 Critic

Actor 观测为 54 维：

| 观测 | 维度 |
| --- | ---: |
| projected gravity | 3 |
| base angular velocity | 3 |
| DOF position error | 12 |
| DOF velocity | 12 |
| previous action | 12 |
| phase sin/cos | 2 |
| foothold goal | 10 |

Actor 不包含世界坐标位置、精确 base 线速度、接触标志、地形真值或 Goal
Sampler 内部参数。

Critic 为 71 维，在 Actor 信息之外增加世界系 base 线速度、脚接触状态、
世界系落脚误差和接触力。

### 3.4 奖励函数

已实现并注册：

- swing foot XY tracking；
- swing foot Z tracking；
- swing foot yaw tracking；
- touchdown accuracy；
- latched stance accuracy；
- phase foot clearance；
- swing knee height。

同时复用了原任务的身体姿态、base 高度、动作变化率、关节速度、关节加速度、
力矩、碰撞、关节限位和 foot slip 等正则项。

### 3.5 规划器适配与平台预留

已建立统一 `FootholdGoalGenerator` 接口，并实现：

- `RandomFootholdGoalSampler`；
- `LIPMFootholdAdapter`；
- `IPC3DFootholdAdapter`。

LIPM/IPC3D 适配器能够接受世界系或基座系批量目标，再统一转换并锁存到站立足
坐标系。当前默认任务使用随机目标采样器，尚未把具体 LIPM/IPC3D planner
callback 设为默认数据源。

已保留 `height_range` 和 `support_platform_generator` 接口。第一阶段按计划只使用
`z=0` 平面，没有实现复杂移动踏石。

## 4. 已完成验证

### 4.1 单元与静态测试

以下七项测试全部通过：

1. 世界坐标到站立足坐标变换；
2. yaw wrap；
3. phase switch；
4. 摆动期目标冻结；
5. 左右脚切换和防交叉；
6. goal tensor shape。
7. R1 资产、12 自由度及 Actor/Critic 信息隔离配置。

新增文件同时通过 Python 编译检查和 `git diff --check`。

### 4.2 原任务回归测试

`humanoid_controller` 使用 8 个环境完成 1 次 PPO 迭代，退出码为 0。原任务
Actor/Critic 仍为 52/52 维，没有出现 NaN、CUDA 或 tensor shape 错误。

### 4.3 新任务 GPU 冒烟测试

执行配置：

```text
task: r1_foothold_tracker
num_envs: 64
max_iterations: 2
physics: GPU PhysX
pipeline: GPU
```

两次迭代均完成，共运行 3072 simulation steps。Actor/Critic 实际创建维度为
54/71，动作维度为 12，没有出现 NaN、CUDA 越界或 tensor shape 错误。

## 5. 当前尚未完成内容

以下内容不能视为已经完成：

- PPO 长时间训练和策略收敛；
- 稳定连续行走和落脚精度验收；
- 奖励权重、课程学习和随机化参数的系统调优；
- 非零高度平台、固定踏石或移动踏石；
- LIPM/IPC3D 规划器在新任务中的端到端训练；
- sim-to-real、实机部署或安全控制验证。

当前新任务已经直接使用 Unitree R1 12 自由度 URDF、关节默认姿态、PD 增益、
力矩限制、左右脚和左右膝刚体映射。原 humanoid 任务仅保留用于兼容和回归检查，
不再属于后续训练路线。

## 6. 下一阶段：平地收敛训练

### 6.1 阶段目标

在 Unitree R1 和 `z=0` 平面上，让策略能够稳定完成交替落脚，并证明目标
跟踪奖励确实可以学习，而不只是代码能够运行。

### 6.2 建议实施顺序

1. 使用固定或很窄的步长、方向范围完成最小训练基线；
2. 记录 episode length、摔倒率、左右脚 XY/Z/yaw 误差和 touchdown 误差；
3. 检查奖励尺度，避免生存、姿态或正则奖励压过落脚点奖励；
4. 确认策略能稳定交替迈步后，逐步扩大步长和方向偏移范围；
5. 再启用 hold-still 混合样本并评估站立稳定性；
6. 固化一组可复现 checkpoint、配置和评估脚本。

### 6.3 建议验收指标

建议在开始三维踏石前至少满足：

- 无 NaN、CUDA 越界和仿真崩溃；
- 多个随机种子均能稳定训练；
- episode length 接近配置上限且摔倒率明显下降；
- 左右脚均被实际训练，不出现单侧拖步或长期不切换；
- 平地 touchdown XY 平均误差建议先达到 5 cm 以内；
- yaw 误差、脚尖离地高度和 foot slip 保持在可控范围；
- hold-still 样本不会引发持续踏步或明显漂移。

具体误差阈值可根据 R1 脚底尺寸和实际任务需求进一步收紧。

### 6.4 下一阶段训练命令

正式训练前建议先进行中等规模观察：

```bash
cd /root/work/Footstep
conda activate footstep

python gym/scripts/train.py \
  --task=r1_foothold_tracker \
  --num_envs=1024 \
  --max_iterations=500 \
  --headless \
  --disable_wandb
```

确认奖励曲线和 episode length 有改善后，再扩大到 4096 个环境。第一次不要直接
进行超长无人值守训练，应先在 50、100、500 次迭代节点检查指标和 checkpoint。

## 7. 后续阶段路线

### 第三阶段：规划器端到端接入

将 LIPM 或 IPC3D 的实际输出接入已实现的 adapter，验证目标 frame、左右脚顺序、
更新时间和 planner 失效回退。随机采样器继续保留作为训练课程和对照组。

### 第四阶段：三维平台与课程学习

按固定静态平台、随机高度平台、稀疏踏石、移动平台的顺序增加难度。每增加一种
地形，都需要重新检查落脚可达性、碰撞、接触奖励和 Critic 特权信息。

### 第五阶段：鲁棒性与部署准备

加入合理的质量、摩擦、延迟、传感噪声和外力随机化，完成批量评估、checkpoint
管理、推理导出和安全限制。实机部署不应直接复用仿真参数而不做额外验证。

## 8. 当前阶段判断

```text
第一阶段：R1 平地工程实现与运行验收   已完成
第二阶段：R1 平地策略收敛与奖励调参   下一步
第三阶段：LIPM/IPC3D 端到端接入       未开始
第四阶段：三维平台与课程学习          未开始
第五阶段：鲁棒性和部署准备            未开始
```

因此，当前最合理的下一步不是继续增加系统复杂度，而是启动受监控的平地训练，
建立落脚误差、摔倒率和 episode length 的量化基线。
