# Footstep 与 LocoMuJoCo 落脚点任务融合技术报告

## 1. 结论摘要

本次工作不是把两个仓库的仿真器和依赖直接合并，而是在
[`celesius/Footstep`](https://github.com/celesius/Footstep) 的 Isaac Gym、PyTorch、
PPO 和 PD 控制运行时中，引入
[`MontenegroAlessandro/loco-mujoco`](https://github.com/MontenegroAlessandro/loco-mujoco)
中 Mind Your Steps 风格的落脚点目标表达与跟踪思想，并将后续训练本体统一为
Unitree R1 12 自由度模型。

融合后的项目仍然是 Footstep 工程：

```text
Isaac Gym Preview 4 / GPU PhysX
  -> Footstep task registry and batched environment
  -> stance-foot-frame foothold goal generator
  -> asymmetric PPO actor/critic
  -> R1 12 joint position targets
  -> scaled PD control
```

没有安装或引入 MuJoCo、LocoMuJoCo、MJX、JAX，也没有改变原
`humanoid_controller`、LIPM 或 IPC3D 任务的运行后端。

当前已经完成工程集成、R1 替换、GPU 冒烟测试和第一轮跳跃问题修复；尚未完成
稳定前进策略的收敛训练。最新 50 轮 checkpoint 是诊断模型，不是可验收的最终
walking policy。

## 2. 两个上游仓库各自提供了什么

### 2.1 Footstep 的保留部分

Footstep 是本项目的主体，继续提供：

- Isaac Gym Preview 4 和 GPU PhysX 批量仿真；
- PyTorch 张量管线和 PPO runner；
- task registry、环境生命周期、reset、reward 注册和日志；
- joint-position action、PD 控制和力矩裁剪；
- gait phase 更新；
- 原 LIPM、IPC3D guidance 与 humanoid 任务；
- headless 训练、play、checkpoint 和视频导出入口。

这些模块没有被 LocoMuJoCo 的实现替换。

### 2.2 从 LocoMuJoCo/Mind Your Steps 借鉴的部分

LocoMuJoCo 本身是 MuJoCo locomotion/imitation-learning 工具链。本次只参考其
foot-placement goal 与 foot-placement reward 的任务语义：策略接收期望落脚点，
按左右脚/步态阶段跟踪目标，并在 touchdown 附近评价落脚精度。

落地到 Footstep 后，相关思想被重新实现为 Torch 批量计算和 Isaac Gym task，
没有复制其 MuJoCo environment、JAX 代码、数据集或训练后端。

## 3. 融合后新增的系统

### 3.1 新任务和 R1 本体

正式任务名为：

```text
r1_foothold_tracker
```

`humanoid_foothold_tracker` 仅保留为兼容别名，加载的也是同一 R1 类和配置。
后续 foothold 训练不再使用原 humanoid 资产。

R1 配置使用 12 个腿部关节：左右各 6 个 hip pitch/roll/yaw、knee、ankle
pitch/roll。头部、手臂和躯干关节不进入 action space。

### 3.2 目标数据流

每只脚的目标定义为：

```text
[x, y, z, sin(yaw), cos(yaw)]
```

左右脚合计 10 维，tensor shape 为 `[num_envs, 10]`。世界系目标只在 reset 或
phase switch 时转换到当时的 stance-foot frame 并锁存，摆动期不会跟随 base
frame 每步重算。

相位约定为：

```text
phase in [0.0, 0.5): left swing, right stance
phase in [0.5, 1.0): right swing, left stance
```

第一版相位由时钟驱动，不用真实接触触发切换。环境保存
`swing_foot_index`、`stance_foot_index`、`previous_swing_foot_index` 和
`phase_switch_mask`。

### 3.3 Goal Generator 与原规划器

统一接口为：

```text
FootholdGoalGenerator.reset(env_ids)
FootholdGoalGenerator.update(env_ids, phase_switch_mask)
FootholdGoalGenerator.get_goal()
```

已实现：

- `RandomFootholdGoalSampler`；
- `LIPMFootholdAdapter`；
- `IPC3DFootholdAdapter`；
- world/base frame 到 stance-foot frame 的统一适配。

默认训练暂时使用随机采样器。LIPM 和 IPC3D 的适配层已就位，但尚未把具体
planner callback 设为默认目标源。

### 3.4 Actor、Critic 与控制输出

Actor 为 54 维：projected gravity 3、base angular velocity 3、DOF position
error 12、DOF velocity 12、previous action 12、phase sin/cos 2、foothold goal 10。

Actor 不包含精确世界位置、精确 base 线速度、真实接触标志、地形真值或 sampler
内部参数。固定 `+0.4 m/s` 前向课程命令只参与 reward，不作为 Actor 输入。

Critic 为 71 维，额外使用世界系 base 线速度、脚接触、世界系脚目标误差和接触力。

Actor 输出 12 维关节位置偏移。R1 的实际 PD 目标为：

```text
q_des = q_default + 0.25 * policy_action
```

原始 action clip 为 4.0，之后由现有 PD 和 torque limit 处理。

## 4. 奖励设计

落脚点任务新增或启用：

- swing foot XY/Z/yaw tracking；
- touchdown accuracy；
- phase-switch latched stance accuracy；
- phase foot clearance；
- swing knee height；
- stance contact；
- flight penalty；
- early swing contact penalty；
- 固定前向速度课程的 world linear velocity tracking。

继续复用 Footstep 的 base orientation/height、heading、action rate、joint velocity、
joint acceleration、torque、foot slip、collision 和 joint limit 正则项。

接触信息可以进入 reward 和 critic，但没有进入 actor observation。

## 5. R1 跳跃问题与已实施修复

### 5.1 已确认的初始弹射

GPU PhysX 零动作扫描显示，`pos[2]=0.68` 的第一个 control step 会产生约
`+6.78 m/s` 竖直速度；`0.69` 和 `0.70` 也存在明显弹射。`0.715` 的第一步为
正常重力下落 `-0.0981 m/s`。

因此已将：

```text
spawn z: 0.715
reset z range: [0.712, 0.718]
base height reward target: 0.68 (保持不变)
```

### 5.2 旧 checkpoint 的策略跳跃

旧 `model_100.pt` 在安全出生高度下仍有：

```text
flight ratio: 90.68%
stance contact ratio: 0.00%
mean abs vertical speed: 0.7312 m/s
mean raw action RMS: 1.4123 rad
```

因此旧模型不能继续使用。动作缩放、接触奖励和出生状态改变后，旧 checkpoint
的 action 语义也已经失效，必须从 iteration 0 重新训练。

### 5.3 修复后的短训结果

第一轮新配置 50 次迭代后：

```text
flight ratio: 0.56%
stance contact ratio: 99.13%
double support ratio: 97.02%
mean abs vertical speed: 0.2599 m/s
mean world forward speed: -0.4825 m/s
mean swing XY error: 0.1847 m
```

这说明大幅腾空跳跃已经被抑制，但策略主要保持双支撑并向后补偿，不能称为行走。

增加前向速度课程并提高 early-swing-contact 惩罚后的第二轮 50 次迭代结果：

```text
flight ratio: 0.06%
stance contact ratio: 98.33%
double support ratio: 95.37%
mean abs vertical speed: 0.2626 m/s
mean world forward speed: -0.5340 m/s
mean swing XY error: 0.1854 m
```

该短训仍未收敛。零动作 200 步基线进一步表明：安全高度消除了出生弹射，但默认
姿态约 0.5 秒后会向前滑倒；50 轮 PPO 学到的后退是对默认前倾不稳定性的过度
补偿，不是目标方向写反。

### 5.4 踝部模型限制

当前 R1 URDF 的 A/B rod 和 constraint 支链由 fixed joints 构成，并在 Isaac Gym
中折叠；真正可动部分仍是串联 ankle pitch/roll。它不是物理闭环平行四边形，
也没有真实 actuator coupling。

这不是 `z=0.68` 初始弹射的唯一原因，但会影响接触和 sim-to-real 精度。本次已
将左右脚原始三角 mesh collision 替换为 `0.18 x 0.07 x 0.02 m` 的足底 box，
并把 sole-center offset 从 `[0.03, 0, -0.035]` 修正为
`[0.04, 0, -0.053]`。落地后的足底 Z 读数已由约 `0.018 m` 对齐到约 0。

简化足底后零动作基线仍会前倾倒地，说明主动平衡和默认姿态仍是主要问题。
后续还可移除装饰支链 collision、保留 visual。实机部署前必须实现真实踝传动
映射。

## 6. 当前验收状态

| 项目 | 状态 |
| --- | --- |
| 原 humanoid/LIPM/IPC3D 未被删除 | 通过 |
| R1 12 自由度新任务可创建 | 通过 |
| Actor/Critic 54/71 维 | 通过 |
| stance-foot-frame 10 维目标 | 通过 |
| phase switch 更新、摆动期冻结 | 通过 |
| 7 项单元测试 | 通过 |
| 新任务 2 次 GPU PPO 冒烟 | 通过 |
| 无 NaN/CUDA 越界/shape 错误 | 通过 |
| 消除初始碰撞弹射 | 通过 |
| 简化足底 collision 并校准 sole offset | 通过 |
| 抑制旧策略大幅腾空 | 通过 |
| 稳定交替落脚 | 未通过 |
| 平均向前行走 | 未通过 |
| touchdown XY 误差小于 5 cm | 未通过 |

## 7. 后续阶段

### 阶段 A：稳定默认站立与接触模型

继续清理踝装饰支链 collision，检查默认姿态的质心投影、足底支撑多边形和 pitch
稳定性。已完成的足底 box 简化没有单独消除前倾。目标是简单站立策略下不发生
持续滑移和快速前倾倒地。

### 阶段 B：前向平地课程收敛

从新配置的 iteration 0 开始，按 50、100、500 次迭代节点评估，不续训旧
`model_100.pt`。重点指标为 forward speed、double support、stance contact、
flight、touchdown XY error、episode length 和 reset rate。

只有平均 `vx > 0`、双支撑比例下降、左右脚实际交替离地并且落脚误差持续下降，
才可判定进入稳定前进阶段。

### 阶段 C：扩大目标范围与规划器接入

平地固定前向课程收敛后，再逐步扩大步长、方向和 yaw，启用 hold-still 混合，
然后分别接入 LIPM/IPC3D adapter 做端到端对照。

### 阶段 D：三维平台与鲁棒性

最后再实现非零高度平台、固定踏石、移动踏石和 domain randomization。当前不应
用复杂地形掩盖平地接触和步态尚未收敛的问题。

## 8. 复现命令

```bash
cd /root/work/Footstep
source /opt/conda/etc/profile.d/conda.sh
conda activate footstep

python -m unittest gym.tests.test_foothold_goal -v

python gym/scripts/diagnose_r1_spawn.py \
  --task=r1_foothold_tracker --headless --disable_wandb

python gym/scripts/train.py \
  --task=r1_foothold_tracker \
  --num_envs=64 \
  --max_iterations=2 \
  --headless \
  --disable_wandb

python gym/scripts/diagnose_r1_policy.py \
  --task=r1_foothold_tracker \
  --load_run=<run_directory> \
  --checkpoint=<iteration> \
  --headless \
  --disable_wandb
```

## 9. 最终阶段判断

```text
工程融合与 R1 任务实现             已完成
GPU 运行链路和两轮冒烟             已完成
初始弹射与大幅腾空问题             已基本解决
R1 接触模型和默认站立稳定性        待优化
R1 前向交替步行策略收敛            进行中，尚未通过
LIPM/IPC3D 默认端到端接入          未开始
三维踏石与移动平台                 未开始
```
