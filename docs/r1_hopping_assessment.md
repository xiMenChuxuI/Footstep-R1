# Unitree R1 跳跃现象评估

## 修复进展更新

本报告下方的“建议”已经实施：spawn z 调整为 `0.715`，reset z 范围调整为
`[0.712, 0.718]`，PD 目标应用 `actuation_scale=0.25`，action clip 调整为 4.0，
并增加 stance contact、flight 和 early swing contact 奖励项。

新模型的 flight ratio 已从旧模型的 `90.68%` 降到低于 `1%`，所以持续腾空跳跃
已经基本压制。但 50 次迭代模型仍有超过 `95%` 的 double support，平均世界系
前向速度仍为负值，尚未形成可验收的前进步态。详细融合关系和最新数据见
`docs/footstep_loco_mujoco_fusion_report_zh.md`。

## 结论

当前跳跃由多个问题叠加造成，不能只归因于 PPO 尚未收敛：

1. `init_state.pos[2]=0.68` 会造成确定性的初始碰撞弹射；
2. R1 动作当前直接作为弧度偏移送入 PD，缺少有效 action scale；
3. 当前奖励没有强制站立脚持续接触，也没有惩罚双脚腾空；
4. `model_100.pt` 仍是很早期的策略，并且已经表现出明显腾空行为；
5. 当前 URDF 没有真实踝关节平行四边形闭环，只是串联踝关节加固定装饰支链。

继续从 `model_100.pt` 训练不合适。应先修复初始状态、动作缩放和接触奖励，再从
头训练新的实验。

## 初始高度实测

使用默认关节姿态、零策略动作、关闭 domain randomization，对不同 pelvis 初始
高度进行 GPU PhysX 对照。第一个 control step 的结果如下：

| 初始高度 | 第一步 base z | 第一步 z 速度 | 判断 |
| ---: | ---: | ---: | --- |
| 0.680 | 0.7424 | +6.8105 m/s | 严重弹射 |
| 0.690 | 0.7389 | +5.2023 m/s | 严重弹射 |
| 0.700 | 0.7124 | +1.1931 m/s | 仍有弹射 |
| 0.710 | 0.7095 | -0.0981 m/s | 正常下落 |
| 0.715 | 0.7145 | -0.0981 m/s | 正常下落 |
| 0.720 | 0.7195 | -0.0981 m/s | 正常下落 |
| 0.730 | 0.7295 | -0.0981 m/s | 正常下落 |
| 0.750 | 0.7495 | -0.0981 m/s | 正常下落 |

`0.715` 没有初始向上弹射，并能较平缓地落到默认弯腿姿态。建议将 reset spawn
高度改为 `0.715`，小范围随机化可使用约 `[0.712, 0.718]`。

这不代表 `base_height_target` 也应改为 `0.715`。零动作落地后，当前默认弯腿
姿态的 pelvis 稳态高度接近 `0.68`。因此第一版建议保持奖励目标 `0.68`，只把
出生高度改到 `0.715`。

## 当前模型行为实测

加载 `model_100.pt`，临时使用安全出生高度 `0.715`，关闭 domain randomization，
在 64 个环境中采样 200 步。结果：

```text
samples:                    12800
flight_ratio:               0.9068
double_support_ratio:       0.0000
stance_contact_ratio:       0.0000
mean_abs_vertical_speed:    0.7312 m/s
mean_action_rms:            1.4123 rad
resets:                     4288
```

因此持续跳跃并非只来自出生弹射。即使安全出生，模型仍有 90.68% 的时间双脚
腾空，且没有形成站立脚承重。播放使用 Actor mean action，不带探索噪声，所以
该行为来自当前策略本身。

## 动作缩放问题

`Hicl12Controller._compute_torques()` 当前直接执行：

```text
desired_position = default_position + policy_action
```

虽然配置中存在 `control.actuation_scale`，该方法没有使用它。R1 配置继承的 scale
还是 `1.0`，action clipping 为 `10.0`。本次模型 action RMS 已达到 `1.4123 rad`，
对关节位置偏移而言过大，容易使力矩饱和并产生跳跃。

建议让 R1 PD 目标使用：

```text
desired_position = default_position + 0.25 * policy_action
```

并把原始 action clipping 收紧到约 `4.0`，使最大目标偏移约为 1 rad。修改后必须
从头训练，旧 checkpoint 的动作语义已经不同。

## 接触奖励漏洞

当前任务具有摆动脚 XY/Z/yaw、clearance、knee height 和 touchdown 正奖励，但：

- `contact_schedule` 权重被设为 0；
- 没有站立脚持续接触奖励；
- 没有双脚腾空惩罚；
- touchdown 只有成功正奖励，没有错过落地或过早接触惩罚。

这允许策略通过整体跳起移动双脚来获取部分跟踪奖励。建议增加：

1. `stance_contact`：当前 stance foot 在整个半周期保持接触；
2. `flight_penalty`：左右脚均无接触时惩罚；
3. `early_swing_contact_penalty`：摆动前段提前触地时惩罚；
4. touchdown event 奖励只在首次落地时结算；
5. 保留并重新标定 base z velocity、foot slip 和 action-rate 惩罚。

## 踝关节闭环评估

当前 `R1_12dof.urdf` 包含：

```text
12 revolute joints
27 fixed joints
1 floating root
```

踝关节 A/B link、rod 和 constraint link 都是 fixed joint 支链。它们没有形成从
驱动杆回到脚部的闭环约束；实际可动踝是独立的 ankle pitch 与 ankle roll 串联
关节。同时配置为：

```text
collapse_fixed_joints = True
apply_humanoid_jacobian = False
```

所以真实 R1 平行四边形传动没有进入仿真，也没有 actuator-to-joint coupling。
它不会以“闭环约束求解失败”的形式直接导致跳跃，但会带来两类问题：

- fixed 支链的质量和碰撞 mesh 被合并进膝/踝刚体，可能形成复杂 compound
  collision；
- 仿真中的独立 pitch/roll 关节与真实传动存在动力学和控制映射差异。

平地 PPO 第一阶段可以继续使用等效串联 ankle pitch/roll，但建议删除装饰连杆的
collision，只保留 visual，并为脚底使用简单、稳定的 box/capsule collision。
进入 sim-to-real 前必须补充真实踝关节 transmission/coupling 映射。

## 修复顺序

建议按以下顺序处理：

1. 将 spawn z 从 `0.68` 改为 `0.715`，保持 base height reward target 为 `0.68`；
2. 让 PD 目标实际应用 `actuation_scale=0.25`，收紧 action clipping；
3. 增加 stance contact、flight 和 early contact 奖励/惩罚；
4. 清理 A/B 装饰支链 collision，简化脚底碰撞体；
5. 重新跑零动作 spawn 诊断和两轮 PPO smoke test；
6. 创建全新训练目录，从 iteration 0 训练，不继续 `model_100.pt`；
7. 在 50/100/500 iteration 分别统计 flight ratio、stance contact ratio、z 速度和
   reset rate，再决定是否扩大训练。

## 可重复诊断命令

初始高度扫描：

```bash
python gym/scripts/diagnose_r1_spawn.py \
  --task=r1_foothold_tracker --headless
```

最新 checkpoint 行为统计：

```bash
python gym/scripts/diagnose_r1_policy.py \
  --task=r1_foothold_tracker --headless \
  --load_run=-1 --checkpoint=-1
```
