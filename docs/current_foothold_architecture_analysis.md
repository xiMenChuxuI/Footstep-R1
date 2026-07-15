# R1 Foothold Tracker 当前架构审计报告

## 1. 审计结论

当前 `r1_foothold_tracker` 已经建立并实际训练了以下闭环：

```text
RandomFootholdGoalSampler（固定向前方向，随机步长/步宽）
    -> world_target_pos / world_target_yaw
    -> phase switch 时转换并锁存为 stance-foot-frame 10 维目标
    -> 54 维 Actor observation（其中最后 10 维为 foothold_goal）
    -> PPO Actor 输出 12 维关节位置残差
    -> clip 到 [-4, 4]，乘 actuation_scale=0.25，加 default_dof_pos
    -> 原 Hicl12Controller PD：Kp(q_des-q)-Kd*qdot
    -> torque limit
    -> Isaac Gym / Unitree R1 12-DOF
```

因此，机器人向前走并非只是原 Footstep Actor 继续读取速度命令。当前 play
把速度命令设为零，而且新 Actor observation 中没有 `commands`。但是当前策略
也不能称为完全由任意落脚目标驱动：训练使用固定 `0.4 m/s` 速度奖励，目标分布
始终接近向前，策略已经学到很强的 phase-driven walking prior。目标清零后仍能以
约 `0.30 m/s` 行走，但改变目标步长会显著改变动作、实际落脚和前进速度。

更准确的定义是：**当前是一个落脚目标条件化、同时受固定速度辅助奖励约束的
步态策略**。新的落脚闭环已经生效，但 LIPM/IPC3D 可插拔规划器和通用方向/地形
目标还没有完成。

## 2. 原始 Footstep 实际控制流程

原始 `humanoid_controller` 注册为 `HumanoidController`，配置为
`HumanoidControllerCfg`。主要调用链如下：

1. `LeggedRobot._resample_commands()` 采样 `commands=[vx, vy, yaw_rate]`。
2. `HumanoidController._post_physics_step_callback()` 更新机器人、CoM、ICP、
   接触和步态相位。
3. `HumanoidController._update_commands()` 在步周期到达时调用
   `_generate_step_command_by_3DLIPM_XCoM()`。
4. 3D LIPM/XCoM 根据 `commands`、CoM、支撑足和步周期产生世界坐标
   `step_commands=[x, y, heading]`。
5. `_set_obs_variables()` 把 `step_commands` 和脚状态从世界坐标转换到移动的
   base frame，形成 `step_commands_right/left`。
6. 原 Actor 同时读取 `base_lin_vel_world`、`commands`、左右脚状态、左右
   `step_commands`、phase、关节状态等；Actor/Critic observation 相同。
7. Actor 输出 `dof_pos_target`，原 `HumanoidController._compute_torques()` 计算
   `q_des=default_dof_pos+dof_pos_target`，再通过 PD 和 torque clip 驱动仿真。

原始流程不是简单的“速度命令直接进入 RL”。实际是：

```text
速度命令
  -> 3D LIPM/XCoM 生成 step_commands
  -> base-frame step command + 速度/机器人状态共同进入 Actor
  -> 关节位置残差
  -> PD
```

相关代码：

- `gym/envs/humanoid/humanoid_controller.py`
  - `_post_physics_step_callback()`
  - `_update_commands()`
  - `_generate_step_command_by_3DLIPM_XCoM()`
  - `_set_obs_variables()`
  - `_compute_torques()`
- `gym/envs/humanoid/humanoid_controller_config.py`
  - `HumanoidControllerRunnerCfg.policy.actor_obs`

## 3. 当前新任务的真实运行流程

### 3.1 任务与 checkpoint 加载

`gym/envs/__init__.py` 将 `r1_foothold_tracker` 和兼容别名
`humanoid_foothold_tracker` 都注册到：

- task class：`R1FootholdTracker`
- env config：`R1FootholdTrackerCfg`
- runner config：`R1FootholdTrackerRunnerCfg`

`gym/scripts/play.py` 的真实流程是：

1. `task_registry.get_cfgs(args)` 取得上述配置。
2. 将环境数限制到 1--9、地形强制为 plane、关闭 domain randomization。
3. 将 `lin_vel_x/lin_vel_y/yaw_vel` 全部设为零。
4. `task_registry.make_env()` 创建 `R1FootholdTracker`。
5. `make_alg_runner()` 根据 `experiment_name=Unitree_R1_Foothold_Tracker`
   加载指定或最新 checkpoint。
6. 每帧执行 `get_inference_actions() -> set_actions() -> env.step()`。

本次审计明确加载：

```text
logs/Unitree_R1_Foothold_Tracker/
  Jul15_12-06-24_r1_stage3_latched_event_trajectory_1000/model_1000.pt
```

为避免“最新目录”变化，复现时应显式传入 `--load_run` 和 `--checkpoint=1000`。

### 3.2 上层目标来源

`R1FootholdTracker._init_buffers()` 直接执行：

```python
self.goal_generator = RandomFootholdGoalSampler(self, self.cfg.foothold_goal)
```

因此当前既不是 LIPM，也不是 IPC3D。它是随机 Goal Sampler，配置为：

```text
movement_direction_deg = [0, 0]
feet_direction_deg     = [0, 0]
step_length_range      = [0.08, 0.16] m
step_width_range       = [0.18, 0.22] m
direction_offset_deg   = [-5, 5]
yaw_offset_deg         = [-5, 5]
hold_probability       = 0.05
height_range           = [0, 0]
```

这不是单一固定落脚点，但属于固定向前方向附近的窄随机分布。

### 3.3 世界目标与站立足坐标目标

`RandomFootholdGoalSampler.update()`：

1. 根据当前 `swing_foot_index/stance_foot_index` 取得站立足 pose。
2. 在站立足附近采样步长、步宽、方向、yaw 和高度。
3. 对左右足分别施加横向符号、reach clamp 和防交叉约束。
4. 保存世界坐标 `world_target_pos[num_envs,2,3]` 和
   `world_target_yaw[num_envs,2]`。
5. 调用 `world_to_stance_foot_goal()`，经 `quat_rotate_inverse()` 得到：

```text
[left x,y,z,sin(yaw),cos(yaw), right x,y,z,sin(yaw),cos(yaw)]
```

即 `foothold_goal[num_envs,10]`。stance entry 是当前站立足相对自身的 pose，
swing entry 是当前有效的下一落脚目标；当前并没有同时规划左右两步的未来 horizon。

### 3.4 phase switch 与目标冻结

`Hicl12Controller._update_robot_states()` 按固定周期推进 `phase`，没有使用真实接触
触发 phase。`R1FootholdTracker._advance_phase_and_goals()` 再执行：

- `phase < 0.5`：左脚 swing、右脚 stance；
- `phase >= 0.5`：右脚 swing、左脚 stance；
- `phase_foot_indices()` 生成 `phase_switch_mask`；
- 仅对 mask 为真的环境调用 sampler update；
- 摆动期每步虽然会读取 `get_goal()`，底层 tensor 没有重算或改变。

GPU 运行审计得到：

```text
frozen_goal_max_change      = 0.00000000
mean_goal_change_on_switch  = 0.215135
```

这证明目标在非切换帧严格冻结，在 phase switch 时发生更新。

### 3.5 Actor 与 Critic observation

`R1FootholdTrackerRunnerCfg.policy` 的 Actor observation 为：

| 变量 | 维数 |
|---|---:|
| `projected_gravity` | 3 |
| `base_ang_vel` | 3 |
| `dof_pos_error` | 12 |
| `dof_vel` | 12 |
| `previous_action` | 12 |
| `phase_sin_cos` | 2 |
| `foothold_goal` | 10 |
| 合计 | 54 |

实际 checkpoint 第一层为 `Linear(in_features=54, out_features=256)`；goal 位于
拼接 observation 的 `[44:54]`。Actor 不包含世界位置、base linear velocity、
`commands`、真实接触或地形真值。

Critic observation 是上述 54 维再加：

| 特权变量 | 维数 |
|---|---:|
| `base_lin_vel_world` | 3 |
| `foot_contact_state` | 2 |
| `world_foot_target_error` | 6 |
| `foot_contact_force` | 6 |
| Critic 合计 | 71 |

实际 checkpoint Critic 第一层为 `Linear(in_features=71, out_features=256)`。
`learning/runners/on_policy_runner.py` 分别通过 `actor_obs` 和 `critic_obs` 构造
两个 tensor，PPO value function 确实读取 71 维输入，因此非对称 observation
不是只写在配置中，而是已经参与训练。

### 3.6 策略输出到 PD

当前策略输出不是绝对关节角，也不是 torque，而是 12 维默认姿态残差：

```text
Actor mean action
  -> OnPolicyRunner.set_actions()
  -> clip_actions = 4.0
  -> BaseTask.set_states(["dof_pos_target"], actions)
  -> Hicl12Controller._compute_torques()
  -> q_des = default_dof_pos + 0.25 * dof_pos_target
  -> tau = Kp*(q_des-q) + Kd*(0-qdot)
  -> clip(tau, +/- URDF torque_limits)
  -> Isaac Gym set_dof_actuation_force_tensor()
```

PD 函数仍沿用 Footstep 的 `Hicl12Controller` 实现。R1 配置只替换了 12-DOF
默认角、Kp/Kd、`actuation_scale=0.25`、URDF 和 torque limits；
`apply_humanoid_jacobian=False`，所以当前没有额外闭环耦合映射。

## 4. 速度命令当前扮演什么角色

需要区分训练与 play：

- 训练配置固定 `commands[:,0]=0.4 m/s`。
- Actor observation 没有 `commands`，所以 Actor 不会接收可变速度指令。
- 继承的 `reward_tracking_lin_vel_world()` 仍以 commands 计算速度奖励，权重为
  `1.5`；`base_heading` 等原奖励也仍在使用 commands。
- Goal Sampler 不读取 commands，而是独立使用 `movement_direction_deg=0` 和
  step-length distribution。
- `play.py` 把 commands 设为 0，但奖励在 inference 时不决定当前 action，
  Goal Sampler 仍产生正 x 目标，所以机器人继续前进。

因此训练时速度是固定辅助监督，play 时前进目标来自 Goal Sampler。当前策略不支持
通过键盘改变 `commands` 来连续调速；要调速，应改变 Goal Generator 的步长/周期，
或未来让上层 planner 根据用户速度命令生成 foothold goal。

## 5. 落脚奖励是否参与训练

以下新奖励均已由非零 config weight 注册并参与 PPO：

- `swing_foot_xy_tracking`
- `swing_foot_z_tracking`
- `swing_foot_yaw_tracking`
- `touchdown_accuracy`
- `stance_accuracy_latched`
- `phase_foot_clearance`
- `swing_knee_height`

Stage 3 进一步使用 smoothstep swing reference，并将 touchdown 定义为晚摆动阶段
首次接触 rising edge，每个 swing 最多记一次。训练日志实际输出了
`rew_swing_foot_xy_tracking`、`rew_touchdown_accuracy` 等字段；1000 轮确定性诊断
记录 `mean_touchdown_xy_error=0.1458 m`。

同时仍复用了原 Footstep/R1 的姿态、高度、速度、关节、torque、slip、collision
和 limit 类奖励。因此它是新落脚奖励与原稳定性奖励的组合，不是完全替换。

## 6. LIPM 和 IPC3D 当前状态

### 已写入但未启用

`LIPMFootholdAdapter` 已实现统一接口，并能接受一个 planner callback。callback
可返回 world 或 base frame 目标，adapter 会转换到 world，再锁存为 stance frame。

### 仅预留接口

`IPC3DFootholdAdapter` 当前只有：

```python
class IPC3DFootholdAdapter(LIPMFootholdAdapter):
    pass
```

它共享 callback contract，但没有把仓库现有 IPC3D controller 的真实输出接进来。

### 当前没有运行

- 新任务硬编码构造 `RandomFootholdGoalSampler`。
- config 没有 `goal_generator_type` 或 factory selection。
- 没有向 LIPM/IPC3D adapter 传入 planner callback。
- 新任务 callback 明确调用 `LeggedRobot._post_physics_step_callback()`，跳过
  `Hicl12Controller._post_physics_step_callback()` 中的旧 `_update_commands()`。
- `_calculate_ICP()` 仍被计算，但当前 Goal Sampler 和 Actor 不读取 ICP。

所以 LIPM/IPC3D 尚未成为真正的可插拔上层生成器。

## 7. 原地面方框的含义与消失原因

### 原方框是什么

原 `HumanoidController._visualization()` 调用：

- `_draw_step_vis()`：绘制 `current_step`
  - 右脚实际/当前 step：粉色
  - 左脚实际/当前 step：青色
- `_draw_step_command_vis()`：绘制 `step_commands`
  - 右脚目标：红色
  - 左脚目标：蓝色

形状由 `gym/envs/hi_cl_12/hi_cl12_utils.py::FootStepGeometry` 产生。它们不是
支撑多边形或 terrain truth；目标框是 `_generate_step_command_by_3DLIPM_XCoM()`
产生的 LIPM/XCoM step command，另外两个是接触脚更新得到的 current step。

### 当前为什么消失

直接原因有两层：

1. `R1FootholdTracker` 继承 `Hicl12Controller`，不是旧 `HumanoidController`；当前
   `Hicl12Controller._visualization()` 函数体只有 `...`，原 draw 调用全部注释。
2. 新任务有意绕过旧 `_update_commands()`，所以旧 `step_commands` 不再被 LIPM
   更新。即使重新打开旧 draw，也只会显示 stale/无效数据，而非新 foothold goal。

这与 `debug_viz` 开关无关。headless 确实不会调用任何 viewer visualization，
但用户的非 headless play 中方框消失的首要原因是继承到 no-op visualization。

### 内部目标是否存在

存在，而且同时保存：

- Actor 使用的 `foothold_goal[num_envs,10]`：stance-foot frame；
- reward/critic/visualization 使用的 `world_target_pos/world_target_yaw`：world frame。

## 8. 新增的当前目标可视化

本次审计已在 `R1FootholdTracker._visualization()` 增加专用绘制，并通过
`R1FootholdTrackerCfg.viewer.draw_foothold_targets=True` 启用。它直接读取 generator
锁存的 world targets，因此没有把局部 goal 当成世界坐标。

颜色说明：

- 左脚实际位置：青色 footprint；
- 右脚实际位置：粉色 footprint；
- 左脚 active target：蓝色 footprint；
- 右脚 active target：红色 footprint；
- 当前 stance foot：黄色球；
- 当前 swing foot：绿色球；
- swing foot 到 active target：橙色误差线；
- 最近首次 touchdown：白色球；
- 最近 touchdown 到其锁存目标：白色误差线。

这里的左右 target 中，一个是当前 stance pose，另一个才是当前 swing 的下一目标，
不是同时显示未来左右两步。运行 `--headless` 时 Isaac Gym 不创建 viewer，因此不会
显示这些线；普通 `play.py` viewer 会在每个 physics step 调用该函数。

## 9. checkpoint 消融证据

审计脚本：`gym/scripts/audit_foothold_checkpoint.py`。它加载同一个 checkpoint，
使用 192 个环境并行测试 normal、goal zero、short、long、narrow、wide 六组。
速度 commands 被明确设为 0，与 `play.py` 一致。

核心结果：

```text
actor_obs_dim                         54
critic_obs_dim                        71
goal_obs_slice                        [44:54]
commands_abs_max                      0.000000
frozen_goal_max_change                0.00000000
mean_goal_change_on_switch            0.215135
same_state_goal_zero_action_rms_delta 0.739545 rad

normal:    speed 0.4103, touchdown local x 0.1280, |y| 0.1956 m
goal_zero: speed 0.3004, touchdown local x 0.1174, |y| 0.1481 m
short x=.08: speed 0.3943, touchdown local x 0.1205 m
long  x=.16: speed 0.4712, touchdown local x 0.1432 m
narrow y=.18: touchdown |local y| 0.1909 m
wide   y=.22: touchdown |local y| 0.2001 m
```

解释：

- 同一状态只屏蔽 goal，Actor 输出平均变化 `0.74 rad RMS`，网络确实使用 goal。
- 长步目标使实际落脚 x 增加约 `2.27 cm`，速度增加约 `0.077 m/s`。
- 宽目标使实际横向落脚增加约 `0.92 cm`；方向正确，但响应增益偏低。
- goal zero 明显改变落脚和速度，但机器人没有立即失去行走能力，证明 phase、
  proprioception 和训练形成的 nominal gait prior 仍占重要比例。

因此证据支持“新架构已参与并影响控制”，不支持“策略完全依赖 foothold goal”。

## 10. 理想架构对照

| 模块 | 当前状态 | 说明 |
|---|---|---|
| 上层 foothold generator | 已运行 | 当前仅 Random Sampler |
| stance-foot-frame 10 维表示 | 已运行 | reset/switch 时转换并锁存 |
| 左右 phase 切换 | 已运行 | 固定 phase，不依赖接触 |
| swing 期间目标冻结 | 已运行 | GPU 验证 max change=0 |
| foothold goal 进入 Actor | 已运行 | observation `[44:54]` |
| 非对称 Actor/Critic | 已运行 | 54/71，PPO 分别读取 |
| foothold tracking rewards | 已运行 | 日志和训练均有记录 |
| joint position residual + PD | 已运行 | 12 维、scale .25、原 PD |
| LIPM adapter | 部分完成 | callback 转换可用，无真实 planner callback |
| IPC3D adapter | 接口占位 | 仅继承 LIPM adapter |
| generator 配置/factory | 未完成 | task 中硬编码 Random sampler |
| 任意速度/方向控制 | 未完成 | 训练固定向前，Actor 不读 command |
| terrain/platform footholds | 未完成 | 当前 z=0 plane |
| contact-adaptive phase | 未完成 | phase 完全按时间推进 |
| 多步目标 horizon | 未完成 | 只含当前 stance 和当前 swing target |
| 新目标 viewer | 已完成 | 本次增加 world-frame 绘制 |

## 11. 推荐的后续验证与开发顺序

1. 保留审计脚本作为每个 checkpoint 的 goal-conditioning 回归测试。
2. 扩大 step length/width/direction curriculum，并要求 long-short、wide-narrow 的
   实际触地差达到设定下限，避免 policy 忽略弱变化。
3. 增加 `goal_generator_type=random|lipm|ipc3d` factory；不要继续在 task 中硬编码。
4. 为现有 LIPM 输出写明确 callback，并验证 world/base frame contract。
5. 为 IPC3D 写独立 adapter，而不是空 subclass。
6. 将用户速度命令只交给上层 generator；低层 Actor 继续只看 foothold goal。
7. 在保持 first-touchdown event 的前提下做 XY precision curriculum。
8. 平地任意目标收敛后，再接 support platform/terrain height。

最终判断：**新的“目标生成--站立足坐标转换--落脚跟踪策略--关节位置残差--PD”
闭环已经真实建立并参与训练和 play；当前向前走同时受该闭环、固定速度训练奖励和
phase-driven nominal gait 三者影响。LIPM/IPC3D 上层可插拔能力仍未完成。**
