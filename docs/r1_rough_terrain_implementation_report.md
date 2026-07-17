# R1 落脚点跟踪复杂地形实现报告

日期：2026-07-15

## 1. 平地基线版本固化

在开始开发复杂地形功能之前，已对 Stage 3 平地版本执行完整回归测试并进行 Git 固化。

- Commit：`2f885c929895d86dc13a5f5c5b5e3b68b503dadb`
- Commit 说明：`feat: complete R1 foothold stage 3 and visualization`
- Tag：`r1-foothold-stage3-flat`
- 保留的平地 checkpoint：`logs/Unitree_R1_Foothold_Tracker/Jul15_12-06-24_r1_stage3_latched_event_trajectory_1000/model_1000.pt`
- checkpoint 继续由 `.gitignore` 排除，没有纳入 Git。

提交前，平地基线通过了 9 项单元测试、Python 语法编译检查以及 `git diff --check` 检查。

## 2. 复杂地形版本实现

原平地任务 `r1_foothold_tracker` 的配置和行为保持不变。复杂地形以新任务形式实现：

```text
r1_foothold_tracker_rough
```

### 2.1 任务与配置

- `gym/envs/humanoid/humanoid_foothold_tracker_rough.py`
  - `R1RoughFootholdTracker` 继承自 `R1FootholdTracker`。
  - 将原 Goal Generator 替换为 `TerrainAwareFootholdGoalSampler`。
  - 基座高度改为相对于当前站立足高度计算。
  - 摆动足离地高度奖励使用地形感知的三维摆动轨迹。
  - 跌倒终止条件使用相对于站立足的基座高度，避免下楼梯时因世界坐标 Z 降低而被错误终止。
- `gym/envs/humanoid/humanoid_foothold_tracker_rough_config.py`
  - 启用 `trimesh`、地形高度测量和 curriculum。
  - 使用 10 行地形难度和 20 列地形类型。
  - 初始地形等级为 0，Stage 1 最大难度为 0.45。
  - 地形比例：坡面 25%、随机粗糙地面 35%、下楼梯 15%、上楼梯 20%、低障碍 5%。
- `gym/envs/__init__.py`
  - 注册新任务 `r1_foothold_tracker_rough`。
  - 原平地任务的注册保持不变。

### 2.2 地形感知落脚目标

`gym/envs/guidance/foothold_goal_sampler.py` 中的 `TerrainAwareFootholdGoalSampler` 使用批量 Tensor 运算完成候选目标生成，不按环境执行 Python 循环。

每次步态相位切换时执行以下流程：

1. 批量采样 8 个可达落脚候选点。
2. 应用原有左右脚可达范围和防止双腿交叉约束。
3. 查询候选点 XY 坐标对应的地形高度。
4. 使用长宽半尺寸为 0.06 m 和 0.035 m 的足底区域，查询中心点及足底四角高度。
5. 拒绝足底高度差超过 0.035 m，或相对站立足高度变化超过 0.12 m 的候选点。
6. 选择第一个有效候选点；如果没有有效候选点，则进入 hold-still 状态。
7. 将地形表面高度写入世界坐标目标的 Z 分量。
8. 将左右脚目标转换为原有的 10 维站立足坐标系表示。

落脚目标仍然只在步态相位切换时更新，并在整个摆动阶段保持冻结。Actor 和 Critic 的观测维度继续保持为 54 和 71，因此可以直接加载平地 checkpoint，不需要修改网络结构。

### 2.3 地形生成与目标可视化

- `gym/utils/terrain.py` 增加渐进式粗糙地形、指定方向的上下楼梯和低障碍地形。
- 选定楼梯地形的名称按照机器人从出生平台向外行走的方向定义：`stairs_up` 表示上楼梯，`stairs_down` 表示下楼梯。
- `gym/envs/hi_cl_12/hi_cl12_utils.py` 支持使用 XYZ 三维坐标绘制落脚点标记。
- `gym/envs/humanoid/humanoid_foothold_tracker.py` 使用真实世界坐标 Z 绘制当前足部位置和目标落脚点。
- `gym/scripts/play.py` 和 `gym/utils/helpers.py` 增加 `--terrain_mode=task`、地形等级、指定地形类型和地形难度参数。

## 3. 测试与验证

### 3.1 单元测试和静态检查

执行命令：

```bash
conda run -n footstep python -m unittest gym.tests.test_foothold_goal -v
```

测试结果：13/13 全部通过。

新增测试覆盖以下内容：

- 高度场批量查询；
- 地形目标 Z 高度；
- 无有效足底支撑面时的 hold 行为；
- Actor 54 维、Critic 71 维的策略接口兼容性。

所有相关 Python 文件均通过编译检查，`git diff --check` 通过。64 环境 GPU 冒烟测试和计划配置下的 1024 环境两轮 GPU 冒烟测试均通过，没有出现 NaN、CUDA 越界或 Tensor 维度错误。

## 4. 训练过程与结果

复杂地形策略从已固化的平地 `model_1000.pt` 热启动：

```bash
python gym/scripts/train.py \
  --task=r1_foothold_tracker_rough \
  --resume \
  --load_run=Jul15_12-06-24_r1_stage3_latched_event_trajectory_1000 \
  --checkpoint=1000 \
  --num_envs=1024 \
  --max_iterations=1000 \
  --run_name=r1_rough_curriculum_stage1_1000 \
  --headless \
  --disable_wandb
```

训练结果：

- Run：`Jul15_15-19-20_r1_rough_curriculum_stage1_1000`
- 最终 checkpoint：`model_2000.pt`
- 本阶段新增训练步数：24,576,000
- 训练耗时：约 1104 秒
- 最终平均奖励：236.59
- 最终平均 episode 长度：1001.00
- 最终 termination 奖励：0.0000
- 最终 touchdown accuracy 奖励：0.8914
- 最终 stance contact 奖励：3.9158
- 最终 swing-foot XY tracking 奖励：3.6550

完整 1000 轮训练顺利结束，没有出现 NaN、CUDA 或 Tensor 维度错误。

## 5. 固定地形评估

每项评估均使用 64 个环境、0.3 地形难度、20 个预热策略步和 600 个正式测量策略步。按约 0.4 m/s 的行走速度计算，该测试距离足以离开宽度为 3 m 的中心平台并实际进入坡面、楼梯或障碍区域。

早期执行的 200 步评估不足以稳定到达楼梯和坡面，因此相关结果已经作废，没有用于本报告结论。

| 地形 | 前进速度（m/s） | 重置次数 | 首次触地 XY 误差（m） | 接触足滑移速度（m/s） | 站立足接触率 |
|---|---:|---:|---:|---:|---:|
| 随机粗糙地面 | 0.3887 | 0 | 0.0674 | 0.0210 | 0.9830 |
| 坡面 | 0.4362 | 0 | 0.0694 | 0.0194 | 0.9889 |
| 上楼梯 | 0.2620 | 49 | 0.1176 | 0.0380 | 0.9776 |
| 下楼梯 | 0.3700 | 4 | 0.1032 | 0.0493 | 0.9730 |
| 低障碍 | 0.2974 | 5 | 0.0873 | 0.0466 | 0.9790 |

五类地形的目标高度与采样地形表面高度之间的平均误差均为 `0.000000 m`。这证明非 hold 状态下的目标 Z 确实来自实际地形高度查询，而不是固定平地高度。

使用 `model_2000.pt` 对原平地任务进行兼容性评估，得到以下结果：

- 世界坐标前进速度：0.4086 m/s；
- 站立足接触率：0.9920；
- 首次触地 XY 误差：0.1365 m；
- 12,800 个采样中发生 2 次重置。

这说明复杂地形续训后，策略仍保留了基本的平地行走能力。

## 6. 使用方法

### 6.1 查看课程混合地形效果

```bash
conda activate footstep
cd /root/work/Footstep
python gym/scripts/play.py \
  --task=r1_foothold_tracker_rough \
  --resume \
  --load_run=Jul15_15-19-20_r1_rough_curriculum_stage1_1000 \
  --checkpoint=2000 \
  --terrain_mode=task
```

### 6.2 查看指定难度的上楼梯效果

```bash
conda activate footstep
cd /root/work/Footstep
python gym/scripts/play.py \
  --task=r1_foothold_tracker_rough \
  --resume \
  --load_run=Jul15_15-19-20_r1_rough_curriculum_stage1_1000 \
  --checkpoint=2000 \
  --terrain_mode=task \
  --terrain_type=stairs_up \
  --terrain_difficulty=0.3
```

`play.py` 默认只创建 1 台机器人，以获得清晰的交互式演示画面。需要同时查看多台机器人时，可以显式传入环境数量：

```bash
python gym/scripts/play.py \
  --task=r1_foothold_tracker_rough \
  --resume \
  --load_run=Jul15_15-19-20_r1_rough_curriculum_stage1_1000 \
  --checkpoint=2000 \
  --terrain_mode=task \
  --terrain_type=stairs_up \
  --terrain_difficulty=0.3 \
  --num_envs=4
```

在指定地形的多机器人模式下，程序会创建足够数量的地形单元，并为每台机器人确定性分配不同的出生原点，不再让多台机器人重叠出生。使用 `--video_2x2` 时至少创建 4 个独立环境，每个视频画面对应一台机器人。

当环境数量大于 1 时，`play.py` 还会自动启用多机器人演示随机化：

- 在各自地形单元中心附近随机偏移出生位置；
- 检查所有机器人世界坐标 XY 间距，默认最小间距为 1.5 m；
- 在默认正负 20 度范围内分层随机采样不同的初始朝向和运动路线；
- 在 0.32 到 0.48 m/s 范围内分层随机采样不同目标速度；
- 根据速度同步调整每个环境的落脚步长倍率；
- 路线方向和步长倍率在 episode reset 后继续保持；
- 多机器人相机自动跟随机器人群体中心并根据群体范围调整观察距离。

可以使用以下参数调整随机化范围：

```text
--multi_spawn_jitter=0.5
--multi_min_spacing=1.5
--multi_heading_range=20
--multi_speed_min=0.32
--multi_speed_max=0.48
```

如需复现旧的同方向演示行为，可添加 `--disable_multi_randomization`。

世界坐标目标标记使用查询得到的地形 Z，因此落脚点方框会显示在真实地形表面，而不是统一显示在平地高度。

指定地形配置和 checkpoint 已通过无界面诊断程序完整运行。受控的无界面 `play.py` 测试成功进入 GPU PhysX，但在 60 秒限制内没有完成 viewer/camera 初始化。因此，交互式画面仍需要在服务器图形桌面中进行最终确认；测试结束后没有遗留后台 `play.py` 进程。

## 7. 当前结论与后续工作

Stage 1 复杂地形策略已经可以在随机粗糙地面和坡面上稳定行走，并具备一定的低障碍和下楼梯能力，可以作为下一阶段训练的基础版本。

但当前版本还不能视为完整的楼梯策略。难度 0.3 下的连续上楼梯仍是主要失败场景，64 个环境在 600 个测量策略步中累计发生 49 次重置。

下一阶段建议按以下顺序继续优化：

1. 逐步提高 curriculum 中上楼梯地形的采样比例；
2. 降低初始楼梯高度，然后通过 curriculum 逐渐增加；
3. 检查楼梯边缘附近的 hold-still 比例和候选点拒绝原因；
4. 在确认目标采样问题后，再决定是否调整奖励权重；
5. 上楼梯稳定后，再考虑加入沟壑、深坑、踏石和移动平台。

沟壑、深坑、踏石和移动平台目前仍按计划保持未启用状态。
