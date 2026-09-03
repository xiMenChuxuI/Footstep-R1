# R1 平地策略 MuJoCo Sim2Sim 实现记录

## 范围

第一阶段只迁移固定世界 X 方向的平地策略，不接入随机路线、复杂地形、DDS 或重新训练。

- Isaac Gym 任务：`r1_foothold_tracker`
- Checkpoint：`Jul15_12-06-24_r1_stage3_latched_event_trajectory_1000/model_1000.pt`
- Actor：54 维输入、12 维输出
- MuJoCo 模型：官方 `unitreerobotics/unitree_mujoco` R1 `scene.xml`
- 官方源码版本：`4134cb5dc7ff1ba7f484deda48b5274b58694519`
- 官方源码压缩包 SHA256：`b2bc4db456598771e65010e94faf27cb5413ce1a9ff158f203a153b25303f6cf`

## 控制链

```text
固定 0.4 m/s 前进命令
-> 0.12 m 步长、0.20 m 步宽的确定性落脚目标
-> 站立足坐标系 10 维目标
-> 54 维 Actor observation
-> 12 维关节残差动作
-> q_des = default_q + 0.25 * action
-> 原训练 Kp/Kd 的 torque PD
-> 官方 R1 MuJoCo 模型
```

`previous_action` 按原代码实际语义保存缩放后的关节残差，即 `0.25 * action`，不是未缩放的网络输出。

## 环境

MuJoCo 使用独立 Conda 环境，避免改动 Isaac Gym 的 Python 3.8、PyTorch 1.10 和 CUDA 11.3 环境。

```bash
conda activate r1_mujoco
python -m pip install -r sim2sim/requirements-mujoco.txt \
  -i https://mirrors.aliyun.com/pypi/simple/
```

实际源记录：Conda 使用清华 `pkgs/main`；清华 PyPI TLS 失败后按既定规则回退到阿里云 PyPI。APT 未修改。官方 `unitree_mujoco` 从 GitHub 官方 codeload 下载，未使用第三方代理。

## 策略导出

```bash
conda activate footstep
cd /root/work/Footstep-sim2sim

python -m sim2sim.mujoco.export_policy \
  --checkpoint /root/work/Footstep/logs/Unitree_R1_Foothold_Tracker/Jul15_12-06-24_r1_stage3_latched_event_trajectory_1000/model_1000.pt \
  --output-dir artifacts/sim2sim/r1_flat_model_1000
```

导出目录包含 TorchScript、纯 NumPy 权重和推理一致性结果。MuJoCo 默认使用 NumPy 权重，从而不依赖旧 PyTorch 运行时。

## 运行

无界面快速验证：

```bash
conda activate r1_mujoco
cd /root/work/Footstep-sim2sim
python -m sim2sim.mujoco.play_r1_flat --headless --duration 5 --stop-on-fall
```

打开 Viewer：

```bash
python -m sim2sim.mujoco.play_r1_flat --duration 60 --realtime
```

相机会自动跟随机器人。无界面录制视频：

```bash
python -m sim2sim.mujoco.play_r1_flat \
  --headless --duration 30 \
  --record artifacts/sim2sim/r1_flat_model_1000/r1_mujoco_walk.mp4
```

零动作 PD 站立对照：

```bash
python -m sim2sim.mujoco.play_r1_flat --zero-action --duration 10 --realtime
```

## 已知物理差异

官方模型包含 29 个执行器。策略只控制前 12 个腿部执行器，其余腰、臂、腕关节由独立 PD 保持零位。训练 URDF 使用 `collapse_fixed_joints=True`，因此上半身在 Isaac Gym 中等效为刚性连接；MuJoCo 适配器使用 `Kp=100/Kd=5` 的保持控制近似这一拓扑。官方 XML 的默认仿真步长为 2 ms，适配器覆盖为训练时的 1 ms，并每 10 个物理步执行一次策略。

参数诊断中，`Kp=30/Kd=2` 的上半身保持控制在约 7.59 秒后失稳；提高到 `Kp=100/Kd=5` 后，同一 checkpoint 在官方 R1 模型中连续运行 30 秒，前进约 14.04 m，未触发跌倒判定。地面摩擦在 0.3 至 1.0 的对照范围内不是该次失稳的主因。

最终 60 秒无界面回归运行完成 6000 个策略周期，前进 28.23 m，横向累计偏移 1.03 m，最终基座高度 0.683 m，未触发跌倒判定；世界 X 平均速度约 0.47 m/s。

## 验证结果

```text
策略导出 NumPy 最大绝对误差：4.29e-6（门槛 1e-5）
Sim2Sim 单元/集成测试：5/5 通过
原 foothold goal 回归测试：9/9 通过
5 秒闭环测试：前进 2.25 m，未跌倒
60 秒闭环测试：前进 28.23 m，未跌倒
10 秒录制测试：MP4 时长 10.0 秒，分辨率 1280x720
```

当前第一阶段已完成“同一 Isaac Gym checkpoint 在官方 R1 MuJoCo 模型中闭环行走”的目标。尚未覆盖随机落脚 checkpoint、随机路线、复杂地形以及 DDS/实机接口，这些内容应在固定方向版本稳定后逐项迁移。

零动作只表示保持默认关节角，并不是平衡策略，因此不能以长时间零动作站立作为该动态行走策略的必要验收项。零样本迁移若不能稳定行走，应优先检查关节方向、足底偏置、四元数/角速度坐标系、相位时序和接触参数，第一阶段不通过重新训练掩盖接口问题。
