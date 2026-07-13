# Unitree R1 Foothold Tracker 使用说明

## 1. 项目与环境位置

项目目录：

```text
/root/work/Footstep
```

Conda 环境：

```text
footstep
```

每次打开新的 SSH 或终端窗口后，先执行：

```bash
cd /root/work/Footstep
source /opt/conda/etc/profile.d/conda.sh
conda activate footstep
```

如果当前终端已经能够直接使用 `conda activate`，第二行可以省略。

检查是否进入正确环境：

```bash
which python
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

预期 Python 路径为 `/opt/conda/envs/footstep/bin/python`，PyTorch 为
`1.10.0+cu113`，CUDA 可用状态为 `True`。

## 2. 正式任务名称

后续训练统一使用：

```text
r1_foothold_tracker
```

它加载 Unitree R1 12 自由度模型。旧名称 `humanoid_foothold_tracker` 只是兼容
别名，不建议在新命令中继续使用。

## 3. 开始训练

### 3.1 首次受监控训练

建议先运行 1024 个环境、500 次迭代：

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

`--headless` 表示训练时不打开图形窗口，可以显著降低渲染开销。不要添加
`--disable_local_saving`，否则不会保存 checkpoint。

模型默认保存到：

```text
logs/Unitree_R1_Foothold_Tracker/<时间>_r1_flat_stance_frame/model_<迭代>.pt
```

### 3.2 使用 tmux 后台训练

远程 SSH 断开时，推荐使用 `tmux`：

```bash
tmux new -s r1_train
cd /root/work/Footstep
source /opt/conda/etc/profile.d/conda.sh
conda activate footstep

python gym/scripts/train.py \
  --task=r1_foothold_tracker \
  --num_envs=1024 \
  --max_iterations=500 \
  --headless \
  --disable_wandb
```

按 `Ctrl+B`，再按 `D`，即可退出 tmux 而不停止训练。

重新进入训练窗口：

```bash
tmux attach -t r1_train
```

查看 tmux 会话：

```bash
tmux ls
```

### 3.3 扩大训练规模

确认前 50、100、500 次迭代没有 NaN、CUDA 报错，并且 episode length 和
落脚奖励开始改善后，可以改为 4096 个环境：

```bash
python gym/scripts/train.py \
  --task=r1_foothold_tracker \
  --num_envs=4096 \
  --max_iterations=2000 \
  --headless \
  --disable_wandb
```

## 4. 查看训练状态

### 4.1 查看控制台指标

训练窗口会逐次输出：

- `Learning iteration`：当前迭代；
- `Mean reward`：平均奖励；
- `Mean episode length`：平均 episode 长度；
- `rew_swing_foot_xy_tracking` 等：各奖励项；
- `Computation`：训练速度；
- `ETA`：预计剩余时间。

当前阶段重点关注：

```text
Mean episode length
rew_swing_foot_xy_tracking
rew_swing_foot_z_tracking
rew_swing_foot_yaw_tracking
rew_touchdown_accuracy
rew_stance_accuracy_latched
rew_foot_slip
rew_termination
```

### 4.2 查看 GPU

在另一个终端运行：

```bash
watch -n 1 nvidia-smi
```

结束查看按 `Ctrl+C`。训练时应能看到 Python 进程、GPU 显存和利用率。

### 4.3 查看 checkpoint

列出所有 R1 foothold 训练目录：

```bash
find logs/Unitree_R1_Foothold_Tracker \
  -maxdepth 2 -type f -name 'model_*.pt' | sort
```

查看最新目录：

```bash
ls -lt logs/Unitree_R1_Foothold_Tracker
```

## 5. 继续训练

自动加载最近一次训练的最新 checkpoint：

```bash
python gym/scripts/train.py \
  --task=r1_foothold_tracker \
  --resume \
  --load_run=-1 \
  --checkpoint=-1 \
  --num_envs=1024 \
  --max_iterations=500 \
  --headless \
  --disable_wandb
```

指定训练目录和 checkpoint：

```bash
python gym/scripts/train.py \
  --task=r1_foothold_tracker \
  --resume \
  --load_run=Jul13_13-58-47_r1_flat_stance_frame \
  --checkpoint=500 \
  --num_envs=1024 \
  --max_iterations=500 \
  --headless \
  --disable_wandb
```

将示例目录名和 checkpoint 数字替换为实际存在的文件。

## 6. 打开图形界面查看策略

查看最新 checkpoint：

```bash
python gym/scripts/play.py --task=r1_foothold_tracker
```

该命令会打开 Isaac Gym 窗口，需要 Ubuntu 图形桌面或可用的 X11 显示。远程网页
桌面较卡时，推荐使用下一节的 headless 视频导出，不需要实时传输仿真窗口。

查看指定 checkpoint：

```bash
python gym/scripts/play.py \
  --task=r1_foothold_tracker \
  --load_run=Jul13_13-58-47_r1_flat_stance_frame \
  --checkpoint=500
```

## 7. 导出 2x2 视频

### 7.1 导出最新 checkpoint

下面的命令不打开桌面窗口，在服务器直接渲染四个 R1 视角并自动拼接 MP4：

```bash
python gym/scripts/play.py \
  --task=r1_foothold_tracker \
  --headless \
  --video_2x2 \
  --video_length=600 \
  --video_fps=50 \
  --video_robot_ids=0,1,2,3
```

默认每个子画面为 640x360，最终视频为 1280x720。600 帧、50 FPS 对应 12 秒。

视频保存到：

```text
logs/Unitree_R1_Foothold_Tracker/videos/
```

查看生成文件：

```bash
ls -lh logs/Unitree_R1_Foothold_Tracker/videos
```

### 7.2 指定 checkpoint 导出

```bash
python gym/scripts/play.py \
  --task=r1_foothold_tracker \
  --headless \
  --video_2x2 \
  --video_length=600 \
  --load_run=Jul13_13-58-47_r1_flat_stance_frame \
  --checkpoint=500
```

### 7.3 修改画面尺寸

每个子画面使用 960x540，最终输出 1920x1080：

```bash
python gym/scripts/play.py \
  --task=r1_foothold_tracker \
  --headless \
  --video_2x2 \
  --video_length=600 \
  --video_width=960 \
  --video_height=540
```

分辨率越高，GPU 渲染和视频编码开销越大。日常检查建议使用默认 1280x720。

### 7.4 2x2 视频实现方式

实现参考了 `Changxi-M/mcx-RL` 的多相机思路，但没有安装该仓库：

1. 为环境 0、1、2、3 分别创建 Isaac Gym camera sensor；
2. 每个 policy step 更新四个跟随相机的位置；
3. 从 GPU PhysX 场景读取四张 RGBA 图像；
4. 使用 OpenCV 横向、纵向拼接为 2x2；
5. 写入 MP4，并在 `video_length` 达到后自动关闭程序。

当前视频会在每个子画面显示环境编号、步态 phase 和摆动脚。

## 8. 停止任务

前台训练或播放按：

```text
Ctrl+C
```

tmux 中可以直接进入会话后按 `Ctrl+C`，或在确认会话名称后执行：

```bash
tmux kill-session -t r1_train
```

## 9. 当前测试 checkpoint 提醒

为验证视频链路，服务器上生成过一个只有 1 次 PPO 迭代的测试 checkpoint。它只能
证明加载和录制功能正常，不代表机器人已经学会落脚跟踪。正式观察训练效果时，
应选择后续 500 次以上受监控训练产生的 checkpoint。
