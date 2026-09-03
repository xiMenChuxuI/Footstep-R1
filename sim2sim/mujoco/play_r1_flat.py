"""Run the fixed-direction R1 foothold policy in official unitree_mujoco."""

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from sim2sim.mujoco.adapter import R1MujocoAdapter
from sim2sim.mujoco.math_utils import yaw_quat
from sim2sim.mujoco.policy import NumpyMLPPolicy


class VideoRecorder:
    def __init__(self, mujoco, model, path, width, height, fps):
        self.mujoco = mujoco
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = 2.4
        self.camera.azimuth = 140.0
        self.camera.elevation = -18.0
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, data):
        self.camera.lookat[:] = data.qpos[:3]
        self.renderer.update_scene(data, camera=self.camera)
        self.process.stdin.write(self.renderer.render().tobytes())

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        self.renderer.close()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")


def add_target_geoms(viewer, adapter):
    mujoco = adapter.mujoco
    scene = viewer.user_scn
    scene.ngeom = 0
    colors = (np.array([0.1, 0.4, 1.0, 0.75]), np.array([1.0, 0.1, 0.25, 0.75]))
    for index, (position, yaw, color) in enumerate(
        zip(
            adapter.goal_generator.world_target_pos,
            adapter.goal_generator.world_target_yaw,
            colors,
        )
    ):
        if index >= scene.maxgeom:
            break
        size = np.array([0.08, 0.04, 0.006])
        quaternion = yaw_quat(yaw)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        mujoco.mjv_initGeom(
            scene.geoms[index],
            mujoco.mjtGeom.mjGEOM_BOX,
            size,
            np.asarray(position) + np.array([0.0, 0.0, 0.008]),
            rotation,
            color,
        )
        scene.ngeom += 1


def run(args):
    if args.record and args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scene_path = Path(args.scene or config["robot"]["official_scene"])
    adapter = R1MujocoAdapter(scene_path, config)
    policy = None if args.zero_action else NumpyMLPPolicy.load(args.policy)
    duration = args.duration or config["simulation"]["duration"]
    steps = int(round(duration / config["simulation"]["policy_dt"]))
    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(adapter.model, adapter.data)
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -18.0
    recorder = None
    if args.record:
        recorder = VideoRecorder(
            adapter.mujoco, adapter.model, args.record,
            args.video_width, args.video_height, args.video_fps,
        )
    initial_position = adapter.data.qpos[:3].copy()
    video_accumulator = 0.0

    started = time.perf_counter()
    max_action = 0.0
    completed = 0
    try:
        for step in range(steps):
            tick = time.perf_counter()
            observation = adapter.observation()
            action = np.zeros(12, dtype=np.float32) if policy is None else policy(observation)
            max_action = max(max_action, float(np.max(np.abs(action))))
            adapter.apply_action(action)
            completed = step + 1
            if viewer is not None:
                viewer.cam.lookat[:] = adapter.data.qpos[:3]
                add_target_geoms(viewer, adapter)
                viewer.sync()
                if not viewer.is_running():
                    break
            if recorder is not None:
                video_accumulator += config["simulation"]["policy_dt"]
                if video_accumulator + 1.0e-12 >= 1.0 / args.video_fps:
                    recorder.write(adapter.data)
                    video_accumulator -= 1.0 / args.video_fps
            if args.realtime:
                remaining = config["simulation"]["policy_dt"] - (time.perf_counter() - tick)
                if remaining > 0.0:
                    time.sleep(remaining)
            if adapter.has_fallen and args.stop_on_fall:
                break
    finally:
        if viewer is not None:
            viewer.close()
        if recorder is not None:
            recorder.close()
    elapsed = time.perf_counter() - started
    displacement = adapter.data.qpos[:3] - initial_position
    sim_time = max(adapter.data.time, 1.0e-9)
    print(
        "sim2sim_result "
        f"steps={completed} sim_time={adapter.data.time:.3f} "
        f"wall_time={elapsed:.3f} base_xyz={adapter.data.qpos[:3].round(4).tolist()} "
        f"displacement_xy={displacement[:2].round(4).tolist()} "
        f"mean_velocity_xy={(displacement[:2] / sim_time).round(4).tolist()} "
        f"fallen={adapter.has_fallen} max_abs_raw_action={max_action:.4f}"
    )
    return 1 if adapter.has_fallen and args.fail_on_fall else 0


def main():
    default_config = Path(__file__).with_name("config") / "r1_flat_model_1000.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--scene", help="Override the official unitree_mujoco R1 scene")
    parser.add_argument(
        "--policy",
        default="artifacts/sim2sim/r1_flat_model_1000/policy_weights.npz",
    )
    parser.add_argument("--duration", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--stop-on-fall", action="store_true")
    parser.add_argument("--fail-on-fall", action="store_true")
    parser.add_argument("--record", help="Write a camera-following MP4 file")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-fps", type=int, default=30)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
