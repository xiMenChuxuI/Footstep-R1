"""Isaac Gym camera recorder producing a four-view 2x2 MP4."""

import os
from datetime import datetime

import cv2
import numpy as np
from isaacgym import gymapi

from gym import LEGGED_GYM_ROOT_DIR


class GridVideoRecorder:
    def __init__(
        self,
        env,
        experiment_name,
        robot_ids,
        width=640,
        height=360,
        fps=50,
        run_name="play",
    ):
        if len(robot_ids) != 4:
            raise ValueError("2x2 video requires exactly four robot ids")
        if any(robot_id < 0 or robot_id >= env.num_envs for robot_id in robot_ids):
            raise ValueError(
                f"Robot ids {robot_ids} must be within [0, {env.num_envs - 1}]"
            )

        self.env = env
        self.robot_ids = robot_ids
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_offset = np.array([1.6, -1.6, 1.0], dtype=np.float32)
        self.lookat_offset = np.array([0.0, 0.0, 0.35], dtype=np.float32)

        properties = gymapi.CameraProperties()
        properties.width = width
        properties.height = height
        properties.horizontal_fov = 60.0
        self.camera_handles = [
            env.gym.create_camera_sensor(env.envs[robot_id], properties)
            for robot_id in robot_ids
        ]

        video_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs", experiment_name, "videos"
        )
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_run_name = run_name or "play"
        self.video_path = os.path.join(
            video_dir, f"{timestamp}_2x2_{safe_run_name}.mp4"
        )
        self.writer = cv2.VideoWriter(
            self.video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width * 2, height * 2),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {self.video_path}")
        self.frame_count = 0
        print(f"2x2 video output: {self.video_path}")

    def _set_camera_locations(self):
        base_positions = self.env.base_pos[self.robot_ids].detach().cpu().numpy()
        for robot_id, camera_handle, base_position in zip(
            self.robot_ids, self.camera_handles, base_positions
        ):
            position = base_position + self.camera_offset
            target = base_position + self.lookat_offset
            self.env.gym.set_camera_location(
                camera_handle,
                self.env.envs[robot_id],
                gymapi.Vec3(*position),
                gymapi.Vec3(*target),
            )

    def _capture(self, robot_id, camera_handle):
        image = self.env.gym.get_camera_image(
            self.env.sim,
            self.env.envs[robot_id],
            camera_handle,
            gymapi.IMAGE_COLOR,
        )
        image = np.asarray(image).reshape(self.height, self.width, 4)
        frame = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        lines = [f"R1 env {robot_id}"]
        if hasattr(self.env, "phase"):
            lines.append(f"phase {self.env.phase[robot_id, 0].item():.2f}")
        if hasattr(self.env, "swing_foot_index"):
            swing = "left" if self.env.swing_foot_index[robot_id].item() == 0 else "right"
            lines.append(f"swing {swing}")
        for row, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (16, 30 + row * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame

    def record_frame(self):
        self._set_camera_locations()
        if self.env.device != "cpu":
            self.env.gym.fetch_results(self.env.sim, True)
        self.env.gym.step_graphics(self.env.sim)
        self.env.gym.render_all_camera_sensors(self.env.sim)
        frames = [
            self._capture(robot_id, camera_handle)
            for robot_id, camera_handle in zip(self.robot_ids, self.camera_handles)
        ]
        top = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], frames[3]))
        self.writer.write(np.vstack((top, bottom)))
        self.frame_count += 1

    def close(self):
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        print(f"Saved {self.frame_count} frames to: {self.video_path}")
