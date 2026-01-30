from real_robot_inference_mock import RealRobotInterfaceMock, EnvSpec
import numpy as np
import cv2
from pathlib import Path


spec = EnvSpec(obs_size=21, privileged_obs_size=None, action_size=7, num_cameras=2)

robot = RealRobotInterfaceMock(spec, True)

kind = "states"
# kind = "actions"
replay_data: np.ndarray = np.load(
    f"videos/AirbotPlayPickCube-20260130-115226/AirbotPlayPickCube-model_2550-{kind}.npy"
)
print(f"Loaded action data with shape: {replay_data.shape}")
video_dir = Path("replayed_videos")
video_dir.mkdir(exist_ok=True)
for rollout in range(0, replay_data.shape[1]):
    robot.reset(np.array([0, -1.1466, 1.1161, 1.5815, -1.4836, 0.0, 0.04]))
    input(f"Robot reset complete for rollout {rollout}. Press Enter to start...")
    rollout_dir = video_dir / f"rollout_{rollout}"
    rollout_dir.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或 'MJPG', 'MP4V' 等
    out = {
        i: cv2.VideoWriter(str(rollout_dir / f"{i}.mp4"), fourcc, 20.0, (640, 480))
        for i in range(spec.num_cameras)
    }
    for step in range(replay_data.shape[0]):
        action = replay_data[step, rollout][:7]
        obs = robot.get_observation()
        i = 0
        for key, value in obs.items():
            if "camera" in key:
                out[i].write(value)
                i += 1
        if kind == "states":
            for _ in range(3):
                robot.send_abs_action(action)
        else:
            robot.send_action(action)
        if input(f"Step {step} complete. Press Enter to continue...") == "q":
            break
    print(f"Rollout {rollout} complete. Videos saved.")
    for i in range(spec.num_cameras):
        out[i].release()
