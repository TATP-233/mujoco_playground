"""Minimal real-robot inference script (mock I/O).

This script loads an RSL-RL ActorCritic checkpoint (e.g. model_*.pt), builds the
matching policy network (MLP or CNN for vision), and runs a simple inference
loop against a mock robot interface.

It is intended to be a minimal starting point for deploying policies trained
with [learning/train_rsl_rl.py](learning/train_rsl_rl.py).

At minimum, this supports vector-observation manipulation envs like
AirbotPlayPickCube (see mujoco_playground/_src/manipulation/airbot_play/pick.py).

You are expected to replace the mock robot methods with your real SDK calls.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def _add_repo_to_path() -> None:
    # Make this script runnable from either repo root or the learning/ dir.
    import sys

    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    learning_dir = here.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(learning_dir))


_add_repo_to_path()


from mujoco_playground import registry  # pylint: disable=wrong-import-position
from mujoco_playground.config import locomotion_params  # pylint: disable=wrong-import-position
from mujoco_playground.config import manipulation_params  # pylint: disable=wrong-import-position
from mujoco_playground._src import wrapper_torch  # pylint: disable=wrong-import-position

from rsl_rl.env import VecEnv  # pylint: disable=wrong-import-position
from rsl_rl.runners import OnPolicyRunner  # pylint: disable=wrong-import-position
from rsl_rl.modules import ActorCritic  # pylint: disable=wrong-import-position
from tensordict import TensorDict  # pylint: disable=wrong-import-position

from actor_critic_cnn import ActorCriticCNN  # pylint: disable=wrong-import-position

import rsl_rl.modules  # pylint: disable=wrong-import-position


from airdc.common.systems.basis import SystemMode, ActionConfigs
from airdc.common.systems.grouped import (
    GroupedComponentsSystem,
    GroupedComponentsSystemConfig,
    SystemSensorComponentGroupsConfig,
    AutoControlConfig,
    GroupsSendActionConfig,
)

from airbot_ie.robots.airbot_play import AIRBOTPlay, AIRBOTPlayConfig
# from airbot_ie.robots.airbot_play_mock import AIRBOTPlay, AIRBOTPlayConfig

from airdc.common.devices.cameras.v4l2 import V4L2Camera, V4L2CameraConfig
# from airdc.common.devices.cameras.mock import (
#     MockCamera as V4L2Camera,
#     MockCameraConfig as V4L2CameraConfig,
# )
from mcap_data_loader.utils.basic import DataStamped
from mcap_data_loader.utils.transformations import quaternion_matrix
from toolz import get
import cv2


# Register the class in rsl_rl.modules so it can be found via eval("rsl_rl.modules.ActorCriticCNN").
rsl_rl.modules.ActorCriticCNN = ActorCriticCNN


def _get_rl_config(env_name: str):
    if env_name in registry.manipulation._envs:
        return manipulation_params.rsl_rl_config(env_name)
    if env_name in registry.locomotion._envs:
        return locomotion_params.rsl_rl_config(env_name)
    raise ValueError(f"No RL config for env_name={env_name}")


@dataclass(frozen=True)
class EnvSpec:
    obs_size: int
    privileged_obs_size: Optional[int]
    action_size: int
    num_cameras: int


def _infer_env_spec(env_name: str, vision: bool) -> EnvSpec:
    env_cfg = registry.get_default_config(env_name)
    if hasattr(env_cfg, "vision"):
        env_cfg.vision = bool(vision)

    # For vision envs, keep the canonical 64x64 used by training.
    if vision and hasattr(env_cfg, "vision_config"):
        env_cfg.vision_config.render_batch_size = 1
        env_cfg.vision_config.render_width = 64
        env_cfg.vision_config.render_height = 64

    env = registry.load(env_name, config=env_cfg, config_overrides={"impl": "jax"})

    obs_size = env.observation_size
    if isinstance(obs_size, dict):
        obs_dim = int(obs_size["state"])
        privileged_dim = int(obs_size["privileged_state"])
    else:
        obs_dim = int(obs_size)
        privileged_dim = None

    num_cameras = int(getattr(env.mj_model, "ncam", 0)) if vision else 0
    return EnvSpec(
        obs_size=obs_dim,
        privileged_obs_size=privileged_dim,
        action_size=int(env.action_size),
        num_cameras=num_cameras,
    )


def _default_obs_groups(spec: EnvSpec, vision: bool) -> Dict[str, list[str]]:
    if vision:
        pixel_views = [f"pixels/view_{i}" for i in range(spec.num_cameras)]
        return {"policy": ["state"] + pixel_views, "critic": ["state"] + pixel_views}
    if spec.privileged_obs_size is not None:
        return {"policy": ["state"], "critic": ["privileged_state"]}
    return {"policy": ["state"], "critic": ["state"]}


def _expected_actor_input_dim(spec: EnvSpec, vision: bool) -> int:
    if not vision:
        return spec.obs_size
    # ActorCriticCNN uses cnn_output_size=16 per camera view.
    cnn_output_size = 16
    return spec.obs_size + spec.num_cameras * cnn_output_size


def _infer_actor_input_dim_from_checkpoint(ckpt: dict, checkpoint_path: Path) -> int:
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"Unexpected checkpoint format at {checkpoint_path}. "
            "Expected a dict with key 'model_state_dict'."
        )
    sd = ckpt["model_state_dict"]
    w = sd.get("actor.0.weight", None)
    if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
        raise ValueError(
            f"Cannot infer actor input dim from {checkpoint_path}: missing 'actor.0.weight'."
        )
    return int(w.shape[1])


class RobotVecEnv(VecEnv):
    """Minimal VecEnv for initializing OnPolicyRunner.

    OnPolicyRunner needs `get_observations()` during init to build the policy.
    We provide a single-environment (num_envs=1) TensorDict with the correct keys.
    """

    def __init__(self, spec: EnvSpec, vision: bool):
        super().__init__()
        self.num_envs = 1
        self.num_actions = int(spec.action_size)
        self.max_episode_length = 1
        self._spec = spec
        self._vision = bool(vision)

        # Used in training mode; set to a valid default.
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

    def get_observations(self) -> TensorDict:
        obs: Dict[str, torch.Tensor] = {
            "state": torch.zeros(
                (self.num_envs, self._spec.obs_size), dtype=torch.float32
            )
        }
        if self._spec.privileged_obs_size is not None:
            obs["privileged_state"] = torch.zeros(
                (self.num_envs, self._spec.privileged_obs_size), dtype=torch.float32
            )
        if self._vision:
            for i in range(self._spec.num_cameras):
                obs[f"pixels/view_{i}"] = torch.zeros(
                    (self.num_envs, 64, 64, 3), dtype=torch.uint8
                )
        return TensorDict(obs, batch_size=[self.num_envs])

    def reset(self) -> TensorDict:  # pragma: no cover
        return self.get_observations()

    def step(self, actions):  # pragma: no cover
        raise NotImplementedError("RobotVecEnv does not implement step().")


class RealRobotInterfaceMock:
    """Mock real-robot interface.

    Replace these methods with your real robot SDK.
    """

    def __init__(self, spec: EnvSpec, vision: bool):
        self._spec = spec
        self._vision = bool(vision)

        config = AIRBOTPlayConfig(port=50051)
        # config = "airdc/airbot_ie/configs/robots/airbot_play.yaml"
        robot = AIRBOTPlay(config)

        grouped = GroupedComponentsSystem(
            GroupedComponentsSystemConfig(
                components=SystemSensorComponentGroupsConfig(
                    instances=[
                        robot,
                        V4L2Camera(
                            V4L2CameraConfig(
                                camera_index="usb-0000:00:14.0-2",
                                width=640,
                                height=480,
                            )
                        ),
                        V4L2Camera(
                            V4L2CameraConfig(
                                camera_index="usb-0000:00:14.0-1",
                                width=640,
                                height=480,
                            )
                        ),
                    ],
                    names=["follow", "left_camera", "front_right_camera"],
                    groups=["/", "/", "/"],
                    roles=["l", "o", "o"],
                    # ignore_roles=["o"],
                ),
                auto_control=AutoControlConfig(groups=[]),
            )
        )
        assert grouped.configure(), "Failed to configure robot grouped system."
        self._grouped = grouped
        self._action = GroupsSendActionConfig(
            groups=["/"],
            action_values=[[]],
            modes=[SystemMode.RESETTING],
        )
        # np.random.uniform()
        """
            minval=jp.array([-0.0, -0.1, 0.03]),
            maxval=jp.array([0.0, 0.1, 0.08]),
        """
        self._target_pos = np.array([0.3, 0.0, 0.03]) + np.array([0.0, 0.0, 0.04])
        self._first_get = True

    def _send_action(self, action: np.ndarray, mode: SystemMode) -> None:
        self._action.modes[0] = mode
        action_list = action.tolist()
        action_list[-1] = action_list[-1] / 0.04 * 0.072
        self._action.action_values[0] = action_list
        self._grouped.send_action(self._action)

    def reset(self, action_abs: np.ndarray) -> None:
        self._ctrl = action_abs
        self._send_action(self._ctrl, SystemMode.RESETTING)
        self._action.modes[0] = SystemMode.SAMPLING

    def get_observation(self) -> Dict[str, np.ndarray]:
        # NOTE: For AirbotPlayPickCube (non-vision), this should be a 1D vector whose
        # length matches env.observation_size.
        obs_data = self._grouped.capture_observation()
        # print(obs_data.keys())
        # convert quat to rot mat
        # quat_keys = ("/follow/eef/pose/orientation",)
        # for k in quat_keys:
        #     obs_data[k]["data"] = quaternion_matrix(obs_data[k]["data"]).flatten()[3:9].tolist()
        eef_keys = ("/follow/eef/joint_state/position",)
        for k in eef_keys:
            value = obs_data[k]["data"][0] / 0.072 * 0.04
            obs_data[k]["data"] = [value] * 2
        _, robot_list = DataStamped.concatenate(
            get(
                [
                    "/follow/arm/joint_state/position",
                    "/follow/eef/joint_state/position",
                    # "/follow/arm/joint_state/velocity",
                    # "/follow/eef/joint_state/velocity",
                    "/follow/eef/pose/position",
                    # "/follow/eef/pose/orientation",
                ],
                obs_data,
            )
        )
        robot_list.extend(self._target_pos)
        # print(robot_list)
        # robot_list.extend(quaternion_matrix([0, 0, 0, 1]).flatten()[:6])
        robot_list.extend(self._ctrl - np.array(robot_list[: len(self._ctrl)]))
        if len(robot_list) != self._spec.obs_size:
            raise ValueError(
                "Robot state obs dim mismatch. "
                f"got={len(robot_list)} expected={self._spec.obs_size}. "
                "Update get_observation() to match the training-time state definition."
            )
        obs: Dict[str, np.ndarray] = {"state": np.array(robot_list, dtype=np.float32)}
        if self._spec.privileged_obs_size is not None:
            obs["privileged_state"] = np.zeros(
                (self._spec.privileged_obs_size,), dtype=np.float32
            )
        if self._vision:
            image_keys = [
                "/left_camera/color/image_raw",
                "/front_right_camera/color/image_raw",
            ]
            for i in range(self._spec.num_cameras):
                image: np.ndarray = obs_data[image_keys[i]]["data"]
                clipped = image[:, : image.shape[0]]
                assert clipped.shape == (480, 480, 3)
                resized = cv2.resize(clipped, (64, 64))
                obs[f"pixels/view_{i}"] = resized[:, :, ::-1].copy()  # BGR to RGB
                obs[image_keys[i]] = image
                if self._first_get:
                    cv2.imwrite(f"pixels/view_{i}.png", resized)
                # cv2.imshow(f"Camera view {i}", resized)
            cv2.waitKey(1)
        if self._first_get:
            self._first_get = False
        return obs

    def send_action(self, action: np.ndarray) -> Dict[str, Any]:
        # Replace with: send joint delta / cartesian delta / gripper cmd.
        # Here we return a mock feedback dict.
        self._ctrl += action * 0.02
        print(f"action: {(action * 0.02).tolist()}")
        print(f"ctrl: {self._ctrl.tolist()}")
        # input("Press Enter to continue...")
        self._send_action(self._ctrl, SystemMode.SAMPLING)
        return {"ok": True, "action_norm": float(np.linalg.norm(action))}

    def send_abs_action(self, action: np.ndarray) -> Dict[str, Any]:
        # Replace with: send joint position / cartesian position / gripper cmd.
        # Here we return a mock feedback dict.
        self._ctrl = action
        print(f"abs action: {action.tolist()}")
        # input("Press Enter to continue...")
        self._send_action(self._ctrl, SystemMode.SAMPLING)
        return {"ok": True, "action_norm": float(np.linalg.norm(action))}

def _obs_to_tensordict(obs: Dict[str, np.ndarray], device: torch.device) -> TensorDict:
    td: Dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        t = torch.from_numpy(v)
        # Ensure stable dtypes.
        if k.startswith("pixels/"):
            if t.dtype != torch.uint8:
                t = t.to(torch.uint8)
        else:
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        else:
            t = t.unsqueeze(0)
        td[k] = t.to(device)
    return TensorDict(td, batch_size=[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="AirbotPlayPickCube",
        help="Must be one of mujoco_playground.registry.ALL_ENVS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--vision", action="store_true", help="Force pixel inputs (CNN policy)."
    )
    parser.add_argument(
        "--auto_vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true (default), auto-detect whether checkpoint expects pixels.",
    )

    # Loading options (match training layout).
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a model_*.pt checkpoint. If omitted, uses --log_root/--load_run_name/--checkpoint_num.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="rslrl-training-logs",
        help="Root directory that contains run folders.",
    )
    parser.add_argument(
        "--load_run_name",
        type=str,
        default="-1",
        help="Run folder name under log_root (or -1 for last run).",
    )
    parser.add_argument(
        "--checkpoint_num",
        type=int,
        default=-1,
        help="Checkpoint number (model_{N}.pt). -1 picks the latest model_*.pt.",
    )

    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only build runner + load checkpoint, then exit (no robot I/O).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Determine checkpoint path.
    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)
    else:
        load_path = wrapper_torch.get_load_path(
            args.log_root, args.load_run_name, args.checkpoint_num
        )
        checkpoint_path = Path(load_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Checkpoint: {checkpoint_path}")

    ckpt = torch.load(
        checkpoint_path.as_posix(), map_location="cpu", weights_only=False
    )
    ckpt_input_dim = _infer_actor_input_dim_from_checkpoint(ckpt, checkpoint_path)

    model_state_dict = (
        ckpt.get("model_state_dict", {}) if isinstance(ckpt, dict) else {}
    )
    ckpt_has_encoder = any(
        k.startswith("encoder.") or k.startswith("critic_encoder.")
        for k in model_state_dict.keys()
    )
    print(
        f"Checkpoint policy: input_dim={ckpt_input_dim} has_encoder={ckpt_has_encoder}"
    )

    spec_no_vision = _infer_env_spec(args.env_name, vision=False)
    spec_vision = _infer_env_spec(args.env_name, vision=True)

    # Decide whether this checkpoint is vision-based.
    # Prefer checkpoint evidence (encoder weights) over current env sizes.
    if args.vision:
        use_vision = True
    elif args.auto_vision:
        use_vision = bool(ckpt_has_encoder)
    else:
        use_vision = False

    if use_vision and not ckpt_has_encoder:
        print(
            "[warn] --vision requested but checkpoint has no encoder weights; "
            "this is likely a non-vision checkpoint."
        )
    if (not use_vision) and ckpt_has_encoder:
        raise ValueError(
            "Checkpoint contains CNN encoder weights (vision policy), but script is in non-vision mode. "
            "Run with --vision or keep --auto_vision (default)."
        )

    # Start from current env-derived spec (for action size / camera count), then adapt
    # observation dimension to match the checkpoint.
    base_spec = spec_vision if use_vision else spec_no_vision

    if use_vision:
        if ckpt_has_encoder:
            # ActorCriticCNN concatenates [state] + sum_i encoder(pixels/view_i).
            cnn_output_size = 16
            inferred_state_dim = (
                ckpt_input_dim - base_spec.num_cameras * cnn_output_size
            )
            if inferred_state_dim <= 0:
                raise ValueError(
                    "Cannot infer state_dim for vision policy. "
                    f"ckpt_input_dim={ckpt_input_dim}, num_cameras={base_spec.num_cameras}, "
                    f"cnn_output_size={cnn_output_size}. "
                    "Check camera count / policy definition."
                )
            spec = EnvSpec(
                obs_size=int(inferred_state_dim),
                privileged_obs_size=base_spec.privileged_obs_size,
                action_size=base_spec.action_size,
                num_cameras=base_spec.num_cameras,
            )
        else:
            # Forced vision mode but checkpoint doesn't have encoder; treat input as plain state.
            spec = EnvSpec(
                obs_size=int(ckpt_input_dim),
                privileged_obs_size=base_spec.privileged_obs_size,
                action_size=base_spec.action_size,
                num_cameras=0,
            )
    else:
        # Non-vision policy input dim is the state dim.
        spec = EnvSpec(
            obs_size=int(ckpt_input_dim),
            privileged_obs_size=base_spec.privileged_obs_size,
            action_size=base_spec.action_size,
            num_cameras=0,
        )
    print(
        f"EnvSpec(vision={use_vision}): obs={spec.obs_size} privileged={spec.privileged_obs_size} "
        f"action={spec.action_size} cams={spec.num_cameras}"
    )

    # Build train config for OnPolicyRunner.
    train_cfg = _get_rl_config(args.env_name).to_dict()
    train_cfg["obs_groups"] = _default_obs_groups(spec, vision=use_vision)
    if use_vision:
        train_cfg["policy"]["class_name"] = "rsl_rl.modules.ActorCriticCNN"

    vecenv = RobotVecEnv(spec, vision=use_vision)
    runner = OnPolicyRunner(vecenv, train_cfg, log_dir=None, device=str(device))
    runner.load(checkpoint_path.as_posix(), load_optimizer=False, map_location="cpu")
    policy = runner.get_inference_policy(device=str(device))

    if args.dry_run:
        print("[dry_run] Runner initialized and checkpoint loaded.")
        return

    robot = RealRobotInterfaceMock(spec, vision=use_vision)
    robot.reset(np.array([0, -1.1466, 1.1161, 1.5815, -1.4836, 0.0, 0.04]))
    input("Robot reset complete. Press Enter to start inference loop...")
    dt = 1.0 / max(args.hz, 1e-6)
    for step in range(args.steps):
        obs_np = robot.get_observation()
        obs_td = _obs_to_tensordict(obs_np, device=device)
        with torch.inference_mode():
            action = policy(obs_td)
        action_np = action.squeeze(0).detach().cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0)

        feedback = robot.send_action(action_np)
        # feedback = None
        # print(f"Action sent: {action_np.tolist()}")
        # if step % 30 == 0:
        #     print(f"step={step} action[0:3]={action_np[:3]} feedback={feedback}")
        time.sleep(dt)


if __name__ == "__main__":
    main()
