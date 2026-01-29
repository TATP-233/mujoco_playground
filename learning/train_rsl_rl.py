# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=wrong-import-position
"""Train a PPO agent using RSL-RL for the specified environment."""

from datetime import datetime
import json
import os

from absl import app
from absl import flags
from absl import logging
import jax
import mediapy as media
from ml_collections import config_dict
import mujoco
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import mujoco_playground

print("MuJoCo Playground path:", mujoco_playground.__path__)
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import numpy as np
from rsl_rl.runners import OnPolicyRunner
import torch
import torch.nn as nn
from tensordict import TensorDict
import warp as wp
from actor_critic_cnn import ActorCriticCNN
import rsl_rl.modules

# Register the class in rsl_rl.modules so it can be found via full path
rsl_rl.modules.ActorCriticCNN = ActorCriticCNN

try:
  import wandb  # pylint: disable=g-import-not-at-top
except ImportError:
  wandb = None

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

# Suppress logs if you want
logging.set_verbosity(logging.WARNING)

# Define flags similar to the JAX script
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "BerkeleyHumanoidJoystickFlatTerrain",
    (
        "Name of the environment. One of: "
        f"{', '.join(mujoco_playground.registry.ALL_ENVS)}"
    ),
)
_LOAD_RUN_NAME = flags.DEFINE_string(
    "load_run_name", None, "Run name to load from (for checkpoint restoration)."
)
_CHECKPOINT_NUM = flags.DEFINE_integer(
    "checkpoint_num", -1, "Checkpoint number to load from."
)
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train."
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode).",
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name.")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed.")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 4096, "Number of parallel envs.")
_DEVICE = flags.DEFINE_string("device", "cuda:0", "Device for training.")
_MULTI_GPU = flags.DEFINE_boolean(
    "multi_gpu", False, "If true, use multi-GPU training (distributed)."
)
_CAMERA = flags.DEFINE_string(
    "camera", None, "Camera name to use for rendering."
)
_WP_KERNEL_CACHE_DIR = flags.DEFINE_string(
    "wp_kernel_cache_dir",
    "/tmp/wp_kernel_cache_playground",
    "Path to the WP kernel cache directory.",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input.")
_USE_DR = flags.DEFINE_boolean("use_dr", False, "Use domain randomization.")

def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in registry.manipulation._envs:
    return manipulation_params.rsl_rl_config(env_name)
  elif env_name in registry.locomotion._envs:
    return locomotion_params.rsl_rl_config(env_name)
  else:
    raise ValueError(f"No RL config for {env_name}")


def tile(img, d):
  """Tiles a batch of images into a single grid image."""
  # img shape: [N, H, W, C]
  n, h, w, c = img.shape
  if n < d * d:
    # Pad with zeros if we don't have enough images
    padding = np.zeros((d * d - n, h, w, c), dtype=img.dtype)
    img = np.concatenate([img, padding], axis=0)
  elif n > d * d:
    img = img[: d * d]
  img = img.reshape((d, d, h, w, c))
  # Swap axes to get [d*H, d*W, C]
  img = img.transpose(0, 2, 1, 3, 4).reshape(d * h, d * w, c)
  return img


def configure_3dgs(env_cfg: config_dict.ConfigDict, env_name: str, num_envs: int):
  env_cfg.vision = True
  env_cfg.vision_config.render_batch_size = num_envs
  env_cfg.vision_config.render_width = 64
  env_cfg.vision_config.render_height = 64
  
  from mujoco_playground._src import mjx_env
  from ml_collections import ConfigDict
  reso = "224"
  
  gaussians_name = {}
  if "Panda" in env_name:
    assets_name = "franka_emika_panda"
    bodies = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger"]
    if env_name == "PandaPickCubeCartesian":
      background_name = "ribbon.ply"
      gaussians_name["box"] = "red_cube.ply"
    elif env_name == "PandaPickCube":
      background_name = "ribbon_blue.ply"
      gaussians_name["box"] = "green_cube.ply"
  elif "AirbotPlay" in env_name:
    assets_name = "airbot_play"
    bodies = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "left", "right"]
    background_name = "ribbon_blue.ply"
    gaussians_name["box"] = "green_cube.ply"

  assets_path = mjx_env.ROOT_PATH / "manipulation" / assets_name / "3dgs"
  print(f"3DGS assets path: {assets_path.as_posix()}")
  body_gaussians = {b: (assets_path / reso / f"{b}.ply").as_posix() for b in bodies}
  
  env_cfg.vision_config.background = (assets_path / background_name).as_posix()
  for k, v in gaussians_name.items():
    body_gaussians[k] = (assets_path / v).as_posix()

  env_cfg.vision_config.body_gaussians = ConfigDict(body_gaussians)

def main(argv):
  """Run training and evaluation for the specified environment using RSL-RL."""
  del argv  # unused

  project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")

  wp.config.kernel_cache_dir = _WP_KERNEL_CACHE_DIR.value

  # Possibly parse the device for multi-GPU
  if _MULTI_GPU.value:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_rank = local_rank
    device = f"cuda:{local_rank}"
    print(f"Using multi-GPU: local_rank={local_rank}, device={device}")
  else:
    device = _DEVICE.value
    device_rank = int(device.split(":")[-1]) if "cuda" in device else 0

  # If play-only, use fewer envs
  if _PLAY_ONLY.value:    
      num_envs = min(64, _NUM_ENVS.value) if _VISION.value else 1
  else:
    num_envs = _NUM_ENVS.value

  # Load default config from registry
  env_cfg = registry.get_default_config(_ENV_NAME.value)

  if _VISION.value:
    configure_3dgs(env_cfg, _ENV_NAME.value, num_envs)
  print(f"Environment config:\n{env_cfg}")

  # Generate unique experiment name
  now = datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Logging directory
  if _PLAY_ONLY.value:
    logdir = os.path.abspath(os.path.join(project_path, "rslrl-eval-logs/", exp_name))
  else:
    logdir = os.path.abspath(os.path.join(project_path, "rslrl-training-logs/", exp_name))
  os.makedirs(logdir, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Checkpoint directory
  ckpt_path = os.path.join(logdir, "checkpoints")
  os.makedirs(ckpt_path, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value and wandb is not None:
    wandb.tensorboard.patch(root_logdir=logdir)
    wandb.init(project="mjxrl", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # Save environment config to JSON
  with open(
      os.path.join(ckpt_path, "config.json"), "w", encoding="utf-8"
  ) as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  # Domain randomization
  randomizer = registry.get_domain_randomizer(_ENV_NAME.value) if _USE_DR.value else None
  print(f"Using domain randomizer: {randomizer}")

  # We'll store environment states during rendering
  render_trajectory = []

  # Callback to gather states for rendering
  def render_callback(_, state):
    render_trajectory.append(state)

  # Create the environment
  raw_env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides={"impl": "jax"}
  )
  if _VISION.value:
    brax_env = wrapper_torch.BatchSplatWrapper(
        raw_env,
        num_envs,
        _SEED.value,
        env_cfg.episode_length,
        1,
        render_callback=render_callback,
        randomization_fn=randomizer,
        device_rank=device_rank,
    )
  else:
    brax_env = wrapper_torch.RSLRLBraxWrapper(
        raw_env,
        num_envs,
        _SEED.value,
        env_cfg.episode_length,
        1,
        render_callback=render_callback,
        randomization_fn=randomizer,
        device_rank=device_rank,
    )

  # Build RSL-RL config
  train_cfg = get_rl_config(_ENV_NAME.value)

  if _VISION.value:
    train_cfg.policy.class_name = "rsl_rl.modules.ActorCriticCNN"
    
    num_cameras = raw_env.mj_model.ncam
    pixel_views = [f"pixels/view_{i}" for i in range(num_cameras)]
    
    train_cfg.obs_groups = {
        "policy": ["state"] + pixel_views,
        "critic": ["state"] + pixel_views,
    }
  else:
    obs_size = raw_env.observation_size
    if isinstance(obs_size, dict):
      train_cfg.obs_groups = {"policy": ["state"], "critic": ["privileged_state"]}
    else:
      train_cfg.obs_groups = {"policy": ["state"], "critic": ["state"]}

  # Overwrite default config with flags
  train_cfg.seed = _SEED.value
  train_cfg.run_name = exp_name
  train_cfg.resume = _LOAD_RUN_NAME.value is not None
  train_cfg.load_run = _LOAD_RUN_NAME.value if _LOAD_RUN_NAME.value else "-1"
  train_cfg.checkpoint = _CHECKPOINT_NUM.value

  train_cfg_dict = train_cfg.to_dict()
  runner = OnPolicyRunner(brax_env, train_cfg_dict, logdir, device=device)

  # If resume, load from checkpoint
  if train_cfg.resume:
    resume_path = wrapper_torch.get_load_path(
        os.path.abspath(os.path.join(project_path, "rslrl-training-logs/")),
        load_run=train_cfg.load_run,
        checkpoint=train_cfg.checkpoint,
    )
    print(f"Loading model from checkpoint: {resume_path}")
    model_name = resume_path.split("/")[-1].split(".")[0]
    print(f"Model name: {model_name}")
    runner.load(resume_path)

  if not _PLAY_ONLY.value:
    # Perform training
    runner.learn(
        num_learning_iterations=train_cfg.max_iterations,
        init_at_random_ep_len=False,
    )
    print("Done training.")
    return

  # If just playing (no training)
  policy = runner.get_inference_policy(device=device)

  # Example: run a single rollout
  eval_env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides={"impl": "jax"}
  )
  base_eval_env = eval_env
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(_SEED.value)
  state = jit_reset(rng)
  rollout = [state]

  # We’ll assume your environment’s observation is in state.obs["state"].
  is_dict_obs = isinstance(eval_env.observation_size, dict)

  def get_obs_dict(state):
    if _VISION.value:
      # We need to render to get pixels for the policy
      # This is a bit slow for play_only but necessary if we want to use the vision policy
      # Alternatively, we could have wrapped eval_env with BatchSplatWrapper
      # For simplicity, we assume the user might want to see the video anyway
      # But wait, raw_env.render returns frames, not the obs pixels.
      # BatchSplatWrapper is the right way to get obs pixels.
      # Let's see if we can just use brax_env instead of eval_env for play_only.
      pass

    if is_dict_obs:
      return {k: wrapper_torch._jax_to_torch(v) for k, v in state.obs.items()}
    return {"state": wrapper_torch._jax_to_torch(state.obs)}

  # If vision, we MUST use a wrapper that provides pixels
  if _VISION.value:
    eval_env = wrapper_torch.BatchSplatWrapper(
        eval_env,
        num_envs,
        _SEED.value,
        env_cfg.episode_length,
        1,
        render_callback=None,
        randomization_fn=randomizer,
        device_rank=device_rank,
    )
    # BatchSplatWrapper's reset/step return TensorDict, not jax state
    obs_torch = eval_env.reset()
    rollout = [eval_env.env_state]
    
    # Capture initial frame with all views
    pixel_keys = sorted([k for k in obs_torch.keys() if k.startswith("pixels/view_")])
    def get_pixel_frame(obs_td):
      # Stack all views: [num_envs, num_cameras, H, W, 3]
      imgs = [obs_td[k].cpu().numpy() for k in pixel_keys]
      return np.stack(imgs, axis=1)
    
    pixel_frames = [get_pixel_frame(obs_torch)]
  else:
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    rng = jax.random.PRNGKey(_SEED.value)
    state = jit_reset(rng)
    rollout = [state]
    obs_torch = get_obs_dict(state)
    pixel_frames = []

  for _ in range(env_cfg.episode_length):
    with torch.no_grad():
      actions = policy(obs_torch)
      actions = torch.clip(actions, -1.0, 1.0)
    # Step environment
    if _VISION.value:
      obs_torch, reward, done, info = eval_env.step(actions)
      rollout.append(eval_env.env_state)
      pixel_frames.append(get_pixel_frame(obs_torch))
      if done.any():
        break
    else:
      state = jit_step(state, wrapper_torch._torch_to_jax(actions.flatten()))
      rollout.append(state)
      obs_torch = get_obs_dict(state)
      if state.done:
        break

  if not _VISION.value:
    reward_sum = sum(s.reward for s in rollout)
    print(f"Rollout reward: {reward_sum}")

  # Render
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  render_every = 2
  # If your environment is wrapped multiple times, adjust as needed:
  base_env = base_eval_env  # or brax_env.env.env.env
  fps = 1.0 / base_env.dt / render_every

  if _VISION.value:
    d = int(np.sqrt(num_envs))
    processed_frames = []
    for f in pixel_frames[::render_every]:
      # f shape: [num_envs, num_cameras, H, W, 3]
      # Concatenate all cameras horizontally for each environment: [num_envs, H, num_cameras * W, 3]
      f_combined = np.concatenate([f[:, i] for i in range(f.shape[1])], axis=2)
      # Tile these environment-wide horizontal strips into a grid: [d*H, d*(num_cameras*W), 3]
      processed_frames.append(tile(f_combined, d))
    frames = processed_frames
  else:
    traj = rollout[::render_every]
    frames = base_eval_env.render(
        traj,
        camera=_CAMERA.value,
        height=480,
        width=640,
        scene_option=scene_option,
    )
  video_dir = f"videos/{_LOAD_RUN_NAME.value}"
  os.makedirs(video_dir, exist_ok=True)
  video_name = f"{video_dir}/{_ENV_NAME.value}-{model_name}-rollout.mp4"
  media.write_video(video_name, frames, fps=fps)
  print(f"Rollout video saved to '{video_name}'.")


if __name__ == "__main__":
  app.run(main)
