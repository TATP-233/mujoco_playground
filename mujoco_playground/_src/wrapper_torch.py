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
"""Wrappers for MuJoCo Playground environments that interop with torch."""

from collections import deque
import functools
import os
import tempfile
from typing import Any

import jax
import mujoco
import numpy as np

try:
  from rsl_rl.env import VecEnv  # pytype: disable=import-error
except ImportError:
  VecEnv = object
try:
  import torch  # pytype: disable=import-error
except ImportError:
  torch = None

from mujoco_playground._src import wrapper
try:
  from tensordict import TensorDict  # pytype: disable=import-error
except ImportError:
  TensorDict = None

import torch
import torch.utils.dlpack as tpack
from etils import epath
from gaussian_renderer import BatchSplatConfig, BatchSplatRenderer, MjxBatchSplatRenderer


def _jax_to_torch(tensor):
  import torch.utils.dlpack as tpack  # pytype: disable=import-error # pylint: disable=import-outside-toplevel

  tensor = tpack.from_dlpack(tensor)
  return tensor


def _torch_to_jax(tensor):
  from jax.dlpack import from_dlpack  # pylint: disable=import-outside-toplevel

  tensor = from_dlpack(tensor)
  return tensor


def get_load_path(root, load_run=-1, checkpoint=-1):
  try:
    runs = os.listdir(root)
    # TODO sort by date to handle change of month
    runs.sort()
    if "exported" in runs:
      runs.remove("exported")
    last_run = os.path.join(root, runs[-1])
  except Exception as exc:
    raise ValueError("No runs in this directory: " + root) from exc
  if load_run == -1 or load_run == "-1":
    load_run = last_run
  else:
    load_run = os.path.join(root, load_run)

  if checkpoint == -1:
    models = [file for file in os.listdir(load_run) if "model" in file]
    models.sort(key=lambda m: m.zfill(15))
    model = models[-1]
  else:
    model = f"model_{checkpoint}.pt"

  load_path = os.path.join(load_run, model)
  return load_path


class RSLRLBraxWrapper(VecEnv):
  """Wrapper for Brax environments that interop with torch."""

  def __init__(
      self,
      env,
      num_actors,
      seed,
      episode_length,
      action_repeat,
      randomization_fn=None,
      render_callback=None,
      device_rank=None,
      full_reset: bool | None = None,
  ):
    import torch  # pytype: disable=import-error # pylint: disable=redefined-outer-name,unused-import,import-outside-toplevel

    self.seed = seed
    self.batch_size = num_actors
    self.num_envs = num_actors

    self.key = jax.random.PRNGKey(self.seed)

    if device_rank is not None:
      gpu_devices = jax.devices("gpu")
      self.key = jax.device_put(self.key, gpu_devices[device_rank])
      self.device = f"cuda:{device_rank}"
      print(f"Device -- {gpu_devices[device_rank]}")
      print(f"Key device -- {self.key.devices()}")

    # split key into two for reset and randomization
    key_reset, key_randomization = jax.random.split(self.key)

    self.key_reset = jax.random.split(key_reset, self.batch_size)

    if randomization_fn is not None:
      randomization_rng = jax.random.split(key_randomization, self.batch_size)
      v_randomization_fn = functools.partial(
          randomization_fn, rng=randomization_rng
      )
    else:
      v_randomization_fn = None

    # NOTE on memory: BraxAutoResetWrapper(full_reset=True) computes env.reset
    # for the entire batch on every step, then selects reset_state for done envs.
    # That can substantially increase peak JAX GPU memory (often enough to OOM).
    # So we only enable full_reset by default when it's actually needed.
    if full_reset is None:
      needs_full_reset = False
      try:
        c = env.unwrapped._config
        if hasattr(c, 'vision_config'):
          # dynamic_bg relies on per-episode camera randomization being applied
          # on auto-reset for done envs.
          needs_full_reset = bool(getattr(c.vision_config, 'dynamic_bg', False))
          # Allow explicit override via config.
          if getattr(c.vision_config, 'full_reset', None) is not None:
            needs_full_reset = bool(c.vision_config.full_reset)
      except Exception:
        needs_full_reset = False

      full_reset = bool(randomization_fn is not None and needs_full_reset)

    self.env = wrapper.wrap_for_brax_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
        full_reset=full_reset,
    )

    self.render_callback = render_callback

    self.asymmetric_obs = False
    obs_shape = self.env.env.unwrapped.observation_size
    print(f"obs_shape: {obs_shape}")

    if isinstance(obs_shape, dict):
      print("Asymmetric observation space")
      self.asymmetric_obs = True
      self.num_obs = obs_shape["state"]
      self.num_privileged_obs = obs_shape["privileged_state"]
    else:
      self.num_obs = obs_shape
      self.num_privileged_obs = None

    self.num_actions = self.env.env.unwrapped.action_size

    self.max_episode_length = episode_length

    # todo -- specific to leap environment
    self.success_queue = deque(maxlen=100)

    print("JITing reset and step")
    self.reset_fn = jax.jit(self.env.reset)
    self.step_fn = jax.jit(self.env.step)
    print("Done JITing reset and step")
    self.env_state = None

  def step(self, action):
    action = torch.clip(action, -1.0, 1.0)  # pytype: disable=attribute-error
    action = _torch_to_jax(action)
    self.env_state = self.step_fn(self.env_state, action)
    critic_obs = None
    if self.asymmetric_obs:
      obs = _jax_to_torch(self.env_state.obs["state"])
      critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
      obs = {"state": obs, "privileged_state": critic_obs}
    else:
      obs = _jax_to_torch(self.env_state.obs)
      obs = {"state": obs}
    reward = _jax_to_torch(self.env_state.reward)
    done = _jax_to_torch(self.env_state.done)
    info = self.env_state.info
    truncation = _jax_to_torch(info["truncation"])

    info_ret = {
        "time_outs": truncation,
        "observations": {"critic": critic_obs},
        "log": {},
    }

    if "last_episode_success_count" in info:
      last_episode_success_count = (
          _jax_to_torch(info["last_episode_success_count"])[done > 0]  # pylint: disable=unsubscriptable-object
          .float()
          .tolist()
      )
      if len(last_episode_success_count) > 0:
        self.success_queue.extend(last_episode_success_count)
      info_ret["log"]["last_episode_success_count"] = np.mean(
          self.success_queue
      )

    for k, v in self.env_state.metrics.items():
      if k not in info_ret["log"]:
        info_ret["log"][k] = _jax_to_torch(v).float().mean().item()

    obs = TensorDict(obs, batch_size=[self.num_envs])
    return obs, reward, done, info_ret

  def reset(self):
    # todo add random init like in collab examples?
    self.env_state = self.reset_fn(self.key_reset)

    if self.asymmetric_obs:
      obs = _jax_to_torch(self.env_state.obs["state"])
      critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
      obs = {"state": obs, "privileged_state": critic_obs}
    else:
      obs = _jax_to_torch(self.env_state.obs)
      obs = {"state": obs}
    return TensorDict(obs, batch_size=[self.num_envs])

  def get_observations(self):
   return self.reset()

  def render(self, mode="human"):  # pylint: disable=unused-argument
    if self.render_callback is not None:
      self.render_callback(self.env.env.env, self.env_state)
    else:
      raise ValueError("No render callback specified")

  def get_number_of_agents(self):
    return 1

  def get_env_info(self):
    info = {}
    info["action_space"] = self.action_space  # pytype: disable=attribute-error
    info["observation_space"] = (
        self.observation_space  # pytype: disable=attribute-error
    )
    return info


def create_rgb_image(num, rgb_color, image_size):
    if not all(0 <= color <= 1 for color in rgb_color):
        raise ValueError("RGB颜色值必须在0到1之间")
    height, width = image_size
    image = np.full((num, height, width, 3), rgb_color, dtype=np.float32)
    return image


class BatchSplatWrapper(RSLRLBraxWrapper):
  """Wrapper for Brax environments that interop with torch and use 3DGS BatchSplatRenderer."""

  def _ensure_nonempty_body_gaussians(
      self, mj_model, body_gaussians, *, purpose: str, warn: bool = True
  ):
    if body_gaussians:
      return body_gaussians

    # gaussian_renderer expects at least one gaussian; provide an invisible dummy.
    target_body = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, 0)  # world

    with tempfile.NamedTemporaryFile(suffix=".ply", mode="w", delete=False) as f:
      f.write("ply\n")
      f.write("format ascii 1.0\n")
      f.write("element vertex 1\n")
      f.write("property float x\n")
      f.write("property float y\n")
      f.write("property float z\n")
      f.write("property float f_dc_0\n")
      f.write("property float f_dc_1\n")
      f.write("property float f_dc_2\n")
      f.write("property float opacity\n")
      f.write("property float scale_0\n")
      f.write("property float scale_1\n")
      f.write("property float scale_2\n")
      f.write("property float rot_0\n")
      f.write("property float rot_1\n")
      f.write("property float rot_2\n")
      f.write("property float rot_3\n")
      f.write("end_header\n")
      # opacity=0 => invisible
      f.write("0 0 0 0 0 0 0 0.01 0.01 0.01 1 0 0 0\n")
      temp_ply_path = f.name

    if warn:
      print(
        f"Warning: body_gaussians is empty for {purpose}. "
        f"Using invisible dummy gaussian at {temp_ply_path} attached to body '{target_body}'."
      )
    return {target_body: temp_ply_path}

  def __init__(
      self,
      env,
      num_actors,
      seed,
      episode_length,
      action_repeat,
      randomization_fn=None,
      render_callback=None,
      device_rank=None,
      full_reset: bool | None = None,
  ):
    super().__init__(
        env,
        num_actors,
        seed,
        episode_length,
        action_repeat,
        randomization_fn,
        render_callback,
        device_rank,
        full_reset,
    )
    self._init_renderer()

  def _init_renderer(self):
    mj_model = self.env.mj_model
    
    self.height = 64
    self.width = 64
    body_gaussians = {}
    background_ply = None
    bg_img_template = None

    self.dynamic_bg = False
    self.pixels_uint8 = False

    if hasattr(self.env.unwrapped, '_config'):
      c = self.env.unwrapped._config
      if hasattr(c, 'vision_config'):
        self.height = c.vision_config.render_height
        self.width = c.vision_config.render_width
        self.dynamic_bg = bool(getattr(c.vision_config, 'dynamic_bg', False))
        self.pixels_uint8 = bool(getattr(c.vision_config, 'pixels_uint8', False))
        background_ply = getattr(c.vision_config, 'background', None)
        if hasattr(c.vision_config, 'body_gaussians'):
          body_gaussians = c.vision_config.body_gaussians
          if hasattr(body_gaussians, 'to_dict'):
            body_gaussians = body_gaussians.to_dict()
        else:
          raise ValueError("BatchSplatWrapper requires body_gaussians in vision_config.")
        if (not self.dynamic_bg) and getattr(c.vision_config, 'bg_img', None) is not None:
          bg = c.vision_config.bg_img
          if isinstance(bg, tuple):
            bg = create_rgb_image(mj_model.ncam, bg, (self.height, self.width))
          if isinstance(bg, np.ndarray):
            bg = torch.from_numpy(bg)
          elif not isinstance(bg, torch.Tensor):
            bg = torch.tensor(bg)

          if bg.dtype == torch.uint8:
            bg = bg.float() / 255.0
          else:
            bg = bg.float()
          
          expected_shape = (mj_model.ncam, self.height, self.width, 3)
          if bg.shape != expected_shape:
            raise ValueError(f"bg_img shape mismatch. Expected {expected_shape}, got {bg.shape}")
          
          bg_img_template = bg

    if self.dynamic_bg and background_ply is None:
      raise ValueError(
          "vision_config.dynamic_bg=True requires vision_config.background to be set "
          "(a background .ply used to re-render bg each reset)."
      )

    fg_body_gaussians = self._ensure_nonempty_body_gaussians(
        mj_model, body_gaussians, purpose="foreground renderer"
    )
    # # Important for VRAM: background_ply can be large. If we're using a static
    # # image background (bg_img) and dynamic_bg is off, we should not also load
    # # the 3DGS background ply into the foreground renderer.
    # using_static_bg_img = (not self.dynamic_bg) and (bg_img_template is not None)
    # if self.dynamic_bg:
    #   fg_background_ply = None
    # else:
    #   fg_background_ply = None if using_static_bg_img else background_ply
    fg_background_ply = None if self.dynamic_bg else background_ply
    fg_cfg = BatchSplatConfig(
        body_gaussians=fg_body_gaussians,
        background_ply=fg_background_ply,
        minibatch=min(self.batch_size, int(256 // mj_model.ncam)),
    )
    self.renderer_fg = MjxBatchSplatRenderer(fg_cfg, mj_model=mj_model)
    self.renderer = self.renderer_fg
    self.fovy_np = np.array(mj_model.cam_fovy)[None, :]

    self.renderer_bg = None
    if self.dynamic_bg:
      bg_body_gaussians = self._ensure_nonempty_body_gaussians(
        mj_model, {}, purpose="background renderer", warn=False
      )
      bg_cfg = BatchSplatConfig(
          body_gaussians=bg_body_gaussians,
          background_ply=background_ply,
          minibatch=min(self.batch_size, int(256 // mj_model.ncam)),
      )
      self.renderer_bg = MjxBatchSplatRenderer(bg_cfg, mj_model=mj_model)

    # Background image storage strategy:
    # - dynamic_bg=True: per-env bg is required (camera pose may differ per env);
    #   keep a batched float32 tensor.
    # - dynamic_bg=False: bg is static; store only per-camera and broadcast at
    #   render time to avoid an always-resident (B,ncam,H,W,3) allocation.
    if self.dynamic_bg:
      self.bg_img = torch.zeros(
          (self.batch_size, mj_model.ncam, self.height, self.width, 3),
          dtype=torch.float32,
          device=self.renderer_fg.device,
      )
      self.bg_img_camera = None
    else:
      if bg_img_template is not None:
        self.bg_img_camera = bg_img_template.to(self.renderer_fg.device)
      else:
        self.bg_img_camera = torch.zeros(
            (mj_model.ncam, self.height, self.width, 3),
            dtype=torch.float32,
            device=self.renderer_fg.device,
        )
      self.bg_img = None

    if self.dynamic_bg:
      self._bg_img_base = torch.zeros(
          (self.batch_size, mj_model.ncam, self.height, self.width, 3),
          dtype=torch.float32,
          device=self.renderer_bg.device,
      )

  def step(self, action):
    action = torch.clip(action, -1.0, 1.0)
    action = _torch_to_jax(action)
    self.env_state = self.step_fn(self.env_state, action)
    
    # Render
    b_pos = _jax_to_torch(self.env_state.data.xpos)
    b_quat = _jax_to_torch(self.env_state.data.xquat)
    c_pos = _jax_to_torch(self.env_state.data.cam_xpos)
    c_xmat = _jax_to_torch(self.env_state.data.cam_xmat)

    # If BraxAutoResetWrapper performed a full reset on done, the returned
    # state.data for done envs is already the reset state with re-randomized
    # camera pose. Refresh background for those envs to keep bg/fg consistent.
    if self.dynamic_bg:
      done_torch = _jax_to_torch(self.env_state.done).to(torch.bool)
      if done_torch.any():
        gsb_bg = self.renderer_bg.batch_update_gaussians(b_pos, b_quat)
        bg_rgb, _ = self.renderer_bg.batch_env_render(
            gsb_bg,
            c_pos,
            c_xmat,
            self.height,
            self.width,
            self.fovy_np,
            self._bg_img_base,
        )
        mask = done_torch.view(-1, 1, 1, 1, 1).to(bg_rgb.device)
        # Only overwrite backgrounds for envs that ended this step.
        self.bg_img = torch.where(mask, bg_rgb, self.bg_img)
    
    gsb = self.renderer_fg.batch_update_gaussians(b_pos, b_quat)
    if self.dynamic_bg:
      bg_img = self.bg_img
    else:
      bg_img = self.bg_img_camera.unsqueeze(0).expand(
          self.batch_size, -1, -1, -1, -1
      )
    rgb, _ = self.renderer_fg.batch_env_render(
      gsb, c_pos, c_xmat, self.height, self.width, self.fovy_np, bg_img
    )

    rgb_obs = rgb
    if self.pixels_uint8:
      rgb_obs = (rgb_obs.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    
    # Construct observations
    critic_obs = None
    if self.asymmetric_obs:
      obs = _jax_to_torch(self.env_state.obs["state"])
      critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
      obs = {"state": obs, "privileged_state": critic_obs}
    else:
      obs = _jax_to_torch(self.env_state.obs)
      obs = {"state": obs}
      
    # Add pixels to observation
    # Assuming obs is a dict, if not, we need to decide how to structure it.
    # RSL-RL usually expects a dict for complex obs.
    if isinstance(obs, dict):
      for i in range(self.env.mj_model.ncam):
        obs[f'pixels/view_{i}'] = rgb_obs[:, i]
    else:
      # If obs was a tensor, convert to dict to add pixels
      obs = {"state": obs}
      for i in range(self.env.mj_model.ncam):
        obs[f'pixels/view_{i}'] = rgb_obs[:, i]

    reward = _jax_to_torch(self.env_state.reward)
    done = _jax_to_torch(self.env_state.done)
    info = self.env_state.info
    truncation = _jax_to_torch(info["truncation"])

    info_ret = {
      "time_outs": truncation,
      "observations": {"critic": critic_obs},
      "log": {},
    }

    if "last_episode_success_count" in info:
      last_episode_success_count = (
          _jax_to_torch(info["last_episode_success_count"])[done > 0]
          .float()
          .tolist()
      )
      if len(last_episode_success_count) > 0:
        self.success_queue.extend(last_episode_success_count)
      info_ret["log"]["last_episode_success_count"] = np.mean(
          self.success_queue
      )

    for k, v in self.env_state.metrics.items():
      if k not in info_ret["log"]:
        info_ret["log"][k] = _jax_to_torch(v).float().mean().item()

    obs = TensorDict(obs, batch_size=[self.num_envs])
    return obs, reward, done, info_ret

  def reset(self):
    self.env_state = self.reset_fn(self.key_reset)
    
    # Render initial state
    b_pos = _jax_to_torch(self.env_state.data.xpos)
    b_quat = _jax_to_torch(self.env_state.data.xquat)
    c_pos = _jax_to_torch(self.env_state.data.cam_xpos)
    c_xmat = _jax_to_torch(self.env_state.data.cam_xmat)
    
    if self.dynamic_bg:
      gsb_bg = self.renderer_bg.batch_update_gaussians(b_pos, b_quat)
      bg_rgb, _ = self.renderer_bg.batch_env_render(
        gsb_bg,
        c_pos,
        c_xmat,
        self.height,
        self.width,
        self.fovy_np,
        self._bg_img_base,
      )
      self.bg_img = bg_rgb

    gsb = self.renderer_fg.batch_update_gaussians(b_pos, b_quat)
    if self.dynamic_bg:
      bg_img = self.bg_img
    else:
      bg_img = self.bg_img_camera.unsqueeze(0).expand(
          self.batch_size, -1, -1, -1, -1
      )
    rgb, _ = self.renderer_fg.batch_env_render(
      gsb, c_pos, c_xmat, self.height, self.width, self.fovy_np, bg_img
    )

    rgb_obs = rgb
    if self.pixels_uint8:
      rgb_obs = (rgb_obs.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

    if self.asymmetric_obs:
      obs = _jax_to_torch(self.env_state.obs["state"])
      critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
      obs = {"state": obs, "privileged_state": critic_obs}
    else:
      obs = _jax_to_torch(self.env_state.obs)
      obs = {"state": obs}
      
    # Add pixels to observation
    if isinstance(obs, dict):
      for i in range(self.env.mj_model.ncam):
        obs[f'pixels/view_{i}'] = rgb_obs[:, i]
    else:
      obs = {"state": obs}
      for i in range(self.env.mj_model.ncam):
        obs[f'pixels/view_{i}'] = rgb_obs[:, i]
            
    return TensorDict(obs, batch_size=[self.num_envs])
