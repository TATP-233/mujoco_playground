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
"""Bring a box to a target and orientation."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
import numpy as np

def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      render_batch_size=1024,
      render_width=64,
      render_height=64,
  )


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.04,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=10., #8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Arm stays close to target pose.
              robot_target_qpos=0.015, #0.3
              # Gripper stays open when approaching the box.
              gripper_open=2.0,
              # Gripper closes when close to the box.
              gripper_close=10.0,
              # Lift the box.
              lift=2.0,
          ),
          lifted_reward=0.5,
          success_reward=2.0,
      ),
      vision=False,
      vision_config=default_vision_config(),
      success_threshold=0.05,
      impl='jax',
      nconmax=24 * 2048,
      njmax=128,
  )
  return config


class PandaPickCube(panda.PandaBase):
  """Bring a box to a target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,
  ):
    self._vision = config.vision

    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
        / "mjx_single_cube.xml"
    )
    super().__init__(
        xml_path,
        config,
        config_overrides,
    )
    self._post_init(obj_name="box", keyframe="init") # "home"
    self._sample_orientation = sample_orientation

    # Contact sensor IDs.
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]

  def reset(self, rng: jax.Array) -> State:
    rng, rng_box, rng_target = jax.random.split(rng, 3)

    # intialize box position
    box_pos = (
        jax.random.uniform(
            rng_box,
            (3,),
            minval=jp.array([-0.1, -0.1, 0.0]),
            maxval=jp.array([0.1, 0.1, 0.0]),
        )
        + self._init_obj_pos
    )
    # box_pos = self._init_obj_pos

    # initialize target position
    target_pos = (
        jax.random.uniform(
            rng_target,
            (3,),
            minval=jp.array([-0.1, -0.1, 0.1]),
            maxval=jp.array([0.1, 0.1, 0.2]),
        )
        + self._init_obj_pos
    )

    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if self._sample_orientation:
      # sample a random direction
      rng, rng_axis, rng_theta = jax.random.split(rng, 3)
      perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
      perturb_axis = perturb_axis / math.norm(perturb_axis)
      perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
      target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)

    # initialize data
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(box_pos)
    )
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    if self._vision:
        data = mjx.forward(self._mjx_model, data)

    # set target mocap position
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
    )

    # initialize env state and info
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    if self._vision:
       metrics.update({
           'reward/lifted': jp.array(0.0, dtype=float),
           'reward/success': jp.array(0.0, dtype=float),
       })

    info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
    if self._vision:
        obs = self._get_obs_vision(data, info)
    else:
        obs = self._get_obs(data, info)

    reward, done = jp.zeros(2)
    state = State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    if self._vision:
        jaw_delta = 0.02  # up to 2 cm movement per ctrl.
        delta = delta.at[-1].set(jp.where(delta[-1] < 0, -jaw_delta, jaw_delta))
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    if self._vision:
        data = mjx.forward(self._mjx_model, data)

    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    if self._vision:
        # Sparse rewards
        box_pos = data.xpos[self._obj_body]
        # lifted = (box_pos[2] > 0.03) * self._config.reward_config.lifted_reward
        # reward += lifted
        success = self._get_success(data, state.info)
        reward += success * self._config.reward_config.success_reward
        state.metrics.update({
            # 'reward/lifted': lifted.astype(float),
            'reward/success': success.astype(float),
        })

    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    # reward = jp.where(jp.isnan(reward), -1e4, reward)

    state.metrics.update(
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    if self._vision:
        obs = self._get_obs_vision(data, state.info)
    else:
        obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state

  def _get_success(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    box_pos = data.xpos[self._obj_body]
    target_pos = info['target_pos']
    return jp.linalg.norm(box_pos - target_pos) < self._config.success_threshold

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    target_pos = info["target_pos"]
    box_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]
    pos_err = jp.linalg.norm(target_pos - box_pos)
    box_mat = data.xmat[self._obj_body]
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

    box_target = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
    
    dist_to_box = jp.linalg.norm(box_pos - gripper_pos)
    gripper_box = 1 - jp.tanh(5 * dist_to_box)
    
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    # Check for collisions with the floor
    hand_floor_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._floor_hand_found_sensor
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)

    # Relaxed threshold for reaching the box
    is_reached = dist_to_box < 0.012
    info["reached_box"] = 1.0 * jp.maximum(info["reached_box"], is_reached)

    # Encourage keeping gripper open when not yet reached the box
    left_finger = data.qpos[self._robot_qposadr[-2]]
    right_finger = data.qpos[self._robot_qposadr[-1]]
    gripper_width = left_finger + right_finger
    gripper_open = ((gripper_width > 0.035) * (~is_reached)).astype(float)
    gripper_close = ((gripper_width < 0.025) * is_reached).astype(float)

    # Explicit lift reward
    lift = 1.0 - jp.tanh(5.0 * jp.abs(box_pos[2] - target_pos[2]))
    lift = lift * info["reached_box"]

    rewards = {
        "gripper_box": gripper_box,
        "box_target": box_target * info["reached_box"],
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
        "gripper_open": gripper_open,
        "gripper_close": gripper_close,
        "lift": lift,
    }
    return rewards

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos[self._robot_qposadr],
        data.qvel[self._robot_qposadr],
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        info["target_pos"] - data.xpos[self._obj_body],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])

    return obs

  def _get_obs_vision(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos[self._robot_qposadr],
        data.qvel[self._robot_qposadr],
        gripper_pos,
        gripper_mat[3:],
        info["target_pos"],
        target_mat.ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])

    return obs

class PandaPickCubeOrientation(PandaPickCube):
  """Bring a box to a target and orientation."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides, sample_orientation=True)
