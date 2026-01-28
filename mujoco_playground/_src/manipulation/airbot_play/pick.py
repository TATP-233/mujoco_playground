"""Bring a box to a target and orientation."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.airbot_play import airbot_play
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
import numpy as np


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      render_batch_size=1024,
      render_width=64,
      render_height=64,
    #   bg_img=(1, 1, 1),
  )


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.02,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=5.0,
              # Box goes to the target mocap.
              box_target=5.0, #8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Do not collide the gripper with the box.
              no_box_collision=0.5,
              # Arm stays close to target pose.
              robot_target_qpos=0.015, #0.3
              gripper_open=0.5,
              # Close the gripper after reaching the box.
              gripper_close=20.0,
              lifted=8.0,
              success=10.0,
            #   # Orientation alignment reward.
            #   reward_ori=0.5,
          ),
      ),
      vision=False,
      vision_config=default_vision_config(),
      success_threshold=0.01,
      impl='jax',
      nconmax=24 * 2048,
      njmax=128,
  )
  return config


class AirbotPlayPickCube(airbot_play.AirbotPlayBase):
  """Bring a box to a target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "airbot_play"
        / "xmls"
        / "mjx_single_cube.xml"
    )
    super().__init__(
        xml_path,
        config,
        config_overrides,
    )
    self._vision = config.vision
    self._post_init(obj_name="box", keyframe="init")
    self._sample_orientation = sample_orientation

    # Contact sensor IDs.
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_box"]
    ]

  def reset(self, rng: jax.Array) -> State:
    rng, rng_box, rng_target = jax.random.split(rng, 3)

    # intialize box position
    box_pos = (
        jax.random.uniform(
            rng_box,
            (3,),
            minval=jp.array([-0.05, -0.05, 0.0]),
            maxval=jp.array([0.05, 0.05, 0.0]),
        )
        + self._init_obj_pos
    )
    # box_pos = self._init_obj_pos
    # print(f"init box pos={box_pos}")
    # initialize target position
    target_pos = (
        jax.random.uniform(
            rng_target,
            (3,),
            minval=jp.array([-0.0, -0.0, 0.02]),
            maxval=jp.array([0.0, 0.0, 0.05]),
        )
        + self._init_obj_pos
    )
    # target_pos = box_pos.at[2].add(0.02)

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
    # jax.debug.print("eef z vector: {v}", v=data.site_xmat[self._gripper_site][:, 2])
    # initialize env state and info
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    if self._vision:
       metrics.update({
           "has_non": False,
           "reached_box": 0.0,
       })

    info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0, "init_box_pos": self._get_box_pos(data), "success": False}
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
        delta = delta.at[-1].set(jp.where(delta[-1] < 0, -1.0, 1.0) * 0.02) # up to 2 cm movement per ctrl.

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
    box_pos = self._get_box_pos(data)
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < (state.info["init_box_pos"][2] - 0.01)
    has_non = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = out_of_bounds | has_non | state.info["success"]
    done = done.astype(float)
    state.metrics.update({"has_non": has_non})
    state.metrics.update({"reached_box": state.info["reached_box"]})
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
    box_pos = self._get_box_pos(data)
    target_pos = info['target_pos']
    return jp.linalg.norm(box_pos - target_pos) < self._config.success_threshold

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    target_pos = info["target_pos"]
    box_pos = self._get_box_pos(data)
    # jax.debug.print("box_pos={b}", b=box_pos)
    gripper_pos = data.site_xpos[self._gripper_site]
    pos_err = jp.linalg.norm(target_pos - box_pos)
    box_mat = data.xmat[self._obj_body]
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

    # Penalize collision with box.
    hand_box = (
        data.sensordata[self._mj_model.sensor_adr[self._box_hand_found_sensor]]
        > 0
    )
    no_box_collision = jp.where(hand_box, 0.0, 1.0)


    box_target = 1 - jp.tanh(10 * (0.9 * pos_err + 0.1 * rot_err))
    gripper_box = 1 - jp.tanh(15 * jp.linalg.norm(box_pos - gripper_pos))
    # robot_target_qpos = 1 - jp.tanh(
    #     jp.linalg.norm(
    #         data.qpos[self._robot_arm_qposadr]
    #         - self._init_q[self._robot_arm_qposadr]
    #     )
    # )

    # Check for collisions with the floor
    hand_floor_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._floor_hand_found_sensor
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)

    # info["reached_box"] = 1.0 * jp.maximum(
    #     info["reached_box"],
    #     (jp.linalg.norm(box_pos - gripper_pos) < 0.005),
    # )
    info["reached_box"] = 1.0 * (jp.linalg.norm(box_pos - gripper_pos) < 0.01)
    info["success"] = self._get_success(data, info)
    # jax.debug.print("reached_box={r}", r=info["reached_box"])

    # jax.debug.print(
    #     "gripper_opening={g}, max_gripper_opening={m}",
    #     g=gripper_opening,
    #     m=max_gripper_opening,
    # )
    gripper_span = self._uppers[-1] - self._lowers[-1]
    gripper_ctrl = data.ctrl[-1]
    gripper_close = gripper_box * (1 - jp.abs(gripper_ctrl - self._lowers[-1]) / gripper_span)
    gripper_open = (1 - gripper_box) * (1 - (jp.abs(self._uppers[-1] - gripper_ctrl) / gripper_span))
    # jax.debug.print(
        # "gripper_open={g}, gripper_close={m}",
        # g=gripper_open,
        # m=gripper_close,
    # )
    rewards = {
        "gripper_box": gripper_box,
        "box_target": box_target * info["reached_box"],
        "no_floor_collision": no_floor_collision,
        "gripper_open": gripper_open,
        # "gripper_close": gripper_close,
        "gripper_close": gripper_close * info["reached_box"],
        "no_box_collision": no_box_collision,
        # "robot_target_qpos": robot_target_qpos,
        "success": info["success"].astype(float),
        "lifted": (box_pos[2] > (info["init_box_pos"][2] + 0.005)) * info["reached_box"],
    }
    return rewards

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    box_pos = self._get_box_pos(data)
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos[self._robot_qposadr],
        data.qvel[self._robot_qposadr],
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        box_pos - data.site_xpos[self._gripper_site],
        info["target_pos"] - box_pos,
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])

    return obs

  def _get_box_pos(self, data: mjx.Data) -> jax.Array:
    box_pos = data.xpos[self._obj_body]
    # return box_pos.at[2].add(-0.01)
    return box_pos

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
