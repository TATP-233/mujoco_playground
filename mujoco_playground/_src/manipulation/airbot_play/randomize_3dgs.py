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
"""Randomization functions."""

import jax
import jax.numpy as jnp
from mujoco import mjx
from typing import Tuple
from mujoco.mjx._src import math


def perturb_orientation(
    key: jax.Array, original: jax.Array, deg: float
) -> jax.Array:
    """Perturbs a 3D or 4D orientation by up to deg."""
    key_axis, key_theta, key_y = jax.random.split(key, 3)
    perturb_axis = jax.random.uniform(key_axis, (3,), minval=-1, maxval=1)
    perturb_axis = perturb_axis.at[1].set(
        jax.random.uniform(key_y, (), minval=0, maxval=1)
    )
    perturb_axis = perturb_axis / (jnp.linalg.norm(perturb_axis) + 1e-6)
    perturb_theta = jax.random.uniform(key_theta, (), minval=0, maxval=jnp.deg2rad(deg))
    rot_offset = math.axis_angle_to_quat(perturb_axis, perturb_theta)    
    if original.shape[-1] == 4:
        return math.quat_mul(rot_offset, original)
    elif original.shape[-1] == 3:
        return math.rotate(original, rot_offset)
    else:
        raise ValueError(f'Invalid input shape: {original.shape}. Expected (3,) or (4,).')


def domain_randomize(
    mjx_model: mjx.Model, num_worlds: int = None, rng=None
) -> Tuple[mjx.Model, mjx.Model]:
  """支持多相机的域随机化，适配 BatchRenderer。"""

  if num_worlds is None:
    assert rng is not None
  else:
    rng = jax.random.split(jax.random.key(0), num_worlds)

  num_cams = mjx_model.ncam  # 获取模型中的相机总数

  @jax.vmap
  def rand(rng: jax.Array):
    _, key = jax.random.split(rng, 2)
    key_pos, key_ori = jax.random.split(key, 2)
    pos_dr = 0.02  # m
    ori_dr = 3  # deg
    cam_offsets = jax.random.uniform(
        key_pos, (num_cams, 3), minval=-pos_dr, maxval=pos_dr
    )
    cam_pos = mjx_model.cam_pos + cam_offsets
    keys_ori = jax.random.split(key_ori, num_cams)
    cam_quat = jax.vmap(perturb_orientation, in_axes=(0, 0, None))(
        keys_ori, mjx_model.cam_quat, ori_dr
    )
    return cam_pos, cam_quat

  cam_pos, cam_quat = rand(rng)

  in_axes: mjx.Model = jax.tree.map(lambda x: None, mjx_model)
  in_axes = in_axes.tree_replace({
      'cam_pos': 0,
      'cam_quat': 0,
  })

  mjx_model = mjx_model.tree_replace({
    'cam_pos': cam_pos,
    'cam_quat': cam_quat,
  })

  return mjx_model, in_axes
