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
from typing import Tuple

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src.manipulation.franka_emika_panda import pick_cartesian

def perturb_orientation(
    key: jax.Array, original: jax.Array, deg: float
) -> jax.Array:
  """Perturbs a 3D or 4D orientation by up to deg."""
  key_axis, key_theta, key = jax.random.split(key, 3)
  perturb_axis = jax.random.uniform(key_axis, (3,), minval=-1, maxval=1)
  # Only perturb upwards in the y axis.
  key_y, key = jax.random.split(key, 2)
  perturb_axis = perturb_axis.at[1].set(
      jax.random.uniform(key_y, (), minval=0, maxval=1)
  )
  perturb_axis = perturb_axis / jp.linalg.norm(perturb_axis)
  perturb_theta = jax.random.uniform(
      key_theta, shape=(1,), minval=0, maxval=np.deg2rad(deg)
  )
  rot_offset = math.axis_angle_to_quat(perturb_axis, perturb_theta)
  if original.shape == (4,):
    return math.quat_mul(rot_offset, original)
  elif original.shape == (3,):
    return math.rotate(original, rot_offset)
  else:
    raise ValueError('Invalid input shape:', original.shape)

def domain_randomize(
    mjx_model: mjx.Model, num_worlds: int
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary axes for the Madrona BatchRenderer."""
  in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
  in_axes = in_axes.tree_replace({
      'cam_pos': 0,
      'cam_quat': 0,
  })
  rng = jax.random.key(0)

  # Simpler logic implementing via Numpy.
  np.random.seed(0)

  @jax.vmap
  def rand(rng: jax.Array):
    """Generate randomized model fields."""
    _, key = jax.random.split(rng, 2)

    #### Cameras ####
    key_pos, key_ori, key = jax.random.split(key, 3)
    cam_offset = jax.random.uniform(key_pos, (3,), minval=-0.05, maxval=0.05)
    assert (
        len(mjx_model.cam_pos) == 1
    ), f'Expected single camera, got {len(mjx_model.cam_pos)}'
    cam_pos = mjx_model.cam_pos.at[0].set(mjx_model.cam_pos[0] + cam_offset)
    cam_quat = mjx_model.cam_quat.at[0].set(
        perturb_orientation(key_ori, mjx_model.cam_quat[0], 10)
    )

    return (
        cam_pos,
        cam_quat,
    )

  cam_pos, cam_quat = rand(jax.random.split(rng, num_worlds))

  mjx_model = mjx_model.tree_replace({
    'cam_pos': cam_pos,
    'cam_quat': cam_quat,
  })

  return mjx_model, in_axes
