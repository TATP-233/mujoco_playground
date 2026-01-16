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
    """对 3D 向量或 4D 四元数进行随机扰动（最大角度为 deg）。"""
    
    # 1. 生成随机旋转轴
    key_axis, key_theta, key_y = jax.random.split(key, 3)
    
    # 生成各向均匀的随机轴
    perturb_axis = jax.random.uniform(key_axis, (3,), minval=-1, maxval=1)
    
    # 按照你原有的逻辑：仅在 y 轴正方向进行扰动 (可能是为了模拟某种特定的抖动)
    perturb_axis = perturb_axis.at[1].set(
        jax.random.uniform(key_y, (), minval=0, maxval=1)
    )
    
    # 归一化轴向量（防止除以 0）
    perturb_axis = perturb_axis / (jnp.linalg.norm(perturb_axis) + 1e-6)
    
    # 2. 生成随机旋转角度 (0 到 deg 之间)
    # 使用 jnp.deg2rad 确保在 JAX 变换中保持兼容
    max_rad = jnp.deg2rad(deg)
    perturb_theta = jax.random.uniform(key_theta, (), minval=0, maxval=max_rad)
    
    # 3. 将轴角转换为四元数
    # 注意：这里需要确保 math.axis_angle_to_quat 的输入形状正确
    rot_offset = math.axis_angle_to_quat(perturb_axis, perturb_theta)
    
    # 4. 根据输入形状应用变换
    if original.shape[-1] == 4:
        # 如果是四元数：使用四元数乘法进行旋转叠加
        return math.quat_mul(rot_offset, original)
    elif original.shape[-1] == 3:
        # 如果是 3D 向量：直接旋转该向量
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

    #### 多相机随机化 ####
    key_pos, key_ori = jax.random.split(key, 2)
    
    # 1. 位置随机化: 生成 (num_cams, 3) 的偏移量
    cam_offsets = jax.random.uniform(
        key_pos, (num_cams, 3), minval=-0.05, maxval=0.05
    )
    cam_pos = mjx_model.cam_pos + cam_offsets

    # 2. 姿态随机化: 使用 vmap 处理每一个相机
    # 假设 perturb_orientation 接受 (key, quat, degrees)
    keys_ori = jax.random.split(key_ori, num_cams)
    
    # 向量化处理所有相机的旋转
    cam_quat = jax.vmap(perturb_orientation, in_axes=(0, 0, None))(
        keys_ori, mjx_model.cam_quat, 10
    )

    return cam_pos, cam_quat

  # 针对所有 world 进行并行计算
  # 结果形状: cam_pos -> (num_worlds, num_cams, 3), cam_quat -> (num_worlds, num_cams, 4)
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
#   jax.debug.print("cam_pos: {c}", c=mjx_model.cam_pos)
#   print("?? why no print")
  return mjx_model, in_axes
