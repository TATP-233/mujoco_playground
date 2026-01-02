"""Randomization functions."""
from typing import Tuple
import jax
import jax.numpy as jnp
from mujoco import mjx
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
  
  in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
  in_axes = in_axes.tree_replace({
      'cam_pos': 0,
      'cam_quat': 0,
  })
  
  # 确保 rng 是有效的 batch key
  if num_worlds is not None:
    # 建议从外部传入变化的 rng，或者这里使用一个不同的 key
    rng = jax.random.split(jax.random.key(42), num_worlds) 
  print(f"{num_worlds=}")
  num_cams = mjx_model.ncam

  @jax.vmap
  def rand(single_rng: jax.Array):
    # 使用唯一的 key
    key_pos, key_ori = jax.random.split(single_rng, 2)
    
    # 【调试】极大化随机范围，确保肉眼可见
    cam_offsets = jax.random.uniform(
        key_pos, (num_cams, 3), minval=-0.5, maxval=0.5 # 增加到 50cm
    )
    # 显式确保我们在原始位置上叠加
    new_cam_pos = mjx_model.cam_pos + cam_offsets

    keys_ori = jax.random.split(key_ori, num_cams)
    new_cam_quat = jax.vmap(perturb_orientation, in_axes=(0, 0, None))(
        keys_ori, mjx_model.cam_quat, 30 # 增加到 30 度
    )
    return new_cam_pos, new_cam_quat

  cam_pos, cam_quat = rand(rng)

  # 调试：打印第一个和第二个 world 的相机位置，看数据是否真的不同
  jax.debug.print("World 0 Cam Pos: {x}", x=cam_pos[0])
  jax.debug.print("World 1 Cam Pos: {x}", x=cam_pos[1])
  jax.debug.print("num cam_pos: {x}", x=len(cam_pos))

  mjx_model = mjx_model.tree_replace({
    'cam_pos': cam_pos,
    'cam_quat': cam_quat,
  })
  
  # --- 开始打印检查 ---
  print("\n" + "="*30)
  print("PyTree 结构检查 (编译时):")
  print(f"Model cam_pos 形状: {mjx_model.cam_pos.shape}") 
  # 预期: (num_worlds, num_cams, 3)
  print(f"Model cam_quat 形状: {mjx_model.cam_quat.shape}") 
  # 预期: (num_worlds, num_cams, 4)

  # 运行时检查具体数值 (确保不同 World 之间真的有差异)
  def debug_check(pos, quat):
    # 打印前两个世界的第一个相机位姿进行对比
    jax.debug.print("--- 运行时随机化数值校验 ---")
    jax.debug.print("World 0 - Cam 0 Pos: {x}", x=pos[0, 0])
    jax.debug.print("World 1 - Cam 0 Pos: {x}", x=pos[1, 0])
    
    # 计算所有世界相机位置的标准差，如果 > 0 说明确实存在随机化
    pos_std = jnp.std(pos)
    jax.debug.print("所有世界相机位置的样本标准差: {std} (应 > 0)", std=pos_std)

  debug_check(mjx_model.cam_pos, mjx_model.cam_quat)
  print("="*30 + "\n")
  # --- 结束打印检查 ---

  print("use dr!")
  return mjx_model, in_axes
