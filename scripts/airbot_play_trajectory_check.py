"""Replay a simple joint-space trajectory for AirbotPlayPickCube.

This script builds a short, safe joint trajectory and feeds it to the
`AirbotPlayPickCube` environment to sanity-check control signals. The
trajectory is defined in actuator (joint) space and converted into the
`action` input expected by the environment (delta control scaled by the
configured `action_scale`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import mediapy as media

from mujoco_playground._src import registry


def _make_targets(
    init_ctrl: jnp.ndarray,
    ctrl_range: jnp.ndarray,
) -> List[Tuple[str, jnp.ndarray]]:
  """Create a handful of gentle waypoints in joint space.

  The arm has 6 joints; the gripper may expose 1 logical control (mimic) or
  more. Offsets are expressed as a fraction of the actuator range. If extra
  actuators exist beyond arm+gripper (unlikely), they stay fixed.
  """
  lower, upper = ctrl_range[:, 0], ctrl_range[:, 1]
  act_dim = init_ctrl.shape[0]
  gripper_dim = act_dim - 6
  if gripper_dim < 1:
    raise ValueError("Expected at least one gripper actuator (mimic)")

  pad_shape = (act_dim - (6 + gripper_dim),)

  def to_target(arm_offsets: Iterable[float], gripper_offset: float) -> jnp.ndarray:
    arm = jnp.array(list(arm_offsets), dtype=jnp.float32)
    if arm.shape[0] != 6:
      raise ValueError("arm_offsets must have length 6")
    grip = jnp.full((gripper_dim,), gripper_offset, dtype=jnp.float32)
    offsets = jnp.concatenate([arm, grip])
    if pad_shape[0] > 0:
      offsets = jnp.concatenate([offsets, jnp.zeros(pad_shape, dtype=jnp.float32)])
    delta = 0.15 * (upper - lower) * offsets
    return jnp.clip(init_ctrl + delta, lower, upper)

  targets = [
      ("lift_open", to_target([0.35, -0.30, 0.30, 0.10, 0.00, 0.00], -0.20)),
      ("reach_forward_close", to_target([0.40, 0.15, -0.15, 0.20, 0.00, 0.00], 0.45)),
      ("hold", to_target([0.38, 0.10, -0.10, 0.18, 0.00, 0.00], 0.50)),
      ("return_home_open", init_ctrl),
  ]
  return targets


def _drive_to_target(
    env, state, target_ctrl: jnp.ndarray, steps: int, action_scale: float
):
  """Move linearly toward a desired control vector over `steps` frames."""
  trajectory = []
  for t in range(steps):
    current_ctrl = state.data.ctrl
    # Linearly interpolate from the current control toward the target.
    desired_ctrl = current_ctrl + (target_ctrl - current_ctrl) / float(steps - t)
    action = (desired_ctrl - current_ctrl) / action_scale
    state = env.step(state, action)
    trajectory.append(state)
  return state, trajectory


def run(seed: int, segment_steps: int, render_path: Path | None) -> None:
  env = registry.load("AirbotPlayPickCube")
  rng = jax.random.PRNGKey(seed)
  state = env.reset(rng)

  init_ctrl = jnp.array(state.data.ctrl)
  ctrl_range = jnp.array(env.mj_model.actuator_ctrlrange)
  action_scale = float(env._config.action_scale)  # pylint: disable=protected-access

  targets = _make_targets(init_ctrl, ctrl_range)

  full_trajectory = [state]
  for name, target_ctrl in targets:
    state, segment = _drive_to_target(env, state, target_ctrl, segment_steps, action_scale)
    full_trajectory.extend(segment)
    print(f"Reached waypoint '{name}':")
    robot_qpos = np.array(state.data.qpos[: env.action_size])
    print(f"  ctrl min/max: {np.min(state.data.ctrl):.3f} / {np.max(state.data.ctrl):.3f}")
    print(f"  first 6 joint qpos: {robot_qpos[:6]}")

  if render_path is not None:
    frames = env.render(full_trajectory, height=480, width=640)
    render_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(render_path, frames, fps=int(1.0 / env.dt))
    print(f"Saved rollout video to {render_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="AirbotPlayPickCube trajectory check")
  parser.add_argument("--seed", type=int, default=0, help="PRNG seed for reset")
  parser.add_argument(
      "--segment-steps",
      type=int,
      default=40,
      help="Steps per waypoint segment",
  )
  parser.add_argument(
      "--render-path",
      type=Path,
      default=Path("outputs/airbot_play_rollout.mp4"),
      help="mp4 path for saving a rendered rollout (use --render-path '' to skip)",
  )
  args = parser.parse_args()

  render_path: Path | None
  if args.render_path == Path(""):
    render_path = None
  else:
    render_path = args.render_path

  run(seed=args.seed, segment_steps=args.segment_steps, render_path=render_path)
