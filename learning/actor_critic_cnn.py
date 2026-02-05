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
"""Actor-Critic with CNN for vision-based RL using RSL-RL."""

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic


class CNNEncoder(nn.Module):
  def __init__(self, input_shape=(3, 64, 64), output_size=256):
    super().__init__()
    self.cnn = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )
    with torch.no_grad():
      n_flatten = self.cnn(torch.zeros(1, *input_shape)).shape[1]
    self.linear = nn.Linear(n_flatten, output_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.relu(self.linear(self.cnn(x)))


class ActorCriticCNN(ActorCritic):
  def __init__(
      self,
      obs: TensorDict,
      obs_groups: dict[str, list[str]],
      num_actions: int,
      **kwargs,
  ) -> None:
    self.cnn_output_size = 16
    self._min_std = float(kwargs.pop("min_std", 1e-6))
    
    # Create a dummy observation with 1D features for the base class initialization
    dummy_obs_dict = {}
    for group, value in obs.items():
      if group.startswith("pixels/"):
        dummy_obs_dict[group] = torch.zeros((value.shape[0], self.cnn_output_size), device=value.device)
      else:
        # Ensure all other groups are 2D as required by base ActorCritic
        if len(value.shape) == 1:
          dummy_obs_dict[group] = value.unsqueeze(-1)
        else:
          dummy_obs_dict[group] = value
          
    super().__init__(
        TensorDict(dummy_obs_dict, batch_size=obs.batch_size),
        obs_groups,
        num_actions,
        **kwargs
    )
    
    # Shared encoders for any pixel inputs
    has_pixels_policy = any(g.startswith("pixels/") for g in obs_groups["policy"])
    has_pixels_critic = any(g.startswith("pixels/") for g in obs_groups["critic"])
    
    self.encoder = CNNEncoder(output_size=self.cnn_output_size) if has_pixels_policy else None
    self.critic_encoder = CNNEncoder(output_size=self.cnn_output_size) if has_pixels_critic else None

  def _update_distribution(self, obs: TensorDict) -> None:
    super()._update_distribution(obs)
    scale = self.distribution.scale
    # torch.normal requires std to be finite and >= 0.
    # nan_to_num keeps training alive in rare numeric blow-ups.
    scale = torch.nan_to_num(scale, nan=self._min_std, posinf=1.0, neginf=self._min_std)
    scale = torch.clamp(scale, min=self._min_std)
    self.distribution = Normal(self.distribution.loc, scale)

  def _process_obs_list(self, obs: TensorDict, group_names: list[str], encoder: nn.Module = None) -> torch.Tensor:
    obs_list = []
    for group in group_names:
      if group.startswith("pixels/"):
        if encoder is not None:
          pixels = obs[group].permute(0, 3, 1, 2).contiguous()
          if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0
          obs_list.append(encoder(pixels))
      else:
        val = obs[group]
        if len(val.shape) == 1:
          val = val.unsqueeze(-1)
        obs_list.append(val)
    return torch.cat(obs_list, dim=-1)

  def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
    return self._process_obs_list(obs, self.obs_groups["policy"], self.encoder)

  def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
    return self._process_obs_list(obs, self.obs_groups["critic"], self.critic_encoder)
