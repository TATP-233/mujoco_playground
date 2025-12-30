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
"""Convert trained RSL-RL model to ONNX format."""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tensordict import TensorDict

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
from actor_critic_cnn import ActorCriticCNN
import rsl_rl.modules

# Register the class
rsl_rl.modules.ActorCriticCNN = ActorCriticCNN

class OnnxPolicyWrapper(nn.Module):
    """Wrapper for ActorCriticCNN to make it compatible with ONNX export."""
    def __init__(self, policy_module, obs_groups, pixel_keys):
        super().__init__()
        self.policy = policy_module
        self.obs_groups = obs_groups
        self.pixel_keys = pixel_keys
        
    def forward(self, *args):
        """
        Inputs will be passed as separate arguments in the order of obs_groups['policy'].
        """
        obs_dict = {}
        for i, group in enumerate(self.obs_groups['policy']):
            obs_dict[group] = args[i]
        
        td_obs = TensorDict(obs_dict, batch_size=[args[0].shape[0]])
        
        # We use act_inference for deterministic output
        actions = self.policy.act_inference(td_obs)
        return actions

def main():
    parser = argparse.ArgumentParser(description="Convert RSL-RL model to ONNX.")
    parser.add_argument("--env_name", type=str, required=True, help="Name of the environment.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file.")
    parser.add_argument("--output", type=str, default="model.onnx", help="Path to save the ONNX model.")
    parser.add_argument("--vision", action="store_true", help="Whether the model is a vision model.")
    args = parser.parse_args()

    device = "cpu"
    
    # 1. Load environment to get metadata
    print(f"Loading environment: {args.env_name}")
    env_cfg = registry.get_default_config(args.env_name)
    
    # Dummy setup for vision if needed to get correct observation sizes
    if args.vision:
        env_cfg.vision = True
        env_cfg.vision_config.render_width = 64
        env_cfg.vision_config.render_height = 64
        # We don't need real assets for just getting metadata
        
    raw_env = registry.load(args.env_name, config=env_cfg, config_overrides={"impl": "jax"})
    num_actions = raw_env.action_size
    
    # 2. Get RL config
    if args.env_name in registry.manipulation._envs:
        train_cfg = manipulation_params.rsl_rl_config(args.env_name)
    elif args.env_name in registry.locomotion._envs:
        train_cfg = locomotion_params.rsl_rl_config(args.env_name)
    else:
        raise ValueError(f"Unknown environment: {args.env_name}")

    # Handle deprecated normalization config mapping
    if "empirical_normalization" in train_cfg:
        if "actor_obs_normalization" not in train_cfg.policy:
            train_cfg.policy.actor_obs_normalization = train_cfg.empirical_normalization
        if "critic_obs_normalization" not in train_cfg.policy:
            train_cfg.policy.critic_obs_normalization = train_cfg.empirical_normalization

    if args.vision:
        obs_groups = {
            "policy": ["state", "pixels/view_0"],
            "critic": ["state", "pixels/view_0"],
        }
        # In actual train_rsl_rl.py we dynamically discover views, 
        # but for ONNX export we usually export for a specific setup.
        # Let's adjust based on ncam if vision is on.
        num_cameras = raw_env.mj_model.ncam
        pixel_views = [f"pixels/view_{i}" for i in range(num_cameras)]
        obs_groups["policy"] = ["state"] + pixel_views
        obs_groups["critic"] = ["state"] + pixel_views
    else:
        obs_size = raw_env.observation_size
        if isinstance(obs_size, dict):
            obs_groups = {"policy": ["state"], "critic": ["privileged_state"]}
        else:
            obs_groups = {"policy": ["state"], "critic": ["state"]}

    # 3. Create dummy observation for initialization
    print("Creating dummy observations...")
    dummy_obs_payload = {}
    obs_size = raw_env.observation_size
    if isinstance(obs_size, dict):
        for k, v in obs_size.items():
            if isinstance(v, (list, tuple)):
                dummy_obs_payload[k] = torch.zeros(1, *v)
            else:
                dummy_obs_payload[k] = torch.zeros(1, v)
    else:
        dummy_obs_payload["state"] = torch.zeros(1, obs_size)
    
    if args.vision:
        for i in range(raw_env.mj_model.ncam):
            dummy_obs_payload[f"pixels/view_{i}"] = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)

    dummy_obs = TensorDict(dummy_obs_payload, batch_size=[1])

    # 4. Instantiate policy
    print("Instantiating policy...")
    if args.vision:
        policy = ActorCriticCNN(
            obs=dummy_obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            **train_cfg.policy
        )
    else:
        from rsl_rl.modules import ActorCritic
        policy = ActorCritic(
            obs=dummy_obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            **train_cfg.policy
        )
    
    # 5. Load weights
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Try loading with strict=True first, fallback to strict=False with warning
    try:
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as e:
        print(f"Warning: Strict loading failed, trying with strict=False. Error: {e}")
        policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    policy.eval()

    # 6. Prepare for export
    print("Preparing for ONNX export...")
    pixel_keys = [k for k in dummy_obs_payload.keys() if k.startswith("pixels/")]
    onnx_wrapper = OnnxPolicyWrapper(policy, obs_groups, pixel_keys)
    
    input_names = obs_groups['policy']
    output_names = ["actions"]
    
    dummy_inputs = []
    dynamic_axes = {}
    for name in input_names:
        val = dummy_obs_payload[name].float() if not name.startswith("pixels/") else dummy_obs_payload[name]
        dummy_inputs.append(val)
        dynamic_axes[name] = {0: "batch_size"}
    dynamic_axes["actions"] = {0: "batch_size"}

    # 7. Export
    print(f"Exporting to {args.output}...")
    torch.onnx.export(
        onnx_wrapper,
        tuple(dummy_inputs),
        args.output,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        # Use legacy trace-based exporter for better compatibility with wrappers and custom libraries
        # Newer dynamo-based exporter (default in some PyTorch versions) can be picky about input structures
        dynamo=False 
    )
    
    print("Conversion successful!")

if __name__ == "__main__":
    main()

