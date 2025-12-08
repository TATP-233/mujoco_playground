"""Convert trained brax PPO model to ONNX format using PyTorch."""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import numpy as np
import torch
import torch.nn as nn
import jax
import jax.numpy as jp
import onnxruntime as rt

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.training.checkpoint import load
from etils import epath

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

import argparse

class PolicyNetwork(nn.Module):
    """PyTorch MLP policy network for ONNX export."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        super().__init__()
        
        # Register normalization parameters as buffers (non-trainable)
        self.register_buffer('obs_mean', torch.from_numpy(mean).float())
        self.register_buffer('obs_std', torch.from_numpy(std).float())
        
        # Build MLP layers
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())  # Swish activation = SiLU in PyTorch
            input_dim = hidden_dim
        
        # Output layer: action_dim * 2 (mean and log_std for Gaussian policy)
        layers.append(nn.Linear(input_dim, action_dim * 2))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, obs):
        """
        Args:
            obs: (batch_size, obs_dim) or (obs_dim,)
        Returns:
            action: (batch_size, action_dim) or (action_dim,)
        """
        # Handle both batched and unbatched inputs
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Normalize observations
        normalized_obs = (obs - self.obs_mean) / self.obs_std
        
        # Forward pass through MLP
        logits = self.mlp(normalized_obs)
        
        # Split into mean and log_std, take only mean (deterministic policy)
        action_dim = logits.shape[-1] // 2
        action_mean = logits[..., :action_dim]
        
        # Apply tanh to bound actions
        action = torch.tanh(action_mean)
        
        if squeeze_output:
            action = action.squeeze(0)
        
        return action


def transfer_weights_from_jax(jax_params: dict, pytorch_model: nn.Module):
    """Transfer weights from JAX parameters to PyTorch model."""
    
    # Get the state dict
    state_dict = pytorch_model.state_dict()
    
    # Map JAX parameter names to PyTorch parameter names
    # JAX format: {'hidden_0': {'kernel': ..., 'bias': ...}, 'hidden_1': {...}, ...}
    # PyTorch format: {'mlp.0.weight': ..., 'mlp.0.bias': ..., 'mlp.2.weight': ..., ...}
    
    layer_idx = 0
    for jax_layer_name in sorted(jax_params.keys()):
        layer_params = jax_params[jax_layer_name]
        
        # JAX uses (input_dim, output_dim) for kernels
        # PyTorch uses (output_dim, input_dim) for weights
        kernel = np.array(layer_params['kernel'])
        bias = np.array(layer_params['bias'])
        
        # Transpose kernel for PyTorch convention
        weight = kernel.T
        
        # Find corresponding PyTorch layer
        # Skip activation layers (SiLU), only count Linear layers
        pytorch_weight_key = f'mlp.{layer_idx}.weight'
        pytorch_bias_key = f'mlp.{layer_idx}.bias'
        
        if pytorch_weight_key in state_dict:
            state_dict[pytorch_weight_key] = torch.from_numpy(weight).float()
            state_dict[pytorch_bias_key] = torch.from_numpy(bias).float()
            print(f"Transferred {jax_layer_name}: weight {weight.shape}, bias {bias.shape}")
            layer_idx += 2  # Skip the activation layer
        else:
            print(f"Warning: Could not find {pytorch_weight_key} in PyTorch model")
    
    # Load the updated state dict
    pytorch_model.load_state_dict(state_dict)
    print("All weights transferred successfully!")


def main():
    print(f"Converting {ENV_NAME} model to ONNX using PyTorch...")
    
    # Find the latest checkpoint
    checkpoint_dir = epath.Path(CHECKPOINT_DIR)
    checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    checkpoints.sort(key=lambda x: int(x.name))
    latest_checkpoint = checkpoints[-1]
    
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Using latest checkpoint: {latest_checkpoint.name}")
    print(f"Output path: {OUTPUT_PATH}")
    
    # Load environment configuration
    env_cfg = registry.get_default_config(ENV_NAME)
    env_cfg["impl"] = "jax"
    env = registry.load(ENV_NAME, config=env_cfg)
    
    obs_size = env.observation_size
    act_size = env.action_size
    print(f"Observation size: {obs_size}")
    print(f"Action size: {act_size}")
    
    # Load PPO parameters
    ppo_params = locomotion_params.brax_ppo_config(ENV_NAME, "jax")
    
    # Create network factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory,
        preprocess_observations_fn=running_statistics.normalize,
    )
    
    # Create PPO network
    ppo_network = network_factory(obs_size, act_size)
    
    # Load checkpoint
    print(f"Loading checkpoint from {latest_checkpoint}...")
    params = load(latest_checkpoint.as_posix())
    params = (params[0], params[1])
    print("Checkpoint loaded successfully.")
    
    # Create JAX inference function
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(params, deterministic=True)
    
    # Extract normalization parameters
    mean = np.array(params[0].mean["state"])
    std = np.array(params[0].std["state"])
    print(f"Normalization params - mean shape: {mean.shape}, std shape: {std.shape}")
    
    # Create PyTorch model
    print("Creating PyTorch model...")
    pytorch_model = PolicyNetwork(
        obs_dim=obs_size["state"][0],
        action_dim=act_size,
        hidden_sizes=ppo_params.network_factory.policy_hidden_layer_sizes,
        mean=mean,
        std=std,
    )
    pytorch_model.eval()  # Set to evaluation mode
    
    # Transfer weights from JAX to PyTorch
    print("Transferring weights from JAX to PyTorch...")
    transfer_weights_from_jax(params[1]['params'], pytorch_model)
    
    # Test PyTorch model
    test_obs = torch.ones(obs_size["state"][0], dtype=torch.float32)
    with torch.no_grad():
        pytorch_pred = pytorch_model(test_obs)
    print(f"PyTorch prediction sample: {pytorch_pred[:5]}")
    
    # Test JAX model for comparison
    test_input_jax = {
        'state': jp.ones(obs_size["state"]),
        'privileged_state': jp.zeros(obs_size["privileged_state"])
    }
    jax_pred, _ = inference_fn(test_input_jax, jax.random.PRNGKey(0))
    print(f"JAX prediction sample: {jax_pred[:5]}")
    
    # Compare predictions
    diff = np.abs(pytorch_pred.numpy() - np.array(jax_pred))
    print(f"Max difference (JAX vs PyTorch): {np.max(diff)}")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    
    # Ensure output directory exists
    output_dir = epath.Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy input for ONNX export (batch size = 1)
    dummy_input = torch.ones(1, obs_size["state"][0], dtype=torch.float32)
    
    # Export to ONNX
    # Note: Setting dynamo=False to use legacy exporter which better supports
    # save_as_external_data parameter for single-file export
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=11,  # Compatible with Isaac Lab and most frameworks
        do_constant_folding=True,
        input_names=['obs'],
        output_names=['continuous_actions'],
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'continuous_actions': {0: 'batch_size'}
        },
        dynamo=False,  # Use legacy exporter for better control
    )
    print(f"ONNX model saved to: {OUTPUT_PATH}")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    providers = ['CPUExecutionProvider']
    session = rt.InferenceSession(OUTPUT_PATH, providers=providers)
    
    # Get input/output info
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"ONNX input name: {input_name}")
    print(f"ONNX output name: {output_name}")
    
    # Run inference
    onnx_input = {input_name: np.ones((1, obs_size["state"][0]), dtype=np.float32)}
    onnx_pred = session.run([output_name], onnx_input)[0][0]
    print(f"ONNX prediction sample: {onnx_pred[:5]}")
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON:")
    print(f"JAX output:     {jax_pred[:5]}")
    print(f"PyTorch output: {pytorch_pred[:5].numpy()}")
    print(f"ONNX output:    {onnx_pred[:5]}")
    print(f"\nMax difference (JAX vs PyTorch): {np.max(np.abs(jax_pred - pytorch_pred.numpy()))}")
    print(f"Max difference (JAX vs ONNX):    {np.max(np.abs(jax_pred - onnx_pred))}")
    print(f"Max difference (PyTorch vs ONNX): {np.max(np.abs(pytorch_pred.numpy() - onnx_pred))}")
    
    if np.allclose(jax_pred, onnx_pred, atol=1e-5):
        print("\n✓ SUCCESS: ONNX model matches JAX model!")
    else:
        print("\n✗ WARNING: ONNX model does not match JAX model closely.")
        print("  This might be acceptable depending on your tolerance.")
    
    print("\n" + "="*60)
    print("Conversion complete!")

if __name__ == "__main__":
    '''
    python convert_onnx_pytorch.py --env_name Go2FlatTerrain --checkpoint_dir /path/to/your/mujoco_playground/mujoco_playground/logs/Go2FlatTerrain-xxxxxx-xxxxxx/checkpoints/ --output_path /path/to/your/mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/xxx_policy.onnx
    参数：
    --env_name: 环境名称
    --checkpoint_dir: 检查点目录
    --output_path: 输出路径
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    ENV_NAME = args.env_name
    CHECKPOINT_DIR = args.checkpoint_dir
    OUTPUT_PATH = args.output_path
    main()
