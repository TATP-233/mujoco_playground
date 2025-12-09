# 指南：在 JAX/MuJoCo 环境中集成 PyTorch 自定义渲染器

## 1. 核心原理
我们需要在 JAX (物理模拟/训练) 和 PyTorch (渲染) 两个完全不同的框架之间建立数据通路。

架构设计：采用“外挂式”渲染。JAX 负责物理计算，算出所有几何体的位置；然后通过“传送门”将数据发给 PyTorch 进行渲染；最后将图像传回 JAX 用于神经网络输入。
数据桥梁：使用 DLPack 协议。它允许 JAX 和 PyTorch 共享同一块 GPU 显存，实现零拷贝 (Zero-Copy) 传输，速度极快，不会经过 CPU。
批处理 (Batching)：为了利用 GPU 并行能力，我们将渲染器包装器（Wrapper）放置在 JAX 的自动向量化（vmap）层之外。这意味着 PyTorch 渲染器一次性接收所有环境（例如 4096 个）的数据，进行批量渲染。

## 2. 实现步骤
第一步：创建渲染桥梁与包装器
在 mujoco_playground 根目录下创建一个新文件 gs_renderer.py。这个文件包含两部分：
1. GSRendererBridge：运行在 Python/PyTorch 环境中，负责“接货”和“发货”。
2. GaussianSplattingWrapper：运行在 JAX 环境中，负责拦截环境状态并调用桥梁。

文件内容 (gs_renderer.py)：
```python
import torch
import jax
import jax.numpy as jp
from jax import dlpack as jdl
import torch.utils.dlpack as tdl
import numpy as np
from mujoco_playground._src import mjx_env

# ==========================================
# Part 1: 运行在 JAX 外面 (Python/PyTorch 环境)
# ==========================================

class My3DGSRenderer:
    """
    这里替换成你真实的 3DGS 渲染器代码。
    """
    def __init__(self, batch_size, width, height):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        print(f"[Renderer] 初始化 3DGS: Batch={batch_size}, Res={width}x{height}")

    def render(self, geom_pos, geom_mat):
        # 输入:
        # geom_pos: (Batch, N_Geoms, 3)
        # geom_mat: (Batch, N_Geoms, 3, 3)
        
        # 模拟渲染输出: (Batch, Camera, Height, Width, 3)
        # 实际使用时，请调用你的 CUDA 渲染核
        fake_images = torch.zeros(
            (self.batch_size, self.height, self.width, 3), 
            device='cuda', 
            dtype=torch.float32
        )
        return fake_images

class GSRendererBridge:
    def __init__(self, batch_size, width=64, height=64):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.renderer = My3DGSRenderer(batch_size, width, height)

    def render_batch_callback(self, geom_xpos_jax, geom_xmat_jax):
        """
        这是 JAX pure_callback 调用的入口。
        输入是 JAX 数组 (在 GPU 上)，输出必须是 JAX 数组。
        """
        # 1. JAX -> PyTorch (零拷贝)
        # 此时 geom_xpos_jax 已经是 Batch 后的数据了 (例如 4096, N, 3)
        geom_pos_torch = tdl.from_dlpack(jdl.to_dlpack(geom_xpos_jax))
        geom_mat_torch = tdl.from_dlpack(jdl.to_dlpack(geom_xmat_jax))

        # 2. 执行渲染 (PyTorch)
        with torch.no_grad():
            image_torch = self.renderer.render(geom_pos_torch, geom_mat_torch)

        # 3. PyTorch -> JAX (零拷贝)
        image_jax = jdl.from_dlpack(tdl.to_dlpack(image_torch))
        
        return image_jax

# ==========================================
# Part 2: 运行在 JAX 里面 (编译后的环境)
# ==========================================

class GaussianSplattingWrapper:
    def __init__(self, env, batch_size, width=64, height=64):
        self.env = env
        self.batch_size = batch_size
        self.width = width
        self.height = height
        
        # 实例化桥梁
        self.bridge = GSRendererBridge(batch_size, width, height)
        
        # 定义 JAX 需要的输出形状信息
        self.image_shape = (batch_size, height, width, 3)
        self.image_dtype = jp.float32

    def reset(self, rng):
        state = self.env.reset(rng)
        return self._add_render_to_state(state)

    def step(self, state, action):
        state = self.env.step(state, action)
        return self._add_render_to_state(state)

    def _add_render_to_state(self, state):
        # 获取几何体数据
        # 因为这个 Wrapper 加在 vmap 之后，所以这里的 state 已经是 Batch 的了
        # Shape: (Batch_Size, N_Geoms, 3)
        geom_xpos = state.data.geom_xpos
        geom_xmat = state.data.geom_xmat

        # 调用传送门
        # jax.pure_callback 会暂停 JAX 流水线，去执行 Python 函数
        images = jax.pure_callback(
            self.bridge.render_batch_callback,
            jax.ShapeDtypeStruct(self.image_shape, self.image_dtype), # 告诉 JAX 返回值长啥样
            geom_xpos, 
            geom_xmat
        )

        # 将图像注入到 obs 中
        # 注意：PPO 网络通常需要字典输入来处理多模态
        if isinstance(state.obs, dict):
            new_obs = state.obs.copy()
            new_obs['pixels'] = images # 这里的 key 要和 PPO 配置对应
        else:
            new_obs = {
                'state': state.obs,
                'pixels': images
            }
            
        return state.replace(obs=new_obs)

    def __getattr__(self, name):
        return getattr(self.env, name)
```

第二步：修改训练脚本
我们需要修改 train_jax_ppo.py，将我们的 Wrapper 插入到正确的位置。

关键原则：必须把 GaussianSplattingWrapper 放在 wrapper.wrap_for_brax_training 之后。因为 wrap_for_brax_training 内部执行了 vmap，只有在它之后，环境的数据才是 Batch 的。

修改 train_jax_ppo.py 的 main 函数：

1. 导入模块：
```python
import sys
# 确保能找到 gs_renderer.py
sys.path.append('/home/yufei/mujoco_playground') 
from gs_renderer import GaussianSplattingWrapper
```
2. 替换环境创建逻辑：
```python
# ... (在 main 函数中，约 250 行左右)

# 1. 禁用内置 Vision
# 我们不需要 Madrona，所以告诉配置不要用 vision
# 这样 env_cfg 就会创建纯状态的 MuJoCo 环境
if _VISION.value:
    env_cfg.vision = False 
    # 注意：可能还需要调整 env_cfg 里的其他参数，确保它不依赖 Madrona

# 2. 加载基础环境
env = registry.load(_ENV_NAME.value, config=env_cfg)

# 3. 应用 Brax 基础包装 (Vmap, Episode, AutoReset)
# 关键：这里 vision=False。
# 这个函数返回后，env.step 接收和返回的都是 (Batch_Size, ...) 的数据
env = wrapper.wrap_for_brax_training(
    env,
    vision=False, 
    num_vision_envs=ppo_params.num_envs,
    episode_length=ppo_params.episode_length,
    action_repeat=ppo_params.action_repeat,
    randomization_fn=training_params.get("randomization_fn"),
)

# 4. 【注入点】挂载自定义渲染器
if _VISION.value:
    print(f"正在挂载自定义 3DGS 渲染器 (Batch={ppo_params.num_envs})...")
    env = GaussianSplattingWrapper(
        env, 
        batch_size=ppo_params.num_envs,
        width=64,  # 根据你的需求调整
        height=64
    )
    
    # 5. 调整 PPO 网络配置
    # 告诉 PPO 网络，现在的 obs 是个字典，图像在 'pixels' 里
    # 这一步取决于你的 PPO 实现细节，通常 brax 的 ppo_networks_vision 会自动处理
    # 如果需要，你可能要手动设置 ppo_params.network_factory 的参数

# ... (后续训练代码保持不变)
```
