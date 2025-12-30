
## 安装

1. `git clone https://github.com/TATP-233/mujoco_playground.git && cd mujoco_playground`
2. 新建conda环境，python >= 3.10
3. `pip install -U "jax[cuda12]"`，验证安装
    `python -c "import jax; print(jax.default_backend())"` 应该打印gpu
4. 安装playground：`pip install -e ".[all]"`，验证安装`python -c "import mujoco_playground"`
5. 安装3dgs依赖：`pip install gsplat`
6. 一般会遇到jax-cudnn不匹配的问题，重新执行一遍第3步即可：`pip install -U "jax[cuda12]"`

## 验证第一个vision-ppo的例子，预计训练5小时左右

设置最大iterations：对于视觉PandaPickCube任务来说，一般设置小一点（2000步），`mujoco_playground/config/manipulation_params.py line273 max_iterations`

这个任务是franka机械臂，依靠两个视角的视觉输入，抓取方块并移动到指定位置，具体的任务定义如下：
`mujoco_playground/_src/manipulation/franka_emika_panda/pick.py`，其中二值的reward见`state.metrics["reward/success"]`

train：

`XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python learning/train_rsl_rl.py --env_name=PandaPickCube --vision=True  --num_envs=2048`

默认开tensorboard。如果爆显存，设置num_envs小一点，还有设置`mujoco_playground/_src/wrapper_torch.py line306 BatchSplatConfig.minibatch` 设置小一点。

play：

`python learning/train_rsl_rl.py --env_name=PandaPickCube --vision=True --play_only --load_run_name PandaPickCube-XXXX`

保存为`rollout.mp4`，视频长这样

![Img](3dgs/image.png)

自动轨迹生成和更多场景马上就ready。