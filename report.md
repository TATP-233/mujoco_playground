# PandaPickCube 和 AirbotPlay 的 VisonPPO训练效果不一致问题修复报告

在使用 PandaPickCube 和 AirbotPlay 进行 VisonPPO 训练时，发现两者的训练效果存在显著差异。具体表现为在相同的训练条件下，PandaPickCube 的训练效果明显优于 AirbotPlay。
训练命令分别为：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python learning/train_rsl_rl.py --env_name=PandaPickCube --vision=True --num_envs=20
```

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python learning/train_rsl_rl.py --env_name=AirbotPlayPickCube --vision=True --num_envs=20
```

推理命令分别为：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python learning/train_rsl_rl.py --env_name=PandaPickCube --vision=True --num_envs=20 --play_only --use_dr --load_run_name=PandaPickCube-20260107-033818
```

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python learning/train_rsl_rl.py --env_name=AirbotPlayPickCube --vision=True --num_envs=20 --play_only --use_dr --load_run_name=AirbotPlayPickCube-20260107-033818
```

可以看到，上述命令仅仅更换了环境名称，其他参数保持一致。然而，经过相同的训练时间，推理结果却大相径庭。具体表现为：

PandaPickCube在接近5h时已经有较大的`reward/lifted`和`reward/success`，并且跑推理能正常抓取并抬高积木：
```txt
################################################################################
                     Learning iteration 685/100000                      

                       Computation: 3972 steps/s (collection: 18.093s, learning 2.530s)
             Mean action noise std: 1.64
          Mean value_function loss: 4231.2563
               Mean surrogate loss: 0.0087
                 Mean entropy loss: 15.2527
                       Mean reward: 2210.38
               Mean episode length: 149.23
           Mean episode box_target: 0.5426
          Mean episode gripper_box: 0.8830
   Mean episode no_floor_collision: 0.9993
        Mean episode out_of_bounds: 0.0000
                     reward/lifted: 1.7979
                    reward/success: 0.6504
    Mean episode robot_target_qpos: 0.2429
--------------------------------------------------------------------------------
                   Total timesteps: 56197120
                    Iteration time: 20.62s
                      Time elapsed: 04:37:23
                               ETA: 21:18:46
```

而AirbotPlayPickCube任务在接近5h时`reward/lifted`和`reward/success`几乎为0，推理时表现为仅仅移动到目标位置但无法成功抓取和抬高积木（要么夹爪关闭撞到物块上，要么夹爪张开，二指位于物块两侧，但不闭合抓取），就算去掉环境reset时对物块位置和目标位置的随机化，设置为固定的可达值，也没有明显改善：

```txt
################################################################################
                     Learning iteration 2860/100000                     

                       Computation: 4967 steps/s (collection: 13.962s, learning 2.528s)
             Mean action noise std: 2.67
          Mean value_function loss: 0.0018
               Mean surrogate loss: 0.0058
                 Mean entropy loss: 16.2607
                       Mean reward: 1731.43
               Mean episode length: 150.00
           Mean episode box_target: 0.7361
          Mean episode gripper_box: 0.9913
   Mean episode no_floor_collision: 1.0000
        Mean episode out_of_bounds: 0.0000
                     reward/lifted: 0.0000
                    reward/success: 0.0000
    Mean episode robot_target_qpos: 0.7082
--------------------------------------------------------------------------------
                   Total timesteps: 234373120
                    Iteration time: 16.49s
                      Time elapsed: 13:01:14
                               ETA: 10:05:29
```

两个环境的文件分别位于：mujoco_playground/_src/manipulation/franka_emika_panda/pick.py和mujoco_playground/_src/manipulation/airbot_play/pick.py。

一些初步洞察：

- 经过文件内容对比，似乎差异仅仅是导入模型不同
- panda机械臂是7自由度，airbot play机械臂是6自由度，可能导致动作空间和控制方式不同，但感觉不至于导致训练效果差异如此之大
- airbot play和panda的夹爪开合距离都是0 - 0.04
- 在mujoco viewer中手动控制airbot play机械臂可以成功抓取和抬高积木，说明夹爪的驱动参数大概率没问题，并且查看endpoint的位置在左右手指之间，从而能正确用于计算相关抓取奖励

目标：找出在几乎相同设置下，airbot play训练效果差的可能原因
