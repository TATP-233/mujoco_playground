阅读参考文件：
文件1：mujoco_playground/_src/manipulation/franka_emika_panda/pick_cartesian.py
文件2：mujoco_playground/_src/manipulation/airbot_play/pick.py
在mujoco_playground/_src/manipulation/airbot_play中编写一个pick_cartesian.py

说明：
1. 模型加载部分参考文件1，其他应严格按照文件2的写法
2. airbot_play是6自由度，panda是7自由度，注意关节数不同带来的影响
3. 注意：执行Python命令前请先激活conda环境：discoverse


参考/home/ghz/Work/Research/roboArena/examples/airbot_play_pick_cube.py中的ik用法，将mujoco_playground/_src/manipulation/airbot_play/pick_cartesian.py中机械臂的关节空间控制改为笛卡尔空间控制