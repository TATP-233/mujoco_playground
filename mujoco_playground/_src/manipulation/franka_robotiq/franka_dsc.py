import os
import mujoco
import torch
import numpy as np
from scipy.spatial.transform import Rotation

os.environ["DISCOVERSE_ASSETS_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)))

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

class FrankaCfg(BaseConfig):
    mjcf_file_path = "xmls/panda_robotiq.xml"
    decimation     = 4
    timestep       = 0.005
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" : 720,
    }
    init_qpos = np.zeros(7)
    obs_rgb_cam_id  = None

    use_gaussian_renderer = True
    gs_model_dict = {
        # "world" : "franka_robotiq.ply",

        "link0" : "franka/link0.ply",
        "link1" : "franka/link1.ply",
        "link2" : "franka/link2.ply",
        "link3" : "franka/link3.ply",
        "link4" : "franka/link4.ply",
        "link5" : "franka/link5.ply",
        "link6" : "franka/link6.ply",
        "link7" : "franka/link7.ply",

        "robotiq_base"      : "robotiq/robotiq_base.ply",
        "left_driver"       : "robotiq/left_driver.ply",
        "left_coupler"      : "robotiq/left_coupler.ply",
        "left_spring_link"  : "robotiq/left_spring_link.ply",
        "left_follower"     : "robotiq/left_follower.ply",

        "right_driver"      : "robotiq/right_driver.ply",
        "right_coupler"     : "robotiq/right_coupler.ply",
        "right_spring_link" : "robotiq/right_spring_link.ply",
        "right_follower"    : "robotiq/right_follower.ply",
    }

class FrankaBase(SimulatorBase):
    def __init__(self, config: FrankaCfg):
        super().__init__(config)

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key("home").id)

    def updateControl(self, action):
        self.mj_data.ctrl[:] = action[:self.mj_model.nu]

    def checkTerminated(self):
        return False

    def getObservation(self):
        return None

    def getPrivilegedObservation(self):
        return None

    def getReward(self):
        return None

if __name__ == "__main__":
    cfg = FrankaCfg()
    cfg.gs_model_dict["background"] = "franka_bg.ply"
    exec_node = FrankaBase(cfg)

    exec_node.reset()
    nu = exec_node.mj_model.nu

    # while exec_node.running:
    #     exec_node.view()

    exec_node.reset()
    nu = exec_node.mj_model.nu
    current_qpos = exec_node.mj_data.qpos[:nu].copy()
    qpos_lowers = np.clip(current_qpos - 0.75, exec_node.mj_model.jnt_range[:nu, 0], exec_node.mj_model.jnt_range[:nu, 1])
    qpos_uppers = np.clip(current_qpos + 0.75, exec_node.mj_model.jnt_range[:nu, 0], exec_node.mj_model.jnt_range[:nu, 1])

    while exec_node.running:
        # Calculate target position based on time for periodic motion
        t = exec_node.mj_data.time
        period = 5.0  # Total period for one complete cycle
        
        # Determine phase: 0-0.33 (current->lower), 0.33-0.67 (lower->upper), 0.67-1.0 (upper->lower)
        phase = (t % period) / period
        
        if phase < 0.33:
            # current -> lower
            alpha = phase / 0.33
            target_qpos = current_qpos * (1 - alpha) + qpos_lowers * alpha
        elif phase < 0.67:
            # lower -> upper
            alpha = (phase - 0.33) / 0.34
            target_qpos = qpos_lowers * (1 - alpha) + qpos_uppers * alpha
        else:
            # upper -> lower
            alpha = (phase - 0.67) / 0.33
            target_qpos = qpos_uppers * (1 - alpha) + qpos_lowers * alpha
        
        # exec_node.mj_data.qpos[:nu] = target_qpos
        exec_node.view()
