import os
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

os.environ["DISCOVERSE_ASSETS_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)))

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

class Ur5eCfg(BaseConfig):
    mjcf_file_path = "xmls/scene_ur5e_robotiq.xml"
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
    rb_link_list   = [
        "base", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link",
        "robotiq_base", "left_coupler", "left_spring_link", "left_follower", "right_driver", "right_coupler", "right_spring_link", "right_follower"
    ]
    obj_list       = []
    use_gaussian_renderer = True
    gs_model_dict = {
        "base"           : "ur5e/base.ply",
        "shoulder_link"  : "ur5e/shoulder_link.ply",
        "upper_arm_link" : "ur5e/upper_arm_link.ply",
        "forearm_link"   : "ur5e/forearm_link.ply",
        "wrist_1_link"   : "ur5e/wrist_1_link.ply",
        "wrist_2_link"   : "ur5e/wrist_2_link.ply",
        "wrist_3_link"   : "ur5e/wrist_3_link.ply",

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

class Ur5eBase(SimulatorBase):
    def __init__(self, config: Ur5eCfg):
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
    cfg = Ur5eCfg()
    cfg.gs_model_dict["background"] = "ur5e_bg.ply"
    exec_node = Ur5eBase(cfg)

    exec_node.reset()
    current_qpos = exec_node.mj_data.qpos[:7].copy()
    qpos_lowers = np.clip(current_qpos - 0.3, exec_node.mj_model.actuator_ctrlrange[:, 0][:7], exec_node.mj_model.actuator_ctrlrange[:, 1][:7])
    qpos_uppers = np.clip(current_qpos + 0.3, exec_node.mj_model.actuator_ctrlrange[:, 0][:7], exec_node.mj_model.actuator_ctrlrange[:, 1][:7])
    qpos_lowers[-1] = 0.0
    qpos_uppers[-1] = 0.82
    qpos_lowers[:-1] = current_qpos[:-1]
    qpos_uppers[:-1] = current_qpos[:-1]

    while exec_node.running:
        # Calculate target position based on time for periodic motion
        t = exec_node.mj_data.time
        period = 2.0  # Total period for one complete cycle
        
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
        
        exec_node.step(action=target_qpos)
