import os
import matplotlib.pyplot as plt
import mujoco
import torch
import numpy as np
from scipy.spatial.transform import Rotation

import cv2
import time
import glfw
import OpenGL.GL as gl

os.environ["DISCOVERSE_ASSETS_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)))

from discoverse.envs import SimulatorBase, SceneAlignmentGUI
from discoverse.utils.base_config import BaseConfig
from discoverse.gaussian_renderer import batch_render

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
        "world" : "franka_robotiq.ply",
        # "world" : "franka_bg.ply",

        # "link0" : "franka/link0.ply",
        # "link1" : "franka/link1.ply",
        # "link2" : "franka/link2.ply",
        # "link3" : "franka/link3.ply",
        # "link4" : "franka/link4.ply",
        # "link5" : "franka/link5.ply",
        # "link6" : "franka/link6.ply",
        # "link7" : "franka/link7.ply",

        # "robotiq_base"      : "robotiq/robotiq_base.ply",
        # "left_driver"       : "robotiq/left_driver.ply",
        # "left_coupler"      : "robotiq/left_coupler.ply",
        # "left_spring_link"  : "robotiq/left_spring_link.ply",
        # "left_follower"     : "robotiq/left_follower.ply",

        # "right_driver"      : "robotiq/right_driver.ply",
        # "right_coupler"     : "robotiq/right_coupler.ply",
        # "right_spring_link" : "robotiq/right_spring_link.ply",
        # "right_follower"    : "robotiq/right_follower.ply",
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

    def render(self):
        self.render_cnt += 1

        if not self.config.headless and self.window is not None:
            current_width_s_, current_height_s_ = glfw.get_framebuffer_size(self.window)
            current_width, current_height = int(current_width_s_/self.screen_scale), int(current_height_s_/self.screen_scale)
            self.update_renderer_window_size(current_width, current_height)
            rgb_gl = self.getRgbImg(self.cam_id)

            cam_pos_fixed = np.empty((0, 3))
            cam_xmat_fixed = np.empty((0, 9))
            fovy_fixed = np.empty((0,))

            self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            trans, quat_wxyz = self.getCameraPose(-1)
            rmat = Rotation.from_quat(quat_wxyz[[1,2,3,0]]).as_matrix().flatten() # (9,)
            fovy = self.mj_model.vis.global_.fovy
            
            # 拼接到固定相机数据后面
            cam_pos = np.vstack([cam_pos_fixed, trans])
            cam_xmat = np.vstack([cam_xmat_fixed, rmat])
            fovy_arr = np.concatenate([fovy_fixed, [fovy]])

            bgimg = 2. * torch.ones((fovy_arr.shape[0], current_height, current_width, 3), dtype=torch.float32, device=cam_pos.device, requires_grad=False)
            bgimg[..., [1,2]] = 0.0

            self.gs_renderer.update_gaussians(self.mj_data)
            rgb_tensor, depth_tensor = batch_render(
                self.gs_renderer.gaussians,
                cam_pos,
                cam_xmat,
                current_height,
                current_width,
                fovy_arr,
                bgimg
            )

            # rgb_gs = (255. * torch.clamp(rgb_tensor, 0.0, 1.0)).cpu().numpy()[0]
            img_vis = torch.clamp(0.5 * 255. * rgb_tensor[0] + 0.5 * (255. - torch.from_numpy(rgb_gl).to(rgb_tensor.device).to(torch.float32)), 0., 255.).to(torch.uint8).cpu().numpy()
            img_vis = torch.clamp(0.5 * 255. * rgb_tensor[0] + 0.5 * (torch.from_numpy(rgb_gl).to(rgb_tensor.device).to(torch.float32)), 0., 255.).to(torch.uint8).cpu().numpy()

            try:
                if glfw.window_should_close(self.window):
                    self.running = False
                    return
                    
                glfw.make_context_current(self.window)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)

                if img_vis is not None:
                    img_vis = img_vis[::-1]
                    img_vis = np.ascontiguousarray(img_vis)
                    gl.glDrawPixels(img_vis.shape[1], img_vis.shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_vis.tobytes())
                
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                
                if self.config.sync:
                    current_time = time.time()
                    wait_time = max(1.0/self.render_fps - (current_time - self.last_render_time), 0)
                    if wait_time > 0:
                        time.sleep(wait_time)
                    self.last_render_time = time.time()
                    
            except Exception as e:
                print(f"渲染错误: {e}")

if __name__ == "__main__":
    cfg = FrankaCfg()
    # cfg.gs_model_dict["background"] = "franka_bg.ply"
    exec_node = FrankaBase(cfg)

    exec_node.reset()
    nu = exec_node.mj_model.nu
    init_qpos = exec_node.mj_data.qpos[:nu].copy()
    init_pos = exec_node.mj_model.body(1).pos.copy()
    init_quat = exec_node.mj_model.body(1).quat.copy()

    # Setup Scene Alignment GUI
    gui = SceneAlignmentGUI(exec_node.mj_model, exec_node.mj_data, init_qpos, init_pos, init_quat)

    while exec_node.running:
        gui.update()
        exec_node.view()