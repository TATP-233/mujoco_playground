import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, TextBox
import mujoco
import torch
import numpy as np
from scipy.spatial.transform import Rotation

import cv2
import time
import glfw
import OpenGL.GL as gl

os.environ["DISCOVERSE_ASSETS_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)))

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig
from discoverse.gaussian_renderer.batch_gs_renderer import batch_render

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

            self.update_gs_scene()
            rgb_tensor, depth_tensor = batch_render(
                self.gs_renderer.renderer.gaussians,
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

    # Setup Matplotlib GUI
    total_sliders = 3 + 4 + nu
    fig = plt.figure(figsize=(8, total_sliders * 0.3 + 2))
    plt.subplots_adjust(left=0.25, bottom=0.1, right=0.95, top=0.95)
    
    sliders_pos = []
    sliders_quat = []
    sliders_joint = []
    textboxes = []

    def create_slider_textbox(idx, label, valmin, valmax, valinit):
        ax_slider = plt.axes([0.20, 0.92 - idx * (0.85 / total_sliders), 0.55, 0.02])
        ax_text = plt.axes([0.80, 0.92 - idx * (0.85 / total_sliders), 0.15, 0.02])
        
        slider = Slider(ax_slider, label, valmin, valmax, valinit=valinit)
        textbox = TextBox(ax_text, '', initial=f"{valinit:.3f}")
        
        def submit(text):
            try:
                val = float(text)
                val = np.clip(val, valmin, valmax)
                slider.set_val(val)
            except ValueError:
                pass
            textbox.set_val(f"{slider.val:.3f}")

        textbox.on_submit(submit)
        
        def update_text(val):
            textbox.set_val(f"{val:.3f}")
            
        slider.on_changed(update_text)
        textboxes.append(textbox)
        return slider

    idx = 0
    
    # Position Sliders
    for i, label in enumerate(['x', 'y', 'z']):
        s = create_slider_textbox(idx, f'Pos {label}', exec_node.mj_model.body(1).pos[i] - 0.1, exec_node.mj_model.body(1).pos[i] + 0.1, init_pos[i])
        sliders_pos.append(s)
        idx += 1
        
    # Quaternion Sliders
    for i, label in enumerate(['w', 'x', 'y', 'z']):
        s = create_slider_textbox(idx, f'Quat {label}', -1.0, 1.0, init_quat[i])
        sliders_quat.append(s)
        idx += 1

    # Joint Sliders
    for i in range(nu):
        low = exec_node.mj_model.joint(i).range[0]
        high = exec_node.mj_model.joint(i).range[1]
        s = create_slider_textbox(idx, exec_node.mj_model.joint(i).name if exec_node.mj_model.joint(i).name else f'Joint {i}', low, high, init_qpos[i])
        sliders_joint.append(s)
        idx += 1

    def update_pos(val):
        p = np.array([s.val for s in sliders_pos])
        exec_node.mj_model.body(1).pos[:] = p

    # Use a mutable object to store state instead of nonlocal
    state = {'is_updating_quat': False}
    
    def update_quat(val):
        if state['is_updating_quat']: return
        state['is_updating_quat'] = True
        
        q = np.array([s.val for s in sliders_quat])
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q /= norm
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        
        exec_node.mj_model.body(1).quat[:] = q
        
        for i, s in enumerate(sliders_quat):
            if s.val != q[i]:
                s.set_val(q[i])
                
        state['is_updating_quat'] = False

    def update_joint(val):
        for i, s in enumerate(sliders_joint):
            exec_node.mj_data.qpos[i] = s.val
            
    for s in sliders_pos: s.on_changed(update_pos)
    for s in sliders_quat: s.on_changed(update_quat)
    for s in sliders_joint: s.on_changed(update_joint)

    def reset(event):
        for s in sliders_pos: s.reset()
        for s in sliders_quat: s.reset()
        for s in sliders_joint: s.reset()

    def print_info(event):
        print("-" * 30)
        print(f"Pos:   {exec_node.mj_model.body(1).pos}")
        print(f"Quat:  {exec_node.mj_model.body(1).quat}")
        print(f"Joint: {exec_node.mj_data.qpos[:nu]}")

    ax_reset = plt.axes([0.25, 0.02, 0.3, 0.05])
    btn_reset = Button(ax_reset, 'Reset')
    btn_reset.on_clicked(reset)

    ax_print = plt.axes([0.6, 0.02, 0.3, 0.05])
    btn_print = Button(ax_print, 'Print')
    btn_print.on_clicked(print_info)

    plt.show(block=False)

    while exec_node.running:
        plt.pause(0.001)
        exec_node.view()