import torch
import math
import genesis as gs
import numpy as np
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # termination
        "termination_if_roll_greater_than": 180,  # degree
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 3.0,
        "termination_if_y_greater_than": 3.0,
        "termination_if_z_greater_than": 2.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "at_target_threshold": 0.1,
        "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "num_obs": 17,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "target": 10.0,
            "smooth": -1e-4,
            "yaw": 0.01,
            "angular": -2e-4,
            "crash": -10.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


class HoverEnv:
    def __init__(
        self,
        device,
        num_envs=1,
        vis=False,
    ):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.done = torch.tensor(([False] * num_envs,), dtype=bool).to(self.device)
        self.action_space = 4
        self.observation_space = 17

        # 基本配置
        self.env_cfg, self.obs_cfg, self.reward_cfg, self.command_cfg = get_cfgs()
        self.num_obs = self.obs_cfg["num_obs"]
        # self.num_privileged_obs = None
        self.num_actions = self.env_cfg["num_actions"]
        self.num_commands = self.command_cfg["num_commands"]

        self.simulate_action_latency = self.env_cfg["simulate_action_latency"]
        self.dt = 0.01  # 100Hz
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_s"] / self.dt)

        self.obs_scales = self.obs_cfg["obs_scales"]
        self.reward_scales = self.reward_cfg["reward_scales"]

        # 构建场景
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=self.env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 1.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=10),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=vis,
        )

        # 添加平面
        self.scene.add_entity(gs.morphs.Plane())

        # 添加目标（可视化选项控制）
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 1.0),
                    ),
                ),
            )
        else:
            self.target = None

        # 添加相机（可视化选项控制）
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # 添加无人机（或 drone 机器人）
        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=self.device, dtype=gs.tc_float
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=self.device, dtype=gs.tc_float
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/cf2x.urdf")
        )

        # 构建场景（多环境）
        self.scene.build(n_envs=self.num_envs)
        # 定义环境索引（这里采用 torch.arange 便于张量索引）
        self.envs_idx = torch.arange(self.num_envs, device=self.device)

        # 初始化奖励函数及累积奖励记录，并将奖励尺度乘以 dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            )

        # 初始化各类 buffer
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = dict()  # 额外日志信息

        # 按 GraspFixedBlockEnv 结构，调用 build_env 进行环境初始化
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.build_env()

    def build_env(self):
        """根据所有环境索引重置无人机及相关状态"""
        self.reset_idx(self.envs_idx)

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # 重置无人机基座状态
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(
            self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.drone.set_quat(
            self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # 重置相对位置信息
        self.rel_pos[envs_idx] = self.commands[envs_idx] - self.base_pos[envs_idx]
        self.last_rel_pos[envs_idx] = (
            self.commands[envs_idx] - self.last_base_pos[envs_idx]
        )

        # 重置动作、计时 buffer
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # 重置每个 episode 的累计奖励，并重采指令
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # return self.obs_buf, None
        return self.obs_buf

    def _resample_commands(self, envs_idx):
        # 分别从给定区间内采样各个维度的指令
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg["pos_x_range"], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["pos_y_range"], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg["pos_z_range"], (len(envs_idx),), self.device
        )
        if self.target is not None:
            self.target.set_pos(
                self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx
            )

    def _at_target(self):
        # 判断相对位置是否小于阈值，返回满足条件的环境索引
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .flatten()
        )
        return at_target

    def step(self, actions):
        # 限幅动作
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        exec_actions = self.actions

        # 控制 propeller 转速（14468 为 hover 时的 rpm）
        self.drone.set_propellels_rpm(
            ((1 + exec_actions * 0.8) * 14468.429183500699).cpu().detach().numpy()
        )
        self.scene.step()

        # 更新 episode 长度及上一步状态
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            )
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # 当接近目标时重采指令
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # 检查终止条件：超过最大 episode 长度或各项物理量超限
        self.crash_condition = (
            (
                torch.abs(self.base_euler[:, 1])
                > self.env_cfg["termination_if_pitch_greater_than"]
            )
            | (
                torch.abs(self.base_euler[:, 0])
                > self.env_cfg["termination_if_roll_greater_than"]
            )
            | (
                torch.abs(self.rel_pos[:, 0])
                > self.env_cfg["termination_if_x_greater_than"]
            )
            | (
                torch.abs(self.rel_pos[:, 1])
                > self.env_cfg["termination_if_y_greater_than"]
            )
            | (
                torch.abs(self.rel_pos[:, 2])
                > self.env_cfg["termination_if_z_greater_than"]
            )
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        self.reset_buf = (
            self.episode_length_buf > self.max_episode_length
        ) | self.crash_condition

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        # 对需要重置的环境执行重置操作
        self.reset_idx((self.reset_buf).nonzero(as_tuple=False).flatten())

        # 计算奖励（各项奖励函数累加）
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # 计算观测值
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        # return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras
        return self.obs_buf, self.rew_buf, self.done

    def get_observations(self):
        return self.obs_buf

    # def get_privileged_observations(self):
    #     return None

    # ------------------- 各奖励函数 -------------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(
            torch.square(self.rel_pos), dim=1
        )
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = (
            torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159
        )  # 转为弧度计算 yaw_reward
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
