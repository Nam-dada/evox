import sys
import os

# 将项目根目录添加到sys.path
os.environ["PYOPENGL_PLATFORM"] = "glx"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evox.workflows import StdWorkflow, EvalMonitor
from evox.algorithms import OpenES
from evox.problems.neuroevolution.genesis import GenesisProblem
import time
import torch
import torch.nn as nn

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

from evox.utils import ParamsAndVector

# (observation space, inner space) () (inner space, action space)


OBS_SIZE = 17
AC_SIZE = 4


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(OBS_SIZE, 4), nn.Tanh(), nn.Linear(4, AC_SIZE)
        )

    def forward(self, x):
        return torch.tanh(self.features(x))


# Make sure that the model is on the same device, better to be on the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# Reset the random seed
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Initialize the MLP model
model = SimpleMLP().to(device)

adapter = ParamsAndVector(dummy_model=model)

# Set the population size
POP_SIZE = 8

model_params = dict(model.named_parameters())
pop_center = adapter.to_vector(model_params)


algorithm = OpenES(
    pop_size=POP_SIZE,
    center_init=pop_center,
    learning_rate=0.1,
    noise_stdev=0.1,
)
algorithm.setup()


# Initialize the Genesis problem
problem = GenesisProblem(
    policy=model,
    task="Hover",
    max_episode_length=100,
    num_episodes=1,
    visual=True,
    pop_size=POP_SIZE,
    device=device,
)

# set an monitor, and it can record the top 3 best fitnesses
monitor = EvalMonitor(
    device=device,
)
monitor.setup()

workflow = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=monitor,
    opt_direction="max",
    solution_transform=adapter,
    device=device,
)

# Set the maximum number of generations
max_generation = 10

times = []
start_time = time.perf_counter()
# Run the workflow
for i in range(max_generation):
    if i % 1 == 0:
        print(f"Generation {i}")
    workflow.step()
    times.append(time.perf_counter() - start_time)

monitor = workflow.get_submodule("monitor")
print(f"Time history: {times}")
print(f"Fitness history: {monitor.get_fitness_history()}")

# env = HoverEnv(
#     num_envs=args.num_envs,
#     env_cfg=env_cfg,
#     obs_cfg=obs_cfg,
#     reward_cfg=reward_cfg,
#     command_cfg=command_cfg,
#     show_viewer=args.vis,
# )


# if __name__ == "__main__":
#     # 初始化 genesis（使用 GPU 和 32 位精度）
#     gs.init(backend=gs.gpu, precision="32")

#     # 构造配置参数（请根据实际需求修改各项参数）
#     env_cfg = {
#         "num_actions": 4,
#         "simulate_action_latency": False,
#         "episode_length_s": 10,
#         "max_visualize_FPS": 60,
#         "visualize_target": True,
#         "visualize_camera": True,
#         "base_init_pos": [0.0, 0.0, 1.0],
#         "base_init_quat": [1.0, 0.0, 0.0, 0.0],
#         "at_target_threshold": 0.1,
#         "termination_if_pitch_greater_than": 30,
#         "termination_if_roll_greater_than": 30,
#         "termination_if_x_greater_than": 2,
#         "termination_if_y_greater_than": 2,
#         "termination_if_z_greater_than": 2,
#         "termination_if_close_to_ground": 0.2,
#         "clip_actions": 1.0,
#     }
#     obs_cfg = {
#         "num_obs": 10,
#         "obs_scales": {
#             "rel_pos": 1.0,
#             "lin_vel": 1.0,
#             "ang_vel": 1.0,
#         },
#     }
#     reward_cfg = {
#         "reward_scales": {
#             "target": 1.0,
#             "smooth": 1.0,
#             "yaw": 1.0,
#             "angular": 1.0,
#             "crash": 1.0,
#         },
#         "yaw_lambda": -0.1,
#     }
#     command_cfg = {
#         "num_commands": 3,
#         "pos_x_range": (-1, 1),
#         "pos_y_range": (-1, 1),
#         "pos_z_range": (0.5, 1.5),
#     }

#     # 实例化环境并启动可视化
#     env = HoverEnv(
#         num_envs=1,
#         env_cfg=env_cfg,
#         obs_cfg=obs_cfg,
#         reward_cfg=reward_cfg,
#         command_cfg=command_cfg,
#         show_viewer=True,
#         device="cuda",
#     )

#     # 示例：调用 reset 和 step
#     obs, _ = env.reset()
#     dummy_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
#     obs, _, rew, reset_buf, extras = env.step(dummy_actions)
#     print("Observation:", obs)
#     print("Reward:", rew)
#     print("Reset flags:", reset_buf)
