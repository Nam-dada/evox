import sys
import os

# 将项目根目录添加到sys.path
os.environ["PYOPENGL_PLATFORM"] = "glx"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evox.workflows import StdWorkflow, EvalMonitor
from evox.algorithms import NSGA2
from evox.problems.neuroevolution.genesis import GenesisProblem
import time
import torch
import torch.nn as nn
from evox.utils import ParamsAndVector


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(nn.Linear(6, 4), nn.Sigmoid(), nn.Linear(4, 8))

    def forward(self, x):
        x = self.features(x)
        return torch.argmax(x)


# Make sure that the model is on the same device, better to be on the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# Reset the random seed
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Initialize the MLP model
model = SimpleMLP().to(device)

adapter = ParamsAndVector(dummy_model=model)

# Set the population size
POP_SIZE = 2
OBJs = 2

model_params = dict(model.named_parameters())
pop_center = adapter.to_vector(model_params)
lower_bound = torch.full_like(pop_center, -5)
upper_bound = torch.full_like(pop_center, 5)

algorithm = NSGA2(
    pop_size=POP_SIZE,
    n_objs=OBJs,
    lb=lower_bound,
    ub=upper_bound,
    device=device,
)
algorithm.setup()

obs_norm = {
    "clip_val": 5.0,
    "std_min": 1e-6,
    "std_max": 1e6,
}

# Initialize the Brax problem
problem = GenesisProblem(
    policy=model,
    task="GraspFixedBlock",
    max_episode_length=1000,
    num_episodes=3,
    visual=True,
    num_envs=1,
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
max_generation = 50

times = []
start_time = time.perf_counter()
# Run the workflow
for i in range(max_generation):
    if i % 10 == 0:
        print(f"Generation {i}")
    workflow.step()
    times.append(time.perf_counter() - start_time)

monitor = workflow.get_submodule("monitor")
print(f"Time history: {times}")
print(f"Fitness history: {monitor.get_fitness_history()}")
