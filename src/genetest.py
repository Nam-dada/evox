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


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(nn.Linear(6, 4), nn.Sigmoid(), nn.Linear(4, 8))

    def forward(self, x):
        x = self.features(x)
        return torch.argmax(x, dim=-1)


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


# Initialize the Brax problem
problem = GenesisProblem(
    policy=model,
    task="GraspFixedBlock",
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
max_generation = 50

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
