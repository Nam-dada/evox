import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch
from torch import nn

from src.core import jit_class, Problem, Algorithm, trace_impl, batched_random
from src.workflows import StdWorkflow
from src.problems.hpo_wrapper import HPOProblemWrapper, HPOFitnessMonitor


if __name__ == "__main__":

    @jit_class
    class BasicProblem(Problem):

        def __init__(self):
            super().__init__()

        def evaluate(self, x: torch.Tensor):
            return (x * x).sum(-1)

    @jit_class
    class BasicAlgorithm(Algorithm):

        def __init__(self, pop_size: int):
            super().__init__()
            self.pop_size = pop_size

        def setup(self, lb: torch.Tensor, ub: torch.Tensor):
            assert (
                lb.ndim == 1 and ub.ndim == 1
            ), f"Lower and upper bounds shall have ndim of 1, got {lb.ndim} and {ub.ndim}"
            assert (
                lb.shape == ub.shape
            ), f"Lower and upper bounds shall have same shape, got {lb.ndim} and {ub.ndim}"
            self.lb = lb
            self.ub = ub
            self.dim = lb.shape[0]
            self.pop = nn.Buffer(
                torch.empty(self.pop_size, lb.shape[0], dtype=lb.dtype, device=lb.device)
            )
            self.fit = nn.Buffer(torch.empty(self.pop_size, dtype=lb.dtype, device=lb.device))
            return self

        def step(self):
            pop = torch.rand(self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device)
            pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
            self.pop.copy_(pop)
            self.fit.copy_(self.evaluate(pop))

        @trace_impl(step)
        def trace_step(self):
            pop = batched_random(
                torch.rand, self.pop_size, self.dim, dtype=self.lb.dtype, device=self.lb.device
            )
            pop = pop * (self.ub - self.lb)[None, :] + self.lb[None, :]
            self.pop = pop
            self.fit = self.evaluate(pop)

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    algo = BasicAlgorithm(pop_size=10)
    algo.setup(-10 * torch.ones(2), 10 * torch.ones(2))
    prob = BasicProblem()
    monitor = HPOFitnessMonitor()
    workflow = StdWorkflow()
    workflow.setup(algo, prob, monitor=monitor)

    hpo_prob = HPOProblemWrapper(iterations=9, num_instances=7)
    hpo_prob.setup(workflow)
    params = HPOProblemWrapper.extract_parameters(hpo_prob.init_state)
    print(params)
    print(hpo_prob.evaluate(params))