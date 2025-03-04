__all__ = ["GenesisProblem"]

# If you meet "OpenGL.error.Error: Attempt to retrieve context when no valid context" on WSL, please try uncomment the following code.
# import os
# os.environ['PYOPENGL_PLATFORM'] = 'glx'
import copy
from typing import Callable, Dict, Optional, Tuple

import genesis as gs

import torch
import torch.nn as nn
import torch.utils.dlpack
from .Genv import *

from ...core import Problem, use_state
from .utils import get_vmap_model_state_forward

task_to_class = {
    "GraspFixedBlock": GraspFixedBlockEnv,
    "GraspFixedRod": GraspFixedRodEnv,
    "GraspRandomBlock": GraspRandomBlockEnv,
    "GraspRandomRod": GraspRandomRodEnv,
    "WaterFranka": WaterFrankaEnv,
    "ShadowHandBase": ShadowHandBaseEnv,
}


class GenesisProblem(Problem):
    """The Genesis problem wrapper."""

    def __init__(
        self,
        policy: nn.Module,
        task: str,
        max_episode_length: int,
        num_episodes: int,
        visual: bool = False,
        num_envs: int | None = None,
        pop_size: int | None = None,
        reduce_fn: Callable[[torch.Tensor, int], torch.Tensor] = torch.mean,
        device: torch.device | None = None,
    ):
        """Construct a Genesis-based problem.
        Firstly, you need to define a policy model.
        Then you need to set the `task name <https://github.com/RochelleNi/GenesisEnvs>`,
        the maximum episode length, the number of episodes to evaluate for each individual.
        For each individual,
        it will run the policy with the environment for num_episodes times,
        and use the reduce_fn to reduce the rewards (default to average).

        :param policy: The policy model whose forward function is :code:`forward(batched_obs) -> action`.
        :param task: The task name.
        :param max_episode_length: The maximum number of time steps of each episode.
        :param num_episodes: The number of episodes to evaluate for each individual.
        :param visual: Indicate whether to show the environment's visualization. Default to `False`.
        :param num_envs: The number of environments to create. If None, we expect the input to have 1 environment.
        :param pop_size: The size of the population to be evaluated. If None, we expect the input to have a population size of 1.
        :param reduce_fn: The function to reduce the rewards of multiple episodes. Default to `torch.mean`.
        :param device: The device to run the computations on. Defaults to the current default device.

        ## Notice
        The initial key is obtained from `torch.random.get_rng_state()`.

        ## Warning
        This problem does NOT support HPO wrapper (`problems.hpo_wrapper.HPOProblemWrapper`) out-of-box, i.e., the workflow containing this problem CANNOT be vmapped.
        *However*, by setting `pop_size` to the multiplication of inner population size and outer population size, you can still use this problem in a HPO workflow.

        ## Examples
        >>> from evox import problems
        >>> problem = problems.neuroevolution.Brax(
        ...    env_name="swimmer",
        ...    policy=model,
        ...    max_episode_length=1000,
        ...    num_episodes=3,
        ...    pop_size=100,
        ...    rotate_key=False,
        ...)
        """
        super().__init__()
        gs.init(backend=gs.gpu, precision="32")
        device = torch.get_default_device() if device is None else device
        pop_size = 1 if pop_size is None else pop_size
        # Create Genesis environment
        num_envs = 1 if num_envs is None else num_envs
        assert task in task_to_class, f"Task '{task}' is not recognized."
        env = task_to_class[task](vis=visual, device=device, num_envs=num_envs)
        self.obs = torch.zeros(
            env.state_dim,
        )

        # Compile Genesis environment
        self.gene_reset = env.reset
        self.gene_step = env.step
        # self.vmap_brax_reset = jax.jit(vmap_env.reset)
        # self.vmap_brax_step = jax.jit(vmap_env.step)

        # JIT stateful model forward
        self.vmap_init_state, self.vmap_state_forward = get_vmap_model_state_forward(
            model=policy,
            pop_size=pop_size,
            in_dims=(0, 0),
            device=device,
        )
        self.state_forward = torch.compile(use_state(policy))

        copied_policy = copy.deepcopy(policy).to(device)
        self.init_state = copied_policy.state_dict()
        for _name, value in self.init_state.items():
            value.requires_grad = False

        self.reduce_fn = reduce_fn
        self.pop_size = pop_size
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.device = device

    def _evaluate_gene(
        self,
        model_state: Dict[str, torch.Tensor],
        record_trajectory: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if not record_trajectory:
            # check the pop_size in the inputs
            # Take a parameter and check its size
            pop_size = next(iter(model_state.values())).size(0)
            assert (
                pop_size == self.pop_size
            ), f"The actual population size must match the pop_size parameter when creating BraxProblem. Expected: {self.pop_size}, Actual: {pop_size}"

        # For each episode, we need a different random key.
        # For each individual in the population, we need the same set of keys.
        # Loop until environment stops

        if record_trajectory:
            total_reward = torch.zeros(()).to(self.device)
            done_array = torch.tensor((), dtype=bool).to(self.device)
        else:
            done_array = torch.tensor(
                ([False] * self.num_envs * self.num_episodes,), dtype=bool
            )
            total_reward = torch.zeros((pop_size * self.num_envs * self.num_episodes,))
        counter = 0
        if record_trajectory:
            gene_state = self.gene_reset()
            trajectory = [gene_state]
        else:
            gene_state = self.gene_reset()

        while counter < self.max_episode_length and ~done_array.all():
            if record_trajectory:
                model_state, action = self.state_forward(model_state, self.obs)
                gene_state, reward, done = self.gene_step(gene_state, action)
            else:
                model_state, action = self.vmap_state_forward(
                    model_state, self.obs.view(pop_size, self.num_episodes, -1)
                )
                action = action.view(pop_size * self.num_episodes, -1)
                gene_state, reward, done = self.gene_step(gene_state, action)

            done_array = torch.logical_or(done_array, done)
            total_reward += reward
            counter += 1
            if record_trajectory:
                trajectory.append(gene_state)
        # Return
        if record_trajectory:
            return model_state, total_reward, trajectory
        else:
            total_reward = total_reward.view(pop_size, self.num_episodes)
            return model_state, total_reward

    def evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Evaluate the final rewards of a population (batch) of model parameters.

        :param pop_params: A dictionary of parameters where each key is a parameter name and each value is a tensor of shape (batch_size, *param_shape) representing the batched parameters of batched models.

        :return: A tensor of shape (batch_size,) containing the reward of each sample in the population.
        """
        # Merge the given parameters into the initial parameters
        model_state = self.vmap_init_state | pop_params
        # Genesis environment evaluation
        model_state, rewards = self._evaluate_gene(model_state)
        rewards = self.reduce_fn(rewards, dim=-1)
        return rewards
