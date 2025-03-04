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
        pop_size: int,
        max_episode_length: int = 1000,
        num_episodes: int = 1,
        visual: bool = False,
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

        self.reduce_fn = reduce_fn
        self.pop_size = pop_size
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.device = device

        # gs.init(backend=gs.gpu, precision="32", logging_level="error")
        gs.init(backend=gs.gpu, precision="32")

        # Create Genesis environment
        assert task in task_to_class, f"Task '{task}' is not recognized."
        self.env = task_to_class[task](
            vis=visual, device=self.device, num_envs=pop_size * num_episodes
        )

        # JIT stateful model forward
        self.vmap_init_state, self.vmap_state_forward = get_vmap_model_state_forward(
            model=policy,
            pop_size=pop_size,
            in_dims=(0, 0),
            device=self.device,
        )

    def _evaluate_gene(
        self,
        model_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        pop_size = next(iter(model_state.values())).size(0)
        assert (
            pop_size == self.pop_size
        ), f"The actual population size must match the pop_size parameter when creating BraxProblem. Expected: {self.pop_size}, Actual: {pop_size}"

        done_array = torch.tensor(
            ([False] * self.pop_size * self.num_episodes,), dtype=bool
        ).to(self.device)
        total_reward = torch.zeros((self.pop_size * self.num_episodes,)).to(self.device)
        counter = 0

        obs = self.env.reset()

        while (counter < self.max_episode_length) and ~done_array.all():

            model_state, action = self.vmap_state_forward(
                model_state, obs.view(self.pop_size, self.num_episodes, -1)
            )
            action = action.view(self.pop_size * self.num_episodes)
            obs, reward, done = self.env.step(action)

            done_array = torch.logical_or(
                done_array, done.view(self.pop_size * self.num_episodes)
            )
            total_reward += reward.view(self.pop_size * self.num_episodes)
            counter += 1

        total_reward = total_reward.view(self.pop_size, self.num_episodes)
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
        print(rewards)
        return rewards
