from abc import ABC
from typing import Dict, Optional

import torch
from torch import nn

from ..core import Problem, Workflow, Monitor, jit_class, use_state, vmap, jit
from .. import utils


class HPOMonitor(Monitor, ABC):
    """The base class for hyper parameter optimization (HPO) monitors used in `HPOProblem.workflow.monitor`."""

    def __init__(self):
        super().__init__()

    def tell_fitness(self) -> torch.Tensor:
        """Get the best fitness found so far in the optimization process that this monitor is monitoring.

        Returns:
            `torch.Tensor`: The best fitness so far.
        """
        raise NotImplementedError("`tell_fitness` function is not implemented. It must be overwritten.")


class HPOFitnessMonitor(HPOMonitor):
    """The monitor for hyper parameter optimization (HPO) that records the best fitness found so far in the optimization process."""

    def __init__(self, multi_obj_indicator: Optional[str] = None):
        """
        Initialize the HPO fitness monitor.

        Args:
            multi_obj_indicator (`str`, optional): The indicator to use for multi-objective optimization, unused in single-objective optimization.
                Currently we only support "IGD" or "HV" for multi-objective optimization. Defaults to `None`.
        """
        super().__init__()
        assert multi_obj_indicator is None or isinstance(
            multi_obj_indicator, str
        ), f"Expect `multi_obj_indicator` to be `None` or `str`, got {multi_obj_indicator}"
        if multi_obj_indicator is not None:
            multi_obj_indicator = multi_obj_indicator.capitalize()
            assert multi_obj_indicator in [
                "IGD",
                "HV",
            ], f"Currently we only support `IGD` or `HV`, got {multi_obj_indicator}"
        self.multi_obj_indicator = multi_obj_indicator

    def setup(self):
        super().setup()
        self.best_fitness = nn.Buffer(torch.tensor(torch.inf))
        return self

    def pre_tell(self, fitness: torch.Tensor):
        if fitness.ndim == 1:
            # single-objective
            self.best_fitness = torch.min(torch.min(fitness), self.best_fitness)
        else:
            pass
            # # multi-objective, TODO: add indicators
            # if self.multi_obj_indicator == "IGD":
            #     self.best_fitness = torch.min(utils.IGD(fitness), self.best_fitness)
            # elif self.multi_obj_indicator == "HV":
            #     self.best_fitness = torch.max(utils.HV(fitness), self.best_fitness)

    def tell_fitness(self) -> torch.Tensor:
        return self.best_fitness


@jit_class
class HPOProblemWrapper(Problem):

    def __init__(self, iterations: int, num_instances: int):
        super().__init__()
        assert iterations > 0, f"`iterations` should be greater than 0, got {iterations}"
        assert num_instances > 0, f"`num_instances` should be greater than 0, got {num_instances}"
        self.iterations = iterations
        self.num_instances = num_instances

    def setup(self, workflow: Workflow):
        self.workflow = workflow
        # check monitor
        monitor = workflow.get_submodule("monitor")
        assert isinstance(
            monitor, HPOMonitor
        ), f"Expect workflow monitor to be `HPOMonitor`, got {type(monitor)}"
        monitor_state = monitor.state_dict(keep_vars=True)
        state_step = use_state(lambda: workflow.step)
        # get monitor's corresponding keys in init_state
        non_batched_init_state = list(state_step.init_state(clone=False).items())
        monitor_keys = {}
        for k, v in non_batched_init_state:
            for sk, sv in monitor_state.items():
                if sv is v:
                    monitor_keys[k] = sk
                    break
        assert len(monitor_keys) == len(
            monitor_state
        ), f"Expect monitor to have {len(monitor_state)} parameters, got {len(monitor_keys)}"

        def get_monitor_fitness(x: Dict[str, torch.Tensor]):
            final_monitor_state = {sk: x[k] for k, sk in monitor_keys.items()}
            monitor.load_state_dict(final_monitor_state)
            return monitor.tell_fitness()

        # JIT workflow step
        vmap_state_step = vmap(state_step)
        init_state = vmap_state_step.init_state(self.num_instances)
        self._workflow_step_: torch.jit.ScriptFunction = jit(
            vmap_state_step, trace=True, example_inputs=(init_state,)
        )
        self._get_monitor_fitness_ = jit(get_monitor_fitness, trace=True, example_inputs=(init_state,))
        monitor.load_state_dict(monitor_state)
        # if no init step
        if type(workflow).init_step == Workflow.init_step:
            self.init_state = init_state
            self._workflow_init_step_ = self._workflow_step_
            return
        # otherwise, JIT workflow init step
        state_init_step = use_state(lambda: workflow.init_step)
        vmap_state_init_step = vmap(state_init_step)
        self.init_state = vmap_state_init_step.init_state(self.num_instances)
        self._workflow_init_step_: torch.jit.ScriptFunction = jit(
            vmap_state_init_step, trace=True, example_inputs=(self.init_state,)
        )

    def evaluate(self, hyper_parameters: Dict[str, nn.Parameter]):
        # hyper parameters check
        for k, v in hyper_parameters.items():
            assert (
                k in self.init_state
            ), f"`{k}` should be in state dict of workflow and is `torch.nn.Parameter`"
            assert isinstance(self.init_state[k], nn.Parameter) and isinstance(
                v, nn.Parameter
            ), f"`{k}` should correspond to a `torch.nn.Parameter`, got {type(self.init_state[k])} and {type(v)}"
        # run the workflow
        state = self.init_state.copy()
        state.update(hyper_parameters)
        state = self._workflow_init_step_(state)
        for _ in range(self.iterations - 1):
            state = self._workflow_step_(state)
        # get final fitness
        return self._get_monitor_fitness_(state)

    @torch.jit.ignore
    def extract_parameters(state: Dict[str, torch.Tensor]):
        """
        Extract all hyper parameters from `state`.

        Args:
            state (`Dict[str, torch.Tensor]`): The state dictionary.

        Returns:
            A dictionary containing all hyper parameters.
        """
        return {k: v for k, v in state.items() if isinstance(v, nn.Parameter)}