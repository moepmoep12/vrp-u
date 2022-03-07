from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import timedelta
from numbers import Number

from vrpu.core import VRPProblem, Solution
from vrpu.core.route_planning.serializable import Serializable


@dataclass
class SolvingSnapshot(Serializable):
    """
    Snapshot during solving. Keeps track of stats.
    """
    runtime: timedelta
    setup_time: timedelta
    step: int
    best_value: Number
    average: Number
    min_value: Number
    max_value: Number

    def serialize(self):
        result = asdict(self)
        result['runtime'] = str(self.runtime)
        result['setup_time'] = str(self.setup_time)
        return result


class ISolver(ABC):
    """
    Interface. Solves VRP Problems.
    """

    @abstractmethod
    def solve(self, problem: VRPProblem) -> Solution:
        """
        Solves the given VRPProblem and produces a solution.
        :param problem: The problem to solve.
        :return: The found solution.
        """
        pass

    @property
    def history(self) -> [SolvingSnapshot]:
        """
        :return: The snapshots collected during solving procedure.
        """
        pass
