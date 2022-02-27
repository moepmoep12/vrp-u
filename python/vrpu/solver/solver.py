from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from numbers import Number

from vrpu.core import VRPProblem, Solution


@dataclass
class SolvingSnapshot:
    """
    Snapshot during solving. Keeps track of stats.
    """
    runtime: timedelta
    step: int
    best_value: Number
    average: Number
    min_value: Number
    max_value: Number


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
