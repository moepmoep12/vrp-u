from abc import ABC, abstractmethod

from vrpu.core import VRPProblem, Solution


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
