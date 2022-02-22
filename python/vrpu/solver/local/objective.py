from abc import ABC, abstractmethod
from numbers import Number

from vrpu.core import Solution


class IObjectiveFunction(ABC):
    """
    Interface. An IObjectiveFunction evaluates a solution.
    """

    @abstractmethod
    def value(self, solution: Solution) -> Number:
        """
        Evaluates a given solution.
        :param solution: The solution to be evaluated.
        :return: The value of the solution as a single number.
        """
        pass


class DistanceObjective(IObjectiveFunction):
    """
    Simple objective function that returns the total distance of the solution.
    """

    def value(self, solution: Solution) -> Number:
        return sum([tour.get_distance() for tour in solution.tours])
