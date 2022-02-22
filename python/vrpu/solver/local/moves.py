from abc import ABC, abstractmethod

from vrpu.solver.solution_encoding import EncodedSolution


class IMove(ABC):
    """
    Interface. An IMove generates a neighbor from a single solution.
    """

    @abstractmethod
    def generate_neighbor(self, current: EncodedSolution) -> EncodedSolution:
        """
        Generates a neighbor from the given solution.
        :param current: The current solution.
        :return: A neighbor from the current solution.
        """
        pass


class SwapMove(IMove):
    """
    Swaps an action with another. Can be between two tours.
    """

    def __init__(self, from_indices: [int], to_indices: [int]):
        assert len(from_indices) == len(to_indices)
        assert len(from_indices) > 0
        self.from_indices = from_indices
        self.to_indices = to_indices

    def generate_neighbor(self, current: EncodedSolution) -> EncodedSolution:
        clone = current.clone()
        # swap the values of the actions
        for from_index, to_index in zip(self.from_indices, self.to_indices):
            clone[to_index].value, clone[from_index].value = clone[from_index].value, clone[to_index].value
        return clone

    def __repr__(self):
        return f"SwapMove(From: {self.from_indices}, To: {self.to_indices})"


class RelocateMove(IMove):
    """
    Moves an action to another position.
    """

    def __init__(self, indices: [int], new_values: [int]):
        self.indices = indices
        self.new_values = new_values

    def generate_neighbor(self, current: EncodedSolution) -> EncodedSolution:
        clone = current.clone()
        for index, new_value in zip(self.indices, self.new_values):
            clone[index].value = new_value
        return clone

    def __repr__(self):
        return f"RelocateMove(Index: {self.indices}, new_value: {self.new_values})"


class ShiftMove(IMove):
    """
    Shifts actions to another vehicle.
    """

    def __init__(self, indices: [int], new_vehicle: int):
        self.indices = indices
        self.new_vehicle = new_vehicle

    def generate_neighbor(self, current: EncodedSolution) -> EncodedSolution:
        clone = current.clone()
        for index in self.indices:
            clone[index].vehicle_index = self.new_vehicle
        return clone

    def __repr__(self):
        return f"ShiftMove(Index: {self.indices}, new_vehicle: {self.new_vehicle})"
