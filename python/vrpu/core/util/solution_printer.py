from abc import ABC, abstractmethod
from overrides import overrides
from datetime import timedelta
import pandas as pd

from vrpu.core import Solution, Tour, NodeAction


class ISolutionPrinter(ABC):
    """
    Interface. Prints out VRP solutions.
    """

    @abstractmethod
    def print_solution(self, solution: Solution) -> None:
        """
        Prints the solution to the console.
        :param solution: The solution to print out.
        """
        pass


class DataFramePrinter(ISolutionPrinter):
    # Columns used in the DataFrame
    COLUMNS = ['Action', 'Node', 'TrqID', 'Time', 'Distance']

    def __init__(self, only_node_actions: bool = True):
        """
        :param only_node_actions: Whether only NodeActions will be printed.
        """
        self._only_node_actions = only_node_actions

    @property
    def only_node_actions(self) -> bool:
        """
        :return: Whether only NodeActions will be printed.
        """
        return self._only_node_actions

    @only_node_actions.setter
    def only_node_actions(self, value: bool) -> None:
        """
        :param value: Whether only NodeActions will be printed.
        """
        self._only_node_actions = value

    @overrides
    def print_solution(self, solution: Solution) -> None:
        total_duration = timedelta()
        total_distance = 0

        for tour in solution.tours:
            df = self._get_dataframe(tour)
            total_duration += tour.get_duration()
            total_distance += tour.get_distance()

            print(f"Tour for vehicle {tour.assigned_vehicle.uid}:\n")
            print(df.to_string(
                justify='center',
                formatters={'Distance': lambda d: f"{d}m" if d or d == 0 else ''}))

            print(f"Tour Distance: {tour.get_distance()}m")
            print(f"Tour Duration: {tour.get_duration()}")
            print("---------------------------------------------\n")

        print(f"Total Distance: {total_distance}m")
        print(f"Total Duration: {total_duration}h")

    def _get_dataframe(self, tour: Tour) -> pd.DataFrame:
        rows = []
        distance = 0

        for action in tour.actions:
            trqid: str = getattr(action, 'trqID', '')
            node: str = getattr(action, 'node', '')
            start_time = tour.start_time + action.start_offset
            distance += getattr(action, 'distance', 0)

            # Skip printing out actions if flag is set
            if self.only_node_actions and not isinstance(action, NodeAction):
                continue

            rows.append([action.name(), node, trqid, start_time.time(), distance])

        return pd.DataFrame(rows, columns=self.COLUMNS)
