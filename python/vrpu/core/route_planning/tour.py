from datetime import datetime, timedelta
from typing import List, Dict
from overrides import overrides

from .vehicle import Vehicle
from .actions import Action, PickUp, DriveAction
from .transport_request import TransportRequest
from .serializable import Serializable


class Tour(Serializable):
    """
       A tour consists of an ordered list of actions. It is assigned to a single vehicle.
       """

    def __init__(self, uid: str,
                 actions: List[Action],
                 assigned_vehicle: Vehicle,
                 start_time: datetime,
                 end_time: datetime):
        self._uid = uid
        self._actions = actions
        self._assigned_vehicle = assigned_vehicle
        self._start_time = start_time
        self._end_time = end_time

    @property
    def uid(self) -> str:
        """
        :return: The unique ID of this tour.
        """
        return self._uid

    @property
    def actions(self) -> List[Action]:
        """
        :return: Ordered list of actions to be executed in this tour.
        """
        return self._actions

    @property
    def start_time(self) -> datetime:
        """
        :return: The start time of this tour.
        """
        return self._start_time

    @property
    def end_time(self) -> datetime:
        """
        :return: The end time of this tour.
        """
        return self._end_time

    @property
    def assigned_vehicle(self) -> Vehicle:
        """
        :return: The vehicle that this tour was assigned to.
        """
        return self._assigned_vehicle

    def get_start_node(self) -> str:
        """
        :return: The start node of this tour.
        """
        for action in self.actions:
            if hasattr(action, 'node'):
                return action.node
        return ''

    def get_end_node(self) -> str:
        """"
        :return: The end node of this tour.
        """
        for action in reversed(self.actions):
            if hasattr(action, 'node'):
                return action.node
        return ''

    def get_action(self, index: int) -> Action:
        """
        :param index: Action index.
        :return: Action at the specified index.
        """
        if 0 <= index < len(self.actions):
            return self.actions[index]
        return None

    def get_transport_requests(self) -> List[TransportRequest]:
        """
        :return: All transport requests of this tour.
        """
        return [a.trq for a in self.actions if isinstance(a, PickUp)]

    def get_duration(self) -> timedelta:
        """
        :return: Total duration of this tour.
        """
        return self.end_time - self.start_time

    def get_distance(self) -> float:
        """
        :return: Total distance of this tour.
        """
        return sum([a.distance for a in self.actions if isinstance(a, DriveAction)])

    def __repr__(self):
        val = f"Tour {self.uid} with Vehicle {self._assigned_vehicle.uid}\n" \
              f" Start:    {self._start_time}\n" \
              f" End:      {self.end_time}\n" \
              f" Duration: {self.get_duration()}\n" \
              f" Actions:\n"
        val += f"{self._dataframe_repr()}\n"

        return val

    @overrides
    def serialize(self) -> Dict[str, object]:
        # TO-DO: handle uturn-state actions in different action?
        start_node = self.get_start_node().split('->')[1] if '->' in self.get_start_node() else self.get_start_node()
        return dict(
            uid=self.uid,
            vehicle=self.assigned_vehicle,
            start_time=str(self.start_time),
            start_node=start_node,
            actions=self.actions,
        )
