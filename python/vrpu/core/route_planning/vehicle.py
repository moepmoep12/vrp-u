from datetime import datetime
from overrides import overrides
from typing import Dict

from .clonable import Clonable
from .serializable import Serializable


class Vehicle(Clonable, Serializable):
    """
    A vehicle in the VRP.
    """

    def __init__(self, uid: str,
                 current_node: str,
                 node_arriving_from: str,
                 available_from: datetime,
                 max_capacity: int,
                 velocity: float):
        """
        :param uid: The unique ID of this vehicle.
        :param current_node: The current node of this vehicle.
        :param node_arriving_from: The node that the vehicle came from.
        :param available_from: The time at which this vehicle becomes available.
        :param max_capacity: The maximum capacity of this vehicle.
        :param velocity: The velocity of the vehicle in km/h.
        """
        self._uid = uid
        self._current_node = current_node
        self._node_arriving_from = node_arriving_from
        self._available_from = available_from
        self._max_capacity = max_capacity
        self._velocity = velocity

    @property
    def uid(self) -> str:
        """
        :return: The unique ID of this vehicle.
        """
        return self._uid

    @property
    def current_node(self) -> str:
        """
        :return: The current node of this vehicle.
        """
        return self._current_node

    @property
    def node_arriving_from(self) -> str:
        """
        :return: The node that the vehicle came from.
        """
        return self._node_arriving_from

    @property
    def available_from(self) -> datetime:
        """
        :return: The time at which this vehicle becomes available.
        """
        return self._available_from

    @property
    def max_capacity(self) -> int:
        """
        :return: The maximum capacity of this vehicle.
        """
        return self._max_capacity

    @property
    def velocity(self) -> float:
        """
        :return: The velocity of the vehicle in km/h.
        """
        return self._velocity

    @overrides
    def clone(self) -> object:
        clone = Vehicle(uid=self.uid, current_node=self.current_node, node_arriving_from=self.node_arriving_from,
                        available_from=self.available_from, max_capacity=self.max_capacity, velocity=self.velocity)
        return clone

    @overrides
    def serialize(self) -> Dict[str, object]:
        return dict(
            uid=self.uid,
            current_node=self.current_node,
            node_arriving_from=self.node_arriving_from,
            available_from=str(self.available_from),
            max_capacity=self.max_capacity,
            velocity=self.velocity
        )

    def __repr__(self) -> str:
        return f"{{ Vehicle: {self.uid} |" \
               f" At: {self.current_node} |" \
               f" Verfügbar ab: {self.available_from.time()} }}"
