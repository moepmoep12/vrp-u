import json
from datetime import datetime
from overrides import overrides
from typing import Dict

from .clonable import Clonable
from .serializable import Serializable


class TransportRequest(Clonable, Serializable):
    """
    A simple transport request that is uniquely identifiable by its ID and has a starting and end node.
    """

    def __init__(self, uid: str, from_node: str, to_node: str, due_date: datetime, quantity: int):
        """
        :param uid: The unique ID of this transport request.
        :param from_node: The starting node of the transport request.
        :param to_node: The end node of the transport request.
        :param due_date: The date & time until the transport request shall be completed.
        """
        self._uid = uid
        self._from_node = from_node
        self._to_node = to_node
        self._due_date = due_date
        self._quantity = quantity

    @property
    def uid(self) -> str:
        """
        :return: The unique ID of this transport request.
        """
        return self._uid

    @property
    def from_node(self) -> str:
        """
        :return: The starting node of the transport request.
        """
        return self._from_node

    @property
    def to_node(self) -> str:
        """
        :return: The end node of the transport request.
        """
        return self._to_node

    @property
    def due_date(self) -> datetime:
        """
        :return: The date & time until the transport request shall be completed.
        """
        return self._due_date

    @property
    def quantity(self) -> int:
        """
        :return: The quantity of this request. Used for capacity constraints.
        """
        return self._quantity

    @overrides
    def clone(self) -> object:
        clone = TransportRequest(self.uid, self.from_node, self.to_node, self.due_date, self.quantity)
        return clone

    @overrides
    def serialize(self) -> Dict[str, object]:
        return dict(
            uid=self.uid,
            from_node=self.from_node,
            to_node=self.to_node,
            due_date=str(self.due_date),
            quantity=self.quantity
        )

    def __repr__(self):
        return f"{{ {self.uid} |" \
               f" {self.from_node} -> {self.to_node} |" \
               f" Q:{self.quantity}x |" \
               f" D:{self.due_date} }}"
