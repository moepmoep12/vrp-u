from typing import TypeVar, Generic, Dict
from overrides import overrides

from vrpu.core.exceptions import NoneValueException
from vrpu.core.route_planning.serializable import Serializable

EdgeData = TypeVar('EdgeData')


class Edge(Generic[EdgeData], Serializable):
    """
    A GraphEdge represents an edge connecting two nodes in a graph.
    It has an associated cost for traversing the edge and some generic payload tls_types.
    """

    def __init__(self, data: EdgeData, cost: float = 0):
        """
        :param data: The payload of the edge.
        :param cost: The cost for traversing.
        """

        if data is None:
            raise NoneValueException(variable_name='data')

        self._data = data
        self._cost = cost

    @property
    def data(self) -> EdgeData:
        return self._data

    @property
    def cost(self) -> float:
        return self._cost

    @cost.setter
    def cost(self, new_cost: float):
        self._cost = new_cost

    @overrides
    def serialize(self) -> Dict[str, object]:
        return dict(
            data=self.data,
            cost=self.cost
        )

    def __repr__(self):
        return f"{{ {str(self.data)} | Cost: {self.cost} }}"

    def __eq__(self, other):
        return self.data == getattr(other, 'data', None) and self.cost == getattr(other, 'cost', None)

    def __hash__(self):
        return hash((self.data, self.cost))
