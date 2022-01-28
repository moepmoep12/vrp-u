from typing import TypeVar, Generic, Dict
from overrides import overrides

from .edge import Edge, EdgeData
from vrpu.core.exceptions import NoneValueException, WrongSubTypeException
from vrpu.core.route_planning.serializable import Serializable

NodeData = TypeVar('NodeData')
UID = str


class Node(Generic[NodeData, EdgeData], Serializable):
    """
    Represents a generic Node in a graph.
    A GraphNode can be uniquely identified by its UID and it has some generic payload.
    """

    def __init__(self, data: NodeData, uid: UID, x: float = 0, y: float = 0):
        """
        :param data: The payload for the node.
        :param uid: The unique identifier to identify this node.
        :param x: X-Position.
        :param y: Y-Position.
        """
        if data is None:
            raise NoneValueException(variable_name='data')

        self._data = data
        self._neighbors = dict()
        self._uid = uid
        self._x = x
        self._y = y

    @property
    def data(self) -> NodeData:
        return self._data

    @property
    def uid(self) -> UID:
        return self._uid

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def neighbors(self) -> dict:
        return self._neighbors

    def add_neighbor(self, neighbor, edge: Edge[EdgeData]):
        """
        Adds a GraphNode as a neighbor to this node.
        May overwrite existing edge.
        :param neighbor: The neighbor to add.
        :param edge: The connecting edge.
        """

        if not isinstance(neighbor, Node):
            raise WrongSubTypeException(Node.__name__, type(neighbor).__name__)

        if not isinstance(edge, Edge):
            raise WrongSubTypeException(Edge.__name__, type(edge).__name__)

        if neighbor == self:
            return

        self._neighbors[neighbor] = edge

    def remove_neighbor(self, neighbor):
        """
        Removes a neighbor from this node.
        :param neighbor: The neighbor to be removed.
        """
        del self._neighbors[neighbor]

    def remove_edge(self, edge):
        """
        Removes a connecting edge from this node and therefore removing a neighbor.
        :param edge: The connecting edge.
        """

        if not isinstance(edge, Edge):
            raise WrongSubTypeException(Edge.__name__, type(edge).__name__)

        for (neighbor, e) in self._neighbors:
            if e == edge:
                del self._neighbors[neighbor]
                return

    @overrides
    def serialize(self) -> Dict[str, object]:
        return dict(
            uid=self.uid,
            data=self.data,
            # neighbors=self.neighbors,
            x=self.x,
            y=self.y
        )

    def __repr__(self):
        return f"{{ {self.uid} | ({self.x},{self.y}) | {self.data.__repr__()} }}"
