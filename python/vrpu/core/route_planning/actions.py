from abc import ABC
from datetime import timedelta
from overrides import overrides
from typing import Dict

from .clonable import Clonable
from .serializable import Serializable
from .transport_request import TransportRequest


class Action(Clonable, ABC, Serializable):
    """
    A planned action has a fixed execution duration.
    """

    def __init__(self, duration: timedelta = timedelta(seconds=0),
                 offset: timedelta = timedelta(seconds=0)):
        self._duration = duration
        self._offset = offset

    @staticmethod
    def name() -> str:
        """
        :return: The name of the action.
        """
        pass

    @property
    def start_offset(self) -> timedelta:
        """
        :return: The offset from the tour start when this actions execution begins.
        """
        return self._offset

    @start_offset.setter
    def start_offset(self, value: timedelta):
        self._offset = value

    @property
    def duration(self) -> timedelta:
        """
        :return: The execution duration in seconds.
        """
        return self._duration

    @duration.setter
    def duration(self, value: timedelta):
        self._duration = value

    def serialize(self) -> Dict[str, object]:
        return dict(
            name=self.name(),
            duration=self.duration.total_seconds(),
            start_offset=self.start_offset.total_seconds()
        )


class DriveAction(Action):
    """
    A drive action represents driving from one node to another node.
    """

    def __init__(self, from_node: str, to_node: str, prior_to_end_node: str, distance: float,
                 duration: timedelta = timedelta(seconds=0),
                 offset: timedelta = timedelta(seconds=0)):
        Action.__init__(self, duration=duration, offset=offset)
        self._from_node = from_node
        self._to_node = to_node
        self._distance = distance
        self._prior_to_end_node = prior_to_end_node

    @property
    def from_node(self) -> str:
        """
        :return: The starting node of the DriveAction.
        """
        return self._from_node

    @property
    def to_node(self) -> str:
        """
        :return: The end node of the DriveAction.
        """
        return self._to_node

    @property
    def prior_to_end_node(self) -> str:
        """
        :return: The node on the path prior to the end node.
        """
        return self._prior_to_end_node

    @prior_to_end_node.setter
    def prior_to_end_node(self, value):
        self._prior_to_end_node = value

    @property
    def distance(self) -> float:
        """
        :return: The total distance of the path.
        """
        return self._distance

    @staticmethod
    @overrides
    def name() -> str:
        return 'DriveAction'

    @overrides
    def clone(self) -> object:
        return DriveAction(from_node=self.from_node,
                           to_node=self.to_node,
                           prior_to_end_node=self.prior_to_end_node,
                           distance=self.distance,
                           duration=self.duration,
                           offset=self.start_offset)

    @overrides
    def serialize(self) -> Dict[str, object]:
        # TO-DO: handle uturn-state actions in different action?
        from_node = self.from_node.split('->')[1] if '->' in self.from_node else self.from_node
        to_node = self.to_node.split('->')[1] if '->' in self.to_node else self.to_node
        return dict(super().serialize(), **dict(
            start_node=from_node,
            end_node=to_node,
            distance=self.distance,
            prior_to_end_node=self.prior_to_end_node,
        ))

    def __repr__(self) -> str:
        return f"{{Drive @{self.from_node}->{self.to_node}) |" \
               f" +{self.start_offset.total_seconds()}s |" \
               f" {self.duration.total_seconds()}s |" \
               f" {self.distance}m}}"


class NodeAction(Action, ABC):
    """
    A NodeAction represents doing something at a specific node.
    """

    @property
    def node(self) -> str:
        """
        :return: The node at which this action is executed.
        """
        pass

    @staticmethod
    @overrides
    def name() -> str:
        return 'NodeAction'

    @overrides
    def serialize(self) -> Dict[str, object]:
        # TO-DO: handle uturn-state actions in different action?
        node = self.node.split('->')[1] if '->' in self.node else self.node
        return dict(super().serialize(), **dict(
            node=node
        ))


class SetupAction(NodeAction):
    """
    A SetupAction is executed upon tour start.
    """

    def __init__(self, start_node: str,
                 duration: timedelta = timedelta(seconds=0),
                 offset: timedelta = timedelta(seconds=0)):
        NodeAction.__init__(self, duration=duration, offset=offset)
        self._start_node = start_node

    @property
    @overrides
    def node(self) -> str:
        return self._start_node

    @staticmethod
    @overrides
    def name() -> str:
        return 'SetupAction'

    @overrides
    def clone(self) -> object:
        return SetupAction(self.node, self.duration, self.start_offset)

    def __repr__(self) -> str:
        return f"{{Start @{self.node} |" \
               f" +{self.start_offset.total_seconds()}s |" \
               f" {self.duration.total_seconds()}s}}"


class VisitAction(NodeAction):
    """
    A VisitAction simply visits a node.
    """

    def __init__(self, node: str,
                 duration: timedelta = timedelta(seconds=0),
                 offset: timedelta = timedelta(seconds=0)):
        NodeAction.__init__(self, duration=duration, offset=offset)
        self._node = node

    @property
    @overrides
    def node(self) -> str:
        return self._node

    @staticmethod
    @overrides
    def name() -> str:
        return 'Visit'

    @overrides
    def clone(self) -> object:
        return VisitAction(self.node, self.duration, self.start_offset)

    @overrides
    def serialize(self) -> Dict[str, object]:
        return dict(super().serialize(), **dict(
            node=self.node,
            place='',
            route_position=''
        ))

    def __repr__(self) -> str:
        return f"{{Visit @{self.node} |" \
               f" +{self.start_offset.total_seconds()}s |" \
               f" {self.duration.total_seconds()}s}}"


class TransportAction(NodeAction, ABC):
    """
    A TransportAction belongs to a TransportRequest and is executed at a node.
    """

    def __init__(self, trq: TransportRequest,
                 duration: timedelta = timedelta(seconds=0),
                 offset: timedelta = timedelta(seconds=0)):
        NodeAction.__init__(self, duration=duration, offset=offset)
        self._trq = trq

    @property
    def trqID(self) -> str:
        """
        :return: The UID of the transport request.
        """
        return self._trq.uid

    @property
    def trq(self) -> TransportRequest:
        """
        :return: The transport request this action belongs to.
        """
        return self._trq

    @staticmethod
    @overrides
    def name() -> str:
        return 'TransportAction'

    @overrides
    def serialize(self) -> Dict[str, object]:
        return dict(super().serialize(), **dict(
            trqID=self.trqID
        ))


class PickUp(TransportAction):
    """
    A PickUp represents picking up something at a node specified by the corresponding transport request.
    """

    def __init__(self, trq: TransportRequest,
                 duration: timedelta = timedelta(seconds=0),
                 offset: timedelta = timedelta(seconds=0)):
        TransportAction.__init__(self, trq, duration, offset)

    @property
    @overrides
    def node(self) -> str:
        return self.trq.from_node

    @staticmethod
    @overrides
    def name() -> str:
        return 'PickUp'

    @overrides
    def clone(self) -> object:
        return PickUp(self.trq, self.duration, self.start_offset)

    def __repr__(self):
        return f"{{PickUp @{self.node} |" \
               f" TRQST: {self.trqID} |" \
               f" +{self.start_offset.total_seconds()}s |" \
               f" {self.duration.total_seconds()}s}}"


class Delivery(TransportAction):
    """
    A Delivery represents delivering something at a node specified by the corresponding transport request.
    """

    def __init__(self, trq: TransportRequest,
                 duration: timedelta = timedelta(seconds=0),
                 offset: timedelta = timedelta(seconds=0)):
        TransportAction.__init__(self, trq, duration, offset)

    @property
    @overrides
    def node(self) -> str:
        return self.trq.to_node

    @staticmethod
    @overrides
    def name() -> str:
        return 'Delivery'

    @overrides
    def clone(self) -> object:
        return Delivery(self.trq, self.duration, self.start_offset)

    def __repr__(self):
        return f"{{Delivery @{self.node} |" \
               f" TRQST: {self.trqID} |" \
               f" Due: {self.trq.due_date.time()} |" \
               f" +{self.start_offset.total_seconds()}s |" \
               f" {self.duration.total_seconds()}s}}"
