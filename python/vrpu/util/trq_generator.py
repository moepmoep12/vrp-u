import random
from datetime import datetime
from abc import ABC, abstractmethod
from overrides import overrides

from vrpu.core import Graph, TransportRequest


class ITransportRequestGenerator(ABC):
    """
    Interface. Responsible for generating transport requests on a graph.
    """

    @abstractmethod
    def generate_transport_requests(self, graph: Graph,
                                    count: int,
                                    depot: str,
                                    nodes_to_skip: [str] = [],
                                    pick_up: bool = False) -> [TransportRequest]:
        """

        :param graph: The graph to generate the transport requests on.
        :param count: Number of transport requests.
        :param depot: The starting depot.
        :param nodes_to_skip: Collection of nodes to skip.
        :param pick_up: Whether pick up is allowed on nodes other than depot.
        :return: Generated Transport Requests.
        """
        pass


class RandomTransportRequestGenerator(ITransportRequestGenerator):
    @overrides
    def generate_transport_requests(self, graph: Graph,
                                    count: int,
                                    depot: str,
                                    nodes_to_skip: [str] = [],
                                    pick_up: bool = False) -> [TransportRequest]:
        forbidden_nodes = {*nodes_to_skip, depot}
        result = []

        for i in range(count):
            available_nodes = set(graph.nodes.keys()) - forbidden_nodes
            from_node = random.choice(list(available_nodes)) if pick_up else depot
            available_nodes = available_nodes - set(from_node)
            direct_neighbors = set(graph.get_node(from_node).neighbors.keys())
            available_nodes = available_nodes - direct_neighbors
            to_node = random.choice(list(available_nodes))

            trq = TransportRequest(str(i), from_node, to_node, datetime(2000, 1, 1), 1)
            result.append(trq)

            forbidden_nodes.add(from_node)
            forbidden_nodes.add(to_node)

        return result
