import sys
from dataclasses import dataclass
from typing import Callable, List, Generic, Collection
from overrides import overrides

from vrpu.core.graph.node import Node, UID, NodeData, EdgeData
from vrpu.core.graph.graph import Graph
from vrpu.core.graph.search import astar, dijkstra


@dataclass
class Distance:
    from_node: UID = ''
    to_node: UID = ''
    distance: int = 0
    path: List = None


MAX_DISTANCE = (sys.maxsize / 100000)


class INodeDistance(object):
    """
    Interface. Defines a method for returning the distance between two nodes.
    """

    def get_distance(self, from_node: UID, to_node: UID) -> int:
        """
        Returns the distance from one node to another.
        :param from_node: The starting node.
        :param to_node: The goal node.
        :return: The distance between both nodes.
        """
        pass


class CachedNodeDistance(INodeDistance):

    def __init__(self, distances):
        self._distances = distances

    def calculate_distances(self, graph: Graph[NodeData, EdgeData], node_subset: Collection[UID] = None):
        """
        :param graph: The underlying graph for calculating the distances.
        :param node_subset: A subset of nodes for which the distances will be calculated.
        """
        pass

    @overrides
    def get_distance(self, from_node: UID, to_node: UID) -> int:
        return self._distances[from_node][to_node].distance


class HeuristicGenerator(Generic[NodeData, EdgeData]):
    """
    Interface for a Heuristic Generator.
    """

    def generate_heuristic(self, goal_node: Node[NodeData, EdgeData]) -> Callable[
        [Node[NodeData, EdgeData]], float]:
        """
        Creates a heuristic function that returns for a given node an estimate for the remaining
        cost to the goal_node
        :param goal_node: The goal_node
        :return: A function that takes a nodes as input and returns a float
        """
        pass


class ZeroHeuristicGenerator(HeuristicGenerator):
    """
    Always returns 0 as the estimate.
    """

    @overrides
    def generate_heuristic(self, goal_node: Node[NodeData, EdgeData]) -> Callable[
        [Node[NodeData, EdgeData]], float]:
        return lambda node: 0


class NodeDistanceAStar(CachedNodeDistance, Generic[NodeData, EdgeData]):
    """
    Pre-calculates the node distances for a graph with A* and caches the results.
    """

    def __init__(self, heuristic_generator: HeuristicGenerator = ZeroHeuristicGenerator()):
        """
        :param heuristic_generator: Optional. Generates a heuristic function for the distance to the goal.
        """
        super().__init__(dict())
        self._heuristic_generator = heuristic_generator

    def calculate_distances(self, graph: Graph[NodeData, EdgeData], node_subset: List[UID] = []):
        nodes = node_subset if node_subset else graph.nodes.keys()
        for start_node_uid in nodes:
            self._distances[start_node_uid] = dict()
            for end_node_uid in nodes:
                def goal_test(node: Node[NodeData, EdgeData]):
                    return node.uid == end_node_uid

                heuristic = self._heuristic_generator.generate_heuristic(graph.get_node(end_node_uid))
                result = astar.search(graph.get_node(start_node_uid), goal_test, heuristic)
                distance = int(result.cost) if result else MAX_DISTANCE
                self._distances[start_node_uid][end_node_uid] = Distance(from_node=start_node_uid,
                                                                         to_node=end_node_uid,
                                                                         distance=distance,
                                                                         path=result.path if result else [])

    @overrides
    def get_distance(self, from_node: UID, to_node: UID) -> int:
        return self._distances[from_node][to_node].distance

    def get_path(self, from_node: UID, to_node: UID):
        return self._distances[from_node][to_node].path

    @property
    def distances(self) -> List[Distance]:
        return [distance for (_, targets) in
                self._distances.items()
                for _, distance in targets.items()]


class NodeDistanceDijkstra(INodeDistance, Generic[NodeData, EdgeData]):
    """
       Pre-calculates the node distances for a graph with Dijkstra and caches the results.
       """

    def calculate_distances(self, graph: Graph[NodeData, EdgeData], node_subset: List[UID] = []):
        nodes = node_subset if node_subset else graph.nodes.keys()

        for start_node_uid in nodes:
            start_node = graph.get_node(start_node_uid)
            search_results = dijkstra.search(start_node, graph)
            relevant_results = [n for n in search_results if n.end_node.uid in nodes]
            self._distances[start_node_uid] = {r.end_node.uid: Distance(from_node=start_node_uid,
                                                                        to_node=r.end_node.uid,
                                                                        distance=int(r.cost),
                                                                        path=[]) for r in relevant_results}

    @overrides
    def get_distance(self, from_node: UID, to_node: UID) -> int:
        return self._distances[from_node][to_node].distance

    @property
    def distances(self) -> List[Distance]:
        return [distance for (_, targets) in
                self._distances.items()
                for _, distance in targets.items()]
