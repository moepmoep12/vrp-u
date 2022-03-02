import random
from abc import ABC, abstractmethod
from overrides import overrides

from vrpu.core import UTurnGraph, UTurnTransitionFunction
from vrpu.core.graph.graph import GridGraph, Graph
from vrpu.core.graph.search import NodeDistanceDijkstra


class IGraphGenerator(ABC):
    """
    Interface. Responsible for generating a graph.
    """

    @abstractmethod
    def generate_graph(self, **kwargs) -> Graph:
        """
        Generates a graph.
        :param kwargs: Key worded arguments.
        :return: The generated graph.
        """
        pass


class RandomGridGraphGenerator(IGraphGenerator):

    @overrides
    def generate_graph(self, **kwargs) -> Graph:
        """
        Generates a graph.
        :param kwargs: Key worded arguments.
            :keyword: size_x: Required. Graph width.
            :keyword: size_y: Required. Graph height.
            :keyword increments_between_nodes: Optional. Distance increments between nodes.
            :keyword: drop_edge_prob: Optional: Probability for dropping edges.
            :keyword drop_node_prob: Optional: Probability for dropping nodes.
            :keyword remove_unreachable_u_nodes: Optional. Whether unreachable nodes are removed.
        :return: The generated graph.
        """
        return self._generate_random_graph(**kwargs)

    @staticmethod
    def _generate_random_graph(size_x: int, size_y: int,
                               increments_between_nodes: [int] = [10],
                               drop_edge_prob: float = 0.0, drop_node_prob: float = 0.0,
                               remove_unreachable_u_nodes: bool = True) -> Graph:
        """
        Generates a graph.
        :param: size_x: Required. Graph width.
        :param: size_y: Required. Graph height.
        :param increments_between_nodes: Optional. Distance increments between nodes.
        :param: drop_edge_prob: Optional: Probability for dropping edges.
        :param drop_node_prob: Optional: Probability for dropping nodes.
        :param remove_unreachable_u_nodes: Optional. Whether unreachable nodes are removed.
        :return: The generated graph.
        """

        graph = GridGraph(size_x, size_y, increments_between_nodes)

        orig_node_count = len(graph.nodes)
        orig_edge_count = len(graph.edges)

        if drop_node_prob > 0:
            nodes_to_remove = []
            for node in graph.nodes.values():
                if random.random() < drop_node_prob and len(node.neighbors) > 3:
                    nodes_to_remove.append(node.uid)
            for node in nodes_to_remove:
                graph.remove_node(node)

            nodes_to_remove = []
            for node in graph.nodes.values():
                if len(node.neighbors) == 0:
                    nodes_to_remove.append(node.uid)
            for node in nodes_to_remove:
                graph.remove_node(node)

        if drop_edge_prob > 0:
            for node in graph.nodes.values():
                if len(node.neighbors) >= 3:
                    for neighbor in node.neighbors.keys():
                        if random.random() < drop_edge_prob:
                            graph.remove_edge(node.uid, neighbor.uid)
                            graph.remove_edge(neighbor.uid, node.uid)
                            break

            nodes_to_remove = []
            for node in graph.nodes.values():
                if len(node.neighbors) == 0:
                    nodes_to_remove.append(node.uid)

            for node in nodes_to_remove:
                graph.remove_node(node)

        if remove_unreachable_u_nodes:
            state_graph = UTurnGraph(UTurnTransitionFunction(graph), graph)
            node_distance = NodeDistanceDijkstra(dict())
            node_distance.calculate_distances(state_graph)

            unreachable_nodes = []
            for start_node, distance_dict in node_distance.distance_dict.items():
                reachable = False
                current_node = start_node
                if '->' in start_node:
                    current_node = start_node.split('->')[1]

                for to_node, distance in distance_dict.items():
                    if distance.reachable and current_node not in to_node:
                        reachable = True
                        break

                if not reachable:
                    unreachable_nodes.append(start_node)

            for u_node in unreachable_nodes:
                node = u_node.split('->')[1]
                print(f"Removing unreachable node {u_node}")
                graph.remove_node(node)

        print(f"Removed {orig_node_count - len(graph.nodes)} nodes")
        print(f"Removed {orig_edge_count - len(graph.edges)} edges")

        return graph
