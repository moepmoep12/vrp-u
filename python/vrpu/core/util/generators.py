import random
from datetime import datetime

from vrpu.core import TransportRequest, Graph
from vrpu.core.graph.graph import GridGraph


def generate_grid_graph(size_x: int, size_y: int, increments: int, drop_edges: bool, drop_nodes: bool) -> Graph:
    graph = GridGraph(size_x, size_y, increments)

    if drop_nodes:

        nodes_to_remove = []
        for node in graph.nodes.values():
            if random.random() < 0.15 and len(node.neighbors) > 3:
                nodes_to_remove.append(node.uid)
        for node in nodes_to_remove:
            graph.remove_node(node)

        nodes_to_remove = []
        for node in graph.nodes.values():
            if len(node.neighbors) == 0:
                nodes_to_remove.append(node.uid)
        for node in nodes_to_remove:
            graph.remove_node(node)

    if drop_edges:
        for node in graph.nodes.values():
            if len(node.neighbors) >= 3:
                for neighbor in node.neighbors.keys():
                    if random.random() < 0.08:
                        graph.remove_edge(node.uid, neighbor.uid)
                        graph.remove_edge(neighbor.uid, node.uid)
                        break

        nodes_to_remove = []
        for node in graph.nodes.values():
            if len(node.neighbors) == 0:
                nodes_to_remove.append(node.uid)

        for node in nodes_to_remove:
            graph.remove_node(node)

    return graph


def generate_transport_requests_d(graph: Graph, count: int, depot: str, nodes_to_skip: [str] = []) \
        -> [TransportRequest]:
    assert len(graph.nodes) - len(nodes_to_skip) >= count

    forbidden_nodes = {*nodes_to_skip, depot}
    result = []

    for i in range(count):
        available_nodes = set(graph.nodes.keys()) - forbidden_nodes
        to_node = random.choice(list(available_nodes))

        trq = TransportRequest(str(i), depot, to_node, datetime(2000, 1, 1), 1)
        result.append(trq)
        forbidden_nodes.add(to_node)

    return result


def generate_transport_requests_pd(graph: Graph, count: int, nodes_to_skip: [str] = []) \
        -> [TransportRequest]:
    assert len(graph.nodes) / 2 - len(nodes_to_skip) >= count

    forbidden_nodes = {*nodes_to_skip}
    result = []

    for i in range(count):
        available_nodes = set(graph.nodes.keys()) - forbidden_nodes
        from_node = random.choice(list(available_nodes))
        available_nodes = available_nodes - set(from_node)
        direct_neighbors = set(graph.get_node(from_node).neighbors.keys())
        available_nodes = available_nodes - direct_neighbors
        to_node = random.choice(list(available_nodes))

        trq = TransportRequest(str(i), from_node, to_node, datetime(2000, 1, 1), 1)
        result.append(trq)

        forbidden_nodes.add(from_node)
        forbidden_nodes.add(to_node)

    return result
