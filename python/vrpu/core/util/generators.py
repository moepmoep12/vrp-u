import random
from datetime import datetime

from vrpu.core import TransportRequest, Graph, UTurnGraph, UTurnTransitionFunction
from vrpu.core.graph.graph import GridGraph
from vrpu.core.graph.search import NodeDistanceDijkstra


def generate_random_graph(size_x: int, size_y: int,
                          increments_between_nodes: [int] = [10],
                          drop_edge_prob: float = 0.0, drop_node_prob: float = 0.0,
                          remove_unreachable_u_nodes: bool = True) -> Graph:
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
