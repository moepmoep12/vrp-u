import heapq
from sys import float_info
from typing import List

from vrpu.core.graph.node import Node, NodeData, EdgeData
from vrpu.core.graph.graph import Graph


class SearchResult(object):
    """
    Structure for returning the result of a graph search.
    """

    def __init__(self, start_node: Node[NodeData, EdgeData], end_node: Node[NodeData, EdgeData],
                 cost: float = 0):
        self.cost = cost
        self.start_node = start_node
        self.end_node = end_node
        self.reachable: bool = cost != float_info.max

    def __repr__(self):
        return f"{{ From: {self.start_node.uid} | To: {self.end_node.uid} | Cost: {self.cost:.2f} }}"


def search(start_node: Node[NodeData, EdgeData], graph: Graph[NodeData, EdgeData]) -> List[SearchResult]:
    """
    Performs the dijkstra algorithm to search for the shortest paths to all other nodes from the starting node.
    :param start_node: The starting node.
    :param graph: The underlying graph.
    :return: Returns a list of SearchResult.
    """
    open_set = [None] * len(graph.nodes)
    search_nodes = dict()

    # wrap the graph nodes in search_vertices used in the search
    for i, n in enumerate(graph.nodes.values()):
        open_set[i] = _SearchVertex(n)
        search_nodes[n] = open_set[i]

    # distance from start to start is 0
    search_nodes[start_node].distance = 0

    heapq.heapify(open_set)

    while len(open_set) > 0:
        # get the node with the lowest distance
        current = heapq.heappop(open_set)

        # process all neighbors
        for neighbor, edge in current.node.neighbors.items():
            assert neighbor in search_nodes

            # only consider unvisited nodes
            if not search_nodes[neighbor].closed:
                distance = current.distance + edge.cost
                # improvement found
                if distance < search_nodes[neighbor].distance:
                    search_nodes[neighbor].distance = distance
                    search_nodes[neighbor].previous = current.node
                    # to-do : improve performance
                    heapq.heapify(open_set)

        # finish node
        search_nodes[current.node].closed = True

    # transform the SearchVertex to SearchResult
    return [SearchResult(start_node, v.node, v.distance) for v in search_nodes.values()]


class _SearchVertex(object):
    """
    Wrapper of graph nodes to be used in  the search process.
    """

    def __init__(self, node: Node[NodeData, EdgeData],
                 distance: float = float_info.max,
                 previous: Node[NodeData, EdgeData] = None,
                 closed=False):
        """
        :param node: The node that is being wrapped.
        :param distance: The distance to this node.
        :param previous: The predecessor on the path to this node.
        :param closed: Whether this node was already processed.
        """
        self.node = node
        self.distance = distance
        self.previous = previous
        self.closed = closed

    def __repr__(self):
        prev = self.previous.uid if self.previous else ''
        return f"{{ {self.node.uid} | previous : {prev} | distance: {self.distance:.2f} | closed: {self.closed} }}"

    def __lt__(self, other):
        return self.distance < other.distance
