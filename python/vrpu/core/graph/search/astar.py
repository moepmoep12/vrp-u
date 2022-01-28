import heapq
from sys import float_info
from typing import Callable, List, Dict

from vrpu.core.graph.node import Node, NodeData, EdgeData


class SearchResult(object):
    """
    Structure for returning the result of a graph search. Contains the whole path and the cost.
    """

    def __init__(self, path: List[Node[NodeData, EdgeData]] = [], cost: float = 0):
        self.path = path
        self.cost = cost


def search(start_node: Node[NodeData, EdgeData],
           goal_test: Callable[[Node[NodeData, EdgeData]], bool],
           heuristic: Callable[[Node[NodeData, EdgeData]], float],
           include_start_node: bool = False) \
        -> SearchResult:
    """
    Performs an A* search to find the goal.
    :param start_node: The starting node.
    :param goal_test: A function that returns whether a given node is the goal.
    :param heuristic: A function that returns an estimation for the remaining cost to the goal.
                      The estimation shall never overestimate the cost.
    :param include_start_node: Whether the start node will be included in the path.
    :return: Returns the result of the search as a SearchResult object.
    """
    came_from = dict()
    open_set = []
    search_nodes = dict()

    search_node = _SearchNode(start_node, heuristic(start_node), 0)
    heapq.heappush(open_set, search_node)
    search_nodes[start_node] = search_node

    while len(open_set) > 0:
        # get the node with the lowest f-score
        current = heapq.heappop(open_set)

        # check if goal was found
        if goal_test(current.node):
            return SearchResult(_reconstruct_path(came_from, current.node, include_start_node), current.g_score)

        # mark current as processed
        current.closed = True

        # process all neighbors
        for neighbor, edge in current.node.neighbors.items():

            node_visited = neighbor in search_nodes

            # check if neighbor was already processed
            if node_visited and search_nodes[neighbor].closed:
                continue

            # discover a new node
            if not node_visited:
                neighbor_search_node = _SearchNode(neighbor)
                search_nodes[neighbor] = neighbor_search_node
            else:
                neighbor_search_node = search_nodes[neighbor]

            # check whether the new path is better than the current path to the neighbor
            g_score_temp = current.g_score + edge.cost
            if not node_visited or g_score_temp < neighbor_search_node.g_score:
                neighbor_search_node.g_score = g_score_temp
                neighbor_search_node.f_score = g_score_temp + heuristic(neighbor)

                # remember this step
                came_from[neighbor] = current.node

                if node_visited:
                    # To-Do: improve performance
                    heapq.heapify(open_set)
                else:
                    heapq.heappush(open_set, neighbor_search_node)


def _reconstruct_path(came_from: Dict[Node[NodeData, EdgeData], Node[NodeData, EdgeData]],
                      current: Node[NodeData, EdgeData],
                      include_start_node: bool) -> List[Node[NodeData, EdgeData]]:
    """
    Reconstructs the path taken to the goal.
    :param came_from: A dictionary representing the taken steps.
    :param current: The current position ( goal node ).
    :param include_start_node: Whether the start node will be included in the path.
    :return: A list of graph nodes.
    """
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    if include_start_node:
        path.append(current)
    path.reverse()
    return path


class _SearchNode(object):
    """
    Structure used for graph search.
    """

    def __init__(self, node: Node[NodeData, EdgeData],
                 f_score: float = float_info.max,
                 g_score: float = float_info.max,
                 closed: bool = False):
        """
        :param node: The node that is being wrapped.
        :param f_score: The actual cost + estimated cost
        :param g_score: The actual cost to this node so far.
        :param closed: Whether this node was already processed.
        """
        self.node = node
        self.f_score = f_score
        self.g_score = g_score
        self.closed = closed

    def __repr__(self):
        return "%s | fscore: %.2f | gscore : %.2f | closed: %d" \
               % (self.node.uid, self.f_score, self.g_score, self.closed)

    def __lt__(self, other):
        return self.f_score < other.f_score
