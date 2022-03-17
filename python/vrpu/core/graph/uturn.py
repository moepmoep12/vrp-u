from abc import ABC, abstractmethod
from typing import Generic, List, Tuple
from overrides import overrides

from .node import NodeData, Node
from .edge import EdgeData, Edge
from .graph import Graph


class UTurnState(object):
    """
    A UTurnState contains the current node and also the node it came from.
    """

    def __init__(self, current_node: str, prev_node: str, current_data: object, prev_data: object):
        self.current_node = current_node
        self.prev_node = prev_node
        self.current_data = current_data
        self.prev_data = prev_data

    def __repr__(self):
        return f"{{ {self.prev_node} -> {self.current_node} }}"


class ITransitionFunction(Generic[NodeData, EdgeData], ABC):
    """
    Interface. A ITransitionFunction is responsible for generating reachable states from a given state.
    """

    @abstractmethod
    def generate_neighbor_states(self, current_state: Node[NodeData, EdgeData]) \
            -> List[Tuple[Node[NodeData, EdgeData], Edge[EdgeData]]]:
        """

        :param current_state: The current state.
        :return: All reachable states from the current state.
        """
        pass


class UTurnTransitionFunction(ITransitionFunction[UTurnState, EdgeData]):
    """
    A TransitionFunction that generates states which prevent U-Turns when traversing a graph.
    A state consists of the current node and the node from which the current node was reached.
    """

    def __init__(self, base_graph: Graph[NodeData, EdgeData]):
        self.base_graph = base_graph

    @overrides
    def generate_neighbor_states(self, current_state: Node[UTurnState, EdgeData]) \
            -> List[Tuple[Node[UTurnState, EdgeData], Edge[EdgeData]]]:
        """
        Generates all reachable (neighbor-) states from the given state.
        :param current_state: The current state.
        :return: A list of reachable states.
        """
        neighbor_states = []
        current_node = self.base_graph.nodes[current_state.data.current_node]

        for neighbor, edge in current_node.neighbors.items():

            # a neighbor node can only be reached if the edge to it is traversable
            # and it's not the node we just came from => no u-turns
            if neighbor.uid != current_state.data.prev_node:
                state = UTurnState(current_node=neighbor.uid, prev_node=current_node.uid, current_data=neighbor.data,
                                   prev_data=current_node.data)
                neighbor_states.append(
                    (Node(data=state,
                          uid=UTurnTransitionFunction.generate_uid(state.current_node, state.prev_node),
                          x=neighbor.x,
                          y=neighbor.y),
                     Edge(edge.data, edge.cost)))

        return neighbor_states

    def generate_initial_states(self) -> List[Node[UTurnState, EdgeData]]:
        """
        Generates a set of initial states.
        """
        initial_states = []

        for uid, node in self.base_graph.nodes.items():
            for neighbor, edge in node.neighbors.items():
                state = UTurnState(current_node=neighbor.uid, prev_node=node.uid, current_data=neighbor.data,
                                   prev_data=node.data)
                initial_states.append(
                    Node(data=state,
                         uid=UTurnTransitionFunction.generate_uid(neighbor.uid, node.uid),
                         x=neighbor.x,
                         y=neighbor.y))

        return initial_states

    @staticmethod
    def generate_uid(current: str, prev: str) -> str:
        """
        Generates a UID that uniquely identifies a state.
        :param current: The current node.
        :param prev: The predecessor of current node.
        :return: A unique ID.
        """
        return "%s->%s" % (prev, current)


class UTurnGraph(Graph):

    def __init__(self, transition_function: UTurnTransitionFunction, base_graph: Graph):
        self._transition_function = transition_function
        self._base_graph = base_graph
        super().__init__()
        self._create_graph()

    @property
    def base_graph(self) -> Graph[NodeData, EdgeData]:
        return self._base_graph

    @property
    def distances(self):
        return self.base_graph.distances

    def _create_graph(self):
        unexplored_states = []

        # initialization
        initial_states = self._transition_function.generate_initial_states()
        for init_state in initial_states:
            unexplored_states.append(init_state)
            self.add_node(init_state.data, init_state.uid, init_state.x, init_state.y)

        while len(unexplored_states) > 0:
            current = unexplored_states.pop()

            neighbor_states = self._transition_function.generate_neighbor_states(current)

            # process all possible neighbor states
            for neighbor, edge in neighbor_states:

                # neighbor hasn't been explored yet
                if neighbor.uid not in self.nodes:
                    self.add_node(neighbor.data, neighbor.uid, neighbor.x, neighbor.y)
                    unexplored_states.append(neighbor)

                self.add_edge(data=edge.data, from_uid=current.uid, to_uid=neighbor.uid, cost=edge.cost)
