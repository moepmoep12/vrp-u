import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from overrides import overrides

from vrpu.core import Solution, Graph, PickUp, Delivery, TransportAction, DriveAction, Node
from vrpu.core.graph.search import astar


class ISolutionRenderer(ABC):
    """
    Interface. Renders a solution.
    """

    @abstractmethod
    def render_solution(self, solution: Solution, graph: Graph, **kwargs) -> None:
        """
        Renders a solution with the given graph.
        :param solution: The solution to render.
        :param graph: The graph to render.
        :param kwargs: Key worded arguments.
        """
        pass


class SolutionRenderer(ISolutionRenderer):

    @overrides
    def render_solution(self, solution: Solution, graph: Graph, **kwargs) -> None:
        """
        Renders a solution with the given graph.
        :param solution: The solution to render.
        :param graph: The graph to render.
        :param kwargs:
            :keyword show_graph_edges: Whether graph edges will be drawn.
            :keyword node_color: Color of nodes.
            :keyword start_node_color: Color of vehicle starting nodes.
            :keyword pick_up_color: Color of pickup nodes.
            :keyword delivery_color: Color of delivery nodes.
            :keyword edge_color: Color of edges.
            :keyword full_screen: Open windows in full screen?
        """
        # retrieve key worded arguments
        show_graph_edges: bool = kwargs.get('show_graph_edges', True)
        node_color: str = kwargs.get('node_color', 'cornflowerblue')
        start_node_color: str = kwargs.get('start_node_color', 'gold')
        pick_up_color: str = kwargs.get('pick_up_color', 'tab:green')
        delivery_color: str = kwargs.get('delivery_color', 'tab:orange')
        edge_color: str = kwargs.get('edge_color', 'maroon')
        full_screen: bool = kwargs.get('full_screen', True)

        for tour_idx, tour in enumerate(solution.tours):
            g = nx.DiGraph()

            node_labels = dict()
            node_labels_order = dict()

            node_labels_order[tour.assigned_vehicle.current_node] = f"  \n  {0}"

            # show all nodes
            for node in self._get_nodes(graph):
                g.add_node(node.uid, pos=(node.x, node.y), color=node_color)
                node_labels[node.uid] = node.uid

            g.nodes[tour.assigned_vehicle.current_node]["color"] = start_node_color

            # show normal edges of the graph
            if show_graph_edges:
                for edge, nodes in self._get_edges(graph):
                    g.add_edge(nodes[0].uid, nodes[1].uid, weight=edge.cost, color='black', width=1, label="")
                    # g.add_edge(nodes[1].uid, nodes[0].uid, weight=edge.cost, color='black', width=1, label="")

            added_edges = []
            edge_traversals = dict()
            edge_labels = dict()
            node_counter = 1

            for action in tour.actions:
                # color nodes
                node = self._get_node_to_action(action)
                if not node:
                    continue

                if isinstance(action, PickUp):
                    g.nodes[node]['color'] = pick_up_color
                if isinstance(action, Delivery):
                    g.nodes[node]['color'] = delivery_color

                if isinstance(action, TransportAction):
                    node_labels_order[node] = f"  \n  {node_counter}"
                    node_counter += 1

            for action in tour.actions:
                if isinstance(action, DriveAction):
                    # show the path taken for each drive action
                    path = self._get_path(from_node=action.from_node, to_node=action.to_node, graph=graph)
                    if path:
                        for i in range(0, len(path) - 1):
                            from_node = self._get_current_node(path[i])
                            to_node = self._get_current_node(path[i + 1])
                            edge = (from_node, to_node)
                            edge_op = (to_node, from_node)
                            added_edges.append(edge)
                            if edge in edge_traversals:
                                edge_traversals[edge].append(len(added_edges))
                            elif edge_op in edge_traversals:
                                edge_traversals[edge_op].append(len(added_edges))
                            else:
                                edge_traversals[edge] = [len(added_edges)]

            nx.draw(G=g, pos=nx.get_node_attributes(g, 'pos'), with_labels=False,
                    edge_color=nx.get_edge_attributes(g, 'color').values(),
                    node_color=nx.get_node_attributes(g, 'color').values(),
                    node_size=600,
                    font_size=10,
                    width=list(nx.get_edge_attributes(g, 'width').values()))

            nx.draw_networkx_labels(G=g, pos=nx.get_node_attributes(g, 'pos'),
                                    labels=node_labels,
                                    font_size=10)

            nx.draw_networkx_labels(G=g, pos=nx.get_node_attributes(g, 'pos'),
                                    labels=node_labels_order,
                                    font_size=14,
                                    font_weight="bold",
                                    verticalalignment='top',
                                    horizontalalignment="left")

            nx.draw_networkx_edges(G=g, pos=nx.get_node_attributes(g, 'pos'),
                                   edgelist=added_edges,
                                   width=7,
                                   alpha=0.75,
                                   edge_color=edge_color,
                                   arrowsize=15,
                                   arrows=True)

            for edge, visits in edge_traversals.items():
                label = f"{visits[0]}"
                for i in range(1, len(visits)):
                    label += f", {visits[i]}"
                edge_labels[edge] = label

            nx.draw_networkx_edge_labels(G=g, pos=nx.get_node_attributes(g, 'pos'),
                                         edge_labels=edge_labels,
                                         font_size=8)

            # windowed full screen
            if full_screen:
                manager = plt.get_current_fig_manager()
                manager.window.state('zoomed')

            plt.show()

    @staticmethod
    def _get_edges(graph: Graph):
        base_graph = getattr(graph, 'base_graph', None)
        if base_graph:
            return base_graph.edges.items()
        return graph.edges.items()

    @staticmethod
    def _get_nodes(graph: Graph) -> [Node]:
        base_graph = getattr(graph, 'base_graph', None)
        if base_graph:
            return base_graph.nodes.values()
        return graph.nodes.values()

    @staticmethod
    def _get_current_node(graph_node: Node) -> str:
        if '->' in graph_node.uid:
            return graph_node.uid.split('->')[1]
        else:
            return graph_node.uid

    @staticmethod
    def _get_node_to_action(action: TransportAction) -> str:
        current_node = getattr(action, 'current_node', '')
        if current_node:
            return current_node
        else:
            return getattr(action, 'node', '')

    @staticmethod
    def _get_path(from_node: str, to_node: str, graph: Graph):
        from_node = graph.get_node(from_node)

        def goal_test(node):
            return node.uid == to_node

        search_result = astar.search(from_node, goal_test, lambda node: 0, True)
        return search_result.path
