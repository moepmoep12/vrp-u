import networkx as nx
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC
from overrides import overrides

from vrpu.core import Graph, TransportRequest


class IGraphRenderer(ABC):
    """
    Interface. Renders a graph.
    """

    @abstractmethod
    def render_graph(self, graph: Graph, **kwargs) -> None:
        """
        Renders a given graph.
        :param graph: The graph to render.
        :param kwargs: Key worded arguments.
        """
        pass


class GraphRenderer(IGraphRenderer):

    @overrides
    def render_graph(self, graph: Graph, **kwargs) -> None:
        """
        :param graph: Graph to render.
        :param kwargs:
            :keyword trqs: Collection of transport requests.
            :keyword node_color: Color of nodes.
            :keyword from_color: Color of nodes where a transport request begins.
            :keyword to_color: Color of nodes where a transport request ends.
            :keyword node_size: Size of nodes.
            :keyword show_trq_id: Whether to show transport request ids at nodes.
            :keyword depot: UID of depot node.
            :keyword depot_color: Color of depot node.
        """
        trqs: [TransportRequest] = kwargs.get('trqs', [])
        node_color: str = kwargs.get('node_color', 'tab:blue')
        from_color: str = kwargs.get('from_color', 'tab:green')
        to_color: str = kwargs.get('to_color', 'tab:orange')
        node_size: int = kwargs.get('node_size', 700)
        show_trq_id: bool = kwargs.get('show_trq_id', True)
        depot: str = kwargs.get('depot', '')
        depot_color: str = kwargs.get('depot_color', 'gold')

        g = nx.DiGraph()

        node_labels = dict()
        node_labels_trq = dict()

        for node in graph.nodes.values():
            g.add_node(node.uid, pos=(node.x, node.y), color=node_color)
            node_labels[node.uid] = node.uid

        for edge, nodes in graph.edges.items():
            g.add_edge(nodes[0].uid, nodes[1].uid, weight=edge.cost)

        for trq in trqs:
            g.nodes[trq.from_node]['color'] = from_color
            node_labels_trq[trq.from_node] = trq.uid
            g.nodes[trq.to_node]['color'] = to_color
            node_labels_trq[trq.to_node] = trq.uid

        if depot:
            g.nodes[depot]['color'] = depot_color
            if depot in node_labels_trq:
                node_labels_trq[depot] = ''

        nx.draw(G=g, pos=nx.get_node_attributes(g, 'pos'), with_labels=False,
                node_color=nx.get_node_attributes(g, 'color').values(),
                node_size=node_size,
                font_size=10)

        nx.draw_networkx_labels(G=g, pos=nx.get_node_attributes(g, 'pos'),
                                labels=node_labels,
                                font_size=10)
        pos = nx.get_node_attributes(g, 'pos')
        pos_adj = {}
        for uid, (x, y) in pos.items():
            pos_adj[uid] = (x + 2, y - 2)

        if show_trq_id:
            nx.draw_networkx_labels(G=g, pos=pos_adj,
                                    labels=node_labels_trq,
                                    font_size=17,
                                    font_weight="bold",
                                    verticalalignment='top',
                                    horizontalalignment="left")

        manager = plt.get_current_fig_manager()
        manager.window.state('zoomed')

        plt.show()
