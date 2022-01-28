import networkx as nx
import matplotlib.pyplot as plt

from vrpu.core import Graph, Solution, DriveAction, PickUp, Delivery, UTurnGraph
from vrpu.core.graph.search import astar

PATH_COLORS = ["tab:pink", "tab:olive", "tab:cyan", "tab:red", "tab:purple", "tab:orange"]


def show_graph(graph: Graph):
    g = nx.DiGraph()

    for node in graph.nodes.values():
        g.add_node(node.uid, pos=(node.x, node.y))

    for edge, nodes in graph.edges.items():
        g.add_edge(nodes[0].uid, nodes[1].uid, weight=edge.cost)

    nx.draw(g, nx.get_node_attributes(g, 'pos'), with_labels=True)
    plt.show()


def show_graph_with_transport_requests(graph: Graph, trqs: []):
    g = nx.DiGraph()

    for node in graph.nodes.values():
        g.add_node(node.uid, pos=(node.x, node.y), color='tab:blue')

    for edge, nodes in graph.edges.items():
        g.add_edge(nodes[0].uid, nodes[1].uid, weight=edge.cost)

    for trq in trqs:
        g.nodes[trq.from_node]['color'] = 'tab:green'
        g.nodes[trq.to_node]['color'] = 'tab:orange'

    nx.draw(g, nx.get_node_attributes(g, 'pos'), with_labels=True,
            node_color=nx.get_node_attributes(g, 'color').values())
    plt.show()


def show_solution(solution: Solution, graph: Graph, show_graph_edges=True):
    for tour_idx, tour in enumerate(solution.tours):
        g = nx.DiGraph()

        # show all nodes
        for node in graph.nodes.values():
            g.add_node(node.uid, pos=(node.x, node.y), color='tab:blue')

        if show_graph_edges:
            # show normal edges of the graph
            for edge, nodes in graph.edges.items():
                g.add_edge(nodes[0].uid, nodes[1].uid, weight=edge.cost, color='black', width=1, label="")
                g.add_edge(nodes[1].uid, nodes[0].uid, weight=edge.cost, color='black', width=1, label="")

        # plt.subplot(100 + 10 * len(solution.tours) + tour_idx + 1)
        added_edges = []
        edge_traversals = dict()
        edge_labels = dict()

        for action in tour.actions:
            # color nodes
            if isinstance(action, PickUp):
                g.nodes[action.node]['color'] = 'tab:green'
            if isinstance(action, Delivery):
                g.nodes[action.node]['color'] = 'tab:orange'

            if isinstance(action, DriveAction):
                # show the path taken for each drive action
                path = get_path(from_node=action.from_node, to_node=action.to_node, graph=graph)
                if path:
                    for i in range(0, len(path) - 1):
                        from_node = path[i].uid
                        to_node = path[i + 1].uid
                        edge = (from_node, to_node)
                        edge_op = (to_node, from_node)
                        added_edges.append(edge)
                        if edge in edge_traversals:
                            edge_traversals[edge].append(len(added_edges))
                        elif edge_op in edge_traversals:
                            edge_traversals[edge_op].append(len(added_edges))
                        else:
                            edge_traversals[edge] = [len(added_edges)]

        nx.draw(G=g, pos=nx.get_node_attributes(g, 'pos'), with_labels=True,
                edge_color=nx.get_edge_attributes(g, 'color').values(),
                node_color=nx.get_node_attributes(g, 'color').values(),
                width=list(nx.get_edge_attributes(g, 'width').values()))

        nx.draw_networkx_edges(G=g, pos=nx.get_node_attributes(g, 'pos'),
                               edgelist=added_edges,
                               width=8,
                               alpha=0.5,
                               edge_color=PATH_COLORS[tour_idx])

        for edge, visits in edge_traversals.items():
            label = f"{visits[0]}"
            for i in range(1, len(visits)):
                label += f", {visits[i]}"
            edge_labels[edge] = label

        nx.draw_networkx_edge_labels(G=g, pos=nx.get_node_attributes(g, 'pos'),
                                     edge_labels=edge_labels)

        plt.show()


def show_uturn_solution(solution: Solution, graph: UTurnGraph, show_graph_edges=True):
    g = nx.DiGraph()

    # show all nodes
    for node in graph.base_graph.nodes.values():
        g.add_node(node.uid, pos=(node.x, node.y), color='tab:blue')

    # show normal edges of the graph
    for edge, nodes in graph.base_graph.edges.items():
        g.add_edge(nodes[0].uid, nodes[1].uid, weight=edge.cost, color='black', width=1, label="")
        g.add_edge(nodes[1].uid, nodes[0].uid, weight=edge.cost, color='black', width=1, label="")

    for tour_idx, tour in enumerate(solution.tours):
        plt.subplot(100 + 10 * len(solution.tours) + tour_idx + 1)
        added_edges = []
        edge_traversals = dict()
        edge_labels = dict()

        for action in tour.actions:
            # color nodes
            if isinstance(action, PickUp):
                g.nodes[action.current_node]['color'] = 'tab:green'
            if isinstance(action, Delivery):
                g.nodes[action.current_node]['color'] = 'tab:orange'

            if isinstance(action, DriveAction):
                # show the path taken for each drive action
                path = get_path(from_node=action.from_node, to_node=action.to_node, graph=graph)
                if path:
                    for i in range(0, len(path) - 1):
                        from_node = path[i].data.current_node
                        to_node = path[i + 1].data.current_node
                        edge = (from_node, to_node)
                        edge_op = (to_node, from_node)
                        added_edges.append(edge)
                        if edge in edge_traversals:
                            edge_traversals[edge].append(len(added_edges))
                        elif edge_op in edge_traversals:
                            edge_traversals[edge_op].append(len(added_edges))
                        else:
                            edge_traversals[edge] = [len(added_edges)]

        nx.draw(G=g, pos=nx.get_node_attributes(g, 'pos'), with_labels=True,
                edge_color=nx.get_edge_attributes(g, 'color').values(),
                node_color=nx.get_node_attributes(g, 'color').values(),
                width=list(nx.get_edge_attributes(g, 'width').values()))

        nx.draw_networkx_edges(G=g, pos=nx.get_node_attributes(g, 'pos'),
                               edgelist=added_edges,
                               width=8,
                               alpha=0.5,
                               edge_color=PATH_COLORS[tour_idx])

        for edge, visits in edge_traversals.items():
            label = f"{visits[0]}"
            for i in range(1, len(visits)):
                label += f", {visits[i]}"
            edge_labels[edge] = label

        nx.draw_networkx_edge_labels(G=g, pos=nx.get_node_attributes(g, 'pos'),
                                     edge_labels=edge_labels)

    plt.show()


def get_path(from_node: str, to_node: str, graph: Graph):
    from_node = graph.get_node(from_node)

    def goal_test(node):
        return node.uid == to_node

    search_result = astar.search(from_node, goal_test, lambda node: 0, True)
    return search_result.path
