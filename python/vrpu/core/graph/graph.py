import json
import random
from typing import Dict, Generic, Tuple
from overrides import overrides

from .node import Node, NodeData
from .edge import Edge, EdgeData
from vrpu.core.route_planning.serializable import Serializable

UID = str


class Graph(Generic[NodeData, EdgeData], Serializable):
    """
    A generic graph consisting of nodes and edges connecting nodes.
    """

    def __init__(self, nodes: Dict[UID, Node[NodeData, EdgeData]] = None,
                 edges: Dict[Edge[EdgeData], Tuple[Node, Node]] = None):
        """
        :param nodes: Initial node set.
        :param edges: Initial edge set.
        """
        self._nodes = nodes if nodes is not None else dict()
        self._edges = edges if edges is not None else dict()

    @property
    def nodes(self) -> Dict[UID, Node[NodeData, EdgeData]]:
        return self._nodes

    @property
    def edges(self) -> Dict[Edge[EdgeData], Tuple[Node, Node]]:
        return self._edges

    def get_node(self, node_uid: UID) -> Node[NodeData, EdgeData]:
        """
        :param node_uid: The unique identifier of the node.
        :raises:
            KeyError: If the node is not present in the graph.
        :return: The retrieved graph node.
        """
        if node_uid in self.nodes:
            return self.nodes[node_uid]
        else:
            raise KeyError(f"Node {node_uid} not found in graph")

    def get_node_by_coord(self, x: float, y: float):
        for node in self.nodes.values():
            if node.x == x and node.y == y:
                return node
        return None

    def get_edge(self, node_from: UID, node_to: UID) -> Edge[EdgeData]:
        for edge, nodes in self.edges.items():
            if nodes[0].uid == node_from and nodes[1].uid == node_to:
                return edge

        raise KeyError(f"Edge ({node_from},{node_to}) not found in graph")

    def add_node(self, node_data: NodeData, uid: UID, x: float = 0, y: float = 0) -> Node[NodeData, EdgeData]:
        """
        Adds a node to the graph by constructing it first.
        :param node_data: The payload for the new node.
        :param uid: The unique identifier for the new node.
        :param x: X-Position.
        :param y: Y-Position.
        :return: The constructed node. If the node is already in the graph it will be returned instead.
        """
        if uid in self.nodes:
            return self.nodes[uid]

        new_node = Node(node_data, uid, x, y)
        self.nodes[uid] = new_node
        return new_node

    def add_edge(self, data: EdgeData, from_uid: UID, to_uid: UID, cost: float = 0) -> Edge[EdgeData]:
        """
        Adds an edge to the graph. The two nodes need to be already in the graph.
        :param data: The payload of the edge.
        :param from_uid: The unique identifier of the starting node.
        :param to_uid: The unique identifier of the ending node.
        :param cost: The cost for edge traversal.
        :param directed: If the edge is directed or not.
        :return: The created and added edge.
        """
        if from_uid not in self.nodes or to_uid not in self.nodes:
            return None

        new_edge = Edge(data, cost)

        from_node = self.nodes[from_uid]
        to_node = self.nodes[to_uid]

        self.edges[new_edge] = (from_node, to_node)

        from_node.add_neighbor(to_node, new_edge)

        return new_edge

    def remove_node(self, node_uid: UID):
        """
        Removes a node from the graph. This also removes the outgoing and incoming edges of the node.
        :param node_uid: The unique identifier of the node to be removed.
        """
        if node_uid not in self.nodes:
            return

        node_to_remove = self.nodes[node_uid]

        # remove all outgoing edge from this node
        neighbors = list(node_to_remove.neighbors.items())
        for neighbor, edge in neighbors:
            self.remove_edge(node_to_remove.uid, neighbor.uid)

        # remove all incoming edges to this node
        for _, node in self.nodes.items():
            self.remove_edge(node.uid, node_to_remove.uid)

        # remove the node
        del self.nodes[node_uid]

    def remove_edge(self, from_uid: UID, to_uid: UID) -> bool:
        """
        Removes an edge from the graph. This may lead to separated nodes.
        :param from_uid: The uid of the starting node.
        :param to_uid: The uid of the ending node.
        :return: Whether removal was successful.
        """
        if from_uid not in self.nodes or to_uid not in self.nodes:
            return False

        from_node = self.nodes[from_uid]
        to_node = self.nodes[to_uid]

        # check if edge exists
        if to_node not in from_node.neighbors:
            return False

        edge = from_node.neighbors[to_node]

        if edge in self.edges:
            del self.edges[edge]

        from_node.remove_neighbor(to_node)
        return True

    @overrides
    def serialize(self) -> Dict[str, object]:
        edges = []
        for edge, nodes in self.edges.items():
            edges.append(
                {
                    "from_node": nodes[0].uid,
                    "to_node": nodes[1].uid,
                    "cost": edge.cost
                }
            )
        return dict(
            nodes=list(self.nodes.values()),
            edges=edges
        )

    @staticmethod
    def from_json(data):
        graph = Graph()

        for n in data['nodes']:
            graph.add_node(node_data=n['data'], uid=str(n['uid']), x=n['x'], y=n['y'])

        for edge in data['edges']:
            edge_data = f"{edge['from_node']}->{edge['to_node']}"
            graph.add_edge(edge_data, edge['from_node'], edge['to_node'], edge['cost'])

        return graph


class GridGraph(Graph):

    def __init__(self, size_x: int, size_y: int, increments: [int] = [10]):
        super().__init__()

        # x_increments = {}
        # for y in range(size_y):
        #     x_increments[y] = random.choice(increments)

        column_coords = {}
        column_coord = 0
        for y in range(size_y):
            column_coords[y] = column_coord
            column_coord += random.choice(increments)

        row_coords = {}
        row_coord = 0
        for x in range(size_x):
            row_coords[x] = row_coord
            row_coord += random.choice(increments)

        for row in range(size_x):
            # y_increment = random.choice(increments)
            for column in range(size_y):
                uid = f"{int(row) * size_y + int(column)}"
                x_coord = column_coords[column]  # x_increments[y] * y
                y_coord = row_coords[row]
                self.add_node(uid, uid, x_coord, y_coord)

        for x in range(size_x):
            for y in range(size_y):
                from_uid = f"{int(x) * size_y + int(y)}"
                if x < size_x - 1:
                    neighbor_up_uid = f"{int(x + 1) * size_y + int(y)}"
                    cost = random.choice(increments)
                    self.add_edge(data=f"{from_uid}->{neighbor_up_uid}", from_uid=from_uid, to_uid=neighbor_up_uid,
                                  cost=cost)
                    self.add_edge(data=f"{neighbor_up_uid}->{from_uid}", from_uid=neighbor_up_uid, to_uid=from_uid,
                                  cost=cost)
                if y < size_y - 1:
                    neighbor_right_uid = f"{int(x) * size_y + int(y + 1)}"
                    cost = random.choice(increments)
                    self.add_edge(data=f"{from_uid}->{neighbor_right_uid}", from_uid=from_uid,
                                  to_uid=neighbor_right_uid,
                                  cost=cost)
                    self.add_edge(data=f"{neighbor_right_uid}->{from_uid}", from_uid=neighbor_right_uid,
                                  to_uid=from_uid,
                                  cost=cost)


class GraphEncoder(json.JSONEncoder):
    def default(self, o: Graph):

        if isinstance(o, Serializable):
            return o.serialize()
        else:
            return {'__{}__'.format(o.__class__.__name__): o.__dict__} if hasattr(o, '__dict__') else str(o)
