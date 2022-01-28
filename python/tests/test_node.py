from unittest import TestCase

data = 'node_data'
uid = 'some_uid'


class TestNode(TestCase):

    def test_init(self):
        from vrpu.core.graph.node import Node
        from vrpu.core.exceptions import NoneValueException

        with self.assertRaises(NoneValueException):
            Node(None, None)

        node1 = Node(data, uid)
        node2 = Node(data, uid)

        self.assertNotEqual(node1, node2)
        self.assertEqual(node1.uid, node2.uid)
        self.assertEqual(node1.data, node2.data)

    def test_add_neighbor(self):
        from vrpu.core.graph.node import Node, Edge
        from vrpu.core.exceptions import WrongSubTypeException

        node = Node(data, uid)
        neighbor = Node('neighbor_data', 'neighbor_uid')
        edge = Edge(data='edge_data')

        with self.assertRaises(WrongSubTypeException):
            node.add_neighbor('', edge)

        with self.assertRaises(WrongSubTypeException):
            node.add_neighbor(neighbor, '')

        node.add_neighbor(node, edge)
        self.assertEqual(len(node.neighbors), 0)

        node.add_neighbor(neighbor, edge)
        self.assertEqual(node.neighbors[neighbor], edge)
