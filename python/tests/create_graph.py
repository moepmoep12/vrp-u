import json

from vrpu.core import Solution
from vrpu.core.graph.graph import GridGraph, GraphEncoder, Graph
from vrpu.core.route_planning.transport_request import TransportRequestEncoder
from vrpu.core.util.visualization import show_graph, show_graph_with_transport_requests
from vrpu.core.util.generators import generate_grid_graph, generate_transport_requests_d, generate_transport_requests_pd

if __name__ == '__main__':
    # graph = generate_grid_graph(13, 13, 25, True, True)
    #
    # serialized = json.dumps(graph, indent=4, cls=GraphEncoder)
    # print(serialized)
    #
    # show_graph(graph)

    with open('../../data/cvrp_1.json') as f:
        data = json.load(f)

    depot = data['depot']

    graph = Graph.from_json(data['graph'])

    # trqs = generate_transport_requests_d(graph, 30, data['depot'])
    trqs = generate_transport_requests_pd(graph, 20, [data['depot']])
    serialized = json.dumps(trqs, indent=4, cls=TransportRequestEncoder)
    print(serialized)

    show_graph_with_transport_requests(graph, trqs)
