from vrpu.core import *
from vrpu.solver.or_tools.or_tools_solver import SolverVRPDPU
from vrpu.core.graph.search.node_distance import NodeDistanceAStar
from vrpu.core.util.solution_printer import DataFramePrinter
from vrpu.core.util.visualization import show_uturn_solution
import json
from datetime import datetime, timedelta

if __name__ == '__main__':
    graph = Graph()
    vehicles = []
    calc_time = datetime.now()
    trqs = []

    with open('../../data/test_vrpdp-u.json') as f:
        data = json.load(f)

        for n in data['nodes']:
            graph.add_node(node_data=n['data'], uid=str(n['uid']), x=n['x'], y=n['y'])

        for edge in data['edges']:
            edge_data = f"{edge['from_node']}<->{edge['to_node']}"
            graph.add_edge(edge_data, edge['from_node'], edge['to_node'], edge['cost'], edge['directional'])

    for i, rq in enumerate(data['requests']):
        trq = TransportRequest(str(i), rq['from_node'], rq['to_node'], calc_time, quantity=1)
        trqs.append(trq)

    for i in range(1):
        v = Vehicle(uid=str(i), current_node='A', node_arriving_from='B', available_from=calc_time, max_capacity=10,
                    velocity=1)
        vehicles.append(v)

    problem = VRPProblem(transport_requests=trqs, vehicles=vehicles, calculation_time=calc_time,
                         pick_duration=timedelta(seconds=0),
                         delivery_duration=timedelta(seconds=0), vehicle_velocity=1)

    turn_graph = UTurnGraph(UTurnTransitionFunction(graph), graph)

    solver = SolverVRPDPU(NodeDistanceAStar(), turn_graph)
    solution = solver.solve(problem)

    printer = DataFramePrinter()
    printer.print_solution(solution)

    show_uturn_solution(solution, turn_graph)
