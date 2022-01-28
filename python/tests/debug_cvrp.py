from vrpu.core import *
from vrpu.solver.or_tools.or_tools_solver import SolverCVRP
from vrpu.core.graph.search.node_distance import NodeDistanceAStar
from vrpu.core.util.solution_printer import DataFramePrinter
from vrpu.core.util.visualization import show_solution
import json
from datetime import datetime, timedelta

if __name__ == '__main__':
    vehicles = []
    calc_time = datetime.now()
    trqs = []

    with open('../../data/cvrp_1.json') as f:
        data = json.load(f)

    depot = data['depot']

    graph = Graph.from_json(data['graph'])

    for i, rq in enumerate(data['requests']):
        trq = TransportRequest(str(i), depot, rq['to_node'], calc_time, 1)
        trqs.append(trq)

    for i in range(data['vehicle_count']):
        v = Vehicle(str(i), depot, '', calc_time, data['max_capacity'], 1)
        vehicles.append(v)

    problem = VRPProblem(transport_requests=trqs, vehicles=vehicles, calculation_time=calc_time,
                         pick_duration=timedelta(seconds=0),
                         delivery_duration=timedelta(seconds=0), vehicle_velocity=1)

    solver = SolverCVRP(NodeDistanceAStar(), graph=graph, open_vrp=False)
    solution = solver.solve(problem)

    printer = DataFramePrinter(only_node_actions=True)
    printer.print_solution(solution)

    show_solution(solution, graph, True)
