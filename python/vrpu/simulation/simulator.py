import json
import os
import re
import tkinter as tk
import yaml
import logging
import logging.config
from datetime import datetime, timedelta
from tkinter import filedialog, ttk

from ortools.constraint_solver import routing_enums_pb2

from vrpu.core import Graph, TransportRequest, Vehicle, VRPProblem, UTurnGraph, UTurnTransitionFunction
from vrpu.core.graph.search import NodeDistanceAStar
from vrpu.core.route_planning.serializable import JSONSerializer
from vrpu.visualization import SolutionRenderer, DataFramePrinter, SolverRenderer, GraphRenderer
from vrpu.solver import *
from vrpu.solver.or_tools.or_tools_solver import SolverParams

logger = logging.getLogger('simulator')
progress_logger = logging.getLogger('progress')

SOLVER_TYPES = [
    "OR-Tools",
    "Genetic Algorithm",
    "Local"
]

PROBLEM_TYPES = [
    "CVRP",
    "CVRPU",
    "VRPDP",
    "VRPDPU"
]

OR_SOLVER_METHODS: dict = {
    routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC: "Automatic",
    routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH: "Tabu Search",
    routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT: "Greedy Descent",
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH: "Guided Local Search",
    routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING: "Simulated Annealing",
}

OR_SOLVER_METHODS_INV: dict = {
    "Automatic": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
    "Tabu Search": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    "Greedy Descent": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
    "Guided Local Search": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    "Simulated Annealing": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
}


class SimulatorGUI:

    def __init__(self):
        self.scenario_path = ''
        self.loaded_graphs = dict()
        self.problem_type = PROBLEM_TYPES[0]
        self.solver_type = SOLVER_TYPES[0]
        self.solver_sub_frames = dict()
        self.solver_sub_frame_methods = {
            SOLVER_TYPES[0]: self._create_or_frame,
            SOLVER_TYPES[1]: self._create_ga_frame,
            SOLVER_TYPES[2]: self._create_local_solver_frame
        }
        self.create_solver_methods = {
            SOLVER_TYPES[0]: self._create_or_tools_solver,
            SOLVER_TYPES[1]: self._create_ga_solver,
            SOLVER_TYPES[2]: self._create_local_solver
        }
        self.ga_settings = {
            'population_size': 300,
            'generations': 400,
            'crossover_prob': 0.7,
            'mutate_prob': 0.03
        }
        self.or_settings = SolverParams()
        self.visualization_settings = {
            'visualize': False,
            'show_edges': True,
            'show_history': False,
            'print_history': True,
            'print_solution': True,
            'save_history': True
        }

        self._local_solver_settings = {
            'greedy': True
        }

    def create_gui(self, root):
        self._create_solver_frame(root)
        self._create_solver_params_frame(root)
        self._create_visualization_frame(root)
        self._create_scenario_frame(root)
        self._create_run_frame(root)

    def _start_simulation(self):
        problem = self._load_problem(self.scenario_path)
        graph = self._load_graph(self.scenario_path)
        solver = self.create_solver_methods[self.solver_type](graph)

        solution = solver.solve(problem)

        if self.visualization_settings['print_solution']:
            printer = DataFramePrinter(only_node_actions=True)
            printer.print_solution(solution)

        if self.visualization_settings['save_history']:
            self._save_history(solver)

        if self.visualization_settings['visualize']:
            solution_renderer = SolutionRenderer()
            solution_renderer.render_solution(solution, graph,
                                              show_graph_edges=self.visualization_settings['show_edges'])

        if self.visualization_settings['show_history']:
            solver_renderer = SolverRenderer()
            solver_renderer.render_solver_history(solver.history)

        if self.visualization_settings['print_history']:
            logger.info("Solver history:")
            for entry in solver.history:
                logger.info(f"   {entry}")

    def _run_all(self):
        data_root_path = f"../../../data/scenarios/"
        logging.disable(logging.DEBUG)

        if self.problem_type == PROBLEM_TYPES[0] or self.problem_type == PROBLEM_TYPES[1]:
            data_root_path += "cvrp"
        else:
            data_root_path += "vrpdp"

        for path, dirs, files in os.walk(os.path.abspath(data_root_path)):
            for k, file_name in enumerate(files):
                progress_logger.info(f"\rSolver '{self.solver_type}'  "
                                     f"Problem type '{self.problem_type}'  "
                                     f"File '{file_name}' {k + 1}/{len(files)}")
                self.scenario_path = os.path.join(path, file_name)

                self._start_simulation()
        progress_logger.info('\n\r')
        logging.disable(logging.NOTSET)

    def _save_history(self, solver: ISolver):
        scenario_name = os.path.basename(self.scenario_path)
        re_result = re.search(r"(.*)(\.json)", scenario_name)
        scenario_name = re_result.group(1)
        file_name = f"result_{scenario_name}.json"
        result = {
            'solver_type': self.solver_type,
            'problem_type': self.problem_type,
            'scenario_name': scenario_name,
            'value': solver.history[-1].best_value,
            'total_runtime': str(solver.history[-1].runtime),
            'setup_time': str(solver.history[0].setup_time),
            'history': solver.history
        }
        if self.solver_type == SOLVER_TYPES[0]:
            result['solver_settings'] = self.or_settings
        if self.solver_type == SOLVER_TYPES[1]:
            result['solver_settings'] = self.ga_settings
        if self.solver_type == SOLVER_TYPES[2]:
            result['solver_settings'] = self._local_solver_settings

        file_path = f"../../../data/results/{self.solver_type}/{self.problem_type}/"
        if not os.path.exists(f"../../../data/results/{self.solver_type}"):
            os.mkdir(f"../../../data/results/{self.solver_type}")
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        file_path = os.path.join(file_path, file_name)

        if os.path.exists(file_path):
            with open(file_path) as file:
                file_data = json.load(file)
                # no improvement -> don't overwrite it
                if file_data['value'] < result['value']:
                    return

        with open(file_path, 'w') as file:
            serialized = json.dumps(result, indent=4, cls=JSONSerializer)
            file.write(serialized)
            # logger.info(f"History saved to {file_path}")

    def _is_pick_and_delivery(self) -> bool:
        return self.problem_type == PROBLEM_TYPES[2] or self.problem_type == PROBLEM_TYPES[3]

    def _is_uturn(self) -> bool:
        return self.problem_type == PROBLEM_TYPES[1] or self.problem_type == PROBLEM_TYPES[3]

    def _load_problem(self, path):
        vehicles = []
        calc_time = datetime.now()
        trqs = []

        with open(path) as f:
            data = json.load(f)

        depot = data['depot']

        for i, rq in enumerate(data['requests']):
            from_node = rq.get('from_node', depot)
            trq = TransportRequest(str(i), from_node, rq['to_node'], calc_time, 1)
            trqs.append(trq)

        for i in range(data['vehicle_count']):
            v = Vehicle(str(i), depot, '', calc_time, data['max_capacity'], 1)
            vehicles.append(v)

        problem = VRPProblem(transport_requests=trqs, vehicles=vehicles, calculation_time=calc_time,
                             pick_duration=timedelta(seconds=0),
                             delivery_duration=timedelta(seconds=0), vehicle_velocity=1, depot=depot)
        return problem

    def _load_graph(self, path):
        with open(path) as f:
            data = json.load(f)

        graph_path = data['graph']
        if graph_path in self.loaded_graphs:
            graph = self.loaded_graphs[graph_path]
        else:
            with open(f"../../../data/graphs/{data['graph']}") as graph_file:
                graph = Graph.from_json(json.load(graph_file))
                graph = UTurnGraph(UTurnTransitionFunction(graph), graph)
                self.loaded_graphs[graph_path] = graph

        if self._is_uturn():
            return graph
        else:
            return graph.base_graph

    def _create_run_frame(self, root):
        frame = ttk.Frame(master=root, relief=tk.RAISED, borderwidth=1)

        def visualize_scenario():
            problem = self._load_problem(self.scenario_path)
            graph = self._load_graph(self.scenario_path)

            graph_renderer = GraphRenderer()
            graph_renderer.render_graph(graph, trqs=problem.transport_requests, depot=problem.depot)

        vis_btn = tk.Button(frame, text='Visualize Scenario', command=visualize_scenario, height=4, width=20)
        vis_btn.grid(column=0, row=0, sticky=tk.W, padx=10)

        run_btn = tk.Button(frame, text='Start Simulation', command=self._start_simulation, height=4, width=20)
        run_btn.grid(column=1, row=0, sticky=tk.W, padx=10)

        run_all_btn = tk.Button(frame, text='Run all scenarios', command=self._run_all, height=4, width=20)
        run_all_btn.grid(column=2, row=0, sticky=tk.W, padx=10)

        frame.pack(fill=tk.X)

    def _create_scenario_frame(self, root):
        frame = ttk.Frame(master=root, relief=tk.RAISED, borderwidth=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=3)
        frame.columnconfigure(2, weight=1)

        label_path = tk.Label(master=frame, text="Scenario")
        label_path.grid(column=0, row=0, sticky=tk.W, padx=10)

        label_path_chosen = tk.Label(master=frame, text="....", relief=tk.SUNKEN, width=45)
        label_path_chosen.grid(column=1, row=0, sticky=tk.W)

        def on_choose_scenario_clicked():
            filename = filedialog.askopenfilename(defaultextension='.json', filetypes=[("json files", '*.json')])
            if not filename:
                return
            label_path_chosen['text'] = filename
            self.scenario_path = filename

        btn_0 = tk.Button(master=frame, text="Choose scenario", relief=tk.RAISED,
                          command=on_choose_scenario_clicked)
        btn_0.grid(column=2, row=0, padx=10, sticky=tk.W)

        frame.pack(fill=tk.X)

    def _create_solver_frame(self, root):
        frame = ttk.Frame(master=root, relief=tk.RAISED, borderwidth=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        lbl_solver = tk.Label(master=frame, text="Solver:")
        lbl_solver.grid(column=0, row=0, sticky=tk.W, padx=5)

        def on_solver_type_changed(*args):
            for name, frame in self.solver_sub_frames.items():
                frame.grid_forget()

            self.solver_sub_frames[var_solver.get()].grid(column=0, row=2, sticky=tk.W)
            self.solver_type = var_solver.get()

        var_solver = tk.StringVar(frame)
        var_solver.set(self.solver_type)
        var_solver.trace("w", on_solver_type_changed)

        drop_down_solver = tk.OptionMenu(frame, var_solver, *SOLVER_TYPES)
        drop_down_solver.grid(column=1, row=0, sticky=tk.W)

        # initialize sub frame for each solver type
        for name, method in self.solver_sub_frame_methods.items():
            self.solver_sub_frames[name] = method(frame)

        def on_problem_type_change(*args):
            self.problem_type = var_type.get()

        var_type = tk.StringVar(frame)
        var_type.set(self.problem_type)
        var_type.trace("w", on_problem_type_change)

        lbl_type = tk.Label(master=frame, text="Problem type:")
        lbl_type.grid(column=0, row=1, sticky=tk.W, padx=5)

        drop_down_solver = tk.OptionMenu(frame, var_type, *PROBLEM_TYPES)
        drop_down_solver.grid(column=1, row=1, sticky=tk.W)

        frame.pack(fill=tk.X)

        on_solver_type_changed()

    def _create_solver_params_frame(self, root):
        pass

    def _create_ga_frame(self, root):
        frame = ttk.Frame(master=root, relief=tk.RAISED, borderwidth=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # population size
        lbl_pop_size = tk.Label(master=frame, text="Population Size:")
        lbl_pop_size.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)

        self.var_pop_size = tk.IntVar()
        self.var_pop_size.set(self.ga_settings['population_size'])
        entry_pop_size = tk.Entry(master=frame, textvariable=self.var_pop_size)
        entry_pop_size.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)

        def on_pop_size_changed(*args):
            self.ga_settings['population_size'] = int(entry_pop_size.get())

        self.var_pop_size.trace_add("write", on_pop_size_changed)

        # generations
        lbl_generations = tk.Label(master=frame, text="Generations:")
        lbl_generations.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)

        self.var_generations = tk.IntVar()
        self.var_generations.set(self.ga_settings['generations'])
        entry_generations = tk.Entry(master=frame, textvariable=self.var_generations)
        entry_generations.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)

        def on_generations_changed(*args):
            self.ga_settings['generations'] = int(entry_generations.get())

        self.var_generations.trace("w", on_generations_changed)

        # crossover
        lbl_crossover = tk.Label(master=frame, text="Crossover Probability:")
        lbl_crossover.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)

        self.var_crossover = tk.StringVar()
        self.var_crossover.set(self.ga_settings['crossover_prob'])
        entry_crossover = tk.Entry(master=frame, textvariable=self.var_crossover)
        entry_crossover.grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)

        def on_crossover_changed(*args):
            self.ga_settings['crossover_prob'] = float(entry_crossover.get())

        self.var_crossover.trace("w", on_crossover_changed)

        # mutation
        lbl_mutate = tk.Label(master=frame, text="Mutation Probability:")
        lbl_mutate.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)

        self.var_mutate = tk.StringVar()
        self.var_mutate.set(self.ga_settings['mutate_prob'])
        entry_mutate = tk.Entry(master=frame, textvariable=self.var_mutate)
        entry_mutate.grid(column=1, row=3, sticky=tk.W, padx=5, pady=5)

        def on_mutate_changed(*args):
            self.ga_settings['mutate_prob'] = float(entry_mutate.get())

        self.var_mutate.trace("w", on_mutate_changed)

        frame.grid()
        return frame

    def _create_visualization_frame(self, root):
        frame = ttk.Frame(master=root, relief=tk.RAISED, borderwidth=1)

        var_visualize = tk.BooleanVar()
        var_visualize.set(self.visualization_settings['visualize'])

        check_visualize = tk.Checkbutton(frame, text="Visualize solution", variable=var_visualize)
        check_visualize.pack(side=tk.LEFT)

        def on_visualize_changed(*args):
            self.visualization_settings['visualize'] = var_visualize.get()

        var_visualize.trace("w", on_visualize_changed)

        var_show_edges = tk.BooleanVar()
        var_show_edges.set(self.visualization_settings['show_edges'])

        check_show_edges = tk.Checkbutton(frame, text="Show graph edges", variable=var_show_edges)
        check_show_edges.pack(side=tk.LEFT)

        def on_show_edges_changed(*args):
            self.visualization_settings['show_edges'] = var_show_edges.get()

        var_show_edges.trace("w", on_show_edges_changed)

        var_visualize_history = tk.BooleanVar()
        var_visualize_history.set(self.visualization_settings['show_history'])

        check_visualize_history = tk.Checkbutton(frame, text="Visualize solver history", variable=var_visualize_history)
        check_visualize_history.pack(side=tk.LEFT)

        def on_visualize_history_changed(*args):
            self.visualization_settings['show_history'] = var_visualize_history.get()

        var_visualize_history.trace("w", on_visualize_history_changed)

        var_print_history = tk.BooleanVar()
        var_print_history.set(self.visualization_settings['print_history'])

        check_print_history = tk.Checkbutton(frame, text="Print solver history", variable=var_print_history)
        check_print_history.pack(side=tk.LEFT)

        def on_print_history_changed(*args):
            self.visualization_settings['print_history'] = var_print_history.get()

        var_print_history.trace("w", on_print_history_changed)

        var_print_solution = tk.BooleanVar()
        var_print_solution.set(self.visualization_settings['print_solution'])

        check_print_solution = tk.Checkbutton(frame, text="Print solution", variable=var_print_solution)
        check_print_solution.pack(side=tk.LEFT)

        def on_print_solution_changed(*args):
            self.visualization_settings['print_solution'] = var_print_solution.get()

        var_print_solution.trace("w", on_print_solution_changed)

        var_save_history = tk.BooleanVar()
        var_save_history.set(self.visualization_settings['save_history'])

        check_save_history = tk.Checkbutton(frame, text="Save history", variable=var_save_history)
        check_save_history.pack(side=tk.LEFT)

        def on_save_history_changed(*args):
            self.visualization_settings['save_history'] = var_save_history.get()

        var_save_history.trace("w", on_save_history_changed)

        frame.pack(fill=tk.X)

    def _create_or_frame(self, root):
        frame = ttk.Frame(master=root, relief=tk.RAISED, borderwidth=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        lbl_solver_method = tk.Label(master=frame, text="Local Search Metaheuristic:")
        lbl_solver_method.grid(column=0, row=0, sticky=tk.W, padx=10, pady=5)

        def on_solver_method_changed(*args):
            self.or_settings.solver_method = OR_SOLVER_METHODS_INV[var_solver_method.get()]

        var_solver_method = tk.StringVar(frame)
        var_solver_method.set(OR_SOLVER_METHODS[self.or_settings.solver_method])
        var_solver_method.trace("w", on_solver_method_changed)

        drop_down_solver = tk.OptionMenu(frame, var_solver_method, *list(OR_SOLVER_METHODS.values()))
        drop_down_solver.grid(column=1, row=0, sticky=tk.W)

        frame.grid()
        return frame

    def _create_local_solver_frame(self, root):
        frame = ttk.Frame(master=root, relief=tk.RAISED, borderwidth=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        var_greedy = tk.BooleanVar()
        var_greedy.set(self._local_solver_settings['greedy'])

        check_greedy = tk.Checkbutton(frame, text="Greedy", variable=var_greedy)
        check_greedy.pack(side=tk.LEFT)

        def on_greedy_changed(*args):
            self._local_solver_settings['greedy'] = var_greedy.get()

        var_greedy.trace("w", on_greedy_changed)

        return frame

    def _create_or_tools_solver(self, graph) -> ISolver:
        if self.problem_type == PROBLEM_TYPES[0]:
            return SolverCVRP(NodeDistanceAStar(distances=graph.distances), graph, False, self.or_settings)

        if self.problem_type == PROBLEM_TYPES[1]:
            return SolverCVRPU(NodeDistanceAStar(distances=graph.distances), graph, self.or_settings)

        if self.problem_type == PROBLEM_TYPES[2]:
            return SolverVRPDP(NodeDistanceAStar(distances=graph.distances), graph, False, self.or_settings)

        if self.problem_type == PROBLEM_TYPES[3]:
            return SolverVRPDPU(NodeDistanceAStar(distances=graph.distances), graph, self.or_settings)

    def _create_ga_solver(self, graph) -> ISolver:
        pop_size = self.ga_settings['population_size']
        generations = self.ga_settings['generations']
        crossover_prob = self.ga_settings['crossover_prob']
        mutate_prob = self.ga_settings['mutate_prob']

        if self.problem_type == PROBLEM_TYPES[0]:
            return GASolverCVRP(NodeDistanceAStar(distances=graph.distances), graph=graph, population_size=pop_size,
                                generations=generations,
                                mutate_prob=crossover_prob,
                                crossover_prob=mutate_prob)

        if self.problem_type == PROBLEM_TYPES[1]:
            return GASolverCVRPU(NodeDistanceAStar(distances=graph.distances), graph=graph, population_size=pop_size,
                                 generations=generations,
                                 mutate_prob=crossover_prob,
                                 crossover_prob=mutate_prob)

        if self.problem_type == PROBLEM_TYPES[2]:
            return GASolverVRPDP(NodeDistanceAStar(distances=graph.distances), graph=graph, population_size=pop_size,
                                 generations=generations,
                                 mutate_prob=crossover_prob,
                                 crossover_prob=mutate_prob)

        if self.problem_type == PROBLEM_TYPES[3]:
            return GASolverVRPDPU(NodeDistanceAStar(distances=graph.distances), graph=graph, population_size=pop_size,
                                  generations=generations,
                                  mutate_prob=crossover_prob,
                                  crossover_prob=mutate_prob)

    def _create_local_solver(self, graph):
        if self.problem_type == PROBLEM_TYPES[0]:
            return LocalSolver(CyclicNeighborhoodGeneratorCVRP(), DistanceObjective(),
                               InitSolverCVRP(NodeDistanceAStar(distances=graph.distances), graph),
                               self._local_solver_settings['greedy'])

        if self.problem_type == PROBLEM_TYPES[1]:
            return LocalSolver(CyclicNeighborhoodGeneratorCVRP(), DistanceObjective(),
                               InitSolverCVRPU(NodeDistanceAStar(distances=graph.distances), graph),
                               self._local_solver_settings['greedy'])

        if self.problem_type == PROBLEM_TYPES[2]:
            return LocalSolver(CyclicNeighborhoodGeneratorVRPDP(), DistanceObjective(),
                               InitSolverVRPDP(NodeDistanceAStar(distances=graph.distances), graph),
                               self._local_solver_settings['greedy'])

        if self.problem_type == PROBLEM_TYPES[3]:
            return LocalSolver(CyclicNeighborhoodGeneratorVRPDP(), DistanceObjective(),
                               InitSolverVRPDPU(NodeDistanceAStar(distances=graph.distances), graph),
                               self._local_solver_settings['greedy'])


if __name__ == '__main__':
    root = tk.Tk(className='VRP Simulator')
    root.geometry("800x400")

    with open(os.path.join(os.path.dirname(__file__), 'logging.yaml')) as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)

    simulator_gui = SimulatorGUI()
    simulator_gui.create_gui(root)

    root.mainloop()
