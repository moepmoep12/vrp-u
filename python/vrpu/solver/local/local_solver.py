from numbers import Number
from overrides import overrides
from timeit import default_timer as timer
from datetime import timedelta

from vrpu.core import VRPProblem, Solution

from vrpu.solver.local.neighborhood import INeighborhoodGenerator, Neighbor
from vrpu.solver.local.objective import IObjectiveFunction
from vrpu.solver.solver import ISolver, SolvingSnapshot
from vrpu.solver.solution_encoding import EncodedAction, EncodedSolution

from vrpu.solver.genetic_algorithm.ga_solver import GASolverCVRP, GASolverVRPDP, GASolverCVRPU, GASolverVRPDPU


class InitSolverCVRP(GASolverCVRP):

    @overrides
    def solve(self, problem: VRPProblem) -> Solution:
        self._current_problem = problem
        self._actions = self._create_actions()
        self._setup_node_distance_function()
        individual = self._init_individual()

        return EncodedSolution([EncodedAction(a[0], a[1]) for a in individual], self._actions,
                               self._current_problem.vehicles,
                               [self._get_start_node_for_vehicle(v) for v in self._current_problem.vehicles],
                               self._node_distance)


class InitSolverVRPDP(GASolverVRPDP):

    @overrides
    def solve(self, problem: VRPProblem) -> Solution:
        self._current_problem = problem
        self._actions = self._create_actions()
        self._setup_node_distance_function()
        individual = self._init_individual()

        return EncodedSolution([EncodedAction(a[0], a[1]) for a in individual], self._actions,
                               self._current_problem.vehicles,
                               [self._get_start_node_for_vehicle(v) for v in self._current_problem.vehicles],
                               self._node_distance)


class InitSolverCVRPU(GASolverCVRPU):

    def _init_individual(self):
        """
        :return: An initial individual consisting of a list of tuples indicating for every action to which
        vehicle it belongs and its value.
        """
        keys = []
        vehicle_index = 0
        load = 0

        for i in range(len(self._actions)):
            vehicle = self._current_problem.vehicles[vehicle_index]
            value = self._generate_value(existing_values=[key[1] for key in keys], min_val=1, max_val=999)
            keys.append((vehicle_index, value))
            load += 1
            if load >= vehicle.max_capacity:
                load = 0
                vehicle_index += 1

        return keys

    @overrides
    def solve(self, problem: VRPProblem) -> Solution:
        self._current_problem = problem
        self._actions = self._create_actions()
        self._setup_node_distance_function()
        individual = self._init_individual()

        return EncodedSolution([EncodedAction(a[0], a[1]) for a in individual], self._actions,
                               self._current_problem.vehicles,
                               [self._get_start_node_for_vehicle(v) for v in self._current_problem.vehicles],
                               self._node_distance)


class InitSolverVRPDPU(GASolverVRPDPU):
    def _init_individual(self):
        keys = []
        vehicle_index = 0

        for i in range(len(self._current_problem.transport_requests)):
            vehicle = self._current_problem.vehicles[vehicle_index]
            pick_value = self._generate_value([key[1] for key in keys], 1, 800)
            keys.append((vehicle_index, pick_value))
            delivery_value = self._generate_value([key[1] for key in keys], pick_value + 1, 999)
            keys.append((vehicle_index, delivery_value))

            max_loads = self._get_max_load(keys)
            if max_loads[vehicle_index] == vehicle.max_capacity:
                vehicle_index += 1

        return keys

    def _individual_to_actions(self, individual: []):
        """
        Transforms an individual to a dict containing the ordered list of actions assigned to each vehicle.
        :param individual: The individual to transform.
        :return: A dict containing the ordered list of actions assigned to each vehicle.
        """
        actions = dict()
        [actions.setdefault(i, []) for i in range(len(self._current_problem.vehicles))]

        for action_index, (vehicle_index, value) in enumerate(individual):
            actions[vehicle_index].append((self._actions[action_index], value))

        # sort actions for each vehicle according to their value
        for vehicle_index, action_list in actions.items():
            actions[vehicle_index] = [a[0] for a in sorted(action_list, key=lambda x: x[1])]

        return actions

    @overrides
    def solve(self, problem: VRPProblem) -> Solution:
        self._current_problem = problem
        self._actions = self._create_actions()
        self._setup_node_distance_function()
        individual = self._init_individual()

        return EncodedSolution([EncodedAction(a[0], a[1]) for a in individual], self._actions,
                               self._current_problem.vehicles,
                               [self._get_start_node_for_vehicle(v) for v in self._current_problem.vehicles],
                               self._node_distance)


class LocalSolver(ISolver):
    """
    A greedy solver that keeps searching until no better solution is found.
    Also called 'Hill Climbing' or 'Greedy Descent'.
    """

    def __init__(self, neighborhood_gen: INeighborhoodGenerator, objective_func: IObjectiveFunction,
                 init_solver: ISolver, greedy: bool = True):
        self.neighborhood_gen: INeighborhoodGenerator = neighborhood_gen
        self.objective_function: IObjectiveFunction = objective_func
        self.initial_solution: EncodedSolution = None
        self.init_solver = init_solver
        self.best_solution: EncodedSolution = self.initial_solution
        self.greedy = greedy
        self.iteration = 0
        self.step = 0
        self.steps_without_improvement = 0
        self.neighbor_history: [Neighbor] = []
        self._history: [SolvingSnapshot] = []

    @overrides
    def solve(self, problem: VRPProblem) -> Solution:
        start_timer = timer()
        self.step = 0
        self.neighbor_history = []
        self.initial_solution = self.init_solver.solve(problem)

        self.best_solution = self.initial_solution
        best_value = self.objective_function.value(self.best_solution)

        self.iteration = 0
        self.steps_without_improvement = 0

        print(f"-- Start Solving with initial value {best_value} and Greedy: {self.greedy} --")

        while self.steps_without_improvement < self.neighborhood_gen.get_max_steps():
            self.iteration += 1

            # Keep track of stats
            self._history.append(
                SolvingSnapshot(
                    runtime=timedelta(seconds=timer() - start_timer),
                    step=self.iteration,
                    best_value=best_value,
                    average=best_value,
                    min_value=best_value,
                    max_value=best_value
                )
            )

            best_neighbor: Neighbor = self._get_best_neighbor(self.best_solution, best_value)
            if best_neighbor is None:
                self.steps_without_improvement += 1
                self.step += 1
                continue

            if best_neighbor.value < best_value:
                best_value = best_neighbor.value
                self.best_solution = best_neighbor.solution
                self.steps_without_improvement = 0
                self.neighbor_history.append(best_neighbor)
            else:
                self.steps_without_improvement += 1
                self.step += 1

            print(f"\r   Iteration: {self.iteration}, Best value: {best_value}", end='')

        print(f"\n-- End of solving after {timer() - start_timer}s --")

        return self.best_solution

    @property
    def history(self) -> [SolvingSnapshot]:
        return self._history

    def _get_best_neighbor(self, solution: EncodedSolution, best_value: Number) -> Neighbor:
        neighborhood = self.neighborhood_gen.generate_neighborhood(solution, iteration=self.step,
                                                                   greedy=self.greedy, best_value=best_value,
                                                                   objective=self.objective_function)
        best_neighbor: Neighbor = None
        if not neighborhood:
            return best_neighbor

        for neighbor in neighborhood:
            if neighbor.value is None:
                neighbor.value = self.objective_function.value(neighbor.solution)
            if best_neighbor is None or neighbor.value < best_neighbor.value:
                best_neighbor = neighbor

        return best_neighbor
