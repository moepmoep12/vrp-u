import sys
from numbers import Number
from overrides import overrides
from timeit import default_timer as timer
from datetime import timedelta

from vrpu.core import VRPProblem, Solution, Delivery, DeliveryUTurnState, Vehicle
from vrpu.core.graph.search.node_distance import CachedNodeDistance

from vrpu.solver.local.neighborhood import INeighborhoodGenerator, Neighbor
from vrpu.solver.local.objective import IObjectiveFunction
from vrpu.solver.solver import ISolver, SolvingSnapshot
from vrpu.solver.solution_encoding import EncodedAction, EncodedSolution

from vrpu.solver.genetic_algorithm.ga_solver import GASolverVRPDP, GASolverVRPDPU


class InitSolverCVRP(ISolver):

    def __init__(self, node_distance: CachedNodeDistance, graph):
        self._node_distance: CachedNodeDistance = node_distance
        self._current_solution: EncodedSolution = None
        self._current_problem: VRPProblem = None
        self._graph = graph
        self._actions = []
        self._history: [SolvingSnapshot] = []
        self._start_nodes = []

    @overrides
    def solve(self, problem: VRPProblem) -> Solution:
        self._current_problem = problem
        self._actions = self._create_actions()
        self._setup_node_distance_function()
        self._start_nodes = [self._get_start_node_for_vehicle(v) for v in self._current_problem.vehicles]
        self._current_solution = EncodedSolution([None] * len(self._actions),
                                                 self._actions,
                                                 self._current_problem.vehicles,
                                                 self._start_nodes,
                                                 self._node_distance)

        actions_inserted = [False] * len(self._actions)
        current_vehicle_idx = 0
        current_vehicle = self._current_problem.vehicles[current_vehicle_idx]
        current_load = 0

        while not all(actions_inserted):

            action_to_insert_indices = self._choose_actions_to_insert(actions_inserted, current_vehicle_idx)
            for action_index in action_to_insert_indices:
                encoded_action = EncodedAction(current_vehicle_idx, sum(actions_inserted))
                self._current_solution[action_index] = encoded_action

                current_load += 1
                actions_inserted[action_index] = True

                if current_load >= current_vehicle.max_capacity:
                    current_vehicle_idx += 1
                    current_vehicle = self._current_problem.vehicles[current_vehicle_idx]
                    current_load = 0

        return self._current_solution

    @property
    def history(self) -> [SolvingSnapshot]:
        return self._history

    def _create_actions(self):
        """
        :return: All actions for the current problem.
        """
        delivery_duration = timedelta(seconds=self._current_problem.delivery_duration.total_seconds())
        deliveries = [Delivery(trq, delivery_duration) for trq in self._current_problem.transport_requests]
        return [*deliveries]

    def _setup_node_distance_function(self):
        """
        Sets up the node distance function to pre-calculate needed distances.
        """
        nodes = [a.node for a in self._actions]
        nodes.extend([v.current_node for v in self._current_problem.vehicles])
        nodes = set(nodes)
        self._node_distance.calculate_distances(self._graph, nodes)

    def _get_start_node_for_vehicle(self, vehicle: Vehicle):
        return vehicle.current_node

    def _choose_actions_to_insert(self, actions_inserted: [bool], vehicle_index: int) -> [int]:
        best_action_index = 0
        best_value = sys.maxsize

        for i, (action, inserted) in enumerate(zip(self._actions, actions_inserted)):
            if inserted:
                continue

            # simulate the action being inserted and calculate the resulting total distance of the solution
            clone = self._current_solution.clone()
            clone[i] = EncodedAction(vehicle_index, sum(actions_inserted))

            distance = sum(t.get_distance() for t in clone.tours)
            if distance < best_value:
                best_action_index = i
                best_value = distance

        return [best_action_index]


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


class InitSolverCVRPU(InitSolverCVRP):

    @overrides
    def _setup_node_distance_function(self):
        """
        Sets up the node distance function to pre-calculate needed distances.
        """
        node_sets = [[a.node for a in a_list] for a_list in self._actions]
        nodes = []
        for node_set in node_sets:
            nodes.extend(node_set)
        nodes.extend([self._get_start_node_for_vehicle(v) for v in self._current_problem.vehicles])
        nodes = set(nodes)
        self._node_distance.calculate_distances(self._graph, nodes)

    def _create_actions(self) -> [[DeliveryUTurnState]]:
        """
        :return: All actions for the current problem.
        Every Delivery can be executed with one of many DeliveryUTurnStates.
        """
        actions = []

        for i, trq in enumerate(self._current_problem.transport_requests):
            actions.append(
                [DeliveryUTurnState(trq, i, node_state, self._current_problem.delivery_duration)
                 for node_state in self._get_possible_state_nodes(trq.to_node)])

        return actions

    def _get_possible_state_nodes(self, current_node: str):
        """
        Returns possible states for a specific node.
        :param current_node: UID of a node.
        :return: All possible UTurn-States for this node.
        """
        result = []

        for uid, node in self._graph.nodes.items():
            if node.data.current_node == current_node:
                result.append(uid)

        return result

    @overrides
    def _get_start_node_for_vehicle(self, vehicle: Vehicle):
        if vehicle.node_arriving_from:
            for uid, node in self._graph.nodes.items():
                if node.data.current_node == vehicle.current_node \
                        and node.data.prev_node == vehicle.node_arriving_from:
                    return uid

        else:
            # free choice
            return self._get_possible_state_nodes(vehicle.current_node)[0]


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
