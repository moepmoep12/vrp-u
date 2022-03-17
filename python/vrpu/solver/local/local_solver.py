import sys
import logging
import random
from dataclasses import dataclass
from numbers import Number
from typing import Tuple, Union

from overrides import overrides
from timeit import default_timer as timer
from datetime import timedelta

from vrpu.core import VRPProblem, Solution, Delivery, DeliveryUTurnState, Vehicle, TransportAction, PickUp, \
    PickUpUTurnState
from vrpu.core.graph.search.node_distance import CachedNodeDistance

from vrpu.solver.local.neighborhood import INeighborhoodGenerator, Neighbor
from vrpu.solver.local.objective import IObjectiveFunction
from vrpu.solver.solver import ISolver, SolvingSnapshot
from vrpu.solver.solution_encoding import EncodedAction, EncodedSolution

progress_logger = logging.getLogger('progress')


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
        start_timer = timer()
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

        setup_time = timedelta(seconds=timer() - start_timer)

        logging.debug('Starting initial solution...')

        while not all(actions_inserted):
            progress_logger.debug(f"\r Inserted {sum(actions_inserted)}/{len(actions_inserted)}")
            action_to_insert_indices = self._choose_actions_to_insert(actions_inserted, current_vehicle_idx)
            current_vehicle = self._current_problem.vehicles[current_vehicle_idx]
            for action_index, value in action_to_insert_indices:
                encoded_action = EncodedAction(current_vehicle_idx, value)
                self._current_solution[action_index] = encoded_action
                actions_inserted[action_index] = True

            max_loads = self._current_solution.get_max_loads()
            if max_loads[current_vehicle_idx] >= current_vehicle.max_capacity:
                current_vehicle_idx += 1

        runtime = timedelta(seconds=timer() - start_timer) - setup_time
        self._history.append(SolvingSnapshot(runtime=runtime,
                                             setup_time=setup_time,
                                             step=0,
                                             best_value=0,
                                             average=0,
                                             min_value=0,
                                             max_value=0))
        progress_logger.debug("\n\r")
        return self._current_solution

    @property
    def history(self) -> [SolvingSnapshot]:
        return self._history

    def _create_actions(self) -> [TransportAction]:
        """
        :return: All actions for the current problem.
        """
        delivery_duration = timedelta(seconds=self._current_problem.delivery_duration.total_seconds())
        deliveries = [Delivery(trq, delivery_duration) for trq in self._current_problem.transport_requests]
        return [*deliveries]

    def _setup_node_distance_function(self) -> None:
        """
        Sets up the node distance function to pre-calculate needed distances.
        """
        nodes = [a.node for a in self._actions]
        nodes.extend([v.current_node for v in self._current_problem.vehicles])
        nodes = set(nodes)
        self._node_distance.calculate_distances(self._graph, nodes)

    def _get_start_node_for_vehicle(self, vehicle: Vehicle) -> str:
        return vehicle.current_node

    def _choose_actions_to_insert(self, actions_inserted: [bool], vehicle_index: int) -> [Tuple[int, int]]:
        best_action_index = 0
        best_value = sys.maxsize

        for i, (action, inserted) in enumerate(zip(self._actions, actions_inserted)):
            if inserted:
                continue

            # simulate the action being inserted and calculate the resulting total distance of the solution
            clone = self._current_solution.clone()
            clone[i] = EncodedAction(vehicle_index, sum(actions_inserted) * 10)

            distance = sum(t.get_distance() for t in clone.tours)
            if distance < best_value:
                best_action_index = i
                best_value = distance

        return [(best_action_index, sum(actions_inserted) * 10)]


class InitSolverVRPDP(InitSolverCVRP):

    def _create_actions(self) -> [TransportAction]:
        """
        :return: All actions for the current problem.
        """
        actions = []

        for trq in self._current_problem.transport_requests:
            actions.append(PickUp(trq, self._current_problem.pick_duration))
            actions.append(Delivery(trq, self._current_problem.delivery_duration))

        return actions

    def _choose_actions_to_insert(self, actions_inserted: [bool], vehicle_index: int) -> [Tuple[int, int]]:
        best_distances = []
        max_actions_to_test = 20
        # if that many actions are already present in a tour, than only a subset of possible positions will be tested
        action_inserted_threshold = 50
        # how many positions for the pickup action will be tried
        max_pickup_pos = 7
        # how many different positions for the delivery will be tried
        # the total amount of tries is max_pickup_pos * max_delivery_pos
        max_delivery_pos = 7

        @dataclass
        class PDDistance:
            pickup_index: int
            delivery_index: int
            value_pickup: int
            value_delivery: int
            distance: Number
            solution: EncodedSolution

        # gather all actions (indices) that are not yet inserted
        actions_to_test = [(i, i + 1) for i in range(0, len(actions_inserted) - 1, 2) if not actions_inserted[i]]
        if len(actions_to_test) > max_actions_to_test:
            actions_to_test = random.sample(max_actions_to_test)

        # go through all actions not already inserted
        for p_idx, d_idx in actions_to_test:

            pd_distance = PDDistance(p_idx, d_idx, 0, 10, sys.maxsize, None)
            best_distances.append(pd_distance)

            # gather all possible values for the pickup action
            possible_values = [e.value for e in self._current_solution if
                               e is not None and e.vehicle_index == vehicle_index]
            if len(possible_values) > 0:
                # allows insertion at tour end
                possible_values.append(max(possible_values) + 1)
            else:
                possible_values.append(1)

            if sum(actions_inserted) > action_inserted_threshold and len(possible_values) > max_pickup_pos:
                possible_values = random.sample(possible_values, max_pickup_pos)

            # go through all possible values (positions) for the pickup action
            for pick_value in possible_values:
                clone_p = self._current_solution.clone()
                # simulate the pickup action being in the tour
                clone_p[p_idx] = EncodedAction(vehicle_index, pick_value)

                # order is disturbed by insertion of the pickup action -> restore it
                for j, encoded_action in enumerate(clone_p):
                    if j == p_idx:
                        continue
                    if encoded_action is None:
                        continue
                    if encoded_action.value >= pick_value:
                        encoded_action.value += 1

                # gather possible values for delivery action
                possible_values = [e.value for e in clone_p if
                                   e is not None and e.vehicle_index == vehicle_index and e.value > pick_value]
                if len(possible_values) > 0:
                    # allows insertion at tour end
                    possible_values.append(max(possible_values) + 1)
                else:
                    possible_values.append(2)

                if sum(actions_inserted) > action_inserted_threshold and len(possible_values) > max_delivery_pos:
                    possible_values = random.sample(possible_values, max_delivery_pos)

                # go through all possible positions for the delivery (that is after the pickup)
                for delivery_value in possible_values:
                    clone_d = clone_p.clone()
                    # simulate the delivery being in the tour, too
                    clone_d[d_idx] = EncodedAction(vehicle_index, delivery_value)

                    # order is disturbed by insertion of the pickup action -> restore it
                    for j, encoded_action in enumerate(clone_d):
                        if j == d_idx:
                            continue
                        if encoded_action is None:
                            continue
                        if encoded_action.value >= delivery_value:
                            encoded_action.value += 1

                    # check capacity
                    max_loads = clone_d.get_max_loads()
                    if max_loads[vehicle_index] > self._current_problem.vehicles[vehicle_index].max_capacity:
                        continue

                    distance = sum(t.get_distance() for t in clone_d.tours)
                    if distance < pd_distance.distance:
                        pd_distance.distance = distance
                        pd_distance.value_pickup = pick_value
                        pd_distance.value_delivery = delivery_value
                        pd_distance.solution = clone_d

        best_distances = sorted(best_distances, key=lambda pd: pd.distance)
        best_pd = best_distances[0]
        for orig, changed in zip(self._current_solution, best_pd.solution):
            if orig is not None and changed is not None:
                orig.value = changed.value
        return [(best_pd.pickup_index, best_pd.value_pickup), (best_pd.delivery_index, best_pd.value_delivery)]


class InitSolverCVRPU(InitSolverCVRP):

    @overrides
    def _setup_node_distance_function(self) -> None:
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

    def _get_possible_state_nodes(self, current_node: str) -> [str]:
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
    def _get_start_node_for_vehicle(self, vehicle: Vehicle) -> str:
        if vehicle.node_arriving_from:
            for uid, node in self._graph.nodes.items():
                if node.data.current_node == vehicle.current_node \
                        and node.data.prev_node == vehicle.node_arriving_from:
                    return uid

        else:
            # free choice
            return self._get_possible_state_nodes(vehicle.current_node)[0]


class InitSolverVRPDPU(InitSolverCVRPU):
    def _create_actions(self) -> [[Union[PickUpUTurnState, DeliveryUTurnState]]]:
        """
        :return: All actions for the current problem.
        Every Delivery can be executed with one of many DeliveryUTurnStates.
        """
        actions = []

        for i, trq in enumerate(self._current_problem.transport_requests):
            actions.append(
                [PickUpUTurnState(trq, i, node_state, self._current_problem.pick_duration)
                 for node_state in self._get_possible_state_nodes(trq.from_node)])
            actions.append(
                [DeliveryUTurnState(trq, i, node_state, self._current_problem.delivery_duration)
                 for node_state in self._get_possible_state_nodes(trq.to_node)])

        return actions

    def _choose_actions_to_insert(self, actions_inserted: [bool], vehicle_index: int) -> [Tuple[int, int]]:
        best_distances = []

        @dataclass
        class PDDistance:
            pickup_index: int
            delivery_index: int
            value_pickup: int
            value_delivery: int
            distance: Number
            solution: EncodedSolution

        # go through all actions not already inserted
        for i in range(0, len(actions_inserted) - 1, 2):
            if actions_inserted[i]:
                continue

            pd_distance = PDDistance(i, i + 1, 0, 10, sys.maxsize, None)
            best_distances.append(pd_distance)

            # gather all possible values for the pickup action
            possible_values = [e.value for e in self._current_solution if
                               e is not None and e.vehicle_index == vehicle_index]
            if len(possible_values) > 0:
                # allows insertion at tour end
                possible_values.append(max(possible_values) + 1)
            else:
                possible_values.append(1)

            # go through all possible values (positions) for the pickup action
            for pick_value in possible_values:
                clone_p = self._current_solution.clone()
                # simulate the pickup action being in the tour
                clone_p[i] = EncodedAction(vehicle_index, pick_value)

                # order is disturbed by insertion of the pickup action -> restore it
                for j, encoded_action in enumerate(clone_p):
                    if j == i:
                        continue
                    if encoded_action is None:
                        continue
                    if encoded_action.value >= pick_value:
                        encoded_action.value += 1

                # gather possible values for delivery action
                possible_values = [e.value for e in clone_p if
                                   e is not None and e.vehicle_index == vehicle_index and e.value > pick_value]
                if len(possible_values) > 0:
                    # allows insertion at tour end
                    possible_values.append(max(possible_values) + 1)
                else:
                    possible_values.append(2)

                # go through all possible positions for the delivery (that is after the pickup)
                for delivery_value in possible_values:
                    clone_d = clone_p.clone()
                    # simulate the delivery being in the tour, too
                    clone_d[i + 1] = EncodedAction(vehicle_index, delivery_value)

                    # order is disturbed by insertion of the pickup action -> restore it
                    for j, encoded_action in enumerate(clone_d):
                        if j == i + 1:
                            continue
                        if encoded_action is None:
                            continue
                        if encoded_action.value >= delivery_value:
                            encoded_action.value += 1

                    # check capacity
                    max_loads = clone_d.get_max_loads()
                    if max_loads[vehicle_index] > self._current_problem.vehicles[vehicle_index].max_capacity:
                        continue

                    distance = sum(t.get_distance() for t in clone_d.tours)
                    if distance < pd_distance.distance:
                        pd_distance.distance = distance
                        pd_distance.value_pickup = pick_value
                        pd_distance.value_delivery = delivery_value
                        pd_distance.solution = clone_d

        best_distances = sorted(best_distances, key=lambda pd: pd.distance)
        best_pd = best_distances[0]
        for orig, changed in zip(self._current_solution, best_pd.solution):
            if orig is not None and changed is not None:
                orig.value = changed.value
        return [(best_pd.pickup_index, best_pd.value_pickup), (best_pd.delivery_index, best_pd.value_delivery)]


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

        logging.debug(f"Start Solving with initial value {best_value} and Greedy: {self.greedy}")
        init_runtime = self.init_solver.history[-1].runtime
        setup_time = timedelta(seconds=timer() - start_timer) - init_runtime

        while self.steps_without_improvement < self.neighborhood_gen.get_max_steps():
            self.iteration += 1

            # Keep track of stats
            self._history.append(
                SolvingSnapshot(
                    runtime=timedelta(seconds=timer() - start_timer) - setup_time + init_runtime,
                    setup_time=setup_time,
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

            progress_logger.debug(f"\r   Iteration: {self.iteration}, Best value: {best_value}")

        progress_logger.debug("\n\r")
        logging.debug(f"End of solving after {timer() - start_timer}s")

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
