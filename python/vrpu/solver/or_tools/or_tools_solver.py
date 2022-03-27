import logging
from dataclasses import dataclass
from typing import List
from datetime import timedelta, datetime
from timeit import default_timer as timer
from overrides import overrides
from ortools.constraint_solver import pywrapcp, routing_parameters_pb2, routing_enums_pb2

from vrpu.core import Tour, DriveAction, VisitAction, VRPProblem, Vehicle, Action, Solution, PickUp, Delivery, \
    NodeAction, TransportAction, SetupAction, DeliveryUTurnState, PickUpUTurnState
from vrpu.solver.solver import ISolver, SolvingSnapshot
from vrpu.core.graph.search.node_distance import INodeDistance

# DIMENSIONS
TIME_DIM: str = 'Time'
DISTANCE_DIM: str = 'Distance'
CAPACITY_DIM: str = 'Capacity'

# MAX VALUES
MAX_TOUR_DISTANCE = 30000
MAX_TOUR_DURATION = 1000000

# WEIGHTINGS
TIME_WEIGHT: int = 100
DISTANCE_WEIGHT: int = 100

# SOLVER SETTINGS
MAX_RUN_TIME: int = 300


@dataclass
class SolverParams:
    solver_method: routing_enums_pb2.LocalSearchMetaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC


class SolverCVRP(ISolver):
    def __init__(self, node_distance: INodeDistance, graph, open_vrp: bool = False,
                 solver_params: SolverParams = SolverParams()):
        self._node_distance: INodeDistance = node_distance
        self._graph = graph
        self._current_problem: VRPProblem = None
        self._vehicles: List[Vehicle] = None
        self._actions: List[Action] = None
        self._index_manager: pywrapcp.RoutingIndexManager = None
        self._routing_model: pywrapcp.RoutingModel = None
        self._assignment: pywrapcp.Assignment = None
        self._solution: Solution = None
        self._open_vrp = open_vrp
        self._solver_params = solver_params
        self._history: [SolvingSnapshot] = []
        self._collector = None

    @overrides
    def solve(self, problem: VRPProblem) -> Solution:
        start_timer = timer()
        self._current_problem = problem
        self._vehicles = problem.vehicles

        if not self._vehicles:
            return

        self._actions = self._create_actions()
        logging.debug(f"Solving with {len(self._actions)} actions")

        # Setup node distance function
        self._node_distance.calculate_distances(self._graph, set([a.node for a in self._actions]))

        start_indices = [i for i in
                         range(len(self._actions) - 2 * len(self._vehicles), len(self._actions) - len(self.vehicles))]
        end_indices = [i for i in range(len(self._actions) - len(self.vehicles), len(self._actions))]
        self._index_manager = pywrapcp.RoutingIndexManager(len(self._actions),
                                                           len(self._vehicles),
                                                           start_indices,
                                                           end_indices
                                                           )

        self._routing_model = pywrapcp.RoutingModel(self._index_manager)

        # Add dimensions & register callbacks
        self._add_time_dimension()
        self._add_capacity_dimension()
        self._add_distance_dimension()

        self._add_actions()

        # Set max run time
        search_parameters: routing_parameters_pb2.RoutingSearchParameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = MAX_RUN_TIME
        search_parameters.local_search_metaheuristic = self.solver_params.solver_method

        # Initial Solution strategy
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION

        setup_time = timedelta(seconds=timer() - start_timer)

        self._routing_model.CloseModel()

        self._collector = self._routing_model.solver().AllSolutionCollector()
        self._collector.AddObjective(self._routing_model.CostVar())
        for end_index in end_indices:
            distance_var = self._routing_model.GetDimensionOrDie(DISTANCE_DIM).CumulVar(end_index)
            self._collector.Add(distance_var)
        self._routing_model.AddSearchMonitor(self._collector)

        # Start solving
        self._assignment = self._routing_model.SolveWithParameters(search_parameters=search_parameters)

        if not self._assignment:
            logging.debug("Failed to solve...")
            return

        self._solution = self._create_solution()

        runtime = timedelta(seconds=timer() - start_timer) - setup_time

        for i in range(self._collector.SolutionCount()):
            solution = self._collector.Solution(i)
            total_distance = 0
            for end_index in end_indices:
                distance_var = self._routing_model.GetDimensionOrDie(DISTANCE_DIM).CumulVar(end_index)
                value = solution.Value(distance_var)
                total_distance += value

            self._history.append(SolvingSnapshot(
                runtime=runtime,
                setup_time=setup_time,
                step=i + 1,
                best_value=total_distance,
                average=total_distance,
                min_value=total_distance,
                max_value=total_distance
            ))

        logging.debug(
            f"Found solution with objective value {self.assignment.ObjectiveValue()} after {timer() - start_timer}s")

        return self._solution

    @property
    def actions(self) -> List[Action]:
        return self._actions

    @property
    def current_problem(self) -> VRPProblem:
        return self._current_problem

    @property
    def history(self) -> [SolvingSnapshot]:
        return self._history

    @property
    def routing_model(self) -> pywrapcp.RoutingModel:
        return self._routing_model

    @property
    def assignment(self) -> pywrapcp.Assignment:
        return self._assignment

    @property
    def solver_params(self) -> SolverParams:
        return self._solver_params

    @property
    def vehicles(self) -> List[Vehicle]:
        return self._vehicles

    @property
    def node_distance(self) -> INodeDistance:
        return self._node_distance

    def get_current_solution(self) -> Solution:
        return self._solution

    def solver_index_to_action(self, solver_index: int) -> Action:
        """
        :param solver_index: The index used by this solver.
        :return: The action corresponding to the solver index.
        """
        return self._actions[self._index_manager.IndexToNode(solver_index)]

    def action_index_to_solver_index(self, action_index: int) -> int:
        """
        :param action_index: The index of the action from self.actions
        :return: The index of the corresponding action w.r.t. the solver
        """
        return self._index_manager.NodeToIndex(action_index)

    def _solution_callback(self, **args):
        solver = self._routing_model.solver()
        pass

    def _create_actions(self):
        """
        :return: All actions for the current problem.
        """
        deliveries = [Delivery(trq, self._current_problem.delivery_duration) for trq in
                      self._current_problem.transport_requests]
        start_actions = [SetupAction(v.current_node, timedelta(seconds=0)) for v in self.vehicles]
        visit_actions = [VisitAction(v.current_node, timedelta(seconds=0)) for v in self.vehicles]

        return [*deliveries, *start_actions, *visit_actions]

    def _add_time_dimension(self):
        """
        Adds the dimension 'time' to the model.
        """
        index = self._routing_model.RegisterPositiveTransitCallback(self._time_callback)
        max_duration = MAX_TOUR_DURATION

        # calculate the offset from which the tours will start
        start_offsets = [int((v.available_from - self._current_problem.calculation_time).total_seconds())
                         for v in self._vehicles]

        # calculate the maximum tour duration for each vehicle w.r.t. offset
        capacities = [int(offset + max_duration)
                      for offset, v in
                      zip(start_offsets, self._vehicles)]

        self._routing_model.AddDimensionWithVehicleCapacity(
            evaluator_index=index,
            slack_max=0,
            vehicle_capacities=capacities,
            fix_start_cumul_to_zero=False,
            name=TIME_DIM
        )

        # set the weight in the objective function
        time_dimension: pywrapcp.RoutingDimension = self._routing_model.GetDimensionOrDie(TIME_DIM)

        # TO-DO: necessary?
        time_dimension.SetGlobalSpanCostCoefficient(0)
        time_dimension.SetSpanCostCoefficientForAllVehicles(0)

        setup_duration = 0

        for vehicle_id, vehicle in enumerate(self._vehicles):
            index = self._routing_model.Start(vehicle_id)
            # set start time (offset) for tours
            time_dimension.CumulVar(index).SetValue(int(start_offsets[vehicle_id] + setup_duration))

    def _time_callback(self, from_index: int, to_index: int) -> int:
        """
        Callback for the time dimension. Called by the solver.
        :param from_index: The from_node w.r.t. Solver indices.
        :param to_index: The to_node w.r.t. Solver indices
        :return: The time it takes to travel from node A to node B.
        """
        distance = self._distance_callback(from_index, to_index)
        time = int(distance / (self._current_problem.vehicle_velocity / 3.6))
        to_action = self.solver_index_to_action(to_index)
        if isinstance(to_action, PickUp):
            time += int(self._current_problem.pick_duration.total_seconds())
        if isinstance(to_action, Delivery):
            time += int(self._current_problem.delivery_duration.total_seconds())
        return time

    def _add_distance_dimension(self):
        """
        Adds the dimension 'distance' to the model.
        """
        index = self._routing_model.RegisterTransitCallback(self._distance_callback)
        self._routing_model.SetArcCostEvaluatorOfAllVehicles(index)

        max_tour_distance = MAX_TOUR_DISTANCE

        # assumption: same distance function for all vehicles
        self._routing_model.AddDimension(
            evaluator_index=index,
            slack_max=0,
            capacity=max_tour_distance,
            fix_start_cumul_to_zero=True,
            name=DISTANCE_DIM
        )

        distance_dimension: pywrapcp.RoutingDimension = self.routing_model.GetDimensionOrDie(DISTANCE_DIM)
        distance_dimension.SetSpanCostCoefficientForAllVehicles(DISTANCE_WEIGHT)

    def _distance_callback(self, from_index: int, to_index: int) -> int:
        """
        Callback for the distance dimension. Called by the solver.
        :param from_index: Index of the from-node w.r.t. to solver indices.
        :param to_index: Index of the to-node w.r.t. to solver indices.
        :return: Distance between from-node and to-node.
        """
        from_node_idx = self._index_manager.IndexToNode(from_index)
        to_node_idx = self._index_manager.IndexToNode(to_index)
        from_action = self._actions[from_node_idx]
        to_action = self._actions[to_node_idx]

        # the distance to the route start is 0, such that tours can end anywhere
        if self._open_vrp and isinstance(to_action, VisitAction):
            return 0

        return self._node_distance.get_distance(from_action.node, to_action.node)

    def _add_capacity_dimension(self):
        """
        Adds the dimension 'capacity' to the model.
        """
        index = self._routing_model.RegisterUnaryTransitCallback(self._capacity_callback)

        self._routing_model.AddDimensionWithVehicleCapacity(
            index,
            0,
            [v.max_capacity for v in self._vehicles],
            True,
            CAPACITY_DIM
        )

    def _capacity_callback(self, node_index: int) -> int:
        """
        Callback for the capacity dimension. Called by the solver.
        :param node_index: Node under consideration.
        """
        action = self.solver_index_to_action(node_index)
        capacity = 0
        # if isinstance(action, PickUp):
        #     capacity = action.trq.quantity
        if isinstance(action, Delivery):
            capacity = action.trq.quantity
        return capacity

    def _add_actions(self) -> None:
        pass

    def _create_solution(self) -> Solution:
        """
        :return: A solution object.
        """
        time_dimension = self.routing_model.GetDimensionOrDie(TIME_DIM)
        tours: List[Tour] = []

        for vehicle_idx, vehicle in enumerate(self.vehicles):
            tour_actions: List[Action] = []
            tour_start: datetime = vehicle.available_from
            index = self.routing_model.Start(vehicle_idx)

            while not self.routing_model.IsEnd(index):

                # Create tour action
                action: NodeAction = self.solver_index_to_action(index).clone()
                action_finished_time = self.current_problem.calculation_time + timedelta(
                    seconds=self.assignment.Value(time_dimension.CumulVar(index)))
                action_start_time = action_finished_time - action.duration
                action.start_offset = action_start_time - tour_start

                # Create drive action
                if len(tour_actions) > 0:
                    from_node = tour_actions[-1].node
                    to_node = action.node
                    distance = self.node_distance.get_distance(from_node, to_node)
                    duration = action_start_time - (
                            tour_start + tour_actions[-1].start_offset + tour_actions[-1].duration)
                    offset = (action_start_time - duration) - tour_start
                    prev_node = ''
                    if '->' in to_node:
                        prev_node = to_node.split('->')[0]
                    tour_actions.append((DriveAction(from_node=from_node,
                                                     to_node=to_node,
                                                     prior_to_end_node=prev_node,
                                                     distance=distance,
                                                     duration=duration,
                                                     offset=offset)))

                tour_actions.append(action)
                index = self.assignment.Value(self.routing_model.NextVar(index))

            # Last Action
            if not self._open_vrp:
                action: Action = self.solver_index_to_action(index).clone()
                # Create tour action
                action: Action = self.solver_index_to_action(index).clone()
                action_finished_time = self.current_problem.calculation_time + timedelta(
                    seconds=self.assignment.Value(time_dimension.CumulVar(index)))
                action_start_time = action_finished_time - action.duration
                action.start_offset = action_start_time - tour_start

                # Create drive action
                if len(tour_actions) > 0:
                    from_node = tour_actions[-1].node
                    to_node = action.node
                    distance = self.node_distance.get_distance(from_node, to_node)
                    duration = action_start_time - (
                            tour_start + tour_actions[-1].start_offset + tour_actions[-1].duration)
                    offset = (action_start_time - duration) - tour_start
                    prev_node = ''
                    if '->' in to_node:
                        prev_node = to_node.split('->')[0]
                    tour_actions.append((DriveAction(from_node=from_node,
                                                     to_node=to_node,
                                                     prior_to_end_node=prev_node,
                                                     distance=distance,
                                                     duration=duration,
                                                     offset=offset)))

                tour_actions.append(action)

            contains_transport_action = False
            for act in tour_actions:
                if isinstance(act, TransportAction):
                    contains_transport_action = True
                    break

            if contains_transport_action:
                tours.append(Tour(str(vehicle_idx), tour_actions, vehicle, tour_start, action_finished_time))

        return Solution(tours)


class SolverVRPDP(SolverCVRP):

    @overrides
    def _create_actions(self):
        """
        :return: All actions for the current problem.
        """
        pickups = [PickUp(trq, self._current_problem.pick_duration) for trq in self._current_problem.transport_requests]
        deliveries = [Delivery(trq, self._current_problem.delivery_duration) for trq in
                      self._current_problem.transport_requests]
        start_actions = [SetupAction(v.current_node, timedelta(seconds=0)) for v in self.vehicles]
        visit_actions = [VisitAction(v.current_node, timedelta(seconds=0)) for v in self.vehicles]

        return [*pickups, *deliveries, *start_actions, *visit_actions]

    @overrides
    def _add_actions(self) -> None:
        """
        Adds pickups & deliveries to the model.
        """
        time_dimension = self._routing_model.GetDimensionOrDie(TIME_DIM)

        for idx, trq in enumerate(self._current_problem.transport_requests):
            pickup_index = self.action_index_to_solver_index(idx)
            delivery_index = self.action_index_to_solver_index(idx + len(self._current_problem.transport_requests))
            self._routing_model.AddPickupAndDelivery(pickup_index, delivery_index)
            self._routing_model.solver().Add(
                self._routing_model.VehicleVar(pickup_index) == self._routing_model.VehicleVar(delivery_index)
            )
            self._routing_model.solver().Add(
                time_dimension.CumulVar(pickup_index) <=
                time_dimension.CumulVar(delivery_index))

    @overrides
    def _capacity_callback(self, node_index: int) -> int:
        """
        Callback for the capacity dimension. Called by the solver.
        :param node_index: Node under consideration.
        """
        action = self.solver_index_to_action(node_index)
        capacity = 0
        if isinstance(action, PickUp):
            capacity = action.trq.quantity
        if isinstance(action, Delivery):
            capacity = -action.trq.quantity
        return capacity


class SolverCVRPU(SolverCVRP):

    def __init__(self, node_distance: INodeDistance, graph, solver_params: SolverParams = SolverParams()):
        super(SolverCVRPU, self).__init__(node_distance, graph, open_vrp=False, solver_params=solver_params)

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
    def _create_actions(self):
        setup_duration = timedelta(seconds=0)
        delivery_actions = []

        for trq in self.current_problem.transport_requests:
            delivery_locations = self._get_possible_state_nodes(trq.to_node)

            for j, d_loc in enumerate(delivery_locations):
                delivery_actions.append(
                    DeliveryUTurnState(trq=trq, idx=0, node=d_loc, duration=self._current_problem.delivery_duration))

        start_actions = [SetupAction(start_node=self._get_start_node_for_vehicle(v), duration=setup_duration)
                         for v in self._vehicles]
        end_actions = [VisitAction(s.node, timedelta(seconds=0)) for s in start_actions]

        return [*delivery_actions, *start_actions, *end_actions]

    @overrides
    def _add_actions(self):
        for trq in self._current_problem.transport_requests:
            added_indices = []

            for j, d in enumerate(self._actions):
                if isinstance(d, Delivery) and d.trqID == trq.uid:
                    d_idx = self.action_index_to_solver_index(j)
                    added_indices.append(d_idx)
            penalty = int(MAX_TOUR_DISTANCE) * len(self.vehicles) * DISTANCE_WEIGHT

            self._routing_model.AddDisjunction(added_indices, penalty, 1)

    def _get_start_node_for_vehicle(self, vehicle: Vehicle):
        if vehicle.node_arriving_from:
            for uid, node in self._graph.nodes.items():
                if node.data.current_node == vehicle.current_node \
                        and node.data.prev_node == vehicle.node_arriving_from:
                    return uid

        else:
            # free choice
            return self._get_possible_state_nodes(vehicle.current_node)[0]


class SolverVRPDPU(SolverVRPDP):

    def __init__(self, node_distance: INodeDistance, graph, solver_params: SolverParams = SolverParams()):
        super(SolverVRPDPU, self).__init__(node_distance, graph, open_vrp=False, solver_params=solver_params)

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
    def _create_actions(self):
        pick_duration = timedelta(seconds=self._current_problem.pick_duration.total_seconds())
        delivery_duration = timedelta(seconds=self._current_problem.delivery_duration.total_seconds())
        setup_duration = timedelta(seconds=0)
        pickup_actions = []
        delivery_actions = []

        for trq in self.current_problem.transport_requests:
            pickup_locations = self._get_possible_state_nodes(trq.from_node)
            delivery_locations = self._get_possible_state_nodes(trq.to_node)

            for i, p_loc in enumerate(pickup_locations):
                for j, d_loc in enumerate(delivery_locations):
                    idx = f"{i}{j}"
                    pickup_actions.append(PickUpUTurnState(trq=trq, idx=idx, node=p_loc, duration=pick_duration))
                    delivery_actions.append(
                        DeliveryUTurnState(trq=trq, idx=idx, node=d_loc, duration=delivery_duration))

        start_actions = [SetupAction(start_node=self._get_start_node_for_vehicle(v), duration=setup_duration)
                         for v in self._vehicles]
        end_actions = [VisitAction(s.node, timedelta(seconds=0)) for s in start_actions]

        return [*pickup_actions, *delivery_actions, *start_actions, *end_actions]

    @overrides
    def _add_actions(self):
        time_dimension = self._routing_model.GetDimensionOrDie(TIME_DIM)

        for trq in self._current_problem.transport_requests:
            added_indices = []

            for i, p in enumerate(self._actions):
                if not isinstance(p, PickUp) or not p.trqID == trq.uid:
                    continue

                for j, d in enumerate(self._actions):
                    if isinstance(d, Delivery) and d.trqID == trq.uid and d.idx == p.idx:
                        p_idx = self.action_index_to_solver_index(i)
                        d_idx = self.action_index_to_solver_index(j)
                        added_indices.append(p_idx)
                        added_indices.append(d_idx)
                        self._routing_model.AddPickupAndDelivery(p_idx, d_idx)
                        self._routing_model.solver().Add(
                            self._routing_model.VehicleVar(p_idx) == self._routing_model.VehicleVar(d_idx)
                        )
                        self._routing_model.solver().Add(
                            time_dimension.CumulVar(p_idx) <=
                            time_dimension.CumulVar(d_idx))
            penalty = int(MAX_TOUR_DISTANCE) * len(self.vehicles) * DISTANCE_WEIGHT

            self._routing_model.AddDisjunction(added_indices, penalty, 2)

    def _get_start_node_for_vehicle(self, vehicle: Vehicle):
        if vehicle.node_arriving_from:
            for uid, node in self._graph.nodes.items():
                if node.data.current_node == vehicle.current_node \
                        and node.data.prev_node == vehicle.node_arriving_from:
                    return uid

        else:
            # free choice
            return self._get_possible_state_nodes(vehicle.current_node)[0]
