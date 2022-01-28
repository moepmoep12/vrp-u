import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from overrides import overrides
from deap import creator, base, tools

from vrpu.core import VRPProblem, Solution, TransportRequest, Vehicle, Graph, PickUp, Delivery, Tour, Action, \
    SetupAction, DriveAction, VisitAction, NodeAction
from vrpu.core.graph.search.node_distance import NodeDistanceAStar, CachedNodeDistance
from vrpu.core.util.solution_printer import DataFramePrinter
from vrpu.core.util.visualization import show_solution
from vrpu.solver.solver import ISolver

FITNESS_FCT_NAME = 'FitnessMax'
INDIVIDUAL_NAME = 'Individual'
POPULATION_NAME = 'Population'
EVALUATION_FCT_NAME = 'Evaluation'
SELECTION_FCT_NAME = 'Selection'
RECOMBINE_FCT_NAME = 'Recombine'
MUTATE_FCT_NAME = 'Mutate'


@dataclass
class InsertionPosition:
    pickup_pos: int
    delivery_pos: int

    def insert(self, actions: List[Action], pickup: PickUp, delivery: Delivery):
        actions.insert(self.delivery_pos, delivery)
        actions.insert(self.pickup_pos, pickup)


class IInsertionPositions(ABC):

    @abstractmethod
    def calc_possible_insertion_positions(self, trq: TransportRequest, actions: List[Action]) \
            -> List[InsertionPosition]:
        pass


class InsertionPositionsAll(IInsertionPositions):

    @overrides
    def calc_possible_insertion_positions(self, trq: TransportRequest, actions: List[Action]) \
            -> List[InsertionPosition]:
        results = []
        range_pick_up = (0, len(actions))
        range_delivery = (0, len(actions))

        for p in range(range_pick_up[0], range_pick_up[1] + 1):
            for d in range(max(p, range_delivery[0]), range_delivery[1] + 1):
                results.append(InsertionPosition(p, d))

        return results


class GASolverCVRP(ISolver):

    def __init__(self, node_distance: CachedNodeDistance, graph, population_size: int = 10,
                 generations: int = 10, crossover_prob: float = 0.7, mutate_prob: float = 0.03):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutate_prob = mutate_prob
        self._node_distance: CachedNodeDistance = node_distance
        self._current_problem: VRPProblem = None
        self._graph = graph
        self._actions = []

    def solve(self, problem: VRPProblem) -> Solution:
        self._current_problem = problem

        self._actions = self._create_actions()

        # Setup node distance function
        nodes = [a.node for a in self._actions]
        nodes.extend([v.current_node for v in self._current_problem.vehicles])
        nodes = set(nodes)
        self._node_distance.calculate_distances(self._graph, nodes)

        # create classes in DEAP
        creator.create(FITNESS_FCT_NAME, base.Fitness, weights=(1.0, -1.0))
        creator.create(INDIVIDUAL_NAME, list, fitness=getattr(creator, FITNESS_FCT_NAME))

        toolbox = base.Toolbox()

        # init individual
        toolbox.register('init', self._init_individual)
        toolbox.register(INDIVIDUAL_NAME, tools.initIterate, getattr(creator, INDIVIDUAL_NAME),
                         getattr(toolbox, 'init'))
        toolbox.register(POPULATION_NAME, tools.initRepeat, list, getattr(toolbox, INDIVIDUAL_NAME),
                         n=self.population_size)

        # evaluation
        toolbox.register(EVALUATION_FCT_NAME, self._evaluate)

        # ga methods
        toolbox.register(SELECTION_FCT_NAME, tools.selRoulette)
        toolbox.register(RECOMBINE_FCT_NAME, self._recombine)
        toolbox.register(MUTATE_FCT_NAME, self._mutate)

        # # initialize population
        population = getattr(toolbox, POPULATION_NAME)(n=self.population_size)

        print('Start of evolution')
        # Evaluate the entire population
        fitnesses = list(map(getattr(toolbox, EVALUATION_FCT_NAME), population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        print(f"   Evaluated {len(population)} individuals")

        # start GA algorithm
        for generation in range(self.generations):

            # choose parents for new generation
            parents = getattr(toolbox, SELECTION_FCT_NAME)(population, len(population))

            # clone
            children = list(map(toolbox.clone, parents))

            # recombination
            for child1, child2 in zip(children[::2], children[1::2]):
                if random.random() < self.crossover_prob:
                    getattr(toolbox, RECOMBINE_FCT_NAME)(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # mutation
            for mutant in children:
                if random.random() < self.mutate_prob:
                    getattr(toolbox, MUTATE_FCT_NAME)(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in children if not ind.fitness.valid]
            fitnesses = map(getattr(toolbox, EVALUATION_FCT_NAME), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # # The population is entirely replaced by the offspring
            population[:] = children

            best_ind = tools.selBest(population, 1)[0]
            print(f"\r   Generation {generation + 1}/{self.generations}"
                  f"   Best Fitness: {best_ind.fitness.values}", end='')

        print('\n-- End of (successful) evolution --')

        best_ind = tools.selBest(population, 1)[0]
        return self._individual_to_solution(best_ind)

    def _create_actions(self):
        """
        :return: All actions for the current problem.
        """
        delivery_duration = timedelta(seconds=self._current_problem.delivery_duration.total_seconds())
        deliveries = [Delivery(trq, delivery_duration) for trq in self._current_problem.transport_requests]
        return [*deliveries]

    def _init_individual(self):
        """
        :return: An initial individual consisting of a list of tuples indicating for every action to which
        vehicle it belongs and its value.
        """
        keys = []
        vehicle_index = 0
        load = 0
        # trqs = random.sample(self._current_problem.transport_requests, len(self._current_problem.transport_requests))

        for i, delivery in enumerate(self._actions):
            vehicle = self._current_problem.vehicles[vehicle_index]
            value = self._generate_value(existing_values=[key[1] for key in keys], min_val=1, max_val=999)
            keys.append((vehicle_index, value))
            load += 1
            if load >= vehicle.max_capacity:
                load = 0
                vehicle_index += 1

        return keys

    def _generate_value(self, existing_values: [int], min_val: int, max_val: int) -> int:
        """
        :param existing_values: Unwanted values.
        :param min_val: Minimum value.
        :param max_val: Maximum value.
        :return: Returns a random value between min_val and max_val excluding elements in existing_values.
        """
        value = int(random.uniform(min_val, max_val))
        if value in existing_values:
            return self._generate_value(existing_values, min_val, max_val)
        return value

    def _evaluate(self, individual: List[Tuple[int, int]]):
        """
        :param individual: The individual to evaluate.
        :return: The fitness value of the individual.
        """
        solution = self._individual_to_solution(individual)
        total_distance = sum([t.get_distance() for t in solution.tours])
        return 1.0 / total_distance, total_distance

    def _recombine(self, child1: List[Tuple[int, int]], child2: List[Tuple[int, int]]):
        """
        :param child1: First child.
        :param child2: Second child.
        :return: Recombination of both children (in place).
        """
        self._two_point_crossover(child1, child2)
        return child1, child2

    def _two_point_crossover(self, ind1: List[Tuple[int, int]], ind2: List[Tuple[int, int]]):
        """
        :param ind1: First individual.
        :param ind2: Second individual.
        :return: Whether crossover was performed.
        """
        from_index = random.randint(0, len(ind1) - 1)
        to_index = random.randint(0, len(ind2) - 1)

        if from_index > to_index:
            from_index, to_index = to_index, from_index

        return self._swap_values(ind1, ind2, from_index, to_index)

    def _mutate(self, individual: List[Tuple[int, int]]):
        """
        :param individual: The individual to mutate.
        :return: Mutates the individual by assigned a random action to another vehicle.
        """
        loads = {}
        [loads.setdefault(v, 0) for v in range(len(self._current_problem.vehicles))]

        for vehicle_idx, value in individual:
            loads[vehicle_idx] += 1

        # choose a random transport request that will be assigned to another vehicle
        i = random.choice(range(len(individual)))

        vehicle_idx = individual[i][0]
        new_vehicle = self._choose_vehicle(vehicle_idx, loads)

        # there might no vehicle available because they all have reached max capacity
        if new_vehicle >= 0:
            individual[i] = new_vehicle, individual[i][1]
            loads[vehicle_idx] -= 1
            loads[new_vehicle] += 1

        return individual

    def _choose_vehicle(self, vehicle_index_origin, loads: Dict[int, int]) -> int:
        """
        :param vehicle_index_origin: Vehicle_index to ignore.
        :param loads: Dict for the load of every vehicle.
        :return: A random vehicle.
        """
        candidates = []
        for vehicle_index, load in loads.items():
            if vehicle_index != vehicle_index_origin \
                    and load < self._current_problem.vehicles[vehicle_index].max_capacity:
                candidates.append(vehicle_index)

        if len(candidates) > 0:
            return random.choice(candidates)
        else:
            return -1

    def _swap_values(self, ind1: List[Tuple[int, int]], ind2: List[Tuple[int, int]], from_index: int,
                     to_index: int) -> bool:
        """
        Swaps the values in the given range in place.
        :param ind1: First individual.
        :param ind2: Second individual.
        :param from_index: Swap start.
        :param to_index: Swap end.
        :return: Whether anything was swapped.
        """
        swapped = False

        for i in range(from_index, to_index + 1):
            ind1[i], ind2[i] = ind2[i], ind1[i]
            swapped = True

        return swapped

    def _individual_to_actions(self, individual: [Tuple[int, int]]) -> Dict[int, List[NodeAction]]:
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

    def _individual_to_solution(self, individual: List[Tuple[int, int]]) -> Solution:
        """
        Transforms an individual to a Solution.
        :param individual: The individual to transform.
        :return: A Solution.
        """
        tours = []
        actions = self._individual_to_actions(individual)

        for vehicle_index, action_list in actions.items():
            # don't create tours for vehicles without assigned transport requests
            if len(action_list) == 0:
                continue
            tour = self._actions_to_tour(action_list, self._current_problem.vehicles[vehicle_index])
            tours.append(tour)
        return Solution(tours)

    def _actions_to_tour(self, actions: List[NodeAction], vehicle: Vehicle) -> Tour:
        """
        :param actions: List of actions assigned to the vehicle.
        :param vehicle: The vehicle for the tour.
        :return: A tour containing all actions.
        """
        setup_time = 0
        end_time = vehicle.available_from + timedelta(seconds=setup_time)
        tour_actions = []
        offset = timedelta(seconds=setup_time)
        node_actions = [SetupAction(start_node=vehicle.current_node, duration=timedelta(seconds=setup_time)),
                        *actions,
                        VisitAction(node=vehicle.current_node, duration=timedelta(seconds=setup_time))]

        for i in range(1, len(node_actions)):
            from_node = node_actions[i - 1].node
            to_node = node_actions[i].node
            distance = self._node_distance.get_distance(from_node, to_node)
            duration = timedelta(seconds=int(distance / self._current_problem.vehicle_velocity / 3.6))
            node_actions[i - 1]._offset = offset
            offset += node_actions[i - 1].duration
            prev_node = ''
            if '->' in to_node:
                prev_node = to_node.split('->')[0]
            tour_actions.append(node_actions[i - 1])
            tour_actions.append((DriveAction(from_node=from_node,
                                             to_node=to_node,
                                             prior_to_end_node=prev_node,
                                             distance=distance,
                                             duration=duration,
                                             offset=offset)))

            offset += duration

        end_time = vehicle.available_from + tour_actions[-1].start_offset + tour_actions[-1].duration

        return Tour(uid=vehicle.uid, actions=tour_actions, assigned_vehicle=vehicle, start_time=vehicle.available_from,
                    end_time=end_time)


class GASolverVRPDP(GASolverCVRP):
    pass

    # @overrides
    # def _init_individual(self, ind, n):
    #     actions = []
    #     pick_duration = timedelta(seconds=self._current_problem.pick_duration.total_seconds())
    #     delivery_duration = timedelta(seconds=self._current_problem.delivery_duration.total_seconds())
    #
    #     trqs = random.sample(self._current_problem.transport_requests, len(self._current_problem.transport_requests))
    #
    #     for trq in trqs:
    #         insertion_positions = InsertionPositionsAll().calc_possible_insertion_positions(trq, actions)
    #         insertion_pos = random.choice(insertion_positions)
    #         insertion_pos.insert(actions, PickUp(trq, pick_duration), Delivery(trq, delivery_duration))
    #
    #     return actions

    @overrides
    def _init_individual(self):
        keys = []
        vehicle_index = 0
        load = 0
        trqs = random.sample(self._current_problem.transport_requests, len(self._current_problem.transport_requests))

        for i, trq in enumerate(trqs):
            vehicle = self._current_problem.vehicles[vehicle_index]
            value = self._generate_value([key[1] for key in keys], 1, 999)
            keys.append((vehicle_index, value))
            load += 1
            if load >= vehicle.max_capacity:
                load = 0
                vehicle_index += 1

        return keys


if __name__ == '__main__':
    vehicles = []
    calc_time = datetime.now()
    trqs = []

    with open('../../../../data/cvrp_1.json') as f:
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

    solver = GASolverCVRP(NodeDistanceAStar(), graph=graph, population_size=100, generations=1000, mutate_prob=0.03,
                          crossover_prob=0.7)
    solution = solver.solve(problem)

    printer = DataFramePrinter(only_node_actions=True)
    printer.print_solution(solution)

    show_solution(solution, graph, True)
