import random
from datetime import timedelta
from operator import itemgetter
from typing import List, Tuple, Dict
from overrides import overrides
from deap import creator, base, tools
from timeit import default_timer as timer

from vrpu.core import VRPProblem, Solution, Vehicle, PickUp, Delivery, Tour, \
    SetupAction, DriveAction, VisitAction, NodeAction, DeliveryUTurnState, \
    PickUpUTurnState
from vrpu.core.graph.search.node_distance import CachedNodeDistance
from vrpu.solver.solver import ISolver, SolvingSnapshot

FITNESS_FCT_NAME = 'FitnessMax'
INDIVIDUAL_NAME = 'Individual'
POPULATION_NAME = 'Population'
EVALUATION_FCT_NAME = 'Evaluation'
SELECTION_FCT_NAME = 'Selection'
RECOMBINE_FCT_NAME = 'Recombine'
MUTATE_FCT_NAME = 'Mutate'


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
        self._history: [SolvingSnapshot] = []
        self._best_individual = None

    def solve(self, problem: VRPProblem) -> Solution:
        self._current_problem = problem

        self._actions = self._create_actions()

        self._setup_node_distance_function()

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

        # Remember best individual
        best_ind = tools.selBest(population, 1)[0]
        self._best_individual = best_ind
        values = [ind.fitness.values[1] for ind in population]
        start_timer = timer()
        self._history.append(
            SolvingSnapshot(runtime=timedelta(seconds=timer() - start_timer),
                            step=0,
                            best_value=best_ind.fitness.values[1],
                            average=sum(values) / len(values),
                            min_value=min(values),
                            max_value=max(values)))

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

            # The population is entirely replaced by the offspring
            population[:] = random.sample(children, len(children))

            # Output currently best individual
            best_ind = tools.selBest(population, 1)[0]
            if best_ind.fitness.values[0] > self._best_individual.fitness.values[0]:
                self._best_individual = best_ind
            print(f"\r   Generation {generation + 1}/{self.generations}"
                  f"   Best Fitness: {best_ind.fitness.values}", end='')

            # Keep track of stats
            values = [ind.fitness.values[1] for ind in population]
            self._history.append(
                SolvingSnapshot(runtime=timedelta(seconds=timer() - start_timer),
                                step=generation + 1,
                                best_value=best_ind.fitness.values[1],
                                average=sum(values) / len(values),
                                min_value=min(values),
                                max_value=max(values)))

        print(f"\n-- End of (successful) evolution after {timer() - start_timer}s --")

        return self._individual_to_solution(self._best_individual)

    @property
    def history(self) -> [SolvingSnapshot]:
        return self._history

    def _setup_node_distance_function(self):
        """
        Sets up the node distance function to pre-calculate needed distances.
        """
        nodes = [a.node for a in self._actions]
        nodes.extend([v.current_node for v in self._current_problem.vehicles])
        nodes = set(nodes)
        self._node_distance.calculate_distances(self._graph, nodes)

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

        for i in range(len(self._actions)):
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

    def _recombine(self, child1: [], child2: []):
        """
        :param child1: First child.
        :param child2: Second child.
        :return: Recombination of both children (in place).
        """
        self._two_point_crossover(child1, child2)
        return child1, child2

    def _uniform_crossover(self, ind1: [], ind2: []) -> bool:

        performed_crossover = False

        for i in range(len(ind1)):
            if random.random() >= 0.5:
                self._swap_values(ind1, ind2, i, i)
                performed_crossover = True

        return performed_crossover

    def _one_point_crossover(self, ind1: [], ind2: []) -> bool:
        from_index = random.randint(0, len(ind1) - 1)
        return self._swap_values(ind1, ind2, from_index, len(ind1) - 1)

    def _two_point_crossover(self, ind1: [], ind2: []) -> bool:
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

    def _mutate(self, individual: []):
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

    def _swap_values(self, ind1: [], ind2: [], from_index: int, to_index: int) -> bool:
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

    def _individual_to_actions(self, individual: []) -> Dict[int, List[NodeAction]]:
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

    def _individual_to_solution(self, individual: []) -> Solution:
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

    def _actions_to_tour(self, actions: [NodeAction], vehicle: Vehicle) -> Tour:
        """
        :param actions: List of actions assigned to the vehicle.
        :param vehicle: The vehicle for the tour.
        :return: A tour containing all actions.
        """
        setup_time = 0
        tour_actions = []
        offset = timedelta(seconds=setup_time)
        node_actions = [
            SetupAction(start_node=self._get_start_node_for_vehicle(vehicle), duration=timedelta(seconds=setup_time)),
            *actions,
            VisitAction(node=self._get_start_node_for_vehicle(vehicle), duration=timedelta(seconds=setup_time))]

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

    def _get_start_node_for_vehicle(self, vehicle: Vehicle):
        return vehicle.current_node


class GASolverVRPDP(GASolverCVRP):
    @overrides
    def _create_actions(self):
        """
        :return: All actions for the current problem.
        """
        actions = []

        for trq in self._current_problem.transport_requests:
            actions.append(PickUp(trq, self._current_problem.pick_duration))
            actions.append(Delivery(trq, self._current_problem.delivery_duration))

        return actions

    @overrides
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

    def _get_max_load(self, individual):
        loads = {}
        [loads.setdefault(v, 0) for v in range(len(self._current_problem.vehicles))]
        loads_max = {}
        [loads_max.setdefault(v, 0) for v in range(len(self._current_problem.vehicles))]

        actions = self._individual_to_actions(individual)

        for vehicle_idx, action_list in actions.items():
            for action in action_list:
                if isinstance(action, PickUp):
                    loads[vehicle_idx] += 1
                    loads_max[vehicle_idx] = max(loads[vehicle_idx], loads_max[vehicle_idx])
                if isinstance(action, Delivery):
                    loads[vehicle_idx] -= 1

        return loads_max

    def _mutate(self, individual: List[Tuple[int, int]]):
        """
        :param individual: The individual to mutate.
        :return: Mutates the individual by assigned a random action to another vehicle.
        """

        loads = self._get_max_load(individual)

        # choose a random transport request that will be assigned to another vehicle
        # first choose a pickup action for that trq
        i = random.choice([n for n in range(len(individual)) if n % 2 == 0])

        vehicle_idx = individual[i][0]
        new_vehicle = self._choose_vehicle(vehicle_idx, loads)

        # there might no vehicle available because they all have reached max capacity
        if new_vehicle >= 0:
            individual[i] = (new_vehicle, individual[i][1])
            individual[i + 1] = (new_vehicle, individual[i + 1][1])

            # check feasibility
            loads = self._get_max_load(individual)
            if loads[new_vehicle] > self._current_problem.vehicles[new_vehicle].max_capacity:
                # revert
                individual[i] = vehicle_idx, individual[i][1]
                individual[i + 1] = (new_vehicle, individual[i + 1][1])

        return individual

    def _two_point_crossover(self, ind1: [], ind2: []) -> bool:
        """
        :param ind1: First individual.
        :param ind2: Second individual.
        :return: Whether crossover was performed.
        """
        from_index = random.choice([n for n in range(len(ind1)) if n % 2 == 0])
        to_index = random.choice([n for n in range(len(ind1)) if n % 2 == 1])

        if from_index > to_index:
            return False

        self._swap_values(ind1, ind2, from_index, to_index)

        # check feasibility
        max_load_ind1 = self._get_max_load(ind1)
        for vehicle_index, max_load in max_load_ind1.items():
            if max_load > self._current_problem.vehicles[vehicle_index].max_capacity:
                self._swap_values(ind1, ind2, from_index, to_index)
                return False

        max_load_ind2 = self._get_max_load(ind2)
        for vehicle_index, max_load in max_load_ind2.items():
            if max_load > self._current_problem.vehicles[vehicle_index].max_capacity:
                self._swap_values(ind1, ind2, from_index, to_index)
                return False

        return True


class GASolverCVRPU(GASolverCVRP):

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

    def _init_individual(self) -> [[int, int, [int]]]:
        """
        :return: An initial individual consisting of a list of tuples indicating for every action to which
        vehicle it belongs and its value.
        """
        keys = []
        vehicle_index = 0
        load = 0

        for i in range(len(self._actions)):
            vehicle = self._current_problem.vehicles[vehicle_index]
            # calculate a value for the order of the delivery
            value = self._generate_value(existing_values=[key[1] for key in keys], min_val=1, max_val=999)

            # calculate a value for every possible Delivery-Spot
            inner_keys = []
            for _ in range(len(self._actions[i])):
                inner_keys.append(self._generate_value(existing_values=inner_keys, min_val=1, max_val=999))

            keys.append((vehicle_index, value, inner_keys))

            # keep track of vehicle capacity
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

    def _individual_to_actions(self, individual: []) -> Dict[int, List[NodeAction]]:
        """
        Transforms an individual to a dict containing the ordered list of actions assigned to each vehicle.
        :param individual: The individual to transform.
        :return: A dict containing the ordered list of actions assigned to each vehicle.
        """
        actions = dict()
        [actions.setdefault(i, []) for i in range(len(self._current_problem.vehicles))]

        for action_index, (vehicle_index, value, action_values) in enumerate(individual):
            index, _ = max(enumerate(action_values), key=itemgetter(1))
            actions[vehicle_index].append((self._actions[action_index][index], value))

        # sort actions for each vehicle according to their value
        for vehicle_index, action_list in actions.items():
            actions[vehicle_index] = [a[0] for a in sorted(action_list, key=lambda x: x[1])]

        return actions

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

    def _two_point_crossover(self, ind1: [], ind2: []) -> bool:
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

    def _mutate(self, individual: []):
        """
        :param individual: The individual to mutate.
        :return: Mutates the individual by assigned a random action to another vehicle.
        """
        loads = {}
        [loads.setdefault(v, 0) for v in range(len(self._current_problem.vehicles))]

        for vehicle_idx, value, _ in individual:
            loads[vehicle_idx] += 1

        # choose a random transport request that will be assigned to another vehicle
        i = random.choice(range(len(individual)))

        vehicle_idx = individual[i][0]
        new_vehicle = self._choose_vehicle(vehicle_idx, loads)

        # there might no vehicle available because they all have reached max capacity
        if new_vehicle >= 0:
            individual[i] = new_vehicle, individual[i][1], random.sample(individual[i][2], len(individual[i][2]))
            loads[vehicle_idx] -= 1
            loads[new_vehicle] += 1
        else:
            # no vehicle available -> do some shuffling
            individual[i] = vehicle_idx, individual[i][1], random.sample(individual[i][2], len(individual[i][2]))

        return individual


class GASolverVRPDPU(GASolverCVRPU):

    def _create_actions(self) -> [[]]:
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

    def _init_individual(self) -> [[int, int, [int]]]:
        """
        :return: An initial individual consisting of a list of tuples indicating for every action to which
        vehicle it belongs and its value.
        """
        keys = []
        vehicle_index = 0

        for i in range(len(self._current_problem.transport_requests)):
            vehicle = self._current_problem.vehicles[vehicle_index]
            # calculate a value for the order of the delivery
            pick_value = self._generate_value(existing_values=[key[1] for key in keys], min_val=1, max_val=800)

            # calculate a value for every possible Pickup-Spot
            inner_keys = []
            for _ in range(len(self._actions[i * 2])):
                inner_keys.append(self._generate_value(existing_values=inner_keys, min_val=1, max_val=999))

            keys.append((vehicle_index, pick_value, inner_keys))

            delivery_value = self._generate_value([key[1] for key in keys], pick_value + 1, 999)
            # calculate a value for every possible Delivery-Spot
            inner_keys = []
            for _ in range(len(self._actions[(i * 2) + 1])):
                inner_keys.append(self._generate_value(existing_values=inner_keys, min_val=1, max_val=999))

            keys.append((vehicle_index, delivery_value, inner_keys))

            # keep track of vehicle capacity
            max_loads = self._get_max_load(keys)
            if max_loads[vehicle_index] == vehicle.max_capacity:
                vehicle_index += 1

        return keys

    def _get_max_load(self, individual):
        loads = {}
        [loads.setdefault(v, 0) for v in range(len(self._current_problem.vehicles))]
        loads_max = {}
        [loads_max.setdefault(v, 0) for v in range(len(self._current_problem.vehicles))]

        actions = self._individual_to_actions(individual)

        for vehicle_idx, action_list in actions.items():
            for action in action_list:
                if isinstance(action, PickUp):
                    loads[vehicle_idx] += 1
                    loads_max[vehicle_idx] = max(loads[vehicle_idx], loads_max[vehicle_idx])
                if isinstance(action, Delivery):
                    loads[vehicle_idx] -= 1

        return loads_max

    def _two_point_crossover(self, ind1: [], ind2: []) -> bool:
        """
        :param ind1: First individual.
        :param ind2: Second individual.
        :return: Whether crossover was performed.
        """
        from_index = random.choice([n for n in range(len(ind1)) if n % 2 == 0])
        to_index = random.choice([n for n in range(len(ind1)) if n % 2 == 1])

        if from_index > to_index:
            return False

        self._swap_values(ind1, ind2, from_index, to_index)

        # check feasibility
        max_load_ind1 = self._get_max_load(ind1)
        for vehicle_index, max_load in max_load_ind1.items():
            if max_load > self._current_problem.vehicles[vehicle_index].max_capacity:
                self._swap_values(ind1, ind2, from_index, to_index)
                return False

        max_load_ind2 = self._get_max_load(ind2)
        for vehicle_index, max_load in max_load_ind2.items():
            if max_load > self._current_problem.vehicles[vehicle_index].max_capacity:
                self._swap_values(ind1, ind2, from_index, to_index)
                return False

        return True

    def _mutate(self, individual: List[Tuple[int, int]]):
        """
        :param individual: The individual to mutate.
        :return: Mutates the individual by assigned a random action to another vehicle.
        """

        loads = self._get_max_load(individual)

        # choose a random transport request that will be assigned to another vehicle
        # first choose a pickup action for that trq
        i = random.choice([n for n in range(len(individual)) if n % 2 == 0])

        vehicle_idx = individual[i][0]
        new_vehicle = self._choose_vehicle(vehicle_idx, loads)

        # there might no vehicle available because they all have reached max capacity
        if new_vehicle >= 0:
            individual[i] = (new_vehicle, individual[i][1], individual[i][2])
            individual[i + 1] = (new_vehicle, individual[i + 1][1], individual[i + 1][2])

            # check feasibility
            loads = self._get_max_load(individual)
            if loads[new_vehicle] > self._current_problem.vehicles[new_vehicle].max_capacity:
                # revert
                individual[i] = vehicle_idx, individual[i][1], individual[i][2]
                individual[i + 1] = (new_vehicle, individual[i + 1][1]), individual[i + 1][2]

        return individual
