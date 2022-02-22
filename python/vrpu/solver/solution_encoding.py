from datetime import timedelta
from typing import Dict, List, Tuple

from vrpu.core import NodeAction, Vehicle, Tour, SetupAction, VisitAction, DriveAction, Solution, PickUp, Delivery
from vrpu.core.graph.search.node_distance import INodeDistance
from vrpu.core.route_planning.clonable import Clonable


class EncodedAction(Clonable):
    """
    An EncodedAction belongs to a tour action. It specifies the vehicle it was assigned to
    and its value determines its position in the tour.
    """

    def __init__(self, vehicle_index: int, value: int):
        self.vehicle_index = vehicle_index
        self.value = value

    def clone(self):
        return EncodedAction(self.vehicle_index, self.value)

    def __repr__(self):
        return f"({self.vehicle_index}, {self.value})"


class EncodedSolution(list, Solution):
    """
    An EncodedSolution encodes a Solution for faster processing.
    The representation as a tour is calculated when requested.
    """

    def __init__(self, iterable: [], actions: [NodeAction], vehicles: [Vehicle], start_nodes: [str],
                 node_distance_func: INodeDistance):
        """
        :param iterable: List of EncodedActions.
        :param actions: The actual actions that this tour encodes.
        :param vehicles: The available vehicles.
        :param start_nodes: The start nodes of those vehicles.
        :param node_distance_func: An INodeDistances that returns shortest distances between nodes.
        """
        super().__init__(iterable)
        self.actions = actions
        self.vehicles = vehicles
        self.start_nodes = start_nodes
        self.node_distance_func = node_distance_func

    @property
    def tours(self) -> List[Tour]:
        return self.to_solution().tours

    def clone(self):
        return EncodedSolution([e.clone() for e in self], self.actions, self.vehicles, self.start_nodes,
                               self.node_distance_func)

    def to_actions(self) -> Dict[int, List[NodeAction]]:
        """
        :return: Representation as a list of ordered actions for each vehicle.
        """
        actions_per_vehicle = dict()

        [actions_per_vehicle.setdefault(i, []) for i in range(len(self.vehicles))]

        for action_index, encoded_action in enumerate(self):
            actions_per_vehicle[encoded_action.vehicle_index].append((self.actions[action_index], encoded_action.value))

        # sort actions for each vehicle according to their value
        for vehicle_index, action_list in actions_per_vehicle.items():
            actions_per_vehicle[vehicle_index] = [a[0] for a in sorted(action_list, key=lambda x: x[1])]

        return actions_per_vehicle

    def to_tour(self, vehicle_index: int, start_node: str, node_distance_func: INodeDistance) -> Tour:
        """
        Creates a tour object for the specified vehicle.
        :param vehicle_index: The index of the vehicle to create a tour. The index refers to this objects vehicle list.
        :param start_node: The start node of the vehicle.
        :param node_distance_func: The function used for getting shortest distances between nodes.
        :return: A tour assigned to the vehicle.
        """
        setup_time = 0
        tour_actions = []
        offset = timedelta(seconds=setup_time)
        vehicle = self.vehicles[vehicle_index]
        actions = self.to_actions()[vehicle_index]
        node_actions = [
            SetupAction(start_node=start_node, duration=timedelta(seconds=setup_time)),
            *actions,
            [VisitAction(node=start_node, duration=timedelta(seconds=setup_time))]]

        from_action = node_actions[0]

        for i in range(1, len(node_actions)):
            to_nodes = [a.node for a in node_actions[i]] if isinstance(node_actions[i], list) else [
                node_actions[i].node]
            distance, to_node_index = self._get_shortest_distance(from_action.node, to_nodes, node_distance_func)
            to_node = to_nodes[to_node_index]
            duration = timedelta(seconds=int(distance / vehicle.velocity / 3.6))
            from_action._offset = offset
            offset += from_action.duration
            prev_node = ''
            if '->' in to_node:
                prev_node = to_node.split('->')[0]
            tour_actions.append(from_action)
            tour_actions.append((DriveAction(from_node=from_action.node,
                                             to_node=to_node,
                                             prior_to_end_node=prev_node,
                                             distance=distance,
                                             duration=duration,
                                             offset=offset)))
            from_action = node_actions[i][to_node_index] if isinstance(node_actions[i], list) else node_actions[i]

            offset += duration

        end_time = vehicle.available_from + tour_actions[-1].start_offset + tour_actions[-1].duration

        return Tour(uid=vehicle.uid, actions=tour_actions, assigned_vehicle=vehicle, start_time=vehicle.available_from,
                    end_time=end_time)

    def _get_shortest_distance(self, from_node: str, to_nodes: [str], node_distance_func: INodeDistance) \
            -> Tuple[float, int]:
        shortest_index = 0
        shortest_distance = None

        for i, to_node in enumerate(to_nodes):
            distance = node_distance_func.get_distance(from_node, to_node)
            if shortest_distance is None or distance < shortest_distance:
                shortest_index = i
                shortest_distance = distance
        return shortest_distance, shortest_index

    def to_solution(self, start_nodes: [str] = None, node_distance_func: INodeDistance = None) -> Solution:
        """
        :return: Decodes this object into a solution object.
        """

        if start_nodes is None:
            start_nodes = self.start_nodes
        if node_distance_func is None:
            node_distance_func = self.node_distance_func

        tours = []
        actions_per_vehicle = self.to_actions()

        for vehicle_index, action_list in actions_per_vehicle.items():
            # don't create tours for vehicles without assigned transport requests
            if len(action_list) == 0:
                continue
            tour = self.to_tour(vehicle_index, start_nodes[vehicle_index], node_distance_func)
            tours.append(tour)
        return Solution(tours)

    def get_max_loads(self) -> Dict[int, int]:
        """
        :return: Maximum load for each vehicle in its tour.
        """
        actions_per_vehicle = self.to_actions()
        result = {}
        for vehicle, actions in actions_per_vehicle.items():
            result[vehicle] = self._get_max_load(actions)
        return result

    @staticmethod
    def _get_max_load(actions: List[NodeAction]) -> int:
        """
        Maximum load for a vehicle in its tour.
        :param actions:
        :return:
        """
        load = 0
        max_load = 0

        for action in actions:
            if isinstance(action, PickUp):
                load += 1
                if load > max_load:
                    max_load = load
            elif isinstance(action, Delivery):
                load -= 1

        # only delivery actions
        if load < 0:
            max_load = -1 * load

        return max_load
