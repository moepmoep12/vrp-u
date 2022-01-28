from typing import List, Dict
from datetime import datetime, timedelta
from overrides import overrides

from .transport_request import TransportRequest
from .vehicle import Vehicle
from .serializable import Serializable


class VRPProblem(Serializable):
    """
    A VRPProblem consists of a set of transport requests and a set of available vehicles.
    """

    def __init__(self, transport_requests: List[TransportRequest],
                 vehicles: List[Vehicle],
                 calculation_time: datetime,
                 pick_duration: timedelta,
                 delivery_duration: timedelta,
                 vehicle_velocity: float):
        """
        :param transport_requests: All transport requests that need to be assigned to tours.
        :param vehicles: All available vehicles.
        :param calculation_time: The time at which solving shall occur.
        :param pick_duration: Execution duration for pickup action of a transport request.
        :param delivery_duration: Execution duration for delivery action of a transport request.
        :param vehicle_velocity: The velocity of the vehicles in km/h.
        """

        self._transport_requests = transport_requests
        self._vehicles = vehicles
        self._calculation_time = calculation_time
        self._pick_duration = pick_duration
        self._delivery_duration = delivery_duration
        self._vehicle_velocity = vehicle_velocity

    @property
    def transport_requests(self) -> List[TransportRequest]:
        """
        :return: All transport requests that need to be assigned to tours.
        """
        return self._transport_requests

    @property
    def vehicles(self) -> List[Vehicle]:
        """
        :return: All available vehicles.
        """
        return self._vehicles

    @property
    def calculation_time(self) -> datetime:
        """
        :return: The time at which solving shall occur.
        """
        return self._calculation_time

    @property
    def pick_duration(self) -> timedelta:
        """
        :return: Execution duration for pickup action of a transport request.
        """
        return self._pick_duration

    @property
    def delivery_duration(self) -> timedelta:
        """
        :return: Execution duration for delivery action of a transport request.
        """
        return self._delivery_duration

    @property
    def vehicle_velocity(self) -> float:
        """
        :return: The velocity of the vehicles in km/h.
        """
        return self._vehicle_velocity

    def __repr__(self) -> str:
        txt = f"Time: {self.calculation_time}\n"

        txt += 'Vehicles:\n'
        for v in self.vehicles:
            txt += f" {v}\n"

        txt += 'Transport Requests:\n'
        for trq in self.transport_requests:
            txt += f" {trq}\n"

        return txt

    @overrides
    def serialize(self) -> Dict[str, object]:
        return dict(
            transport_requests=self.transport_requests,
            vehicles=self.vehicles,
            calculation_time=str(self.calculation_time),
            pick_duration=str(self.pick_duration),
            delivery_duration=str(self.delivery_duration),
            vehicle_velocity=self.vehicle_velocity
        )
