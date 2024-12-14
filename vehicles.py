from typing import Tuple, Union, List, Optional

import numpy as np

from highway_env.road.road import Road, Route, LaneIndex
from highway_env.utils import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env import utils
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle


# 其他的车辆 不准变道至合并车道
class OthersIDMVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 0.1  # [s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY
        self.color = VehicleGraphics.BLUE
        self.controlled_vehicles = None  # 控制车辆列表

    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1])

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.
        """
        if self.crashed:
            self.color = VehicleGraphics.RED
            self.crashed = False
            self.recover_from_stop(self.ACC_MAX)
        else:
            self.color = VehicleGraphics.BLUE

        action = {}
        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()

        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # Longitudinal: IDM
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.target_lane_index)
            target_idm_acceleration = self.acceleration(ego_vehicle=self,
                                                        front_vehicle=front_vehicle,
                                                        rear_vehicle=rear_vehicle)
            action['acceleration'] = min(action['acceleration'], target_idm_acceleration)
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        Vehicle.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.

    def step(self, dt: float):
        """
        Step the simulation.
        """
        self.timer += dt
        super().step(dt)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = abs(utils.not_zero(getattr(ego_vehicle, "target_speed", 0)))
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * \
                            np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.
        """
        # If a lane change is already ongoing
        if self.lane_index != self.target_lane_index:
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        if self.in_control_vehicles(self.lane_index):
            self.target_lane_index = ("a", "b", 1)
            self.color = VehicleGraphics.GREEN
            return
        else:
            self.color = VehicleGraphics.BLUE

        for lane_index in self.road.network.side_lanes(self.lane_index):
            if lane_index == ("b", "c", 3):
                continue
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            if np.abs(self.speed) < 1:
                continue

            if self.in_control_vehicles(lane_index):
                continue

            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    # 判断是否在控制车辆同车道范围内
    def in_control_vehicles(self, lane_index):
        # 假设控制车辆列表不为空
        control_lane_index = self.controlled_vehicles[0].lane_index
        is_same_lane = (control_lane_index[2] == lane_index[2])
        if not is_same_lane:
            return False
        control_max_x = max([v.position[0] for v in self.controlled_vehicles])
        control_min_x = self.controlled_vehicles[0].position[0]
        self_x = self.position[0]
        out_distance = 100
        if self_x > control_min_x - out_distance and self_x < control_max_x + out_distance:
            return True
        return False

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model
        """
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)

        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                    self.route[0][2] - self.target_lane_index[2]):
                return False
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        If stopped on the wrong lane, try a reversing maneuver.
        """
        stopped_speed = 5
        safe_distance = 200
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.network.get_lane(self.target_lane_index))
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                return -self.COMFORT_ACC_MAX / 2
        return acceleration


class IDMVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies (原CACCVehicle已重命名为IDMVehicle).

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    LENGTH = 10  # 车长10 [m]

    DISTANCE_WANTED = 10.0 + LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 0  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 idx=None,
                 distance_wanted=DISTANCE_WANTED,  # 与前车的期望距离
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.idx = idx
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY
        self.distance_wanted = distance_wanted
        self.acceleration_action = 0

    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1])

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def act(self, action: Union[dict, str] = None):
        if self.crashed:
            return
        action = {}
        self.follow_road()
        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # 根据和前车的距离与期望距离相比较 获得加速度
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)

        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        self.acceleration_action = action['acceleration']
        Vehicle.act(self, action)

    def step(self, dt: float):
        self.timer += dt
        super().step(dt)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = abs(utils.not_zero(getattr(ego_vehicle, "target_speed", 0)))
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            if d < self.DISTANCE_WANTED:
                dv = ego_vehicle.speed - front_vehicle.speed
                th = 6
                acceleration = 0
                if dv > 0:
                    acceleration = -self.ACC_MAX
            else:
                acceleration -= self.COMFORT_ACC_MAX * \
                                np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.distance_wanted
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star


class MergeVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.
    """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    COMFORT_ACC_MAX = 3.0  # [m/s2]
    COMFORT_ACC_MIN = -5.0  # [m/s2]

    LENGTH = 10  # [m]

    DISTANCE_WANTED = 10.0 + LENGTH  # [m]
    TIME_WANTED = 0  # [s]

    DELTA_MERGE = 4.0  # []
    DELTA = 4.0  # []
    DELTA_RANGE = [3.5, 4.5]

    POLITENESS = 0.
    LANE_CHANGE_MIN_ACC_GAIN = 0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0
    LANE_CHANGE_DELAY = 1.0

    def __init__(self,
                 road: Road,
                 position: Vector,
                 idm_vehicles: List[IDMVehicle] = [],
                 insert_idx: int = None,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY
        self.idm_vehicles = idm_vehicles
        self.insert_idx = insert_idx
        self.acceleration_action = 0

    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1])

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def act(self, action: Union[dict, str] = None):
        if self.crashed:
            return
        action = {}
        self.follow_road()
        self.merge_policy()

        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        if self.lane_index[2] != 2:  # 合并前
            action['acceleration'] = self.acceleration_merge_before(self)
        else:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
            action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                       front_vehicle=front_vehicle,
                                                       rear_vehicle=rear_vehicle)
            self.adjust_all_speed()  # 汇合后 重新调整每辆车的目标速度
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        self.acceleration_action = action['acceleration']
        Vehicle.act(self, action)

    def adjust_all_speed(self):
        speeds = [120] * len(self.idm_vehicles) + [100]
        idy = len(self.idm_vehicles) - self.insert_idx
        c = self.idm_vehicles[:idy] + [self] + self.idm_vehicles[idy:len(self.idm_vehicles)]
        for i, v in enumerate(c):
            v.target_speed = speeds[i]

    def get_insert_position_x_and_rear_vehicle(self):
        if self.insert_idx == len(self.idm_vehicles):
            rear_vehicle = self.idm_vehicles[0]
            x = rear_vehicle.position[0] - IDMVehicle.LENGTH
        else:
            rear_vehicle = self.idm_vehicles[len(self.idm_vehicles) - self.insert_idx - 1]
            x = rear_vehicle.position[0] + IDMVehicle.LENGTH * 2
        return x, rear_vehicle

    def merge_policy(self):
        if self.lane_index != self.target_lane_index:
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0
        target_lane_index = ("b", "c", 2)
        if self.lane_index == ("b", "c", 3):
            if np.abs(self.speed) < 1:
                return
            if self.insert_idx == 0:
                if self.idm_vehicles[-1].position[0] < self.position[0] - IDMVehicle.LENGTH:
                    self.target_lane_index = target_lane_index
                return
            if self.insert_idx == len(self.idm_vehicles):
                if self.idm_vehicles[0].position[0] - IDMVehicle.LENGTH > self.position[0]:
                    dv = abs(self.idm_vehicles[0].speed - self.speed)
                    if dv < 10:
                        self.target_lane_index = target_lane_index
                return
            front_vehicle = self.idm_vehicles[len(self.idm_vehicles) - self.insert_idx]
            rear_vehicle = self.idm_vehicles[len(self.idm_vehicles) - self.insert_idx - 1]
            if front_vehicle.position[0] - IDMVehicle.LENGTH > self.position[0] and rear_vehicle.position[0] < \
                    self.position[0] - IDMVehicle.LENGTH:
                dv1 = abs(rear_vehicle.speed - self.speed)
                dv2 = abs(front_vehicle.speed - self.speed)
                if dv1 < 10 and dv2 < 10:
                    self.target_lane_index = target_lane_index

    def step(self, dt: float):
        self.timer += dt
        super().step(dt)

    def acceleration_merge_before(self,
                                  ego_vehicle: ControlledVehicle) -> float:

        ego_target_speed = abs(utils.not_zero(getattr(ego_vehicle, "target_speed", 0)))

        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA_MERGE))

        target_x, rear_vehicle = self.get_insert_position_x_and_rear_vehicle()

        d = target_x - ego_vehicle.position[0]

        if d > 0:
            dv = np.dot(ego_vehicle.velocity - rear_vehicle.velocity, ego_vehicle.direction)
            th = 6
            acceleration = 0
            if dv > th:
                acceleration = - self.ACC_MAX * 0.7
            elif dv < -th:
                acceleration = self.ACC_MAX * 0.7
            elif dv < th and dv > 0:
                acceleration = - (dv / abs(dv)) * dv
        else:
            dv = np.dot(ego_vehicle.velocity - rear_vehicle.velocity, ego_vehicle.direction)
            if dv > self.ACC_MAX:
                acceleration = - self.ACC_MAX * 0.7
            elif dv >= 0:
                acceleration = -dv
            elif dv < -self.ACC_MAX:
                acceleration = self.ACC_MAX
            elif dv < 0:
                acceleration = dv
        return acceleration

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = abs(utils.not_zero(getattr(ego_vehicle, "target_speed", 0)))
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))
        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * \
                            np.power(
                                self.desired_gap(ego_vehicle, front_vehicle, projected=True) / utils.not_zero(d),
                                2)
        return acceleration

    def desired_gap_wanted_distance(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None,
                                    d0=IDMVehicle.DISTANCE_WANTED, projected: bool = True) -> float:
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def change_lane_policy(self) -> None:
        if self.lane_index != self.target_lane_index:
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        for lane_index in self.road.network.side_lanes(self.lane_index):
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            if np.abs(self.speed) < 1:
                continue
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                    self.route[0][2] - self.target_lane_index[2]):
                return False
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        return True

    def recover_from_stop(self, acceleration: float) -> float:
        stopped_speed = 5
        safe_distance = 200
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.network.get_lane(self.target_lane_index))
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                return -self.COMFORT_ACC_MAX / 2
        return acceleration
