import glob
import math
import os
import random
from typing import Dict, Text, Tuple
import pandas as pd

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.envs.common.action import Action
from typing import Optional, TypeVar

from matplotlib import pyplot as plt

from vehicles import CACCVehicle, MergeVehicle, OthersIDMVehicle

Observation = TypeVar("Observation")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(46)


def get_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    def __init__(self, config: dict = None, render_mode: Optional[str] = None):
        self.total_traffic = set()  # 某道路段总车量
        super().__init__(config, render_mode)

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
        })
        return cfg

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.total_traffic = set()  # 某道路段总车量

    def _create_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        # 之前，收敛，合并，之后  四段组成
        '''
        之前：150 米 
        收敛：80 米 匝道合并的斜路
        合并：80 米 匝道合并后缓冲区
        之后：150 米 匝道消失
        
        '''
        start_x = 1000  # 汇合路段开始
        end_x = 5000  # 汇合路段结束

        ends = [400, 300, 300, 400]  # Before, converging, merge, after

        '''
        c：实线
        s：虚线
        n：无线
        '''
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        # 四条车道
        y = [0, StraightLane.DEFAULT_WIDTH, StraightLane.DEFAULT_WIDTH * 2]
        line_type = [[c, s], [s, s], [n, c]]
        line_type_merge = [[c, s], [s, s], [n, s]]

        for i in range(3):
            net.add_lane("s", "a", StraightLane([0, y[i]], [start_x, y[i]], line_types=line_type[i]))
            net.add_lane("a", "b",
                         StraightLane([start_x, y[i]], [start_x + sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c",
                         StraightLane([start_x + sum(ends[:2]), y[i]], [start_x + sum(ends[:3]), y[i]],
                                      line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([start_x + sum(ends[:3]), y[i]], [start_x + sum(ends), y[i]],
                                                line_types=line_type[i]))
            net.add_lane("d", "e", StraightLane([start_x + sum(ends), y[i]], [start_x + sum(ends) + end_x, y[i]],
                                                line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([start_x, 6.5 + sum(y)], [start_x + ends[0], 6.5 + sum(y)], line_types=[c, c],
                           forbidden=True)

        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)

        lbc = StraightLane(lkb.position(ends[1], 0),
                           lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)

        left_lane = StraightLane([0, 6.5 + sum(y)], [start_x, 6.5 + sum(y)], line_types=[c, c], forbidden=True)

        net.add_lane("s", "j", left_lane)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def get_mean_speed(self):
        '''
        获得路上车平均速度
        :return:
        '''
        speeds = []
        for v in self.road.vehicles:
            # 只统计环境车
            if v not in self.controlled_vehicles:
                speeds.append(v.speed)
        return np.mean(speeds)

    # 获取某段道路上交通流量
    def get_road_traffic(self):
        lane_index = ("a", "b")

        # 记录该段路的所有车
        for v in self.road.vehicles:
            if (v not in self.controlled_vehicles
                    and v not in self.total_traffic and v.lane_index[:2] == lane_index):
                self.total_traffic.add(v)
    def get_traffic_flow(self):
        return (len(self.total_traffic)/self.time)*3600

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        self.get_road_traffic()  # 计算路上车流量
        return super().step(action)

    def _create_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road

        self.controlled_vehicles = []
        cacc_init_spacing = self.config["cacc_init_spacing"]  # 车辆初始间距
        cacc_num = self.config["cacc_num"]  # 编队车辆数

        CACCVehicle.DISTANCE_WANTED = cacc_init_spacing
        MergeVehicle.DISTANCE_WANTED = cacc_init_spacing
        longitudinals = list(range(0, cacc_num * cacc_init_spacing, cacc_init_spacing))

        self.cacc_vehicles = []
        # 添加车道正常行驶三个车辆
        for idx in range(cacc_num):
            vehicle = CACCVehicle(road, road.network.get_lane(("a", "b", 2))
                                  .position(longitudinals[idx], 0)
                                  , idx=idx, speed=90, target_speed=90)
            self.cacc_vehicles.append(vehicle)
            road.vehicles.append(vehicle)

        # 设置第一辆车辆的目标速度为110  会限制后面的车辆的速度并调整距离
        self.cacc_vehicles[-1].target_speed = 80

        if len(self.cacc_vehicles) == self.config["insert_index"]:
            init_pos = - cacc_init_spacing - 10
        else:
            # 添加车道合并行驶的车辆 初始位置为插入位置 后面一个5+车辆长度位置
            init_pos = longitudinals[len(self.cacc_vehicles) - self.config["insert_index"] - 1] - 10
        merging_controlled_vehicle = MergeVehicle(road,
                                                  road.network.get_lane(("j", "k", 0)).position(init_pos, 0),
                                                  cacc_vehicles=self.cacc_vehicles,  # 插入位置前面的车辆
                                                  insert_idx=self.config["insert_index"],
                                                  speed=95, target_speed=130)

        road.vehicles.append(merging_controlled_vehicle)

        self.controlled_vehicles.extend(self.cacc_vehicles)
        self.controlled_vehicles.append(merging_controlled_vehicle)

        other_vehicles_type = OthersIDMVehicle
        others = self.config["vehicles_count"]

        # 指定随机车辆生成的车道
        gen_lanes = [
            ("s", "a", 0.1),
            ("a", "b", 0.1),
            ("c", "d", 0.3),
            ("d", "e", 0.2),
            ("b", "c", 0.1),
        ]
        speeds = [100, 110, 120, 140]
        for lane_from, lane_to, w in gen_lanes[:-1]:
            lane_others = int(self.config["vehicles_count"] * w)
            others -= lane_others
            for _ in range(lane_others):
                vehicle = other_vehicles_type.create_random(self.road, lane_from=lane_from, lane_to=lane_to,
                                                            speed=random.choice(speeds),
                                                            spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                vehicle.controlled_vehicles = self.controlled_vehicles
                road.vehicles.append(vehicle)
        for _ in range(others):
            vehicle = other_vehicles_type.create_random(self.road, lane_from="b", lane_to="c",
                                                        speed=random.choice(speeds),
                                                        spacing=1 / self.config["vehicles_density"])
            vehicle.randomize_behavior()
            vehicle.controlled_vehicles = self.controlled_vehicles
            road.vehicles.append(vehicle)

    def get_next_actions(self):
        return tuple([(0, 0)] * len(self.controlled_vehicles))

    def _reward(self, action: Action):
        multi_rewards = {}
        _rewards = self._rewards(action)
        for vehicle_id, rewards in _rewards.items():
            _reward = []
            for name, reward in rewards.items():
                _reward.append(self.config.get(name, 0) * reward)
            reward = sum(_reward)
            if self.config["normalize_reward"]:
                reward = utils.lmap(reward,
                                    [self.config.get("collision_reward", 0),
                                     self.config.get("high_speed_reward", 0) + self.config.get(
                                         "right_lane_reward", 0) + self.config.get(
                                         "on_road_reward", 0)],
                                    [0, 1])
            reward *= rewards['on_road_reward']
            multi_rewards[vehicle_id] = reward
        return multi_rewards

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": [vehicle.speed for vehicle in self.controlled_vehicles],
            "crashed": [vehicle.crashed for vehicle in self.controlled_vehicles],
            "action": action,
            "head_distance": [self.get_head_distance(vehicle_id) for vehicle_id in range(len(self.controlled_vehicles))]
        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info

    def get_head_distance(self, control_vehicle_id):
        '''
        控制车辆同车道前方是否有车 有的话获得间距
        :return:
        '''
        control_vehicle = self.controlled_vehicles[control_vehicle_id]
        head_distance = float("inf")  # 前方车距
        control_lane_id = control_vehicle.lane_index[2]
        for idx, vehicle in enumerate(self.controlled_vehicles):
            if idx == control_vehicle_id:
                continue
            lane_id = vehicle.lane_index[2]
            if lane_id == control_lane_id:
                if vehicle.position[0] > control_vehicle.position[0]:
                    distance = vehicle.position[0] - control_vehicle.position[0]
                    if head_distance > distance:
                        head_distance = distance
        return head_distance if head_distance != float("inf") else None

    def multi_rewards_func(self, vehicle_id, control_vehicle: ControlledVehicle):
        '''
        获得受控制车辆的奖励信息
        :param control_vehicle: 受控制车辆
        :return: 受控制车辆的奖励信息
        '''
        neighbours = self.road.network.all_side_lanes(control_vehicle.lane_index)
        lane = control_vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = control_vehicle.speed * np.cos(control_vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(control_vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(control_vehicle.on_road),
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle)
            )
        }

    # 判断合并是否完成
    def merge_is_complete(self):
        # 1. 判断所有车辆是否在同一个车道
        lane_index = [c_v.lane_index[2] for c_v in self.controlled_vehicles]
        f1 = False
        if len(set(lane_index)) == 1:
            f1 = True
        # 2.判断车辆之间的距离是否差不多相同
        f2 = False
        v_x_list = [c_v.position[0] for c_v in self.controlled_vehicles]
        x_d = np.diff(v_x_list, 2)
        if np.array(x_d).all() < 5:  # 每辆车间距比较差异小于5米
            f2 = True
        # 3.判断速度是否差不多
        f3 = False
        v_speed_list = [c_v.speed for c_v in self.controlled_vehicles]
        v_speed_d = np.diff(v_speed_list)
        if np.array(v_speed_d).all() < 3:
            f3 = True

        # 4, 汇入后角度是否为0
        f4 = abs(self.controlled_vehicles[-1].heading) < 1e-4
        return f1 and f2 and f3 and f4

    def _rewards(self, action: Action):
        rewards = {}
        for vehicle_id, controlled_vehicle in enumerate(self.controlled_vehicles):
            rewards[vehicle_id] = self.multi_rewards_func(vehicle_id, controlled_vehicle)
        return rewards

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _is_terminated(self) -> bool:
        for controlled_vehicle in self.controlled_vehicles:
            if controlled_vehicle.crashed:
                return True
            if self.config["offroad_terminal"] and not controlled_vehicle.on_road:
                return True

        merge_is_complete = self.merge_is_complete()
        return False or merge_is_complete


# 每次生成不同的文件夹 run/exp1 run/exp2 ...
def get_log_path(log_root="run", sub_dir="exp"):
    os.makedirs(log_root, exist_ok=True)
    files = glob.glob(os.path.join(log_root, f"{sub_dir}") + "*", recursive=False)
    log_dir = os.path.join(log_root, f"{sub_dir}{len(files) + 1}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


if __name__ == '__main__':

    default_config = {
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",  # 每个控制车辆的状态信息类型
                "vehicles_count": 12,  # 观测到的车辆最大数目
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "ContinuousAction",
                "acceleration_range": [-5, 5],  # 加速度范围
                "steering_range": [-np.pi / 4, np.pi / 4],  # 转向值范围
                "speed_range": [20, 180],  # 速度范围
                "longitudinal": True,  # 启用油门控制
                "lateral": True  # 启用转弯控制
            },
        },
        "insert_index": 1,  # 插入位置 从0开始
        "cacc_num": 3,  # 编队货车数量
        "cacc_init_spacing": 50,  # 货车编队初始间距 [m]
        "vehicles_count": 50,  # 非控制车辆数目
        "duration": 60,  # 仿真时长 [s]  不是真实时长
        "reward_speed_range": [20, 40],  # 该速度范围才有速度奖励 超过最大值奖励达到最大
        "vehicles_density": 3,  # 车辆密度
        # 受控制的车辆数目以及配置
        "normalize_reward": True,  # 是否对奖励进行归一化
        "offroad_terminal": True,  # 是否在离开路面时终止仿真
        "simulation_frequency": 15,  # 仿真频率，每秒进行24次仿真步骤 [Hz]
        "policy_frequency": 5,  # 策略更新频率，每秒进行1次策略更新 [Hz]
        "screen_width": 1000,  # 屏幕宽度，用于图形渲染 [px]
        "screen_height": 200,  # 屏幕高度，用于图形渲染 [px]
        "centering_position": [0.2, 0.5],  # 屏幕中心位置的归一化坐标，x坐标为0.3，y坐标为0.5
        "scaling": 3,  # 屏幕缩放因子，用于图形渲染
        "show_trajectories": False,  # 是否显示车辆轨迹
        "render_agent": True,  # 是否渲染代理车辆
        "real_time_rendering": False  # 是否实时渲染
    }

    env = MergeEnv(default_config)
    env.reset()

    control_vehicles = len(env.controlled_vehicles)
    print(control_vehicles)
    eposides = 10
    rewards = [0 for _ in range(control_vehicles)]
    # 0: 'LANE_LEFT',
    # 1: 'IDLE',
    # 2: 'LANE_RIGHT',
    # 3: 'FASTER',
    # 4: 'SLOWER'
    print(env.action_space)
    save_dir = get_log_path()

    for eq in range(eposides):

        steps = []
        step = 0
        env.config["insert_index"] = eq % len(env.controlled_vehicles)
        acceleration_list = [None] * (env.config["cacc_num"] + 1)
        # env.config["insert_index"] = len(env.controlled_vehicles)-1
        obs = env.reset()

        # print(obs)
        env.render()
        done = False
        truncated = False

        while not done and not truncated:
            action = env.get_next_actions()
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            if env.merge_is_complete():
                print(f"合并至位置{env.config['insert_index']}完成\n"
                            f"耗时：{env.time:.2f}\n"
                            f"环境车平均速度：{env.get_mean_speed():.2f} m/s\n"
                            f"环境车车流量{env.get_traffic_flow():.2f} 辆/小时\n")
                # 保存数据至文本文件
                with open(os.path.join(save_dir, f"merge_{eq}_{env.config['insert_index']}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"合并至位置{env.config['insert_index']}完成\n"
                            f"耗时：{env.time:.2f}\n"
                            f"环境车平均速度：{env.get_mean_speed():.2f} m/s\n"
                            f"环境车车流量{env.get_traffic_flow():.2f} 辆/小时\n")

            steps.append(step)
            for i, v in enumerate(env.controlled_vehicles):
                if acceleration_list[i] is None:
                    acceleration_list[i] = [v.acceleration_action]
                else:
                    acceleration_list[i].append(v.acceleration_action)
            step += 1
        cols = 4
        table = {}
        table["steps"] = steps
        # 绘制
        plt.figure(figsize=(12, 8))
        for i, v in enumerate(env.controlled_vehicles[:-1]):
            plt.subplot(math.ceil((env.config["cacc_num"] + 1) / cols), cols, i + 1)
            table["truck" + str(i + 1)] = acceleration_list[i]
            plt.plot(steps, acceleration_list[i], label=f"Truck {i + 1}")
            plt.xlabel("Step")
            plt.ylabel("Acceleration")
            plt.legend()

        plt.subplot(math.ceil((env.config["cacc_num"] + 1) / cols), cols, len(env.controlled_vehicles))
        plt.plot(steps, acceleration_list[-1], label="Merge Truck")
        table["merge_truck"] = acceleration_list[-1]
        plt.xlabel("Step")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,
                                 f"acceleration_curve_{env.config['insert_index']}_{env.config['cacc_num']}.png"))
        plt.close()

        # 保存数据为xlsx
        df = pd.DataFrame(table)
        df.to_excel(
            os.path.join(save_dir, f"acceleration_curve_{env.config['insert_index']}_{env.config['cacc_num']}.xlsx"))
