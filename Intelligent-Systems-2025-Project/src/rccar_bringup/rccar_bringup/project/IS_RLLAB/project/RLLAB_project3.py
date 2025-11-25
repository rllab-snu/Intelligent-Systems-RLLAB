import os
import time
from datetime import datetime
import argparse
import traceback

import numpy as np
from ruamel.yaml import YAML
from easydict import EasyDict

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix

from message.msg import Result, Query
from rccar_gym.env_wrapper import RCCarWrapper
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

###################################################
########## YOU CAN ONLY CHANGE THIS PART  #########

"""
TODO:
Freely import modules, define methods and classes, etc.

You may add extra Python scripts, but remember to push them to GitHub.
Notify TA if any additional modules need to be installed on the evaluation server.
(For deep learning, please use PyTorch.)

However, it is NOT ALLOWED to use pre-built PPO implementations (e.g.: Stable-Baselines3 or RLlib).
You must implement the training process on your own.
"""

TEAM_NAME = "TA"

import random

import torch

### you can use different algorithms other than ppo if you want ###
### Feel free to import any necessary modules except predefined Libraries such as stable-baselines
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.vector import NormalizeObservation, NormalizeReward, ClipReward
from gymnasium.wrappers import RecordEpisodeStatistics

from .wrapper import RCCarEnvTrainWrapper
from .algo import PPO


def make_env_fn(index, args, maps, max_steer, min_speed, max_speed, time_limit):
    def _thunk():
        base_env = RCCarWrapper(args=args, maps=maps, render_mode=None, seed=args.seed + index)
        env = RCCarEnvTrainWrapper(base_env, max_steer=max_steer, min_speed=min_speed, max_speed=max_speed, time_limit=time_limit)
        return RecordEpisodeStatistics(env)

    return _thunk


################### CHANGE END  ###################
###################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Set seed number.")
    parser.add_argument("--env_config", default="configs/env.yaml", type=str, help="Path to environment config file (.yaml)")  # or ../..
    parser.add_argument("--dynamic_config", default="configs/dynamic.yaml", type=str, help="Path to dynamic config file (.yaml)")  # or ../..
    parser.add_argument("--render", default=True, action="store_true", help="Whether to render or not.")
    parser.add_argument("--no_render", default=False, action="store_true", help="No rendering.")
    parser.add_argument("--mode", default="val", type=str, help="Whether train new model or not")
    parser.add_argument("--traj_dir", default="trajectory", type=str, help="Saved trajectory path relative to 'IS_TEAMNAME/project/'")
    parser.add_argument("--model_dir", default="model", type=str, help="Model path relative to 'IS_TEAMNAME/project/'")
    parser.add_argument("--num_obstacles", default=10, type=int, help="number of obstacles spawned inside track") # note that in order to spawn different obstacles you need to change the seed
    
    ###################################################
    ########## YOU CAN ONLY CHANGE THIS PART ##########
    """
    TODO:
    Change the model name as you want.
    Note that this will used for evaluation by the server as well.
    You can add any arguments you want.
    """

    parser.add_argument("--model_name", default="model.pt", type=str, help="Model name to save and load")
    parser.add_argument("--checkpoint_freq", default=int(2e5), type=int, help="Save every N timesteps during training")
    parser.add_argument("--wandb", default=False, action="store_true", help="(Optional) Enable wandb logging")
    parser.add_argument("--tb", default=False, action="store_true", help="(Optional) Enable tensorboard logging")

    ################### CHANGE END  ###################
    ###################################################

    args = parser.parse_args()
    args = EasyDict(vars(args))

    # render
    if args.no_render:
        args.render = False

    ws_path = os.path.join(get_package_prefix("rccar_bringup"), "../..")

    # map files
    args.maps = os.path.join(ws_path, "maps")
    args.maps = [map for map in os.listdir(args.maps) if os.path.isdir(os.path.join(args.maps, map))]

    # configuration files
    args.env_config = os.path.join(ws_path, args.env_config)
    args.dynamic_config = os.path.join(ws_path, args.dynamic_config)

    with open(args.env_config, "r") as f:
        task_args = EasyDict(YAML().load(f))
    with open(args.dynamic_config, "r") as f:
        dynamic_args = EasyDict(YAML().load(f))

    args.update(task_args)
    args.update(dynamic_args)

    # Trajectory & Model Path
    project_path = os.path.join(ws_path, f"src/rccar_bringup/rccar_bringup/project/IS_{TEAM_NAME}/project")
    args.project_path = project_path
    args.traj_dir = os.path.join(project_path, args.traj_dir)
    args.model_dir = os.path.join(project_path, args.model_dir)
    args.model_path = os.path.join(args.model_dir, args.model_name)

    return args


class RCCarPolicy(Node):
    def __init__(self, args):
        super().__init__(f"{TEAM_NAME}_project3")
        self.args = args
        self.mode = args.mode

        self.query_sub = self.create_subscription(Query, "/query", self.query_callback, 10)
        self.result_pub = self.create_publisher(Result, "/result", 10)

        self.dt = args.timestep
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer
        self.maps = args.maps
        self.render = args.render
        self.time_limit = 180.0

        self.project_path = args.project_path
        self.traj_dir = args.traj_dir
        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.model_path = args.model_path

        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########
        """
        TODO:
        Freely change the codes (__init__, train, load, get_action) to increase the performance.
        You can also add attributes, methods, etc.
        """

        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        seed = int(self.args.seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)

        self.model = None

        self.load()
        self.get_logger().info(f">>> Running Project 3 for TEAM {TEAM_NAME}")

    def train(self):
        self.get_logger().info(">>> Start model training")
        """
        Train and save your model.
        You can either use this part or explicitly train using other python codes.
        """
        ### you can use the same training setting of Project 2 or you could freely use other algorithms
        ### you can change any structure of the code except the query callback function
        pass

    def load(self):
        """
        Load your trained model.
        Make sure not to train a new model when self.mode == 'val'.
        """
        if self.mode == "val":
            if not os.path.exists(self.model_path):
                raise AssertionError(f"Model not found: {self.model_path}")
            ### load your trained model ###

        elif self.mode == "train":
            self.train()

        else:
            raise AssertionError("mode should be one of 'train' or 'val'.")

    def get_action(self, scan_obs):
        """
        Predict action using obs - 'scan' data.
        Be sure to satisfy the limitation of steer and speed values.
        """
        
        action_arr = self.model.predict(scan_obs.reshape(1, -1), deterministic=True)
        action_arr = action_arr[0]

        steer = float(np.clip(action_arr[0], -self.max_steer, self.max_steer))
        speed = float(np.clip(action_arr[1], self.min_speed, self.max_speed))

        action = np.array([[steer, speed]])

        return action

    ################### CHANGE END  ###################
    ###################################################

    def query_callback(self, query_msg):

        id = query_msg.id
        team = query_msg.team
        map = query_msg.map
        trial = query_msg.trial
        _exit = query_msg.exit

        result_msg = Result()

        START_TIME = time.time()

        try:
            if team != TEAM_NAME:
                return

            if map not in self.maps:
                END_TIME = time.time()
                result_msg.id = id
                result_msg.team = team
                result_msg.map = map
                result_msg.trial = trial
                result_msg.time = END_TIME - START_TIME
                result_msg.waypoint = 0
                result_msg.n_waypoints = 20
                result_msg.success = False
                result_msg.fail_type = "Invalid Track"
                self.get_logger().info(">>> Invalid Track")
                self.result_pub.publish(result_msg)
                return

            self.get_logger().info(f"[{team}] START TO EVALUATE! MAP NAME: {map}")

            ### New environment
            
            env = RCCarWrapper(args=self.args, maps=[map], render_mode="human_fast" if self.render else None, seed = self.args.seed)
            track = env._env.unwrapped.track
            if self.render:
                env.unwrapped.add_render_callback(track.centerline.render_waypoints)

            obs, _ = env.reset(seed=self.args.seed)
            _, _, scan = obs
           
            step = 0
            terminate = False

            while True:
                act = self.get_action(scan)
                steer = np.clip(act[:, 0], -self.max_steer, self.max_steer)
                speed = np.clip(act[:, 1], self.min_speed, self.max_speed)

                obs, _, terminate, _, info = env.step(np.stack([steer, speed], axis=1))
                _, _, scan = obs
                step += 1
                

                if self.render:
                    env.render()

                if time.time() - START_TIME > self.time_limit:
                    END_TIME = time.time()
                    result_msg.id = id
                    result_msg.team = team
                    result_msg.map = map
                    result_msg.trial = trial
                    result_msg.time = step * self.dt
                    result_msg.waypoint = info["waypoint"]
                    result_msg.n_waypoints = 20
                    result_msg.success = False
                    result_msg.fail_type = "Time Out"
                    self.get_logger().info(f">>> Time Out: {map}")
                    self.result_pub.publish(result_msg)
                    env.close()
                    break

                if terminate:
                    END_TIME = time.time()
                    result_msg.id = id
                    result_msg.team = team
                    result_msg.map = map
                    result_msg.trial = trial
                    result_msg.time = step * self.dt
                    result_msg.waypoint = info["waypoint"]
                    result_msg.n_waypoints = 20
                    if info["waypoint"] == 20:
                        result_msg.success = True
                        result_msg.fail_type = "-"
                        self.get_logger().info(f">>> Success: {map}")
                    else:
                        result_msg.success = False
                        result_msg.fail_type = "Collision"
                        self.get_logger().info(f">>> Collision: {map}")
                    self.result_pub.publish(result_msg)
                    env.close()
                    break

        except Exception as e:
            END_TIME = time.time()
            result_msg.id = id
            result_msg.team = team
            result_msg.map = map
            result_msg.trial = trial
            result_msg.time = END_TIME - START_TIME
            result_msg.waypoint = 0
            result_msg.n_waypoints = 20
            result_msg.success = False
            result_msg.fail_type = "Script Error"
            error_message = traceback.format_exc()
            self.get_logger().error(f">>> Script Error - {str(e)}\n{error_message}")
            self.result_pub.publish(result_msg)

        if _exit:
            exit(0)
        return


def main():
    args = get_args()
    rclpy.init()
    node = RCCarPolicy(args)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
