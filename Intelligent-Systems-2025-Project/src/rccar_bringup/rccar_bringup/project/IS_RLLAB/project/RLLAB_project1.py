import os
import time
import argparse
import numpy as np
from ruamel.yaml import YAML
from easydict import EasyDict

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix

from message.msg import Result, Query
from rccar_gym.env_wrapper import RCCarWrapper

from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from joblib import dump, load
import traceback

###################################################
########## YOU MUST CHANGE THIS PART ##############

TEAM_NAME = "RLLAB"

###################################################
###################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help="Set seed number.")
    parser.add_argument("--env_config", default="configs/env.yaml", type=str, help="Path to environment config file (.yaml)") # or ../..
    parser.add_argument("--dynamic_config", default="configs/dynamic.yaml", type=str, help="Path to dynamic config file (.yaml)") # or ../..
    parser.add_argument("--render", default=True, action='store_true', help="Whether to render or not.")
    parser.add_argument("--no_render", default=False, action='store_true', help="No rendering.")
    parser.add_argument("--mode", default='val', type=str, help="Whether train new model or not")
    parser.add_argument("--traj_dir", default="trajectory", type=str, help="Saved trajectory path relative to 'IS_TEAMNAME/project/'") #you can change the path where your trajecotry file is saved
    parser.add_argument("--model_dir", default="model", type=str, help="Model path relative to 'IS_TEAMNAME/project/'")
    
    ###################################################
    ########## YOU CAN ONLY CHANGE THIS PART ##########
    """
    Change the name as you want.
    Note that this will used for evaluation by the server as well.
    """
    parser.add_argument("--model_name", default="default_model.pkl", type=str, help="model name to save and use")
    ###################################################
    ###################################################
    
    args = parser.parse_args()
    args = EasyDict(vars(args))
    
    # render
    if args.no_render:
        args.render = False
    
    ws_path = os.path.join(get_package_prefix('rccar_bringup'), "../..")

    # map files
    args.maps = os.path.join(ws_path, 'maps')
    args.maps = [map for map in os.listdir(args.maps) if os.path.isdir(os.path.join(args.maps, map))]
    
    # configuration files
    args.env_config = os.path.join(ws_path, args.env_config)
    args.dynamic_config = os.path.join(ws_path, args.dynamic_config)
    
    with open(args.env_config, 'r') as f:
        task_args = EasyDict(YAML().load(f))
    with open(args.dynamic_config, 'r') as f:
        dynamic_args = EasyDict(YAML().load(f))

    args.update(task_args)
    args.update(dynamic_args)
    
    # Trajectory & Model Path
    project_path = os.path.join(ws_path, f"src/rccar_bringup/rccar_bringup/project/IS_{TEAM_NAME}/project")
    args.traj_dir = os.path.join(project_path, args.traj_dir)
    args.model_dir = os.path.join(project_path, args.model_dir)
    args.model_path = os.path.join(args.model_dir, args.model_name)

    return args


class GaussianProcess(Node):
    def __init__(self, args):
        super().__init__(f"{TEAM_NAME}_project1")
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

        self.traj_dir = args.traj_dir
        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.model_path = args.model_path
        
        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########
        """
        1) Choose the maps to use for training as expert demonstration. You can add more maps for training 
        2) Define your model and other configurations for pre/post-processing.
        """
        self.train_maps = ['map1', 'map2']
        self.kernel = None
        self.alpha = None
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
        ###################################################
        ###################################################
        
        self.load()
        self.get_logger().info(">>> Running Project 1 for TEAM {}".format(TEAM_NAME))
        
    def train(self):
        self.get_logger().info(">>> Start model training")
        
        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########
        """
        1) load your expert demonstration.
        2) Fit GP model, which gets lidar observation as input, and gives action as output.
           We recommend to pre/post-process observations and actions for better performance (e.g. normalization).
        """
        
        ###################################################
        ###################################################

        os.makedirs(self.model_dir, exist_ok=True)

        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########
        """
        Save the file containing trained model and configuration for pre/post-processing.
        """

        ###################################################
        ###################################################
        
        self.get_logger().info(">>> Trained model {} is saved".format(self.model_name))
        
            
    def load(self):
        if self.mode == 'val':
            assert os.path.exists(self.model_path)
            ###################################################
            ########## YOU CAN ONLY CHANGE THIS PART ##########
            """
            Load the trained model and configurations for pre/post-processing.
            """
            
            ###################################################
            ###################################################
        elif self.mode == 'train':
            self.train()
        else:
            raise AssertionError("mode should be one of 'train' or 'val'.")   

    def get_action(self, obs):
        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########
        """
        1) Pre-process the observation input, which is current 'scan' data.
        2) Get predicted action from the model.
        3) Post-process the action. Be sure to satisfy the limitation of steer and speed values.
        """
        
        ###################################################
        ###################################################
        return action


    ######################################################
    #####           DO NOT CHANGE THIS PART          #####
    ##### THIS IS A PART OF THE TA's EVALUATION CODE #####
    ######################################################
    def query_callback(self, query_msg):
        
        id = query_msg.id
        team = query_msg.team
        map = query_msg.map
        trial = query_msg.trial
        exit = query_msg.exit 
        
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
            env = RCCarWrapper(args=self.args, maps=[map], render_mode="human_fast" if self.render else None)
            track = env._env.unwrapped.track
            if self.render:
                env.unwrapped.add_render_callback(track.centerline.render_waypoints)
            
            obs, _ = env.reset(seed=self.args.seed)
            _, _, scan = obs

            step = 0
            terminate = False

            while True:  

                act = self.get_action(scan)
                steer = np.clip(act[0][0], -self.max_steer, self.max_steer)
                speed = np.clip(act[0][1], self.min_speed, self.max_speed)
                
                obs, _, terminate, _, info = env.step(np.array([steer, speed]))
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
                    result_msg.waypoint = info['waypoint']
                    result_msg.n_waypoints = 20
                    result_msg.success = False
                    result_msg.fail_type = "Time Out"
                    self.get_logger().info(">>> Time Out: {}".format(map))
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
                    result_msg.waypoint = info['waypoint']
                    result_msg.n_waypoints = 20
                    if info['waypoint'] == 20:
                        result_msg.success = True
                        result_msg.fail_type = "-"
                        self.get_logger().info(">>> Success: {}".format(map))
                    else:
                        result_msg.success = False
                        result_msg.fail_type = "Collision"
                        self.get_logger().info(">>> Collision: {}".format(map))
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
        
        if exit:
            rclpy.shutdown()
        return

def main():
    args = get_args()
    rclpy.init()
    node = GaussianProcess(args)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
