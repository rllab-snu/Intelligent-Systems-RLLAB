import os
import argparse
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from easydict import EasyDict

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix

from std_msgs.msg import Bool
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D
from ackermann_msgs.msg import AckermannDriveStamped

from rccar_gym.env_wrapper import RCCarWrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help="Set seed number.")
    parser.add_argument("--env_config", default="configs/env.yaml", type=str, help="Path to environment config file (.yaml)")
    parser.add_argument("--dynamic_config", default="configs/dynamic.yaml", type=str, help="Path to dynamic config file (.yaml)")
    parser.add_argument("--render", default=True, action='store_true', help="Whether to render or not.")
    args = parser.parse_args()
    args = EasyDict(vars(args))
    
    ws_path = os.path.join(get_package_prefix('rccar_bringup'), "../..")
    args.env_config = os.path.join(ws_path, args.env_config)
    args.dynamic_config = os.path.join(ws_path, args.dynamic_config)
    
    # configuration files
    with open(args.env_config, 'r') as f:
        task_args = EasyDict(YAML().load(f))
    with open(args.dynamic_config, 'r') as f:
        dynamic_args = EasyDict(YAML().load(f))

    args.update(task_args)
    args.update(dynamic_args)

    return args


class RCCarBridge(Node):
    def __init__(self, args):
        super().__init__('rccar_bridge')

        self.args = args
        self.env = None
        self.map = None
        self.query_steer = 0.0
        self.query_speed = 0.0
        self.render = args.render
        self.running = False

        ### Obs publishing timer
        self.state_pub_timer = self.create_timer(0.005, self.state_pub_callback)
        
        ### Publishers
        # running state
        self.state_publisher = self.create_publisher(Bool, "/running", 10)
        
        # Terminate signal
        ####################################################
        ### TODO 1. Create '/terminate' topic publisher. ###
        
        ####################################################
        
        ### Subscribers
        self.reset_subscriber = self.create_subscription(String, "/reset", self.reset_callback, 1)
        ##################################################
        ### TODO 2. Create '/action' topic subscriber. ###
        
        ##################################################
        self.get_logger().info(">>> RCCar Bridge Node Activated!")
        
    def state_pub_callback(self):
        running_msg = Bool()
        running_msg.data = self.running
        self.state_publisher.publish(running_msg)

    def reset_callback(self, map_msg):
        if self.running:
            self.env.close()

        ### New environment
        self.map = map_msg.data
        maps = [self.map]
        self.env = RCCarWrapper(args=self.args, maps=maps, render_mode="human_fast" if self.render else None)
        track = self.env._env.unwrapped.track

        _, _ = self.env.reset(seed=self.args.seed)
        
        
        if self.render:
            self.env.unwrapped.add_render_callback(track.centerline.render_waypoints)
        if self.render:
            self.env.render()
        
        self.get_logger().info(">>> Current Map: {}".format(self.map))
        self.running = True
    
    def action_callback(self, action_msg):
        if self.running:
            self.query_steer = action_msg.drive.steering_angle
            self.query_speed = action_msg.drive.speed
            
            _, _, terminate, _, _ = self.env.step(np.array([[self.query_steer, self.query_speed]]))
            
            if self.render:
                self.env.render()
                
            if terminate:
                ####################################################################
                ### TODO 3. Publish '/terminate' topic to keyboard_control node. ###
                # ">>> Terminated" should be printed in terminal in which keyboard_control node is running.
                


                ###################################################################
                self.env.close()

                self.get_logger().info(">>> Terminated: {}".format(self.map))
                self.running = False

def main():
    args = get_args()

    rclpy.init()
    rccar_bridge = RCCarBridge(args)
    rclpy.spin(rccar_bridge)


if __name__ == "__main__":
    main()
