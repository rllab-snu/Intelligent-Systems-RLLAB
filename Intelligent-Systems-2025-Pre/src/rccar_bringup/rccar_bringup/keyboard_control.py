import os
import sys
import tty
import termios
import argparse
import threading
from ruamel.yaml import YAML
from easydict import EasyDict

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix

from std_msgs.msg import Bool
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped

MAP_NAME = 'square'

msg = """>>> Keyboard Controller
To control the rccar, press the following buttons.

Moving foward   : w
Turn left       : a
Turn Right      : d
Stop            : s

Quit            : q (When CTRL-C doesn't work)

Note that there would be some delay from action input to desired driving movement.

Start Controlling!"""


def getKey(settings):
    # Only For Linux
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    
    return key


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", default="configs/env.yaml", type=str, help="Path to environment config file (.yaml)")
    parser.add_argument("--dynamic_config", default="configs/dynamic.yaml", type=str, help="Path to dynamic config file (.yaml)")
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


class KeyboardControl(Node):
    def __init__(self, args):
        super().__init__("keyboard_control_node")
        
        self.args = args
        self.max_speed = args.max_speed
        self.max_steer = args.max_steer
        self.env_ready_to_init = False
        self.terminate = False
        self.initialized = False
        
        ### Publishers
        self.reset_publisher = self.create_publisher(String, "/reset", 10)
        self.action_publisher = self.create_publisher(AckermannDriveStamped, "/action", 10)
        
        ### Subscribers
        self.state_subscriber = self.create_subscription(Bool, "/running", self.state_callback, 10)
        self.terminator_subscriber = self.create_subscription(Bool, "/terminate", self.terminate_callback, 10)

        print(msg)
        
    def state_callback(self, state_msg):
        state = state_msg.data
        prev_state = self.env_ready_to_init
        self.env_ready_to_init = not state
        
        if not prev_state and self.env_ready_to_init:
            self.reset()
        
    def reset(self):
        map_name = MAP_NAME
        
        map_msg = String()
        map_msg.data = map_name
        
        self.reset_publisher.publish(map_msg)
        self.terminate = False
        self.initialized = True

    def run_control(self):
        while True:
            if self.initialized:
                settings = termios.tcgetattr(sys.stdin)
                key = getKey(settings)

                ###################################################################
                ### TODO 4. Define 'steer' and 'speed' which are going to
                #           published to 'rccar_bridge' node for controlling RC-car.
                # 'w': move forward, 's': stop, 'a': turn left, 'd': turn right
                # 'steer' must be in [-self.max_steer, self.max_steer]
                # 'speed' must be in [0, self.max_speed]
                
                
                








                if key == 'q':
                    print("Keyborad Control Finished")
                    rclpy.shutdown()
                    sys.exit(0)
                if key == '\x03':
                    rclpy.shutdown()
                    raise KeyboardInterrupt
                ###################################################################

                #################################################################
                ### TODO 5. Publish '/action' topic containing steer & speed. ###
                



                #################################################################
                
            else:
                steer = 0.0
                speed = 0.0

    def terminate_callback(self, terminate_msg):
        terminate = terminate_msg.data
        if terminate:
            print(">>> Terminated\r")
            self.terminate = True
            self.initialized = False


def main():
    args = get_args()
    
    rclpy.init()
    action_node = KeyboardControl(args)
    spinner = threading.Thread(target=rclpy.spin, args=(action_node,))
    spinner.start()
    
    action_node.run_control()


if __name__ == "__main__":
    main()
