# Introduction to Intelligent Systems [430.457]
This is the repository for the student projects of the 'Introduction to Intelligent Systems' course at Seoul National University.
rccar_gym environment codes are originated from [F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym) repository.

> Original author of F1TENTH Gym: *Hongrui Zheng*
 
> (Special Thanks to *Hyeokjin Kwon, Geunje Cheon, Junseok Kim* for editing rccar_gym)

> Authors of this repo: *Minsoo Kim, Yoseph Park, Subin Shin*

## Fall 2025 
> TAs for this class: *Hyeondal Son, Jooyoung Kim, Hosung Lee*

## RCCar Gym Environment Setting
We recommend you install packages inside a virtual environment such as [Anaconda](https://www.anaconda.com) (or virtualenv).

```shell
conda create -n rccar python=3.8
conda activate rccar

git clone https://github.com/rllab-snu/Intelligent-Systems-RLLAB.git
cd Intelligent-Systems-2024-Pre/rccar_gym
pip install -e .
```
This will install a gym environment for the RC car and its dependencies.

## ROS2 Setting
We use ‘ROS2 Foxy’ to run the gym environment and project codes.

First, install ROS2 foxy by following the [documentation](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html).

To build the ROS2 packages to use our specific python virtual environment, we should install colcon building tools to our python environment.

Assuming the virtual environment is activated, i.e. `conda activate rccar`,

```shell
pip install colcon-common-extensions
```
This enables installed files resulting from colcon build to use desired package in our environment.

Now, install dependencies and build the packages.

```shell
cd Intelligent-Systems-2025-Pre
rosdep update --rosdistro foxy
rosdep install -i --from-path src --rosdistro foxy -y
colcon build --symlink-install
```
Note that `--rosdistro foxy` is required for `rosdep update` since foxy is an end-of-life version.

Note that `--symlink-install` is required to use modified python files directly without building again.

After building the package, we should use following command in every terminal we want to use our packages.

```shell
source install/setup.bash
```

## Running System by ROS2 Commands
To run the node activating rccar gym, use following command in the first terminal.

```shell
ros2 run rccar_bringup rccar_bringup
```
To run the node which enables keyboard control, use following command in the second terminal.

```shell
ros2 run rccar_bringup keyboard_control
```
