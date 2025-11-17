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
cd Intelligent-Systems-RLLAB/Intelligent-Systems-2025-Project/rccar_gym
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
cd Intelligent-Systems-RLLAB/Intelligent-Systems-2025-Project
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

## Project
The rccar gym environment has slight modifications so you need to install the dependencies again.(commands above)

To train your PPO agent and evaluate, run project code with `--mode train` argument.
```shell
ros2 run rccar_bringup RLLAB_project2 --mode train
#replace RLLAB with your team name
```
To load a trained model without training, just run without `--mode` argument since the default value is `val`
```shell
ros2 run rccar_bringup RLLAB_project2
#replace RLLAB with your team name
```
## Manually publishing map topic
For each project code, you need to publish `/query` topic manually using the following command in another terminal

```shell
ros2 topic pub --once /query message/msg/Query "{id: '0', team: 'RLLAB', map: 'map1', trial: 0, exit: false}"
# don't forget to change RLLAB to your team name
```
Note that you can publish the topic once with `--once` argument
## Map generation
You can change the parameters defined in `random_trackgen.py` and randomly generate your own map with `--seed` argument. You can run `random_trackgen.py` using the following command.
```shell
cd Intelligent-Systems-RLLAB/Intelligent-Systems-2025-Project/maps
python random_trackgen.py --seed your_seed --name your_map_name
```