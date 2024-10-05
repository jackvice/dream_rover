python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs atari \
  --run.train_ratio 32
  

python dreamerv3/main.py --logdir ~/logdir/{timestamp} --configs bipedalwalker



python dreamerv3/main.py --logdir ~/logdir/{timestamp} --configs dmc_proprio

python dreamerv3/main.py --configs turtlebot --logdir ~/logdir/turtlebot



################ ROS2

ros2 run teleop_twist_keyboard teleop_twist_keyboard
ros2 control list_controllers

ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py

Fix contollers:
ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{start_controllers: ['joint_state_broadcaster'], stop_controllers: [], strictness: 1, start_asap: false, timeout: {sec: 5, nanosec: 0}}"

ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{start_controllers: ['diffdrive_controller'], stop_controllers: [], strictness: 1, start_asap: false, timeout: {sec: 5, nanosec: 0}}"

