import functools
import numpy as np
import embodied
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import time


class Turtlebot(embodied.Env):
    def __init__(self, task, size=(64, 64), length=100, scan_topic='/scan',
                 cmd_vel_topic='/cmd_vel', connection_check_timeout=30):
        super().__init__()
        rclpy.init()
        self.node = rclpy.create_node('turtlebot_controller')
        self.publisher = self.node.create_publisher(Twist, cmd_vel_topic, 10)
        self.lidar_subscriber = self.node.create_subscription(
            LaserScan, 
            scan_topic, 
            self.lidar_callback, 
            10
        )
        self.lidar_data = None # np.zeros(640, dtype=np.float32)  # Initial lidar data placeholder
        self._done = True
        self._length = length
        self._step = 0
        self._received_scan = False
        
        # Check for actual connection to the robot
        self._robot_connected = self._check_robot_connection(timeout=connection_check_timeout)
        if not self._robot_connected:
            self.node.get_logger().warn("No actual robot detected. Running in simulation mode.")
            self.lidar_data = np.zeros(640, dtype=np.float32)  # Default to 640 for simulation

    def _check_robot_connection(self, timeout):
        start_time = time.time()
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                return False
            if self._received_scan:
                return True
        return False

    def lidar_callback(self, msg):
        self.lidar_data = np.array(msg.ranges)
        self._received_scan = True
        self.node.get_logger().debug(f"Received scan data. Shape: {self.lidar_data.shape}")

    @functools.cached_property
    def obs_space(self):
        lidar_shape = self.lidar_data.shape if self.lidar_data is not None else (640,)
        spaces = {
            'lidar': embodied.Space(np.float32, lidar_shape),
        }
        return {
            **spaces,
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @functools.cached_property
    def act_space(self):
        return {
            'linear_velocity': embodied.Space(np.float32, (), -1, 1),
            'angular_velocity': embodied.Space(np.float32, (), -1, 1),
            'reset': embodied.Space(bool),
        }

    def step(self, action):
        if action['reset'] or self._done:
            self._step = 0
            self._done = False
            return self._obs(0.0, is_first=True)
        
        twist = Twist()
        twist.linear.x = 0.1 #action['linear_velocity']
        twist.angular.z = 0.0 #action['angular_velocity']
        self.publisher.publish(twist)

        if self._robot_connected:
            # Wait for scan data only if robot is connected
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if not self._received_scan:
                self.node.get_logger().warn("No scan data received")
        else:
            # Simulate scan data if no robot is connected
            self.lidar_data = np.random.rand(640)  # Random data for simulation

        self._step += 1
        reward = self._calculate_reward()
        self._done = (self._step >= self._length)
        return self._obs(
            reward,
            is_last=self._done,
            is_terminal=self._done)

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        obs = {
            'lidar': self.lidar_data,
        }
        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal)
        return obs

    def _calculate_reward(self):
        # Implement your reward calculation here
        # For now, let's return a dummy reward
        return 0.0

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

class Turtlebot_no_sim_data(embodied.Env):
    def __init__(self, task, size=(64, 64), length=100, scan_topic='/scan',
                 cmd_vel_topic='/cmd_vel', topic_wait_timeout=30):
        super().__init__()
        rclpy.init()
        self.node = rclpy.create_node('turtlebot_controller')
        self.publisher = self.node.create_publisher(Twist, cmd_vel_topic, 10)
        self.lidar_subscriber = self.node.create_subscription(
            LaserScan, 
            scan_topic, 
            self.lidar_callback, 
            10
        )
        self.lidar_data = np.zeros(640)  # Initial lidar data placeholder
        self._done = True
        self._length = length
        self._step = 0
        self._received_scan = False
        
        # Wait for the scan topic to become available
        self._wait_for_scan_topic(timeout=topic_wait_timeout)

    def _wait_for_scan_topic(self, timeout):
        start_time = time.time()
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                self.node.get_logger().warn(f"Timeout waiting for scan topic: {self.lidar_subscriber.topic}")
                break
            if self._received_scan:
                self.node.get_logger().info(f"Received data on scan topic: {self.lidar_subscriber.topic}")
                break

    def lidar_callback(self, msg):
        self.lidar_data = np.array(msg.ranges)
        self._received_scan = True
        self.node.get_logger().debug(f"Received scan data. Shape: {self.lidar_data.shape}")

    @functools.cached_property
    def obs_space(self):
        spaces = {
            'lidar': embodied.Space(np.float32, (640,)),
        }
        return {
            **spaces,
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @functools.cached_property
    def act_space(self):
        return {
            'linear_velocity': embodied.Space(np.float32, (), -1, 1),
            'angular_velocity': embodied.Space(np.float32, (), -1, 1),
            'reset': embodied.Space(bool),
        }

    def step(self, action):
        if action['reset'] or self._done:
            self._step = 0
            self._done = False
            return self._obs(0.0, is_first=True)
        
        twist = Twist()
        twist.linear.x = 0.1 #action['linear_velocity']
        twist.angular.z = 0.0 #action['angular_velocity']
        self.publisher.publish(twist)

        # Wait for scan data
        rclpy.spin_once(self.node, timeout_sec=0.1)
        if not self._received_scan:
            self.node.get_logger().warn("No scan data received")

        self._step += 1
        reward = self._calculate_reward()
        self._done = (self._step >= self._length)
        return self._obs(
            reward,
            is_last=self._done,
            is_terminal=self._done)

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        obs = {
            'lidar': self.lidar_data,
        }
        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal)
        return obs

    def _calculate_reward(self):
        # Implement your reward calculation here
        # For now, let's return a dummy reward
        return 0.0

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
