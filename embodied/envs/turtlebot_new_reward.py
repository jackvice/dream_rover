import functools
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import embodied
import time

class Turtlebot(embodied.Env):
    def __init__(self, task, size=(64, 64), length=100, scan_topic='/scan',
                 cmd_vel_topic='/cmd_vel', connection_check_timeout=30,
                 lidar_points=1080, max_lidar_range=100.0):
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
        self.lidar_points = lidar_points
        self.max_lidar_range = max_lidar_range
        self.lidar_data = np.zeros(self.lidar_points, dtype=np.float32)
        self._done = False  # Changed from True to False
        self._length = length
        self._step = 0
        self._received_scan = False
        self.first = False
        self.desired_distance = 1.0
        
        # Check for actual connection to the robot
        self._robot_connected = self._check_robot_connection(timeout=connection_check_timeout)
        if not self._robot_connected:
            self.node.get_logger().warn("No actual robot detected. Running in simulation mode.")


    def get_reward(self):
        if self.lidar_data is None:
            return 0.0
        
        # Get the minimum distance to an obstacle from the LIDAR data
        min_distance = np.nanmin(self.lidar_data)
        
        # Calculate reward based on how close the min_distance is to the desired_distance
        reward = -((min_distance - self.desired_distance) ** 2)  # Gaussian-shaped reward
        return reward

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
        if not self.first:
            print("First scan received:")
            print(f"Number of points: {len(msg.ranges)}")
            print(f"Angle min: {msg.angle_min}, Angle max: {msg.angle_max}")
            print(f"Angle increment: {msg.angle_increment}")
            print(f"Range min: {msg.range_min}, Range max: {msg.range_max}")
            print(f"First 10 ranges: {msg.ranges[:10]}")
            print(f"Last 10 ranges: {msg.ranges[-10:]}")
            # Store angle information
            self.angle_min = msg.angle_min
            self.angle_max = msg.angle_max
            self.angle_increment = msg.angle_increment
            self.first = True

    # Rest of your existing lidar_callback code...       
        # Convert incoming data to numpy array and handle infinite values
        lidar_data = np.array(msg.ranges, dtype=np.float32)
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_range)
        
        # Resize if necessary
        if len(lidar_data) != self.lidar_points:
            lidar_data = np.resize(lidar_data, (self.lidar_points,))

        self.lidar_data = lidar_data
        self._received_scan = True
        self.node.get_logger().debug(f"Received scan data. Shape: {self.lidar_data.shape}")

    @property
    def obs_space(self):
        spaces = {
            'lidar': embodied.Space(np.float32, (self.lidar_points,), 0, self.max_lidar_range),
        }
        return {
            **spaces,
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @property
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
        twist.linear.x = action['linear_velocity']
        twist.angular.z = action['angular_velocity']
        self.publisher.publish(twist)
        # Wait for some time to simulate step duration
        time.sleep(0.1)
        

        if self._robot_connected:
            # Wait for scan data only if robot is connected
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if not self._received_scan:
                self.node.get_logger().warn("No scan data received")
        else:
            # Simulate more realistic scan data if no robot is connected
            self.lidar_data = np.random.uniform(0.1, self.max_lidar_range,
                                                self.lidar_points).astype(np.float32)

        # Calculate reward
        reward = self._calculate_reward()

            
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
        # Ensure angle information is available
        if not hasattr(self, 'angle_min'):
            return 0.0  # Cannot compute reward without angle information

        # Compute the angles for each lidar point
        angles = np.arange(self.lidar_points) * self.angle_increment + self.angle_min

        # Set parameters for wall-following on the right side
        side_angle = -np.pi / 2  # Right side (-90 degrees)
        angle_tolerance = np.pi / 8  # 22.5 degrees

        # Find indices of lidar points within the desired angle range
        side_indices = np.where(np.abs(angles - side_angle) <= angle_tolerance)[0]

        # If no points are found, return a neutral reward
        if len(side_indices) == 0:
            return 0.0

        # Get distances at those indices
        side_distances = self.lidar_data[side_indices]

        # Filter out invalid distances
        valid_distances = side_distances[(side_distances > 0.1) & (side_distances < self.max_lidar_range)]
        if len(valid_distances) == 0:
            return 0.0

        # Calculate the mean distance to the wall
        mean_distance = np.mean(valid_distances)

        # Calculate the error from the desired distance
        error = mean_distance - self.desired_distance

        # Define maximum acceptable error
        max_error = self.max_lidar_range - self.desired_distance

        # Normalize error to be between -1 and 1
        normalized_error = error / max_error
        normalized_error = np.clip(normalized_error, -1.0, 1.0)

        # Invert and shift to make reward range between -1 and 1
        reward = 1.0 - np.abs(normalized_error) * 2  # Reward is between -1 and 1
        
        return reward

    
    def _calculate_reward_minus1_to_zero(self):
        # Ensure angle information is available
        if not hasattr(self, 'angle_min'):
            return 0.0  # Cannot compute reward without angle information

        # Compute the angles for each lidar point
        angles = np.arange(self.lidar_points) * self.angle_increment + self.angle_min

        # Set parameters for wall-following on the right side
        side_angle = -np.pi / 2  # Right side (-90 degrees)
        angle_tolerance = np.pi / 8  # 22.5 degrees

        # Find indices of lidar points within the desired angle range
        side_indices = np.where(np.abs(angles - side_angle) <= angle_tolerance)[0]

        # If no points are found, return a neutral reward
        if len(side_indices) == 0:
            return 0.0

        # Get distances at those indices
        side_distances = self.lidar_data[side_indices]

        # Filter out invalid distances
        valid_distances = side_distances[(side_distances > 0.1) & (side_distances < self.max_lidar_range)]
        if len(valid_distances) == 0:
            return 0.0

        # Calculate the mean distance to the wall
        mean_distance = np.mean(valid_distances)

        # Calculate the maximum possible error
        max_error = self.max_lidar_range - self.desired_distance

        # Calculate the error from the desired distance
        error = mean_distance - self.desired_distance

        # Normalize the error to be between -1 and 1
        normalized_error = error / max_error
        normalized_error = np.clip(normalized_error, -1.0, 1.0)

        # Reward is negative squared normalized error
        reward = - (normalized_error ** 2)

 
    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()


