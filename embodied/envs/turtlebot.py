import functools
import numpy as np
import rclpy
from geometry_msgs.msg import Twist #, Pose, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
import embodied
import time
import math
from dreamerv3.utils import l2_distance
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from transforms3d.euler import quat2euler
import jax.numpy as jnp

class Turtlebot(embodied.Env):
    def __init__(self, task, size=(64, 64), length=200, scan_topic='/scan',
                 cmd_vel_topic='/cmd_vel', odom_topic='/odom', connection_check_timeout=30,
                 lidar_points=640, max_lidar_range=12.0):
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
        self.odom_subscription = self.node.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )
 
        # Define QoS profile
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        # Create the subscription using the custom QoS profile
        self.pose_subscription = self.node.create_subscription(
            Odometry,
            '/sim_ground_truth_pose',
            self.pose_callback,
            qos_profile
        )

        self.lidar_points = lidar_points
        self.max_lidar_range = max_lidar_range
        self.lidar_data = np.zeros(self.lidar_points, dtype=np.float32)
        self._done = False  # Changed from True to False
        self._length = length
        self._step = 0
        self._received_scan = False
        self.first = False
        self.total_steps = 0
        self.highest_reward = -1.0
        self.lowest_reward = 1.0
        self.last_linear_velocity = 0.0
        self.turtle_position = (0.0, 0.0, 0.0)
        self.old_distance_to_goal = 100
        self.current_pitch = 0.0  # Initialize pitch angle
        self.current_roll = 0.0   # Initialize roll angle
        self.current_yaw = 0.0   # Initialize roll angle
        self.point_num = 0
        #self.point_nav_point = [( -14.0, -19.0 ), ( 0.0, 0.5 ), ( -5.0, 13.0), ( 0.0, 0.5 )]
        self.point_nav_point = [( -13.0, -20.0 ), ( -13, -4.5 )]
        
        # Check for actual connection to the robot
        self._robot_connected = self._check_robot_connection(timeout=connection_check_timeout)
        if not self._robot_connected:
            self.node.get_logger().warn("No actual robot detected. Running in simulation mode.")

    def step(self, action):
        if action['reset'] or self._done:
            self._step = 0
            self._done = False
            return self._obs(0.0, is_first=True)
        self.total_steps += 1
        
        twist = Twist()
        twist.linear.x = float(action['linear_velocity'])
        twist.angular.z = float(action['angular_velocity'])
        self.publisher.publish(twist)

        # Store the last linear velocity for reward calculation
        self.last_linear_velocity = twist.linear.x
        
        # Wait for some time to simulate step duration
        #time.sleep(0.1)
        #time.sleep(0.03)
        
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
        reward = self.get_reward()
        if reward > self.highest_reward:
            print('################################# new high reward of', reward)
            self.highest_reward = reward

        if reward < self.lowest_reward:
            self.lowest_reward = reward
            
        if self.total_steps % 1000 == 0:
            print('steps', self.total_steps,'highest reward',self.highest_reward,
                  'lowest reward',self.lowest_reward, 'current reward', reward)
            #print('lidar', self.lidar_data)
            print('Minimum lidar value:', np.nanmin(self.lidar_data))

        self._step += 1
        self._done = (self._step >= self._length)
        return self._obs(
            reward,
            is_last=self._done,
            is_terminal=self._done)

    def get_reward(self):
        """             bump      back        closer           goal
        Reward scaling -1.0, . . -0.001, 0.0 , 0.01, . . . .  . 1.0 
        """
        if self.lidar_data is None:
            return 0.0
        
        collision_threshold = 0.35  # Threshold distance is applied (e.g., 20 cm)
        reverse_threshold = 0.25    # Allow small backward movements
        
        if self.last_linear_velocity < reverse_threshold:
            print('too much reverse', self.last_linear_velocity)
            return -0.001

        min_distance = np.nanmin(self.lidar_data)  # Get the min distance to an obstacle 
        # Collision penalty if too close to an obstacle
        if min_distance < collision_threshold:
            self.node.get_logger().info(
                f"Step {self.total_steps}: Collision, Distance {min_distance}"
            )
            return -1.0  # negative reward for collision to strongly discourage it

        pointnav_reward = self.calc_point_nav_reward()
            
        return jnp.clip(pointnav_reward, -1, 1)


    def calc_point_nav_reward(self):
        # Check if we are close enough to goal for success
        current_goal = self.point_nav_point[self.point_num]
        distance_to_goal = l2_distance((self.turtle_position[0], self.turtle_position[1]),
                                       current_goal)

        if distance_to_goal < 0.3:  # success
            print('###################################  PointNav Goal achieved!')
            self.point_num += 1
            if self.point_num >= len(self.point_nav_point):
                self.point_num = 0
            self.old_distance_to_goal = l2_distance((self.turtle_position[0],
                                                     self.turtle_position[1]),
                                                    self.point_nav_point[self.point_num])
            self._done = True
            return 1.0  # max reward for achieving goal.

        # Calculate if we are closer
        if distance_to_goal < self.old_distance_to_goal:
            progress = self.old_distance_to_goal - distance_to_goal
            p_reward = progress * 10.0 # scaling factor for max 0.3
        else:
            p_reward = 0.005  # small neg reward for moving further

        time_penality = -0.03

        total_reward = p_reward + time_penality
        
        if self.total_steps % 100 == 0 or total_reward >= 1.0:
            print('linear_velocity', self.last_linear_velocity, 'pointnav_reward:',
                  p_reward, 'distance_to_goal:', distance_to_goal,
                  'self.old_distance_to_goal:', self.old_distance_to_goal, 'my x:',
                  self.turtle_position[0], 'my y:', self.turtle_position[1],
                  'goal:', current_goal, 'total_reward', total_reward)
            
        self.old_distance_to_goal = distance_to_goal  # Update the old distance

        return total_reward
    

    def _check_robot_connection(self, timeout):
        start_time = time.time()
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                return False
            if self._received_scan:
                return True
        return False


    def odom_callback(self, msg):
        # Get the current linear velocity from odometry data
        self.last_linear_velocity = msg.twist.twist.linear.x


    def pose_callback(self, msg):
        position = msg.pose.pose.position
        self.turtle_position = (position.x, position.y, position.z)

        # Extract quaternion components into a NumPy array for efficiency
        quat = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
                         msg.pose.pose.orientation.y, msg.pose.pose.orientation.z])
        # Log the received quaternion
        self.node.get_logger().debug(f"Received Quaternion: {quat}")
        
        # Compute the norm (magnitude) of the quaternion
        norm = np.linalg.norm(quat)
        if norm == 0:
            raise ValueError("Received a zero-length quaternion")
        
        # Normalize the quaternion to unit length
        quat_normalized = quat / norm
        
        # Log the normalized quaternion
        self.node.get_logger().debug(f"Normalized Quaternion: {quat_normalized}")
        
        # Convert the normalized quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = quat2euler(quat_normalized, axes='sxyz')
        
        # Update the turtle's current pitch and roll
        self.current_pitch = pitch
        self.current_roll = roll
        self.current_yaw = yaw         
        
        # Log the computed Euler angles
        self.node.get_logger().debug(f"Roll: {roll:.3f}, Pitch: {pitch:.3f}, Yaw: {yaw:.3f}")


    def lidar_callback(self, msg):
        if not self.first:
            print("First scan received:")
            print(f"Number of points: {len(msg.ranges)}")
            print(f"Angle min: {msg.angle_min}, Angle max: {msg.angle_max}")
            print(f"Angle increment: {msg.angle_increment}")
            print(f"Range min: {msg.range_min}, Range max: {msg.range_max}")
            print(f"First 20 ranges: {msg.ranges[:20]}")
            print(f"Last 20 ranges: {msg.ranges[-20:]}")
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
            'odom': embodied.Space(np.float32, (3,)),
            'imu': embodied.Space(np.float32, (3,)),
            'goal': embodied.Space(np.float32, (2,)),
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

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        obs = {
            'lidar': self.lidar_data,
            'odom': np.array(self.turtle_position, dtype=np.float32),
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw],
                            dtype=np.float32),
            'goal': np.array(self.point_nav_point[self.point_num], dtype=np.float32),# x, y tuple
        }
        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal)
        return obs

