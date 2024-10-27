import functools
import numpy as np
import rclpy
from geometry_msgs.msg import Twist #, Pose, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
import embodied
import time
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from transforms3d.euler import quat2euler


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
        self.imu_subscriber = self.node.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback, 10
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
        self.desired_distance = 1.3
        self.total_steps = 0
        self.highest_reward = -1.0
        self.lowest_reward = 1.0
        self.last_linear_velocity = 0.0
        self.position = [0.0, 0.0, 0.0]
        
        # Check for actual connection to the robot
        self._robot_connected = self._check_robot_connection(timeout=connection_check_timeout)
        if not self._robot_connected:
            self.node.get_logger().warn("No actual robot detected. Running in simulation mode.")



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
        robot_quat = msg.pose.pose.orientation
        self.position[0] = position.x
        self.position[1] = position.y
        self.position[2] = position.z


        
    def pose_callbackbroke(self, msg):
        # Get the current linear velocity from odometry data
        
        self.position = msg.pose.pose.position


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
        # Check if we need to reset the environment
        if action['reset'] or self._done:
            self._step = 0
            self._done = False
            return self._obs(0.0, is_first=True)
        
        # Increment total steps
        self.total_steps += 1
        
        # Create and publish Twist message for robot movement
        twist = Twist()
        twist.linear.x = float(action['linear_velocity'])
        twist.angular.z = float(action['angular_velocity'])
        self.publisher.publish(twist)

        # Store the last linear velocity for reward calculation
        self.last_linear_velocity = twist.linear.x
        
        # Wait for some time to simulate step duration
        #time.sleep(0.1)
        time.sleep(0.03)
        
        # Handle LIDAR data
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
        
        # Update highest and lowest rewards
        if reward > self.highest_reward:
            self.highest_reward = reward
        if reward < self.lowest_reward:
            self.lowest_reward = reward
            
        # Print debug information every 1000 steps
        if self.total_steps % 1000 == 0:
            print('steps', self.total_steps,'highest reward',self.highest_reward,
                  'lowest reward',self.lowest_reward, 'current reward', reward)
            #print('lidar', self.lidar_data)
            print('Minimum lidar value:', np.nanmin(self.lidar_data))

        # Increment step counter and check if episode is done
        self._step += 1
        self._done = (self._step >= self._length)
        
        # Return observation
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

    def get_reward(self):
        if self.lidar_data is None:
            return 0.0
        good_distance = 0.1
        
        # Get the minimum distance to an obstacle from the LIDAR data
        min_distance = np.nanmin(self.lidar_data)

        # Collision penalty if too close to an obstacle
        collision_threshold = 0.35  # Threshold distance is applied (e.g., 20 cm)
        if min_distance < collision_threshold:
            return -0.1  # small negative reward for collision to strongly discourage it

        
        # Calculate reward based on how close the min_distance is to the desired_distance
        error = np.abs(min_distance - self.desired_distance)
        max_error = self.max_lidar_range - self.desired_distance
        if max_error == 0:
            max_error = 1e-6  # Prevent division by zero

        # Normalize error to be between 0 and 1
        normalized_error = error / max_error
        normalized_error = np.clip(normalized_error, 0.0, 1.0)

        # Invert and shift to make reward range between 0 and 1
        distance_reward = 1.0 - normalized_error

        # Encourage forward movement
        forward_velocity = self.last_linear_velocity
        max_velocity = self.act_space['linear_velocity'].high
        if max_velocity == 0:
            max_velocity = 1e-6  # Prevent division by zero

        # Normalize the forward velocity to [0, 1]
        normalized_velocity = forward_velocity / max_velocity
        normalized_velocity = np.clip(normalized_velocity, 0.0, 1.0)
        
        # Weight factors for distance and velocity rewards
        alpha = 0.5  # Weight for distance reward
        beta = 0.4   # Weight for velocity reward

        # Combined reward
        combined_reward = alpha * distance_reward + beta * normalized_velocity
        if 1.1 < min_distance < 1.5:
            combined_reward += good_distance

        
        if self.total_steps % 1000 == 0:
            print('alpha', alpha, '* distance_reward', round(distance_reward, 3),
                  '   +     beta', beta, '* normalized_velocity', normalized_velocity,)
        combined_reward = np.clip(combined_reward, -1.0, 1.0) # clip just in case
        return combined_reward

    
    def get_reward_right_side_wall(self):
        # Ensure angle information is available
        if not hasattr(self, 'angle_min'):
            return 0.0  # Cannot compute reward without angle information

        # Compute the angles for each lidar point
        angles = np.arange(self.lidar_points) * self.angle_increment + self.angle_min

        # Set parameters for wall-following on the right side
        side_angle = -np.pi / 2  # Right side (-90 degrees)
        angle_tolerance = np.pi / 4  # 45 degrees

        # Find indices of lidar points within the desired angle range
        side_indices = np.where(np.abs(angles - side_angle) <= angle_tolerance)[0]

        # If no points are found, set distance_reward to zero
        if len(side_indices) == 0:
            distance_reward = 0.0
        else:
            # Get distances at those indices
            side_distances = self.lidar_data[side_indices]

            # Filter out invalid distances
            valid_distances = side_distances[(side_distances > 0.1) &
                                         (side_distances < self.max_lidar_range)]
            if len(valid_distances) == 0:
                distance_reward = 0.0
            else:
                # Calculate the mean distance to the wall
                mean_distance = np.mean(valid_distances)
                
                # Calculate the error from the desired distance
                error = mean_distance - self.desired_distance

                # Define maximum acceptable error
                max_error = 1.0  # Assume 1 meter is a reasonable maximum error for wall-following

                # Normalize error to be between -1 and 1
                normalized_error = error / max_error
                normalized_error = np.clip(normalized_error, -1.0, 1.0)

                # Calculate reward with a sharper penalty for deviation
                distance_reward = np.exp(-np.abs(normalized_error) * 3)

        # Encourage forward movement
        forward_velocity = self.last_linear_velocity
        max_velocity = self.act_space['linear_velocity'].high
        if max_velocity == 0:
            max_velocity = 1e-6  # Prevent division by zero

        # Normalize the forward velocity to [0, 1]
        normalized_velocity = forward_velocity / max_velocity
        normalized_velocity = np.clip(normalized_velocity, 0.0, 1.0)

        # Weight factors for distance and velocity rewards
        alpha = 0.5  # Weight for distance reward
        beta = 0.5   # Weight for velocity reward

        # Combined reward
        combined_reward = (alpha * distance_reward) + (beta * normalized_velocity)

        # Collision penalty - penalize if the robot is too close to any obstacle
        collision_threshold = 0.2  # Define a threshold for collision (e.g., 20 cm)
        min_distance = np.nanmin(self.lidar_data)  # Get the minimum distance from the LIDAR data
        if min_distance < collision_threshold:
            collision_penalty = -0.5  # Apply a negative reward for being too close to an obstacle
            combined_reward += collision_penalty

        # Print debug info every 1000 steps
        if self.total_steps % 1000 == 0:
            print('alpha', alpha, '* distance_reward', round(distance_reward, 3),
                  '   +     beta', beta, '* normalized_velocity', normalized_velocity,
                  '   +     collision_penalty',
                  collision_penalty if min_distance < collision_threshold else 0)

        return combined_reward

    def imu_callback(self, msg):
        """
        Processes incoming IMU data by converting quaternion orientation to Euler angles.
        Updates the current pitch and roll of the rover.
        """
        try:
            # Extract quaternion components into a NumPy array for efficiency
            quat = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y,
                             msg.orientation.z])
        
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
        
            # Update the rover's current pitch and roll
            self.current_pitch = pitch
            self.current_roll = roll
            self.current_yaw = yaw
            
        
            # Log the computed Euler angles
            self.node.get_logger().debug(f"Euler Angles - Roll: {roll:.3f}, Pitch: {pitch:.3f}, Yaw: {yaw:.3f}")
        
        except Exception as e:
            self.node.get_logger().error(f"Error processing IMU data: {e}")