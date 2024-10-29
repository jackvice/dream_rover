import functools
import numpy as np
import rclpy
import cv2
import embodied
import time
import math
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from dreamerv3.utils import l2_distance
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from transforms3d.euler import quat2euler
from sensor_msgs.msg import Image


class Rover(embodied.Env):
    def __init__(self,
                 task,
                 size=(64, 64),
                 shape=(64, 64, 3),
                 repeat = 4,
                 length=200, #episode length
                 scan_topic='/scan',
                 imu_topic='/imu/data',
                 cmd_vel_topic='/cmd_vel',
                 odom_topic='/odometry/wheels',
                 camera_topic='/camera/image_raw',
                 connection_check_timeout=30,
                 lidar_points=640,
                 max_lidar_range=12.0,
                 environment_type='simualtion'):
        super().__init__()
        rclpy.init()
        self.bridge = CvBridge() 
        self.node = rclpy.create_node('turtlebot_controller')

        self.publisher = self.node.create_publisher(
            Twist,
            cmd_vel_topic,
            10
        )
        
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
            imu_topic,
            self.imu_callback,
            10
        )
        
        self.camera_subscriber = self.node.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            10
        )
        self.repeat = repeat
        self.image_size = size
        self.image_shape = shape
        self.current_image = None  # Store the current camera image
        self.lidar_points = lidar_points
        self.max_lidar_range = max_lidar_range
        self.lidar_data = np.zeros(self.lidar_points, dtype=np.float32)
        self._done = False  # Changed from True to False
        self._length = length
        self._step = 0
        self._received_scan = False
        self.first = False
        self.desired_distance = .5 #wall following
        self.total_steps = 0
        self.highest_reward = -1.0
        self.lowest_reward = 1.0
        self.last_linear_velocity = 0.0
        self.current_pitch = 0.0  # Initialize pitch angle
        self.current_roll = 0.0   # Initialize roll angle
        self.current_yaw = 0.0   # Initialize roll angle
        
        # Cooldown mechanism to prevent oscillations
        self.cooldown_steps = 100  # Number of steps to wait before allowing another correction
        self.steps_since_correction = self.cooldown_steps  # Initialize to allow immediate correction
        #self.point_nav_point = [( 0.5, 0.5 ), ( 0.5, 19.5 ), ( 19.5, 19.5 ), ( 19.5, 0.5 )]
        self.point_num = 0
        self.point_nav_point = [( -9.0, 9.0 ), ( 0.0, 0.0 ), ( -9.0, -9.0), ( 0.0, 0.0 )]
        #self.pos_x = 0
        #self.pos_y = 0
        self.rover_position = (0,0,0)
        self.old_distance_to_goal = 100 #some high starting number

        
        # Check for actual connection to the robot
        self._robot_connected = self._check_robot_connection(timeout=connection_check_timeout)
        if not self._robot_connected:
            self.node.get_logger().warn("No actual robot detected. Running in simulation mode.")

    def step(self, action):
        """
        Perform the specified action and return the new observation, reward, done flag, and additional info.
        Adds the current 64x64 camera image to the observation.
        
        :param action: Action to perform.
        :return: Tuple (observation, reward, done, info).
        """
        if action['reset'] or self._done:
            self._step = 0
            self._done = False
            return self._obs(0.0, is_first=True)
        self.total_steps += 1

        total_reward = 0  # Initialize total reward for action repeat
        for _ in range(self.repeat):
            # **New Logic: Check for climbing and adjust movement with cooldown**
            climbing_status, climbing_severity = self.is_climbing_wall()  # Modify function to return severity if needed
            if climbing_status and self.steps_since_correction >= self.cooldown_steps:
                # Execute corrective action instead of agent's action
                twist = Twist()
                if climbing_status == 'forward':
                    # Execute a small reverse movement with angular correction
                    twist.linear.x = -0.1  # Reduced reverse velocity
                    twist.angular.z = -self.current_roll * 1.0  # Example angular correction
                    self.node.get_logger().info(
                        f"Step {self.total_steps}: Forward climb. doing reverse. angle{self.current_pitch}"
                    )
                elif climbing_status == 'reverse':
                    # Execute a small forward movement with angular correction
                    twist.linear.x = 0.1  # Reduced forward velocity
                    twist.angular.z = self.current_roll * 1.0  # Example angular correction
                    self.node.get_logger().info(
                        f"Step {self.total_steps}: Reverse climb. doing forward. angle{self.current_pitch}"
                    )
                self.publisher.publish(twist)
                # Reset cooldown
                self.steps_since_correction = 0
            else:
                # Normal operation: publish agent's action twist
                twist = Twist()
                twist.linear.x = float(action['linear_velocity'])
                twist.angular.z = float(action['angular_velocity'])
                self.publisher.publish(twist)
                # Store the last linear velocity for reward calculation
                self.last_linear_velocity = twist.linear.x
                # Increment cooldown counter
                self.steps_since_correction += 1

            # Process sensor data
            if self._robot_connected:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if not self._received_scan:
                    self.node.get_logger().warn("No scan data received")
            else:
                self.lidar_data = np.random.uniform(0.1, self.max_lidar_range,
                                                    self.lidar_points).astype(np.float32)

            # Calculate reward and accumulate it for each repeated step
            reward = self.get_reward()
            total_reward += reward

            # Check if maximum steps have been reached
            self._step += 1
            if self._step >= self._length:
                self._done = True
                break  # Exit loop early if done

        # Update highest and lowest rewards for logging
        if total_reward > self.highest_reward:
            self.highest_reward = total_reward
        if total_reward < self.lowest_reward:
            self.lowest_reward = total_reward
        if self.total_steps % 1000 == 0:
            print('steps', self.total_steps, 'highest reward', self.highest_reward,
                  'lowest reward', self.lowest_reward, 'current reward', total_reward)
            print('Minimum lidar value:', np.nanmin(self.lidar_data))

        return self._obs(
            total_reward,
            is_last=self._done,
            is_terminal=self._done)


    def step_no_repeat(self, action):
        """
        Perform the specified action and return the new observation, reward, done flag, and additional info.
        Adds the current 96x96 camera image to the observation.
        
        :param action: Action to perform.
        :return: Tuple (observation, reward, done, info).
        """
        if action['reset'] or self._done:
            self._step = 0
            self._done = False
            return self._obs(0.0, is_first=True)
        self.total_steps += 1

        # **New Logic: Check for climbing and adjust movement with cooldown**
        climbing_status, climbing_severity = self.is_climbing_wall()  # Modify function to return severity if needed
        if climbing_status and self.steps_since_correction >= self.cooldown_steps:
            # Execute corrective action instead of agent's action
            twist = Twist()
            if climbing_status == 'forward':
                # Execute a small reverse movement with angular correction
                twist.linear.x = -0.1  # Reduced reverse velocity
                twist.angular.z = -self.current_roll * 1.0  # Example angular correction
                self.node.get_logger().info(
                    f"Step {self.total_steps}: Forward climb. doing reverse. angle{self.current_pitch}"
                )
            elif climbing_status == 'reverse':
                # Execute a small forward movement with angular correction
                twist.linear.x = 0.1  # Reduced forward velocity
                twist.angular.z = self.current_roll * 1.0  # Example angular correction
                self.node.get_logger().info(
                    f"Step {self.total_steps}: Reverse climb. doiing forward. angle{self.current_pitch}"
                )
            self.publisher.publish(twist)
            # Reset cooldown
            self.steps_since_correction = 0
        else:
            # Normal operation: publish agent's action twist
            twist = Twist()
            twist.linear.x = float(action['linear_velocity'])
            twist.angular.z = float(action['angular_velocity'])
            self.publisher.publish(twist)
            # Store the last linear velocity for reward calculation
            self.last_linear_velocity = twist.linear.x
            # Increment cooldown counter
            self.steps_since_correction += 1

        # Process sensor data
        if self._robot_connected:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if not self._received_scan:
                self.node.get_logger().warn("No scan data received")
        else:
            self.lidar_data = np.random.uniform(0.1, self.max_lidar_range,
                                                self.lidar_points).astype(np.float32)

        # Calculate reward
        reward = self.get_reward()
        # Update highest and lowest rewards for logging
        if reward > self.highest_reward:
            self.highest_reward = reward
        if reward < self.lowest_reward:
            self.lowest_reward = reward
        if self.total_steps % 1000 == 0:
            print('steps', self.total_steps, 'highest reward', self.highest_reward,
                  'lowest reward', self.lowest_reward, 'current reward', reward)
            print('Minimum lidar value:', np.nanmin(self.lidar_data))

        self._step += 1
        self._done = (self._step >= self._length)
        return self._obs(
            reward,
            is_last=self._done,
            is_terminal=self._done)


    def get_reward(self):
        """
        Calculate the reward based on the current state of the rover.
        
        :return: The calculated reward.
        """
        if self.lidar_data is None:
            return 0.0
        
        collision_threshold = 0.35  # Threshold distance is applied (e.g., 20 cm)
        reverse_threshold = 0.0    # Allow small backward movements
        
        if self.last_linear_velocity < reverse_threshold:
            return -0.001

        min_distance = np.nanmin(self.lidar_data)  # Get the min distance to an obstacle 
        # Collision penalty if too close to an obstacle
        if min_distance < collision_threshold:
            self.node.get_logger().info(
                f"Step {self.total_steps}: Collision, Distance {min_distance}"
            )
            return -0.4  # small negative reward for collision to strongly discourage it

        #pointnav_reward = self.calc_point_nav_reward()
        # Wall-following reward
        wall_reward = self.calc_wall_following_reward()
        
        #return pointnav_reward
        return wall_reward

    def calc_wall_following_reward(self):
        desired_distance = 0.5  # Desired distance from the wall on the right (in meters)
        lidar_ranges = self.lidar_data  # Assuming this is a NumPy array
        num_readings = len(lidar_ranges)
        
        # Define the angle range for the right side
        # Right side corresponds to angles around 270 degrees
        # Since LiDAR scans from front (0 degrees) and increases counter-clockwise,
        # right side indices are around index 480 (270 degrees)
        
        # Determine indices for right side (e.g., from 250 degrees to 290 degrees)
        right_start_angle_deg = 250
        right_end_angle_deg = 290
        
        # Degrees per index
        degrees_per_index = 360 / num_readings
        
        # Convert degrees to indices
        right_start_idx = int(right_start_angle_deg / degrees_per_index) % num_readings
        right_end_idx = int(right_end_angle_deg / degrees_per_index) % num_readings
        
        if right_start_idx <= right_end_idx:
            right_side_indices = np.arange(right_start_idx, right_end_idx + 1)
        else:
            # Handle wrapping around the array
            right_side_indices = np.concatenate((
                np.arange(right_start_idx, num_readings),
                np.arange(0, right_end_idx + 1)
            ))
            
        # Extract distances on the right side
        right_distances = lidar_ranges[right_side_indices]
        right_distances = right_distances[np.isfinite(right_distances)]  # Remove inf values
        
        if len(right_distances) == 0:
            return 0.0  # No valid data on the right side
        
        average_distance = np.mean(right_distances)

        # Calculate reward based on how close the average_distance is to the desired_distance
        error = np.abs(average_distance - desired_distance)
        max_error = lidar_ranges[np.isfinite(lidar_ranges)].max() - desired_distance
        if max_error == 0:
            max_error = 1e-6  # Prevent division by zero

        # Normalize error to be between 0 and 1
        normalized_error = error / max_error
        normalized_error = np.clip(normalized_error, 0.0, 1.0)

        # Invert to make reward decrease with increasing error
        distance_reward = 1.0 - normalized_error

        return distance_reward

    def camera_callback(self, msg: Image):
        """
        Callback function for processing incoming camera images. Stores the most recent image in 96x96 size.
        
        :param msg: Image message from the camera.
        """
        try:
            # Convert the ROS Image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Resize the image to 96x96
            resized_image = cv2.resize(cv_image, self.image_size)
            # Store the resized image as the current observation
            self.current_image = resized_image
        except Exception as e:
            self.node.get_logger().error(f"Failed to process image: {str(e)}")


            
    def calc_point_nav_reward(self):
        # Check if we are close enough to goal for success
        current_goal = self.point_nav_point[self.point_num]
        distance_to_goal = l2_distance((self.rover_position[0], self.rover_position[1]), current_goal)

        if distance_to_goal < 0.3:  # success
            print('PointNav Goal achieved!')
            self.point_num += 1
            if self.point_num >= len(self.point_nav_point):
                self.point_num = 0
            self.old_distance_to_goal = 10  # Reset to high value for the next goal
            return 1.0  # max reward for achieving goal.

        # Calculate if we are closer
        if distance_to_goal < self.old_distance_to_goal:
            p_reward = 0.1
        else:
            p_reward = 0.0  # no reward for moving further

        self.old_distance_to_goal = distance_to_goal  # Update the old distance

        if self.total_steps % 10000 == 0:
            print('pointnav_reward:', p_reward, 'distance_to_goal:', distance_to_goal,
                  'self.old_distance_to_goal:', self.old_distance_to_goal, 'my x:',
                  self.rover_position[0], 'my y:', self.rover_position[1], 'goal:', current_goal)
        return p_reward
    
    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        """
        Create and return the current observation of the environment.
        
        :param reward: The reward obtained.
        :param is_first: Whether this is the first step of an episode.
        :param is_last: Whether this is the last step of an episode.
        :param is_terminal: Whether this is a terminal state.
        :return: The current observation.
        """
        obs = {
            'camera_image': self.current_image if self.current_image is not None else np.zeros(self.image_shape, dtype=np.uint8),
            #'lidar': self.lidar_data,
            'odom': np.array(self.rover_position, dtype=np.float32),
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw], dtype=np.float32),
            'goal': np.array(self.point_nav_point[self.point_num], dtype=np.float32),
        }
        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal)
        return obs
    

    def imu_callback(self, msg):
        """
        Processes incoming IMU data by converting quaternion orientation to Euler angles.
        Updates the current pitch and roll of the rover.
        """
        try:
            # Extract quaternion components into a NumPy array for efficiency
            quat = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        
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


    def odom_callback(self, msg):
        # Get the current linear velocity from odometry data
        self.rover_position = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
        #self.pos_x = msg.pose.pose.position.x
        #self.pos_y = msg.pose.pose.position.y
        self.last_linear_velocity = msg.twist.twist.linear.x


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
            'camera_image': embodied.Space(np.uint8, self.image_shape),
            #'lidar': embodied.Space(np.float32, (self.lidar_points,)),
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
            'linear_velocity': embodied.Space(np.float32, (),-0.2, 0.3), # -1, 1),
            'angular_velocity': embodied.Space(np.float32, (), -1, 1),
            'reset': embodied.Space(bool),
        }


    
    def _check_robot_connection(self, timeout):
        start_time = time.time()
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                return False
            if self._received_scan:
                return True
        return False

    def is_climbing_wall(self):
        """
        Determines if the rover is climbing a wall based on LIDAR data and IMU orientation.
        Returns:
        tuple: (climbing_status, severity)
           climbing_status: 'forward', 'reverse', or False
           severity: float indicating the degree of climbing
        """
        if self.lidar_data is None:
            return False, 0.0
        
        min_distance = np.nanmin(self.lidar_data)
        collision_threshold = 0.2  # meters
        pitch_threshold = 0.2      # radians (~11.5 degrees)
        roll_threshold = 0.2       # radians (~11.5 degrees)
        #if self.total_steps % 10 == 0:
        #    print('pitch', self.current_pitch)
        
        is_too_close = min_distance < collision_threshold
        is_pitch_steep = abs(self.current_pitch) > pitch_threshold
        is_roll_steep = abs(self.current_roll) > roll_threshold
        
        climbing_status = False
        severity = 0.0
        
        if is_too_close and (is_pitch_steep or is_roll_steep):
            if self.current_pitch > pitch_threshold:
                # Climbing forward
                climbing_status = 'reverse'
                severity = self.current_pitch
            elif self.current_pitch < -pitch_threshold:
                # Climbing backward
                climbing_status = 'forward'
                severity = abs(self.current_pitch)
            else:
                # If pitch is not steep but roll is
                if self.current_roll > roll_threshold:
                    climbing_status = 'right_tilt'
                    severity = self.current_roll
                elif self.current_roll < -roll_threshold:
                    climbing_status = 'left_tilt'
                    severity = abs(self.current_roll)

        return climbing_status, severity

        
    def get_reward_old(self):
        if self.lidar_data is None:
            return 0.0
        
        forward_velocity = self.last_linear_velocity
        collision_threshold = 0.2  # Threshold distance is applied (e.g., 20 cm)
        reverse_threshold = -0.1  # Allow small backward movements
        
        if forward_velocity < reverse_threshold:
            return 0.0

        min_distance = np.nanmin(self.lidar_data) # Get the min distance to an obstacle 
        # Collision penalty if too close to an obstacle
        if min_distance < collision_threshold:
            self.node.get_logger().info(
                f"Step {self.total_steps}: Collision, Distance {min_distance}"
            )
            return -0.1  # small negative reward for collision to strongly discourage it

        return point_nav_reward()
        
        # Encourage forward movement
        max_velocity = self.act_space['linear_velocity'].high
        if max_velocity == 0:
            max_velocity = 1e-6  # Prevent division by zero

        distance_reward = wall_distance_reward(min_distance)
        
        # Normalize the forward velocity to [0, 1]
        normalized_velocity = forward_velocity / max_velocity
        normalized_velocity = np.clip(normalized_velocity, 0.0, 1.0)
        
        # Weight factors for distance and velocity rewards
        alpha = 0.5  # Weight for distance reward
        beta = 0.4   # Weight for velocity reward
        combined_reward = alpha * distance_reward + beta * normalized_velocity


        # Additional penalty for climbing
 
        if self.total_steps % 10000 == 0:
            print('alpha', alpha, '* distance_reward', round(distance_reward, 3),
                  '   +     beta', beta, '* normalized_velocity', normalized_velocity,)
        combined_reward = np.clip(combined_reward, -1.0, 1.0) # clip just in case
        return combined_reward
    
    def wall_distance_reward(self, min_distance):
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
        return distance_reward




