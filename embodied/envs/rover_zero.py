import functools
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
import embodied
import time
from nav_msgs.msg import Odometry
import math

from transforms3d.euler import quat2euler

class Rover(embodied.Env):
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
            self.imu_callback,
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
        self.desired_distance = .5
        self.total_steps = 0
        self.highest_reward = -1.0
        self.lowest_reward = 1.0
        self.last_linear_velocity = 0.0
        self.current_pitch = 0.0  # Initialize pitch angle
        self.current_roll = 0.0   # Initialize roll angle

        # Cooldown mechanism to prevent oscillations
        self.cooldown_steps = 100  # Number of steps to wait before allowing another correction
        self.steps_since_correction = self.cooldown_steps  # Initialize to allow immediate correction

        
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
            'linear_velocity': embodied.Space(np.float32, (),-0.2, 0.3), # -1, 1),
            'angular_velocity': embodied.Space(np.float32, (), -1, 1),
            'reset': embodied.Space(bool),
        }




    def step(self, action):
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
                twist.linear.x = -0.05  # Reduced reverse velocity
                twist.angular.z = -self.current_roll * 1.0  # Example angular correction
                self.node.get_logger().info(
                    f"Step {self.total_steps}: Forward climbing detected. Executing small reverse movement."
                )
            elif climbing_status == 'reverse':
                # Execute a small forward movement with angular correction
                twist.linear.x = 0.05  # Reduced forward velocity
                twist.angular.z = self.current_roll * 1.0  # Example angular correction
                self.node.get_logger().info(
                    f"Step {self.total_steps}: Reverse climbing detected. Executing small forward movement."
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


    
    def stepold(self, action):
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
        #time.sleep(0.025)
        

        if self._robot_connected:
            # Wait for scan data only if robot is connected
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if not self._received_scan:
                self.node.get_logger().warn("No scan data received")
        else:
            # Simulate more realistic scan data if no robot is connected
            self.lidar_data = np.random.uniform(0.1, self.max_lidar_range,
                                                self.lidar_points).astype(np.float32)

        # **New Logic: Check for climbing and adjust movement with cooldown**
        climbing_status, climbing_severity = self.is_climbing_wall()  # Modify function to return severity if needed
        if climbing_status and self.steps_since_correction >= self.cooldown_steps:
            corrective_twist = Twist()
            if climbing_status == 'forward':
                # Execute a small reverse movement with angular correction
                corrective_twist.linear.x = -0.05  # Reduced reverse velocity
                corrective_twist.angular.z = -self.current_roll * 1.0  # Example angular correction
                self.publisher.publish(corrective_twist)
                self.node.get_logger().info(
                    f"Step {self.total_steps}: Forward climbing detected. Executing small reverse movement."
                )
            elif climbing_status == 'reverse':
                # Execute a small forward movement with angular correction
                corrective_twist.linear.x = 0.05  # Reduced forward velocity
                corrective_twist.angular.z = self.current_roll * 1.0  # Example angular correction
                self.publisher.publish(corrective_twist)
                self.node.get_logger().info(
                f"Step {self.total_steps}: Reverse climbing detected. Executing small forward movement."
                )
        
            # Reset cooldown
            self.steps_since_correction = 0
        else:
            self.steps_since_correction += 1


        # Calculate reward
        reward = self.get_reward()
        if reward > self.highest_reward:
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
        
        forward_velocity = self.last_linear_velocity
        collision_threshold = 0.2  # Threshold distance is applied (e.g., 20 cm)
        reverse_threshold = -0.1  # Allow small backward movements
        
        if forward_velocity < reverse_threshold:
            return 0.0

        min_distance = np.nanmin(self.lidar_data) # Get the min distance to an obstacle 
        # Collision penalty if too close to an obstacle
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
        max_velocity = self.act_space['linear_velocity'].high
        if max_velocity == 0:
            max_velocity = 1e-6  # Prevent division by zero

        # Normalize the forward velocity to [0, 1]
        normalized_velocity = forward_velocity / max_velocity
        normalized_velocity = np.clip(normalized_velocity, 0.0, 1.0)
        
        # Weight factors for distance and velocity rewards
        alpha = 0.5  # Weight for distance reward
        beta = 0.4   # Weight for velocity reward
        combined_reward = alpha * distance_reward + beta * normalized_velocity


        # Additional penalty for climbing
 
        if self.total_steps % 1000 == 0:
            print('alpha', alpha, '* distance_reward', round(distance_reward, 3),
                  '   +     beta', beta, '* normalized_velocity', normalized_velocity,)
        combined_reward = np.clip(combined_reward, -1.0, 1.0) # clip just in case
        return combined_reward


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
        print('pitch', self.current_pitch)
        
        is_too_close = min_distance < collision_threshold
        is_pitch_steep = abs(self.current_pitch) > pitch_threshold
        is_roll_steep = abs(self.current_roll) > roll_threshold
        
        climbing_status = False
        severity = 0.0
        
        if is_too_close and (is_pitch_steep or is_roll_steep):
            if self.current_pitch > pitch_threshold:
                # Climbing forward
                climbing_status = 'forward'
                severity = self.current_pitch
            elif self.current_pitch < -pitch_threshold:
                # Climbing backward
                climbing_status = 'reverse'
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


    
    def is_climbing_wallOld(self):
        """
        Determine if the rover is climbing a wall based on lidar data and IMU orientation.
        Returns:
            str: 'forward' if climbing while moving forward,
                 'reverse' if climbing while moving in reverse,
                 False otherwise.
        """
        if self.lidar_data is None:
            return False
        
        min_distance = np.nanmin(self.lidar_data)
        collision_threshold = 0.2  # Threshold distance in meters
        pitch_threshold = 0.2  # Threshold pitch in radians (~5.7 degrees)
        roll_threshold = 0.1  # Threshold roll in radians (~5.7 degrees)
        
        # Define criteria for climbing
        is_too_close = min_distance < collision_threshold
        is_pitch_steep = abs(self.current_pitch) > pitch_threshold
        is_roll_steep = abs(self.current_roll) > roll_threshold
        is_moving_forward = self.last_linear_velocity > 0
        is_moving_reverse = self.last_linear_velocity < 0
        
        # Climbing while moving forward
        if is_too_close and (is_pitch_steep or is_roll_steep) and is_moving_forward:
            return 'forward'
        
        # Climbing while moving in reverse
        if is_too_close and (is_pitch_steep or is_roll_steep) and is_moving_reverse:
            return 'reverse'
        
        return False
    
    def imu_callbackOld(self, msg):
        """
        Processes incoming IMU data by converting quaternion orientation to Euler angles.
        Updates the current pitch and roll of the rover.
        """
        try:
            # Extract quaternion components
            quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        
        
            # Normalize the quaternion
            norm = np.linalg.norm(quat)
            if norm == 0:
                raise ValueError("Received a zero-length quaternion")
            quat_normalized = quat / norm
        

            # Convert to Euler angles (roll, pitch, yaw)
            roll, pitch, yaw = quat2euler(*quat_normalized, axes='sxyz')
        
            # Update rover's pitch and roll
            self.current_pitch, self.current_roll = pitch, roll
        
            # Log Euler angles
            self.node.get_logger().debug(f"Euler Angles - Roll: {roll:.3f}, Pitch: {pitch:.3f}, Yaw: {yaw:.3f}")
        
        except Exception as e:
            self.node.get_logger().error(f"Error processing IMU data: {e}")


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
        
            # Log the computed Euler angles
            self.node.get_logger().debug(f"Euler Angles - Roll: {roll:.3f}, Pitch: {pitch:.3f}, Yaw: {yaw:.3f}")
        
        except Exception as e:
            self.node.get_logger().error(f"Error processing IMU data: {e}")


"""
        # **New Logic: Check for climbing and adjust movement with cooldown**
        climbing_status = self.is_climbing_wall()
        if climbing_status and self.steps_since_correction >= self.cooldown_steps:
            if climbing_status == 'forward':
                # Execute a small reverse movement
                corrective_twist = Twist()
                corrective_twist.linear.x = -0.1  # Small reverse velocity
                corrective_twist.angular.z = 0.0
                self.publisher.publish(corrective_twist)
                self.node.get_logger().info(
                    f"Step {self.total_steps}: Forward climbing detected. doing small reverse movement."
                )
            elif climbing_status == 'reverse':
                # Execute a small forward movement
                corrective_twist = Twist()
                corrective_twist.linear.x = 0.1  # Small forward velocity
                corrective_twist.angular.z = 0.0
                self.publisher.publish(corrective_twist)
                self.node.get_logger().info(
                    f"Step {self.total_steps}: Reverse climbing detected. doing small forward movement."
                )
            
            # Reset cooldown
            self.steps_since_correction = 0
        else:
            self.steps_since_correction += 1


"""
