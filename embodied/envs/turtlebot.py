import functools
import numpy as np
import embodied
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class Turtlebot(embodied.Env):
    def __init__(self, task, size=(64, 64), length=100):
        super().__init__()
        rclpy.init()
        self.node = rclpy.create_node('turtlebot_controller')
        self.publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.lidar_subscriber = self.node.create_subscription(LaserScan, '/scan', 
                                                              self.lidar_callback, 10)
        self.lidar_data = np.zeros(270)  # Initial lidar data placeholder
        self._done = True
        self._length = length
        self._step = 0

    def lidar_callback(self, msg):
        self.lidar_data = np.array(msg.ranges[:270])  # Assuming 270 lidar points

    @functools.cached_property
    def obs_space(self):
        spaces = {
            'lidar': embodied.Space(np.float32, (270,)),
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


class Turtlebot_old(embodied.Env):
    def __init__(self, task, size=(64, 64), length=100):
        #print("Turtlebot.__init__ kwargs:", kwargs)
        super().__init__()
        rclpy.init()
        self.node = rclpy.create_node('turtlebot_controller')
        self.publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.lidar_subscriber = self.node.create_subscription(LaserScan, '/scan',
                                                              self.lidar_callback, 10)
        self.lidar_data = np.zeros(270)  # Initial lidar data placeholder
        self._done = False
        self._step = 0
        self._length = length
        self._reward = 0  # Initialize reward

    def lidar_callback(self, msg):
        self.lidar_data = np.array(msg.ranges[:270])  # Assuming 270 lidar

    def step(self, action):
        #obs, reward, self._done, self._info = self._env.step(action)
        if action['reset'] or self._done:
            self._step = 0
            self._done = False
            return self._obs(0, is_first=True)

        twist = Twist()
        twist.linear.x = action['linear_velocity']
        twist.angular.z = action['angular_velocity']
        self.publisher.publish(twist)

        self._step += 1
        self._done = (self._step >= self._length)
        self._reward = self._calculate_reward()  # Calculate reward based on your task
        return self._obs(self._reward, is_last=self._done, is_terminal=self._done)

    @property
    def act_space(self):
        return {
            'linear_velocity': embodied.Space(np.float32, (), -1, 1),
            'angular_velocity': embodied.Space(np.float32, (), -1, 1),
            'reset': embodied.Space(bool),
        }

    @property
    def obs_space(self):
        return {
            'lidar': embodied.Space(np.float32, (270,)),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        return dict(
            lidar=self.lidar_data,
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
    def _calculate_reward(self):
        # Implement your reward calculation here
        # For now, let's return a dummy reward
        return 0.0


class TurtlebotOlder(embodied.Env):
    def __init__(self, task, size=(64, 64), length=100):
        super().__init__()
        self.publisher = None
        rclpy.init()
        self.node = rclpy.create_node('turtlebot_controller')
        self.publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)

    def step(self, action):
        if action['reset'] or self._done:
            self._step = 0
            self._done = False
            return self._obs(0, is_first=True)

        twist = Twist()
        twist.linear.x = action['linear_velocity']
        twist.angular.z = action['angular_velocity']
        self.publisher.publish(twist)

        self._step += 1
        self._done = (self._step >= self._length)
        return self._obs(1, is_last=self._done, is_terminal=self._done)

    @property
    def act_space(self):
        return {
            'action': embodied.Space(np.float32, (), 0, 2),
            'other': embodied.Space(np.float32, (6,)),
            'reset': embodied.Space(bool),
        }

    @property
    def obs_space(self):
        return {
            'lidar': embodied.Space(np.float32, (270,)),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }
    
    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        return dict(
            lidar=np.zeros(270, np.float32),
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
