import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class PoseSubscriber(Node):
    def __init__(self):
        super().__init__('pose_subscriber')
        self.position = [0.0, 0.0, 0.0]
        self.total_steps = 0

        # Define QoS profile
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Create the subscription using the custom QoS profile
        self.pose_subscription = self.create_subscription(
            Odometry,
            '/sim_ground_truth_pose',
            self.pose_callback,
            qos_profile
        )
        print("Subscription successfully created")

    def pose_callback(self, msg):
        print("Pose callback triggered")
        position = msg.pose.pose.position
        print(f"Received position - x: {position.x}, y: {position.y}, z: {position.z}")
        self.position[0] = position.x
        self.position[1] = position.y
        self.position[2] = position.z

        self.total_steps += 1
        if self.total_steps % 10 == 0:
            print('#########position', self.position[0], self.position[1], self.position[2])

def main(args=None):
    rclpy.init(args=args)
    pose_subscriber = PoseSubscriber()
    rclpy.spin(pose_subscriber)
    pose_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
