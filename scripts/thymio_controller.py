import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from transforms3d import euler

import sys

class ControllerNode(Node):
    def __init__(self, *args, update_step=1/60, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.update_step = update_step 
        self.odom_pose = None
        self.odom_velocity = None
        self.name = 'thymio0'

        self.vel_publisher = self.create_publisher(Twist, f'/{self.name}/cmd_vel', 10)
        self.odom_subscriber = self.create_subscription(Odometry, f'/{self.name}/odom', self.pose_refresh_callback, 10)
       
        
    def start(self):
        self.timer = self.create_timer(self.update_step, self.refresh_callback)
    
    def stop(self):
        cmd_vel = Twist()
        self.vel_publisher.publish(cmd_vel)
    
    def pose_refresh_callback(self, msg):
        self.odom_pose = msg.pose.pose
        self.odom_valocity = msg.twist.twist
        
        pose2d = self.pose_convers(self.odom_pose)
    
    
    def pose_convers(self, pose3):
        quaternion = (
            pose3.orientation.x,
            pose3.orientation.y,
            pose3.orientation.z,
            pose3.orientation.w
        )
        
        roll, pitch, yaw = euler.quat2euler(quaternion)
        
        pose2D = (
            pose3.position.x,  
            pose3.position.y,  
            yaw                
        )
        
        return pose2D
        
    def refresh_callback(self):
        cmd_vel = Twist() 
        cmd_vel.linear.x  = 0.2
        cmd_vel.angular.z = 0.0
        
        self.vel_publisher.publish(cmd_vel)


def main():

    rclpy.init(args=sys.argv)
    

    node = ControllerNode()
    node.start()
    

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    
    node.stop()


if __name__ == '__main__':
    main()
