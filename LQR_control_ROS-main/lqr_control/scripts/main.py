#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import numpy as np
from scipy.integrate import odeint
from lqr_controller import lqr
from robot_dynamics import robot_dynamics
from tf.transformations import quaternion_from_euler

def main():
    rospy.init_node('lqr_control')

    # System parameters
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
    B = np.array([[0], [0], [0], [1]])

    Q = np.eye(4)
    R = np.array([[1]])

    # Initial state
    y0 = np.array([0, 0, 0.5, 0])  # [x, x_dot, theta, theta_dot]

    # Time span
    t = np.linspace(0, 10, 100)

    # Design LQR controller
    K = lqr(A, B, Q, R)

    # Solve the system
    sol = odeint(robot_dynamics, y0, t, args=(A, B, K))

    # ROS publishers
    pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    pub_odom = rospy.Publisher('/odom', Odometry, queue_size=10)
    pub_marker = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        for i, state in enumerate(sol):
            twist_msg = Twist()
            twist_msg.linear.x = state[1]
            twist_msg.angular.z = state[2]

            odom_msg = Odometry()
            odom_msg.pose.pose.position.x = state[0]
            quaternion = quaternion_from_euler(0, 0, state[2])
            odom_msg.pose.pose.orientation.x = quaternion[0]
            odom_msg.pose.pose.orientation.y = quaternion[1]
            odom_msg.pose.pose.orientation.z = quaternion[2]
            odom_msg.pose.pose.orientation.w = quaternion[3]
            
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.type = marker.CUBE
            marker.action = marker.ADD
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.position.x = state[0]
            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]

            pub_cmd_vel.publish(twist_msg)
            pub_odom.publish(odom_msg)
            pub_marker.publish(marker)

            rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
