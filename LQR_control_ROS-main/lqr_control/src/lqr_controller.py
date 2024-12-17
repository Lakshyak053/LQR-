#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# Define constants
max_linear_velocity = 0.22  # meters per second (TurtleBot3 max velocity)
max_angular_velocity = 2.84  # radians per second (TurtleBot3 max angular velocity)

class LQRController:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('lqr_controller', anonymous=True)

        # Publisher for velocity commands
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscriber to the odometry data
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Define desired state [x, y, yaw]
        self.desired_state_xf = np.array([2.0, 2.0, np.pi/4])

        # Current state [x, y, yaw]
        self.actual_state_x = np.array([0.0, 0.0, 0.0])

        # Define LQR parameters
        self.A = np.array([[1.0, 0, 0],
                           [0, 1.0, 0],
                           [0, 0, 1.0]])

        self.Q = np.array([[100.0, 0, 0],  # Penalize X position error 
                           [0, 100.0, 0],  # Penalize Y position error 
                           [0, 0, 0.1]])  # Penalize YAW ANGLE heading error 

        self.R = np.array([[0.1, 0],  # Penalize linear velocity effort
                           [0, 0.1]])  # Penalize angular velocity effort

        # Set time step
        self.dt = 0.1  # 100 ms

        # Rate at which the control loop runs
        self.rate = rospy.Rate(10)  # 10 Hz

        # Lists to log linear and angular velocities
        self.linear_velocities = []
        self.angular_velocities = []
        self.time_steps = []

    def odom_callback(self, data):
        """Callback function to update the current state of the robot from odometry data."""
        position = data.pose.pose.position
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion(orientation_list)

        self.actual_state_x = np.array([position.x, position.y, yaw])

    def getB(self, yaw, dt):
        """Calculate and return the B matrix."""
        B = np.array([[np.cos(yaw) * self.dt, 0],
                      [np.sin(yaw) * self.dt, 0],
                      [0, self.dt]])
        return B

    def state_space_model(self, state, B, control_input):
        """Update the robot's state based on the LQR control input."""
        control_input[0] = np.clip(control_input[0], -max_linear_velocity, max_linear_velocity)
        control_input[1] = np.clip(control_input[1], -max_angular_velocity, max_angular_velocity)
        state_estimate = np.dot(self.A, state) + np.dot(B, control_input)
        return state_estimate

    def lqr(self, actual_state, desired_state, B):
        """LQR controller to calculate the optimal control input."""
        x_error = actual_state - desired_state
        N = 50  # Number of time steps to solve backwards
        P = [None] * (N + 1)
        P[N] = self.Q

        # Solve for P matrix backwards in time
        for i in range(N, 0, -1):
            P[i-1] = self.Q + self.A.T @ P[i] @ self.A - (self.A.T @ P[i] @ B) @ np.linalg.pinv(self.R + B.T @ P[i] @ B) @ (B.T @ P[i] @ self.A)

        # Calculate the optimal control input using the last step
        K = -np.linalg.pinv(self.R + B.T @ P[1] @ B) @ B.T @ P[1] @ self.A
        u_star = K @ x_error

        return u_star

    def control_loop(self):
        """Main control loop that runs continuously."""
        start_time = rospy.get_time()

        while not rospy.is_shutdown():
            rospy.loginfo("Node Started !")
            
            # Calculate the B matrix based on current yaw
            B = self.getB(self.actual_state_x[2], self.dt)

            # Get the optimal control input from the LQR controller
            optimal_control_input = self.lqr(self.actual_state_x, self.desired_state_xf, B)

            # Log the linear and angular velocities
            self.linear_velocities.append(optimal_control_input[0])
            self.angular_velocities.append(optimal_control_input[1])

            # Log the time step
            self.time_steps.append(rospy.get_time() - start_time)

            # Update the robot's state using the control input
            self.actual_state_x = self.state_space_model(self.actual_state_x, B, optimal_control_input)

            # Publish the control input to /cmd_vel
            twist = Twist()
            twist.linear.x = optimal_control_input[0]  # Linear velocity
            twist.angular.z = optimal_control_input[1]  # Angular velocity
            self.cmd_pub.publish(twist)

            # Log the current position (x, y, yaw)
            x, y, yaw = self.actual_state_x
            rospy.loginfo(f"Current Position: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad")

            # Stop when the error is small
            state_error_magnitude = np.linalg.norm(self.actual_state_x - self.desired_state_xf)
            if state_error_magnitude < 0.1:
                rospy.loginfo("Goal reached!")
                break

            # Sleep to maintain loop rate
            self.rate.sleep()

        # After the loop, plot the velocity graphs
        self.plot_velocities()


    def plot_velocities(self):
        """Plot the linear and angular velocities over time."""
        plt.figure()

        # Plot linear velocity
        plt.subplot(2, 1, 1)
        plt.plot(self.time_steps, self.linear_velocities, label='Linear Velocity')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title('Linear Velocity vs Time')
        plt.grid(True)

        # Plot angular velocity
        plt.subplot(2, 1, 2)
        plt.plot(self.time_steps, self.angular_velocities, label='Angular Velocity', color='r')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Angular Velocity vs Time')
        plt.grid(True)

        # Show the plot
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    try:
        controller = LQRController()
        controller.control_loop()
        controller.plot_velocities()
    except rospy.ROSInterruptException:
        pass
