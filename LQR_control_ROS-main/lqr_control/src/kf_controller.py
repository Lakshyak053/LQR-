#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt

# Adjust the LQR gain matrix to match 3 states (x, y, theta)
K = np.array([[1.0, 0.0, 0.0],  # Linear velocity gain for x, y, theta
              [0.0, 0.0, 1.0]])  # Angular velocity gain for theta


# Set the desired goal position (Point B) [x_goal, y_goal, theta_goal]
goal = np.array([2.0, 2.0, 0.0])  # Example goal: x=2.0, y=2.0, theta=0.0

# Kalman Filter Parameters
dt = 0.1  # Time step (assuming 10 Hz update rate)
A = np.eye(3)  # State transition matrix (identity for simplicity)
B = np.array([[dt, 0], [0, dt], [0, 0]])  # Control matrix
C = np.eye(3)  # Measurement matrix (direct observation)
Q_kf = np.eye(3) * 0.01  # Process noise covariance
R_kf = np.eye(3) * 0.1  # Measurement noise covariance

# Initialize the Kalman filter state
x_kf = np.zeros((3, 1))  # Initial state (x, y, theta)
P_kf = np.eye(3)  # Initial covariance matrix

# Initialize lists to store control inputs and errors
control_inputs = []
errors = []

# Maximum angular velocity (rad/s) to limit excessive rotation
MAX_ANGULAR_VEL = 1.0  # Adjust this value to your needs

def kalman_filter_predict(x, P, u):
    """
    Kalman Filter Predict Step.
    :param x: Current state estimate
    :param P: Current covariance matrix
    :param u: Control input (linear and angular velocity)
    :return: Predicted state and covariance
    """
    # Predict the next state
    x_pred = np.dot(A, x) + np.dot(B, u)
    # Predict the next covariance
    P_pred = np.dot(A, np.dot(P, A.T)) + Q_kf
    return x_pred, P_pred

def kalman_filter_update(x_pred, P_pred, z):
    """
    Kalman Filter Update Step.
    :param x_pred: Predicted state
    :param P_pred: Predicted covariance
    :param z: Measurement (from odometry)
    :return: Updated state and covariance
    """
    # Compute Kalman Gain
    S = np.dot(C, np.dot(P_pred, C.T)) + R_kf
    K_kf = np.dot(P_pred, np.dot(C.T, np.linalg.inv(S)))

    # Update state estimate
    x_updated = x_pred + np.dot(K_kf, (z - np.dot(C, x_pred)))

    # Update covariance estimate
    P_updated = P_pred - np.dot(K_kf, np.dot(C, P_pred))

    return x_updated, P_updated

def lqr_control(state):
    """
    Compute the LQR control input based on the state.
    :param state: current state of the robot [x, y, theta]
    :return: control input [linear_velocity, angular_velocity]
    """
    error = state - goal  # Error between current state and goal
    errors.append(error)  # Store error for plotting

    # Control law (u = -K * error)
    control_input = -np.dot(K, error)
    control_inputs.append(control_input)  # Store control input for plotting

    linear_velocity = control_input[0]
    angular_velocity = control_input[1]

    # Limit angular velocity to avoid excessive spinning
    angular_velocity = np.clip(angular_velocity, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

    return linear_velocity, angular_velocity

def odom_callback(msg):
    """
    Callback to process odometry data and apply Kalman filter for state estimation.
    """
    global x_kf, P_kf

    # Extract position and velocity from the odometry message
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    # Convert quaternion to Euler angles to get theta (yaw)
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (_, _, theta) = euler_from_quaternion(orientation_list)

    # Measurement from odometry
    z = np.array([[x], [y], [theta]])

    # Control input from the last step (linear and angular velocity)
    v = msg.twist.twist.linear.x
    w = msg.twist.twist.angular.z
    u = np.array([[v], [w]])

    # Predict the next state using the Kalman filter
    x_pred, P_pred = kalman_filter_predict(x_kf, P_kf, u)

    # Update the state estimate using the new measurement
    x_kf, P_kf = kalman_filter_update(x_pred, P_pred, z)

    # Use the Kalman-filtered state for LQR control
    linear_vel, angular_vel = lqr_control(x_kf.flatten())

    # Publish control input to cmd_vel
    cmd_vel = Twist()
    cmd_vel.linear.x = linear_vel
    cmd_vel.angular.z = angular_vel
    pub.publish(cmd_vel)

def plot_control_inputs():
    """
    Plot the control inputs (linear and angular velocity) over time.
    """
    if len(control_inputs) > 0:
        # Extract control inputs
        linear_velocities = [u[0] for u in control_inputs]
        angular_velocities = [u[1] for u in control_inputs]

        # Plot the control inputs
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(linear_velocities, label='Linear Velocity')
        plt.xlabel('Time Step')
        plt.ylabel('Linear Velocity')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(angular_velocities, label='Angular Velocity')
        plt.xlabel('Time Step')
        plt.ylabel('Angular Velocity')
        plt.legend()

        plt.show()

if __name__ == '__main__':
    rospy.init_node('lqr_controller_with_kalman')

    # Publisher for velocity commands
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # Subscriber for odometry data
    rospy.Subscriber('/odom', Odometry, odom_callback)

    # Run the node and make the TurtleBot move
    rospy.spin()

    # After the node finishes, plot the control inputs
    plot_control_inputs()
