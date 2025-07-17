import numpy as np
import math

class SensorFusion:
    def __init__(self, alpha=0.98, dt=0.01):
        """
        Initializes the sensor fusion class for combining IMU and depth sensor data.

        Args:
            alpha (float): Complementary filter constant (0.98-0.99).
            dt (float): Time step for integration (in seconds).
        """
        self.alpha = alpha  # Complementary filter constant
        self.dt = dt  # Time step

        # IMU initial values
        self.theta_gyro = 0  # Initial pitch angle from gyroscope (radians)
        self.theta_acc = 0  # Initial pitch angle from accelerometer (radians)
        
        # Depth sensor initial value
        self.depth = 0  # Depth value from the sensor (meters)
        
        # Position and velocity estimates
        self.position = np.array([0.0, 0.0, 0.0])  # [x, y, z] in meters
        self.velocity = np.array([0.0, 0.0, 0.0])  # [vx, vy, vz] in m/s

    def update_orientation(self, acc_data, gyro_data):
        """
        Updates the orientation based on accelerometer and gyroscope data.

        Args:
            acc_data (tuple): Accelerometer data (ax, ay, az).
            gyro_data (tuple): Gyroscope data (gx, gy, gz).
        """
        # Extract accelerometer data
        ax, ay, az = acc_data
        gx, gy, gz = gyro_data

        # Normalize accelerometer data to get the pitch angle
        acc_angle = math.atan2(ay, az)  # Pitch angle from accelerometer

        # Integrate gyroscope data to get the pitch angle
        self.theta_gyro += gx * self.dt  # Gyroscope angle (in radians)

        # Apply complementary filter
        self.theta = self.alpha * (self.theta_gyro) + (1 - self.alpha) * acc_angle
        self.theta_acc = acc_angle

    def update_depth(self, depth_data):
        """
        Updates the depth (altitude) based on the depth sensor data.

        Args:
            depth_data (float): Depth from the depth sensor in meters.
        """
        self.depth = depth_data  # Update depth from sensor

    def update_position(self, imu_data, depth_data):
        """
        Updates the position of the ROV based on IMU and depth sensor data.

        Args:
            imu_data (tuple): IMU data containing accelerometer and gyroscope values (ax, ay, az, gx, gy, gz).
            depth_data (float): Depth data from the depth sensor.
        """
        acc_data = imu_data[:3]  # (ax, ay, az)
        gyro_data = imu_data[3:]  # (gx, gy, gz)

        # Update orientation
        self.update_orientation(acc_data, gyro_data)

        # Update depth (altitude)
        self.update_depth(depth_data)

        # Simple motion model for position and velocity estimation using accelerometer data
        ax, ay, az = acc_data
        self.velocity += np.array([ax, ay, az]) * self.dt  # Update velocity based on accelerometer data
        self.position += self.velocity * self.dt  # Update position based on velocity

    def get_position(self):
        """
        Returns the current estimated position of the ROV.
        """
        return self.position

    def get_orientation(self):
        """
        Returns the current estimated orientation (pitch angle) of the ROV.
        """
        return self.theta

    def get_depth(self):
        """
        Returns the current depth (altitude) of the ROV.
        """
        return self.depth

# Example Usage

# Initialize the SensorFusion object with a complementary filter and time step
fusion = SensorFusion(alpha=0.98, dt=0.1)

# Example sensor data (accelerometer (ax, ay, az) and gyroscope (gx, gy, gz))
acc_data = (0.0, 9.81, 0.0)  # Accelerometer readings (e.g., static condition on the surface)
gyro_data = (0.0, 0.0, 0.0)  # Gyroscope readings (no movement)
depth_data = 10.0  # Depth from the depth sensor (in meters)

# Update the sensor fusion with the data
fusion.update_position(acc_data + gyro_data, depth_data)

# Get the current position, orientation, and depth
print("Position (x, y, z):", fusion.get_position())
print("Orientation (pitch):", fusion.get_orientation())
print("Depth:", fusion.get_depth())
