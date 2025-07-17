class PID:
    """
    A simple PID controller.
    
    Attributes:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        setpoint (float): Desired value.
        prev_error (float): The previous error for derivative calculation.
        integral (float): The accumulated error for integral calculation.
        output (float): The output of the PID controller.
    """
    
    def __init__(self, Kp, Ki, Kd, setpoint):
        """
        Initializes the PID controller with gains and setpoint.
        
        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            setpoint (float): Desired value.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.output = 0

    def compute(self, current_value, dt):
        """
        Computes the PID control output based on the current value and time step.

        Args:
            current_value (float): The current value (e.g., current position, velocity, etc.).
            dt (float): Time step (change in time between two control updates).

        Returns:
            float: The control output.
        """
        # Calculate error
        error = self.setpoint - current_value
        
        # Proportional term
        P_term = self.Kp * error
        
        # Integral term (accumulated error over time)
        self.integral += error * dt
        I_term = self.Ki * self.integral
        
        # Derivative term (rate of change of error)
        D_term = self.Kd * (error - self.prev_error) / dt if dt > 0 else 0
        
        # Compute total output
        self.output = P_term + I_term + D_term
        
        # Save current error for next derivative calculation
        self.prev_error = error
        
        return self.output


# Example usage of the PID controller

# Set the PID constants (you'll need to tune these for your system)
Kp = 1.0  # Proportional gain
Ki = 0.1  # Integral gain
Kd = 0.01  # Derivative gain

# Setpoint is the desired value (e.g., target position, velocity, etc.)
setpoint = 10.0  # Example setpoint (e.g., 10 meters)

# Initialize the PID controller
pid = PID(Kp, Ki, Kd, setpoint)

# Simulate a system that is moving towards the setpoint
current_value = 0.0  # Start at 0 meters (for example)
dt = 0.1  # Time step in seconds (e.g., 100 ms)

# Run the controller for 100 iterations (simulating 10 seconds)
for i in range(100):
    control_output = pid.compute(current_value, dt)
    
    # Simulate system response: for example, it moves by control_output amount
    current_value += control_output * dt  # Update the current value
    
    # Print the results every 10 iterations
    if i % 10 == 0:
        print(f"Iteration {i}, Current Value: {current_value:.2f}, Control Output: {control_output:.2f}")

# After running the loop, you'll see the PID controller minimizing the error
