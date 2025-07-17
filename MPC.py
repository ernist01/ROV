import numpy as np
import cvxpy as cp  # For solving optimization problems
import matplotlib.pyplot as plt

# Define the system dynamics (1D case: position and velocity)
# x_k+1 = A * x_k + B * u_k
A = np.array([[1, 1],  # Position and velocity dynamics (discrete-time)
              [0, 1]])
B = np.array([[0.5],   # Control input effect (acceleration)
              [1]])

# Define MPC parameters
N = 10  # Prediction horizon
dt = 0.1  # Time step (seconds)
target_position = 10  # Target position for the ROV (m)

# Define the cost function
Q = np.diag([10, 1])  # Cost for position and velocity
R = np.array([[0.1]])  # Cost for control input (thrust)

# Define constraints
max_thrust = 2  # Maximum thrust (control input) (m/s^2)
max_position = 15  # Max position (m)
min_position = 0  # Min position (m)

# Initial state (position = 0, velocity = 0)
x0 = np.array([0, 0])

# Define the MPC optimization problem
x = cp.Variable((2, N+1))  # State trajectory
u = cp.Variable((1, N))  # Control trajectory (thrusts)
cost = 0  # Initialize cost

# Formulate the cost function and constraints
for k in range(N):
    # State cost (minimize position error and velocity)
    cost += cp.quad_form(x[:, k] - np.array([target_position, 0]), Q)
    # Control effort cost (minimize thrust)
    cost += cp.quad_form(u[:, k], R)
    
    # System dynamics constraints
    if k < N-1:
        # x_{k+1} = A * x_k + B * u_k
        constraints = [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]
    
    # Control input constraints (thrust limits)
    constraints += [u[:, k] <= max_thrust, u[:, k] >= -max_thrust]
    
    # Position constraints (bounds)
    constraints += [x[0, k] <= max_position, x[0, k] >= min_position]

# Initial condition constraint
constraints += [x[:, 0] == x0]

# Solve the optimization problem
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve()

# Extract the control inputs and state trajectory
u_optimal = u.value
x_optimal = x.value

# Plot the results
time = np.arange(0, (N+1) * dt, dt)
plt.figure(figsize=(10, 5))

# Plot position and velocity
plt.subplot(2, 1, 1)
plt.plot(time, x_optimal[0, :], label="Position")
plt.plot(time, np.ones_like(time) * target_position, label="Target Position", linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time[:-1], u_optimal[0, :], label="Thrust (Control Input)")
plt.xlabel('Time (s)')
plt.ylabel('Thrust (m/s^2)')
plt.legend()

plt.tight_layout()
plt.show()
