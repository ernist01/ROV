import numpy as np
import math

class ROV:
    def __init__(self, position, goal, obstacles, repulsive_strength=1.0, attractive_strength=1.0):
        self.position = np.array(position)  # Robot's current position
        self.goal = np.array(goal)  # Goal position
        self.obstacles = [np.array(o) for o in obstacles]  # List of obstacle positions
        self.repulsive_strength = repulsive_strength  # Strength of repulsive force
        self.attractive_strength = attractive_strength  # Strength of attractive force
    
    def attractive_force(self):
        """
        Computes the attractive force toward the goal.
        """
        vector_to_goal = self.goal - self.position
        distance_to_goal = np.linalg.norm(vector_to_goal)
        force = self.attractive_strength * vector_to_goal / distance_to_goal
        return force
    
    def repulsive_force(self):
        """
        Computes the total repulsive force from all obstacles.
        """
        total_repulsive_force = np.array([0.0, 0.0])
        
        for obs in self.obstacles:
            vector_to_obstacle = self.position - obs
            distance_to_obstacle = np.linalg.norm(vector_to_obstacle)
            
            # Avoid obstacles that are too close
            if distance_to_obstacle < 1:  # Minimum safe distance
                repulsive_magnitude = self.repulsive_strength * (1.0 / distance_to_obstacle - 1.0)
                repulsive_force = repulsive_magnitude * vector_to_obstacle / distance_to_obstacle
                total_repulsive_force += repulsive_force
        
        return total_repulsive_force
    
    def compute_net_force(self):
        """
        Computes the total net force as the sum of attractive and repulsive forces.
        """
        attractive = self.attractive_force()
        repulsive = self.repulsive_force()
        net_force = attractive + repulsive
        return net_force
    
    def move(self, dt=0.1):
        """
        Moves the ROV based on the computed net force.
        """
        net_force = self.compute_net_force()
        # Update the ROV's position based on the net force
        self.position += net_force * dt

# Example Usage
start = [0, 0]
goal = [10, 10]
obstacles = [[5, 5], [6, 6], [3, 3]]

rov = ROV(start, goal, obstacles)

for i in range(100):
    rov.move(dt=0.1)
    print(f"Step {i}: Position {rov.position}")
