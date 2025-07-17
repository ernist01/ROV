import heapq

class Node:
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x  # x-coordinate
        self.y = y  # y-coordinate
        self.g = g  # cost from start to current node
        self.h = h  # heuristic cost from current node to goal
        self.f = g + h  # total cost
        self.parent = parent  # parent node for path reconstruction

    def __lt__(self, other):
        return self.f < other.f

def a_star(start, goal, grid):
    """
    A* pathfinding algorithm.

    Args:
        start (tuple): Starting point (x, y).
        goal (tuple): Goal point (x, y).
        grid (list): 2D grid of the environment with 0 as walkable and 1 as obstacles.

    Returns:
        list: List of nodes that form the shortest path from start to goal.
    """
    # Initialize the open and closed lists
    open_list = []
    closed_list = set()
    
    # Start node
    start_node = Node(start[0], start[1], g=0, h=heuristic(start, goal))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        
        # If we reached the goal, reconstruct the path
        if (current_node.x, current_node.y) == goal:
            return reconstruct_path(current_node)

        closed_list.add((current_node.x, current_node.y))

        # Generate neighbors (4 directions: up, down, left, right)
        neighbors = [(current_node.x + 1, current_node.y), 
                    (current_node.x - 1, current_node.y),
                    (current_node.x, current_node.y + 1), 
                    (current_node.x, current_node.y - 1)]
        
        for nx, ny in neighbors:
            # Check if the neighbor is within bounds and not an obstacle
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                if (nx, ny) in closed_list:
                    continue
                
                g = current_node.g + 1  # Assume each step costs 1
                h = heuristic((nx, ny), goal)
                neighbor_node = Node(nx, ny, g=g, h=h, parent=current_node)

                # Add the neighbor to the open list if not already there
                heapq.heappush(open_list, neighbor_node)

    return None  # No path found

def heuristic(node, goal):
    """
    Calculate the heuristic (Manhattan distance) for A*.
    
    Args:
        node (tuple): The current node (x, y).
        goal (tuple): The goal node (x, y).
    
    Returns:
        float: Heuristic cost.
    """
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def reconstruct_path(node):
    """
    Reconstruct the path from the goal node to the start node.
    
    Args:
        node (Node): The goal node.
    
    Returns:
        list: The path as a list of (x, y) tuples.
    """
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    path.reverse()  # The path is constructed backwards, so reverse it
    return path

# Example grid environment (0 = walkable, 1 = obstacle)
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

# Starting and goal positions
start = (0, 0)
goal = (4, 4)

# Run A* algorithm
path = a_star(start, goal, grid)
print("Path found:", path)
