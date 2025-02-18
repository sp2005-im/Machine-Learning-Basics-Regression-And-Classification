#Sri Rama Jayam
import numpy as np
import random as random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math as m

class Node:
  def __init__(self,x,y):
    self.x = x
    self.y = y
    self.parent = None

class RRT:
  def __init__(self,start,goal,obstacles,boundary,step_size=0.1,max_iterations=5000):
    # Function to initialize the different variables and parameters
    self.start = Node(start[0],start[1])
    self.goal = Node(goal[0],goal[1])
    self.obstacles = obstacles # List of [x,y,width,height] for each rectangular obstacle
    self.boundary = boundary # [min_x, min_y, max_x, max_y]
    self.step_size = step_size
    self.max_iterations = max_iterations
    self.nodes = [self.start]
    self.goal_reached = False

  def plan(self):
    for i in range(self.max_iterations):
      if random.random() < 0.05:  # Bias towards goal
          rand_node = Node(self.goal.x, self.goal.y)
      else:
          rand_node = self._get_random_node()
                
      nearest_node = self._find_nearest(rand_node)
      new_node = self._steer(nearest_node, rand_node)
            
      if new_node and not self._collision_check(nearest_node, new_node):
          self.nodes.append(new_node)
          new_node.parent = nearest_node
                
          if self._distance_calculate(new_node, self.goal) < self.step_size:
              self.goal.parent = new_node
              self.nodes.append(self.goal)
              self.goal_reached = True
              break
    return self.goal_reached, self.nodes

  def _get_random_node(self):
    #Function to generate random nodes within the defined boundaries
    x = random.uniform(self.boundary[0],self.boundary[2])
    y = random.uniform(self.boundary[1],self.boundary[3])
    return Node(x,y)

  

  def _distance_calculate(self,node1,node2):
    #Function to calculate the distance between two nodes
    x1 = node1.x
    x2 = node2.x
    y1 = node1.y
    y2 = node2.y
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist
  
  def _find_nearest(self,node):
    #Function to find the nearest node
    min_dist =  float('inf')
    nearest_node = None
    for n in self.nodes:
      dist = self._distance_calculate(node,n)
      if (dist<min_dist):
        min_dist = dist
        nearest_node = n

    return nearest_node

  def _slope_calculate(self,node1,node2):
    #Function to calculate the slope of a line joining two points
    x1 = node1.x
    x2 = node2.x
    y1 = node1.y
    y2 = node2.y
    if (y2!=y1):
      theta = m.atan2((y2-y1),(x2-x1))
    else:
      theta = (m.pi)/2
    return theta

  def _steer(self,from_node,to_node):
    #Function to return new_node based on the from_node and to_node
    distance = self._distance_calculate(from_node,to_node)
    if distance < self.step_size:
      return to_node
    theta = self._slope_calculate(from_node,to_node)
    x_new = from_node.x + (self.step_size)*m.cos(theta)
    y_new = from_node.y + (self.step_size)*m.sin(theta)

    #Check if the new point is within the defined boundaries
    if (x_new  < self.boundary[0] or x_new > self.boundary[2]) or (y_new < self.boundary[1] or y_new > self.boundary[3]):
      return None

    return Node(x_new, y_new)

  def _point_in_obstacle(self, point, obstacle, epsilon = 1):
    '''
    Function to check if a point is within the rectangular obstacles. Also to improve this, we make sure that the point is not too close to the obstacle by
    fixing the parameter epsilon..this parameter is added into the function so that it can be throttled to any value.

    Returns False if the point is not within the obstacle

    Returns True if the point is within the obstacle
    '''
    x,y = point.x, point.y
    x_obs, y_obs, width, height = obstacle
    if (x_obs-epsilon <= x <= x_obs+width+epsilon) and (y_obs-epsilon <= y <= y_obs+height+epsilon):
      return True
    return False

  def _collision_check(self, node1, node2):
    '''
    To check if either of the nodes are within the obstacles or whether the line joining the nodes passes through the obstacles

    Returns False if neither the line nor the point(s) are within the obstacle
    Returns True otherwise
    '''

    #To check if the nodes are within the obstacle
    for obs in self.obstacles:
      if self._point_in_obstacle(node1,obs) or self._point_in_obstacle(node2,obs):
        return True

    # To check if the line joining the nodes has a part lying within the obstacle
    # Discretizing the line...dividing it into ten parts

    for i in range(10):
      t = i/10
      node_x = node1.x + (node2.x-node1.x)*t
      node_y = node1.y + (node2.y-node1.y)*t
      node_part = Node(node_x, node_y) # A segment joining two points broken up into ten segments

      for obs in self.obstacles:
        if self._point_in_obstacle(node_part,obs):
          return True

    return False
  
  def get_path(self):
    if not self.goal_reached:
      return []
        
    path = []
    node = self.goal
    while node:
      path.append((node.x, node.y))
      node = node.parent
        
    return path[::-1]  # Reverse to get start-to-goal path
 
def visualize_rrt(rrt, path=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot obstacles
    for obs in rrt.obstacles:
        x, y, width, height = obs
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='k', facecolor='blue')
        ax.add_patch(rect)

    # Plot nodes and edges
    for node in rrt.nodes:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], 'b-', alpha=0.2)

    # Plot path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, 'r-', linewidth=2)

    # Plot start and goal
    plt.plot(rrt.start.x, rrt.start.y, 'go', markersize=10)
    plt.plot(rrt.goal.x, rrt.goal.y, 'ro', markersize=10)

    # Set bounds
    plt.xlim(rrt.boundary[0], rrt.boundary[2])
    plt.ylim(rrt.boundary[1], rrt.boundary[3])
    plt.grid(False)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RRT Path Planning')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define start and goal positions
    start = (1.0, 1.0)
    goal = (18.0, 15.0)

    # Define square obstacles as [x, y, width, height]
    obstacles = [
        [4.0, 4.0, 5.0, 5.0],
        [10.0, 6.0, 8.0, 8.0],
        #[15.0, 20.0, 7.0, 7.0]
    ]

    # Define boundary [min_x, min_y, max_x, max_y]
    boundary = [0.0, 0.0, 20.0, 20.0]

    # Initialize and run RRT
    rrt = RRT(start, goal, obstacles, boundary, step_size=0.5)
    success, nodes = rrt.plan()

    if success:
        path = rrt.get_path()
        print(f"Path found with {len(nodes)} nodes explored")
        visualize_rrt(rrt, path)
    else:
        print("No path found")
        visualize_rrt(rrt)                    
  
