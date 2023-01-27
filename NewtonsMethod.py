import numpy as np
import matplotlib.pyplot as plt


# Objective function: f(x,y) = 100(y - x^2)^2 + (1 - x)^2
# Utilize Newton's Method

# Define objective function
def obj_fun(position):
    x = position[0]
    y = position[1]
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


# Define gradient function
def gradient(position):
    x = position[0]
    y = position[1]
    return np.array([-400 * x * (y - x ** 2) - 2 * (1 - x), 200 * (y - x ** 2)])


# Define hessian function
def hessian(position):
    x = position[0]
    y = position[1]
    return np.array([[-400 * (y - 3 * x ** 2) + 2, -400 * x], [-400 * x, 200]])


# Define initial parameters
x0 = -2.5
y0 = 2
alpha = 0.1
ans = np.array([1, 1])  # Computed by hand
dist = 0.001    # Distance from the ans

# Create arrays and define variables for plotting
pos = np.array([x0, y0])
trajectory = []
euc_dist = []
iterations = 0

while np.linalg.norm(ans - pos) > dist:
    trajectory.append(pos)
    euc_dist.append(np.linalg.norm(ans - pos))
    iterations += 1
    pos = pos - alpha * np.matmul(np.linalg.inv(hessian(pos)), gradient(pos))


# Trajectory
traj_x = [vector[0] for vector in trajectory]
traj_y = [vector[1] for vector in trajectory]

f1 = plt.figure(1)
plt.plot(traj_x, traj_y)
plt.title("Newton's Method Trajectory")
plt.xlabel("x")
plt.ylabel("y")

# Euclidian Distance
f2 = plt.figure(2)
plt.plot(range(iterations), euc_dist)
plt.title("Euclidian Distance vs. Number of Iterations")
plt.xlabel("Iterations")
plt.ylabel("Euclidian Distance")

plt.show()
