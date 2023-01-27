import numpy as np
import matplotlib.pyplot as plt

# Objective function: f(x,y) = 100(y - x^2)^2 + (1 - x)^2
# Utilize Monte Carlo Sampling

np.random.seed(15)


# Define initial parameters
x0 = -2.5
y0 = 2
alpha = 0.001
T = 0.0001
dist = 0.001    # Distance from the ans
ans = np.array([1, 1])  # Computed by hand


# Generate random angle
def random_angle():
    return 2 * np.pi * np.random.random()


# Define objective function
def obj_fun(position):
    x = position[0]
    y = position[1]
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


# Define Monte Carlo acceptance function
def mc_fun(new, previous, prob):
    if prob < np.e ** (-(obj_fun(new) - obj_fun(previous)) / T):
        return True
    else:
        return False


# Create arrays and define variables for plotting
new_pos = np.array([x0, y0])
trajectory = []
euc_dist = []
iterations = 0

while np.linalg.norm(ans - new_pos) > dist:
    trajectory.append(new_pos)
    euc_dist.append(np.linalg.norm(ans - new_pos))
    iterations += 1
    prev_pos = new_pos

    # Generate new potential position vector
    beta = random_angle()
    pot_pos = prev_pos - alpha * np.array([np.cos(beta), np.sin(beta)])

    # Generate random probability p
    p = np.random.random()

    # Accept or reject trial
    if mc_fun(pot_pos, prev_pos, p):
        new_pos = pot_pos


# Trajectory
traj_x = [vector[0] for vector in trajectory]
traj_y = [vector[1] for vector in trajectory]

f1 = plt.figure(1)
plt.plot(traj_x, traj_y)
plt.title("Monte Carlo Method Trajectory")
plt.xlabel("x")
plt.ylabel("y")


# Euclidian Distance
f2 = plt.figure(2)
plt.plot(range(iterations), euc_dist)
plt.title("Euclidian Distance vs. Number of Iterations")
plt.xlabel("Iterations")
plt.ylabel("Euclidian Distance")

plt.show()
