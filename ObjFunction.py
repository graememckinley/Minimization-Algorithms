import numpy as np
import matplotlib.pyplot as plt


# Define objective function
def obj_fun(x1, y1):
    return 100 * (y1 - x1 ** 2) ** 2 + (1 - x1) ** 2


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = obj_fun(X, Y)

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none")

ax.set_title("Objective Function")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()
