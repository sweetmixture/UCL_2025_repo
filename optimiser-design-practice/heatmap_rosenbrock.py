import numpy as np
import matplotlib.pyplot as plt

# Define Rosenbrock function
def rosenbrock(var, params):
    x, y = var     # <tuple>
    a, b = params  # <tuple> # Getting Multiple Input Parameters
    return (a - x)**2 + b * (y - x**2)**2

# Set parameters for the Rosenbrock function
params = (1, 100)  # a = 1, b = 100

# Create a grid of x and y values
x_range = np.linspace(0.5, 2, 1000)  # X values from -2 to 2
y_range = np.linspace(0.5, 2, 1000)  # Y values from -1 to 3

# Create meshgrid of X and Y values
X, Y = np.meshgrid(x_range, y_range)

# Initialize the Z values (the Rosenbrock function)
Z = np.zeros_like(X)

# Compute the function values over the grid
for i in range(len(x_range)):
    for j in range(len(y_range)):
        Z[j, i] = rosenbrock([X[j, i], Y[j, i]], params)

# Plot the heatmap
fig, ax = plt.subplots(figsize=(8, 6))
c = ax.imshow(Z,
extent=[x_range.min(), x_range.max(), y_range.min(), y_range.max()], origin='lower', cmap='viridis')

# Add color bar to show the value scale
fig.colorbar(c, ax=ax)

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Heatmap of Rosenbrock Function')

# Display the plot
plt.show()

