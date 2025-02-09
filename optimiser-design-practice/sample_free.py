from matplotlib import pyplot as plt
from scipy.optimize import minimize
import numpy as np
import sys

#
# Rosenbrock function Objective function
#

# Input Parameters
# a = 1
# b = 100
# GM = (1,1)
params = (1, 100)

def rosenbrock(var,params):
    x, y = var     # <tuple>
    a, b = params  # <tuple> # Getting Multiple Input Parameters
    return (a - x)**2 + b * (y - x**2)**2

# store optimisation profile
optprofile = []
xkprofile = []

def record_optprofile(xk):
    optprofile.append(rosenbrock(xk,params)) # xk is automatic -> paramter at kth iteration
    xkprofile.append(xk)

# Initial guess
x0 = [2,1]  
#x0 = [3,3]  
# Tolerance
xtol = 1e-6

res = minimize(
    rosenbrock,
    x0,
    args=(params,),
    method='BFGS',
    #method='CG',
    tol = xtol,
    options={'disp': True},
    callback=record_optprofile, # record_optprofile <callable> optional
    )

for key,value in zip(res.keys(),res.values()):
    print(f'{key:12.6s} {value}') 

#
# show opt profile
#
fig, ax = plt.subplots(1,2)
cm = 1/2.54
xl, yl = 16, 12
fig.set_size_inches((xl*cm,yl*cm))
plt.subplots_adjust(
left = 0.10,
bottom = 0.1,
right = 0.94,
top = 0.96,
wspace = 0.200,
hspace = 0.0
)
fs, lfs = 12, 14 #fonts

ax[0].plot(optprofile, marker='o', linestyle='-')
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Function Value")
ax[0].set_title("Optimization Progress")
ax[0].grid()



xkprofile.insert(0,x0)

xkprofile_x = [ p[0] for p in xkprofile ]
xkprofile_y = [ p[1] for p in xkprofile ]
print(xkprofile)
print(xkprofile_x)
print(xkprofile_y)


ax[1].plot(xkprofile_x,xkprofile_y,marker='o', linestyle='-')

# ------------------------------------------------------------
# Set parameters for the Rosenbrock function
params = (1, 100)  # a = 1, b = 100

# Create a grid of x and y values
x_range = np.linspace(0.5, 2.5, 1000)  # X values from -2 to 2
y_range = np.linspace(0.5, 2.5, 1000)  # Y values from -1 to 3

# Create meshgrid of X and Y values
X, Y = np.meshgrid(x_range, y_range)

# Initialize the Z values (the Rosenbrock function)
Z = np.zeros_like(X)

# Compute the function values over the grid
for i in range(len(x_range)):
    for j in range(len(y_range)):
        Z[j, i] = rosenbrock([X[j, i], Y[j, i]], params)

ax[1].imshow(Z,
    extent=[x_range.min(), x_range.max(), y_range.min(), y_range.max()],
    origin='lower',
    cmap='viridis',
    )




plt.show()

sys.exit()
    
#print("Optimized x:", res.x[0])  # Optimized x
#print("Fixed y:", res.x[1])  # Should remain 2
#print("Function value at minimum:", res.fun)
