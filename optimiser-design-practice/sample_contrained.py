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

# Original ObjectiveF
def rosenbrock(var,params):
    x, y = var     # <tuple>
    a, b = params  # <tuple>
    return (a - x)**2 + b * (y - x**2)**2

# store optimisation profile
optprofile = []
xkprofile = []

def record_optprofile(xk):
    optprofile.append(rosenbrock(xk,params)) # xk is automatic -> paramter at kth iteration
    xkprofile.append(xk)
    
# Initial guess
x0 = [10, 1]  
# Tolerance
xtol = 1e-6

#
# Define Parameter Bound
#
bounds = [(None,None),tuple( [x0[1] for i in range(len(x0))] )]
#         ^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#         No Bound    Bound x0[1] value i.e., fix 'y' variable
#


res = minimize(
    rosenbrock,
    x0,
    args=(params,),
    bounds=bounds,
    method='L-BFGS-B',
    tol = xtol,
    options={'disp': True},
    callback=record_optprofile,
    )

for key,value in zip(res.keys(),res.values()):
    print(f'{key:12.6s} {value}') 
print(res)

# sort out x, y profiles from xkprofile
xlist = [ p[0] for p in xkprofile ]
ylist = [ p[1] for p in xkprofile ] 
step  = [ i+1 for i in range(len(xkprofile)) ]
stepxlist = np.array([ [s,x] for s,x in zip(step,xlist) ])
stepxlist = [ [s,x] for s,x in zip(step,xlist) ]
#print(stepxlist)
#print(step)

#
# show opt profile
#

fig, ax = plt.subplots(1,2)
cm = 1/2.54
xl,yl = 16,12
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
ax[0].set_yscale('log')
ax[0].grid()

#print(type(optprofile),type(stepxlist))
#print(stepxlist)
ax[1].plot(step,xlist, marker='o', linestyle='-')
ax[1].axhline(y=1,linestyle='--',color='r')
ax[1].set_title('x value profile')

plt.show()

# print function values
#for item in optprofile:
#    print(item)


sys.exit()
