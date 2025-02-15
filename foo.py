# # Data Loading
# https://pubs.rsc.org/en/content/articlelanding/2018/ra/c8ra04074e
#
# Q1(n) = C * n^z
# -----------------------------------------------------------------------------
#
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
import sys

exp_clist = np.array([0.37,0.36,0.46,0.39,0.27,0.32])
exp_zlist = np.array([0.49,0.50,0.50,0.56,0.63,0.63])

#
# later used for the model evaluation!
#
Templist  = np.array([ 298, 298, 298, 298, 298, 298 ])
Cratelist = np.array([ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ])
Dratelist = np.array([ 0.5, 1.0, 2.0, 3.0, 4.0, 5.0 ])

# data order                 0.5  1.0  2.0  3.0  4.0  5.0
exp_maxcyclelist = np.array([3300,3000,2000,1300,1000,600])
rand_normal_std = 0.5
np.random.seed(42)
#
# data creation
#
def ExpDataCreator(n,C,z):
    return C*np.power(n,z)
    
#np.random.normal(0, rand_norm_std, size=len(x_data))
#
# Generate loss data
#
_stride = 50
loss_data = [
    np.column_stack(
            (cycles := np.arange(0,maxcycle,_stride), ExpDataCreator(cycles,C,z) + np.random.normal(0,rand_normal_std,size=len(cycles)))
            )
        for C, z, maxcycle in zip(exp_clist, exp_zlist, exp_maxcyclelist)
    ]
# print(loss_data[0]) # shape size check
print(len(loss_data))

# --------------------------------------------------------
fig, ax = plt.subplots(1,2)
cm = 1/2.54
xl,yl = 32,12
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

markers = ['o', 's', 'd', '^', 'v', 'x']
colors  = ['b', 'g', 'r', 'c', 'm', 'y']

Dratelist = np.array([ 0.5, 1.0, 2.0, 3.0, 4.0, 5.0 ])

for k,Drate in enumerate(Dratelist):
    x_data = loss_data[k][:,0]
    y_data = loss_data[k][:,1]
    ax[0].plot(x_data,y_data,
            linestyle='',
            marker=markers[k],
            color=colors[k],
            label=f'Dch rate : {Drate}',
            )
    
ax[0].legend()
#plt.show()

# --------------------------------------------------------
# define Q components
#

# predefine constansts
_n = 0.5
_Ea = 10000
_R = 8.314
_T0 = 298.15
_T = 300

# T C D functions parameters
T1_initial_guess = [ 1, 1, 0.05 ] # h, DT, a : T0 = T0
C1_initial_guess = [ 1, 10 ] # AC, RhoC
D1_initial_guess = [ 50, 30 ] # AD, RhoD
B0_initial_guess = [ 1. ]

#
# Note: missing items DOD, SOC info
#
def Q1Temp( params, Temp ): # expected length params : 3
    h, DT, a = params
    return h + DT * np.power( 1. - np.exp( a * (Temp - _T0) ), 2. )

def Q1ChCrate( params, Crate ): # expected length params : 2
    A, Rho = params
    return A * np.exp( -Crate / Rho )

def Q1DchCrate( params, Crate ): # expected length params : 2
    A, Rho = params
    return A * np.exp( -Crate / Rho )

def Q1loss( params , condition, cyc ):
    # params : component function parameters
    # cond   : experimental condition

    T_params = params[0:3]
    Ch_params = params[3:5]
    Dch_params = params[5:7]
    B0 = params[-1]

    Temp, ChCrate, DchCrate = condition
    
    return B0 * Q1Temp( T_params, Temp ) * Q1ChCrate( Ch_params, ChCrate ) * Q1DchCrate( Dch_params, DchCrate ) * np.exp(-_Ea / (Temp*_R)) * np.power(cyc,_n)
    
#
# Parameter Initiaul Guess : length 8
#
# h DT a / A Rho / A Rho / B0
initial_guess = T1_initial_guess + C1_initial_guess + D1_initial_guess + B0_initial_guess

# function testing condition - temporal later will be in a form of condition data <list or numpy array>
condition = [ _T, 0.5, 1.0 ] # temp ChCrate DchCrate
#Qcyc = Q1loss(initial_guess, condition, 21)
#print(Qcyc)

#
# DEFINE LOSS FUNCTION FOR THE DATA SET
#

# dummy data test
#dummy_data = np.column_stack(
#    (cycles := np.array([ i*30 for i in range(100) ]), Q1loss(initial_guess,condition,cycles))
#)
#dx_data = dummy_data[:,0]
#dy_data = dummy_data[:,1]
#ax[1].plot(dx_data,dy_data,)

#
# dataset : loss_data
#
print('--- working loss function')

def set_residual( params, data, condition ):
    '''
        data   : experimental data set
        params : model parameter
        conds  : data set sepcific parameters

        > data - condition (bound) : improve this by object coupling later * * *
    '''
    cycles = data[:,0] # 0 column -> cycles
    Qloss = data[:,1]  # 1 column -> Q loss for this data set
    rsd = Qloss - Q1loss(params, condition, cycles)
    #return rsd # rsd vector
    return np.linalg.norm(rsd)**2.

#rsd = set_residual( params, data, condition )
#print(rsd)

def objective_function( params, loss_data, condition_data ):

    rsd_sum = 0
    for data, condition in zip(loss_data,condition_data):
        rsd_sum += set_residual( params, data, condition )
        #
        # form of condition : e.g., Temp, ChCrate, DchCrate = cond
        #
    return rsd_sum

#
# need to construct condition data
#
Templist  = np.array([ 298, 298, 298, 298, 298, 298 ])
Cratelist = np.array([ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ])
Cratelist = np.array([ 0.1, 0.4, 0.3, 0.2, 0.8, 1.0 ])
Dratelist = np.array([ 0.5, 1.0, 2.0, 3.0, 4.0, 5.0 ])

condition_data = np.column_stack( (Templist,Cratelist,Dratelist) ) # or use vstack + transpose *.T


# ------- optimiser
xtol = 1e-6
_maxiter = 5000
optprofile = []
xkprofile = []
def record_optprofile(xk):
    optprofile.append(objective_function(xk,loss_data,condition_data))
    xkprofile.append(xk)

result = opt.minimize(
        objective_function,
        initial_guess,
        args=(loss_data, condition_data),      # taking multiple inputs
        method='L-BFGS-B',
        tol = xtol,
        options={
            'disp':True,
            'maxiter': _maxiter,
            'gtol': 1e-8
        },
        callback=record_optprofile, # record_optprofile <callable> optional
    )

print(result)
optimised_params = result.x
print(optimised_params)
print(condition_data)



#exp_maxcyclelist = [ 600 for i in range(6) ]

fit_result = [
    np.column_stack(
            (cycles := np.arange(0,maxcycle,_stride), Q1loss(optimised_params,condition,cycles))
    )
        for condition, maxcycle in zip(condition_data, exp_maxcyclelist)
    ]

for k,fit in enumerate(fit_result):
    dx_data = fit[:,0]
    dy_data = fit[:,1]
    ax[1].plot(dx_data,dy_data,linestyle='',marker=markers[k])
    ax[0].plot(dx_data,dy_data)
    

#for i in range(6):
#    for j in range(i+1,6):
#        resi = fit_result[i][:,1] - fit_result[j][:,1]
#        print(i,j,end='')
#        print(np.linalg.norm(resi))
plt.show()
