import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt

# -------------------------
# Data Loading and Generation
# -------------------------
exp_clist = np.array([0.37, 0.36, 0.46, 0.39, 0.27, 0.32])
exp_zlist = np.array([0.49, 0.50, 0.50, 0.56, 0.63, 0.63])
Templist  = np.array([298, 298, 298, 298, 298, 298])
Cratelist = np.array([0.53, 0.5, 0.8, 0.8, 1.0, 1.0])
Dratelist = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
exp_maxcyclelist = np.array([3300, 3000, 2000, 1300, 1000, 600])



exp_clist = np.array([0.37, 0.36, 0.46, 0.39, 0.32 ])
exp_zlist = np.array([0.49, 0.50, 0.50, 0.56, 0.63 ])
Templist  = np.array([298, 298, 298, 298, 298 ])
Cratelist = np.array([0.53, 0.5, 0.8, 0.8, 1.0 ])
Dratelist = np.array([0.5, 1.0, 2.0, 3.0, 5.0 ])
exp_maxcyclelist = np.array([3300, 3000, 2000, 1300, 600 ])
rand_normal_std = 0.1
np.random.seed(42)

def ExpDataCreator(n, C, z):
    return C * np.power(n, z)

_stride = 50
loss_data = [
    np.column_stack(
        (cycles := np.arange(0, maxcycle, _stride),
         ExpDataCreator(cycles, C, z) + np.random.normal(0, rand_normal_std, size=len(cycles)))
    )
    for C, z, maxcycle in zip(exp_clist, exp_zlist, exp_maxcyclelist)
]

# -------------------------
# Model Definition and Constants
# -------------------------
_n = 0.5
_Ea = 10000
_R = 8.314
_T0 = 298.15

def Q1Temp(params, Temp):
    """Temperature-dependent component.
       params: [h, DT, a]"""
    h, DT, a = params
    # Taking absolute values in DT and a to help keep function positive
    return h + np.abs(DT) * np.power(1. - np.exp(np.abs(a) * (Temp - _T0)), 2.)

def Q1ChCrate(params, Crate):
    """Charge rate component.
       params: [ca, A, cb, Rho]"""
    ca, A, cb, Rho = params
    return ca * np.power(A, Crate) + cb * np.exp(Crate / Rho)

def Q1DchCrate(params, Crate):
    """Discharge rate component.
       params: [ca, A, cb, Rho]"""
    ca, A, cb, Rho = params
    return ca * np.power(A, Crate) + cb * np.exp(Crate / Rho)

def CombinedQloss(params, condition, cyc):
    """
    Combined model output.
    
    The parameter vector is assumed to be of length 34:
      - Q1 component uses params[0:12]:
            T_params = params[0:3]   -> for Q1Temp
            Ch_params = params[3:7]  -> for Q1ChCrate (used here for discharge)
            Dch_params = params[7:11] -> for Q1DchCrate (used here for discharge)
            B0 = params[11]          -> scale factor
      - Q3 component uses params[12:34]:
            Tm0_params = params[12:15]
            Chm0_params = params[15:19]
            Dhm0_params = params[19:23]
            TFcm_params = params[23:26]
            ChFcm_params = params[26:30]
            DchFcm_params = params[30:34]
    
    condition: [Temp, ChCrate, DchCrate]
    cyc: cycle number(s)
    
    The function returns Q1_out + Q3_out.
    """
    # Q1 component
    T_params = params[0:3]
    Ch_params = params[3:7]
    Dch_params = params[7:11]
    B0 = params[11]
    Temp, ChCrate, DchCrate = condition
    Q1_out = (B0 * Q1Temp(T_params, Temp) *
              Q1DchCrate(Ch_params, ChCrate) *
              Q1DchCrate(Dch_params, DchCrate) *
              np.exp(-_Ea / (Temp * _R)) *
              np.power(cyc, _n))
    
    # Q3 component
    Tm0_params   = params[12:15]
    Chm0_params  = params[15:19]
    Dhm0_params  = params[19:23]
    TFcm_params  = params[23:26]
    ChFcm_params = params[26:30]
    DchFcm_params= params[30:34]

    Q3_inner = ( Q1Temp(Tm0_params, Temp) +
                 Q1DchCrate(Chm0_params, ChCrate) +
                 Q1DchCrate(Dhm0_params, DchCrate) )
    Q3_factor = np.exp( cyc /(Q1Temp(TFcm_params, Temp) *
                       Q1DchCrate(ChFcm_params, ChCrate) *
                       Q1DchCrate(DchFcm_params, DchCrate))) - 1
    #
    # comment WKJEE 19.02.2025
    # Q_factor exponent function (T, Crates) must control the timing of accel. decay
    #

    Q3_out = np.exp(-Q3_inner) * Q3_factor
    
    return Q1_out + Q3_out

def print_Q3_component(params, condition):

    Temp, ChCrate, DchCrate = condition
    # Q3 component
    Tm0_params   = params[12:15]
    Chm0_params  = params[15:19]
    Dhm0_params  = params[19:23]
    TFcm_params  = params[23:26]
    ChFcm_params = params[26:30]
    DchFcm_params= params[30:34]
    Q3_inner = ( Q1Temp(Tm0_params, Temp) +
                 Q1DchCrate(Chm0_params, ChCrate) +
                 Q1DchCrate(Dhm0_params, DchCrate) )
    Q3_factor = 1./( Q1Temp(TFcm_params, Temp) *
                       Q1DchCrate(ChFcm_params, ChCrate) *
                       Q1DchCrate(DchFcm_params, DchCrate) )

    print(f'Q3_inner : {Q3_inner}')
    print(f'Q3_factor: {Q3_factor}')
    
    return None

def Q3_component(params, condition, cyc):

    Temp, ChCrate, DchCrate = condition
    # Q3 component
    Tm0_params   = params[12:15]
    Chm0_params  = params[15:19]
    Dhm0_params  = params[19:23]
    TFcm_params  = params[23:26]
    ChFcm_params = params[26:30]
    DchFcm_params= params[30:34]
    Q3_inner = ( Q1Temp(Tm0_params, Temp) +
                 Q1DchCrate(Chm0_params, ChCrate) +
                 Q1DchCrate(Dhm0_params, DchCrate) )
    Q3_factor = np.exp( cyc / ( Q1Temp(TFcm_params, Temp) *
                       Q1DchCrate(ChFcm_params, ChCrate) *
                       Q1DchCrate(DchFcm_params, DchCrate) ) ) - 1
    
    Q3_out = np.exp(-Q3_inner) * Q3_factor
    
    return Q3_out


# -------------------------
# Initial Guess and Condition Data
# -------------------------
# Q1 Initial Guess (12 parameters)
T1_initial_guess = [1, 0.2, 0.07]    # for Q1Temp
C1_initial_guess = [1, 1, 1, 20]      # for Q1ChCrate (used in Q1 part)
D1_initial_guess = [1, 4, 24, 30]     # for Q1DchCrate (used in Q1 part)
B0_initial_guess = [5000.]            # scale factor
Q1_initial_guess = T1_initial_guess + C1_initial_guess + D1_initial_guess + B0_initial_guess

# Q3 Initial Guess (22 parameters)
Tm0_initial_guess  = [1, 0.3, 0.5]
Chm0_initial_guess = [2, 1, 2, 16]
Dhm0_initial_guess = [0.5, 4, 0.2, 24]
TFcm_initial_guess = [0.13, 0.5, 0.043]
ChFcm_initial_guess = [2, 1, 2, 16]
DchFcm_initial_guess = [0.5, 4, 0.2, 24]
Q3_initial_guess = (Tm0_initial_guess + Chm0_initial_guess + Dhm0_initial_guess +
                    TFcm_initial_guess + ChFcm_initial_guess + DchFcm_initial_guess)

# Combined initial guess: total length 34
initial_guess = Q1_initial_guess + Q3_initial_guess

# Construct condition data: each row corresponds to one dataset's [Temp, ChCrate, DchCrate]
condition_data = np.column_stack((Templist, Cratelist, Dratelist))

# -------------------------
# Weighted Objective Function with Penalty (for IRLS)
# -------------------------
_penalty = 1e3
def weighted_objective(params, weights, loss_data, condition_data):
    total_loss = 0.0
    # Sum weighted squared residuals over all datasets
    for i, (data, condition) in enumerate(zip(loss_data, condition_data)):
        cycles = data[:, 0]
        Qloss_data = data[:, 1]
        model_output = CombinedQloss(params, condition, cycles)
        residual = Qloss_data - model_output
        total_loss += weights[i] * np.linalg.norm(residual)**2

    # Penalty: enforce that Q1Temp, Q1ChCrate, Q1DchCrate (for all three model parts) are > 0.
    penalty = 0.0
    # For Q1 part:
    for condition in condition_data:
        Temp, ChCrate, DchCrate = condition
        f_temp   = Q1Temp(params[0:3], Temp)
        f_charge = Q1ChCrate(params[3:7], ChCrate)
        f_dch    = Q1DchCrate(params[7:11], DchCrate)
        if f_temp <= 0:
            penalty += _penalty * (abs(f_temp) + 1.0)
        if f_charge <= 0:
            penalty += _penalty * (abs(f_charge) + 1.0)
        if f_dch <= 0:
            penalty += _penalty * (abs(f_dch) + 1.0)
            
    # For Q3 part:
    for condition in condition_data:
        Temp, ChCrate, DchCrate = condition
        fm0_temp   = Q1Temp(params[12:15], Temp)
        fm0_charge = Q1ChCrate(params[15:19], ChCrate)
        fm0_dch    = Q1DchCrate(params[19:23], DchCrate)
        if fm0_temp <= 0:
            penalty += _penalty * (abs(fm0_temp) + 1.0)
        if fm0_charge <= 0:
            penalty += _penalty * (abs(fm0_charge) + 1.0)
        if fm0_dch <= 0:
            penalty += _penalty * (abs(fm0_dch) + 1.0)
        fFcm_temp   = Q1Temp(params[23:26], Temp)
        fFcm_charge = Q1ChCrate(params[26:30], ChCrate)
        fFcm_dch    = Q1DchCrate(params[30:34], DchCrate)
        if fFcm_temp <= 0:
            penalty += _penalty * (abs(fFcm_temp) + 1.0)
        if fFcm_charge <= 0:
            penalty += _penalty * (abs(fFcm_charge) + 1.0)
        if fFcm_dch <= 0:
            penalty += _penalty * (abs(fFcm_dch) + 1.0)
    return total_loss + penalty

# -------------------------
# IRLS with Basin Hopping and Penalty
# -------------------------
_bhop_iter= 40
max_irls_iter = 5   # Maximum number of IRLS iterations
tol = 1e-3           # Convergence tolerance
epsilon = 1e-6       # Small constant to avoid division by zero

weights = np.ones(len(loss_data))
params = np.array(initial_guess)

print("Starting IRLS with Basin Hopping (with penalty) for global optimization:")
for it in range(max_irls_iter):
    minimizer_kwargs = {'method': 'L-BFGS-B', 'tol': tol, 'args': (weights, loss_data, condition_data)}
    result = opt.basinhopping(weighted_objective, params, minimizer_kwargs=minimizer_kwargs, niter=_bhop_iter, disp=True)
    new_params = result.x

    # Update weights: inverse of squared residual plus epsilon for each dataset
    new_weights = []
    for data, condition in zip(loss_data, condition_data):
        cycles = data[:, 0]
        Qloss_data = data[:, 1]
        model_output = CombinedQloss(new_params, condition, cycles)
        residual = Qloss_data - model_output
        res_norm2 = np.linalg.norm(residual)**2
        new_weight = 1.0 / (res_norm2 + epsilon)
        new_weights.append(new_weight)
    new_weights = np.array(new_weights)

    print(f"Iteration {it}:")
    print("  Params =", new_params)
    print("  Weights =", new_weights)

    if np.linalg.norm(new_params - params) < tol and np.linalg.norm(new_weights - weights) < tol:
        params = new_params
        weights = new_weights
        break

    params = new_params
    weights = new_weights

print("\nOptimized Parameters after IRLS with Basin Hopping and Penalty:")
print(params)

# -------------------------
# Plotting the Results
# -------------------------
fig, ax = plt.subplots(3, 2, figsize=(12, 12))
markers = ['o', 's', 'd', '^', 'v', 'x']
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Plot experimental data on the upper left subplot.
for k, (Drate, data) in enumerate(zip(Dratelist, loss_data)):
    x_data, y_data = data[:, 0], data[:, 1]
    ax[0,0].plot(x_data, y_data, linestyle='', marker=markers[k],
                 color=colors[k], label=f'Dch rate: {Drate}')
ax[0,0].legend()
ax[0,0].set_title("Experimental Data")

# Generate fitted model curves using the optimized parameters.
fit_result = [
    np.column_stack(
        (cycles := np.arange(0, maxcycle + int(maxcycle*2.), _stride),
         CombinedQloss(params, condition, cycles))
    )
    for condition, maxcycle in zip(condition_data, exp_maxcyclelist)
]

for k, fit in enumerate(fit_result):
    dx_data, dy_data = fit[:, 0], fit[:, 1]
    ax[0,0].plot(dx_data, dy_data, linestyle='-', color=colors[k],
                 label=f'Fit: {Dratelist[k]}')
    ax[0,1].plot(dx_data, dy_data, linestyle='-', color=colors[k],
                 label=f'Fit: {Dratelist[k]}')
ax[0,0].set_ylim(0, 30)

ax[0,1].legend()
ax[0,1].set_title("Fitted Model")

temp = np.column_stack((tx := np.linspace(280,310,100), Q1Temp(params[0:3], tx)))
ax[1,0].plot(temp[:,0], temp[:,1], linestyle='-')
ax[1,0].set_xlim(280,310)
ax[1,0].set_ylim(0,300)

ch_crate = np.column_stack((cr := np.linspace(0,1,100), Q1ChCrate(params[3:7], cr)))
ax[1,1].plot(ch_crate[:,0], ch_crate[:,1], linestyle='-')

dch_crate = np.column_stack((cr := np.linspace(0,1,100), Q1DchCrate(params[7:11], cr)))
ax[2,0].plot(dch_crate[:,0], dch_crate[:,1], linestyle='-')


Q3_fit_result = [
    np.column_stack(
        #(cycles := np.arange(0, maxcycle + int(maxcycle*2.), _stride),
        (cycles := np.arange(0, maxcycle + int(maxcycle*2.), 0.01),
         Q3_component(params, condition, cycles))
    )
    for condition, maxcycle in zip(condition_data, exp_maxcyclelist)
]
for k, fit in enumerate(Q3_fit_result):
    dx_data, dy_data = fit[:, 0], fit[:, 1]
    ax[2,1].plot(dx_data, dy_data, linestyle='-', color=colors[k],
                 label=f'Fit: {Dratelist[k]}')
    ax[2,1].plot(dx_data, dy_data, linestyle='-', color=colors[k],
                 label=f'Fit: {Dratelist[k]}')

for condition in condition_data:
    print(condition, '   ', end='   ')
    print_Q3_component(params,condition)

plt.show()

