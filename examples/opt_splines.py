# %% [markdown]
# # Levenberg-Marquardt Optimization with AADC Derivatives
#
# This notebook demonstrates using Levenberg-Marquardt optimization with both SciPy and the AADC implementation.

# %% [markdown]
# ## 1. Standard Implementation with SciPy
#
# First, let's import the necessary libraries:

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import aadc
import aadc.ndarray
from aadc.scipy.interpolate import CubicSpline

# %matplotlib inline
plt.ioff()  # Turn off interactive mode

# %% [markdown]
# Now we'll define our residual function:

# %%
# Define the function to be minimized (residuals)
def residual_func(params, x_data, y_data):
    # Example: fitting y = a * exp(-b * x) + c
    a, b, c = params
    y_model = a * np.exp(-b * x_data) + c
    return y_model - y_data

# %% [markdown]
# Let's generate some sample data for our optimization problem:

# %%
# Generate some sample data
np.random.seed(42)  # For reproducibility
x_data = np.linspace(0, 10, 10)
true_params = [5.0, 0.5, 1.0]
y_true = true_params[0] * np.exp(-true_params[1] * x_data) + true_params[2]
noise = np.random.normal(0, 0.2, x_data.shape)
y_data = y_true + noise

print("x_data:", x_data)
print("y_data:", y_data)

# Define data for spline fitting
spline_x_data = np.linspace(0, 10, 5)
initial_params = np.ones(np.shape(spline_x_data))

def spline_residual_func(params, x_data, y_data):
    spline = CubicSpline(spline_x_data, aadc.ndarray.AADCArray(params))
    return spline(x_data) - y_data

# %% [markdown]
# Now we'll create an optimization function:

# %%
def run_optimization(x_data, y_data):
    # Perform Levenberg-Marquardt optimization
    result = least_squares(
        spline_residual_func,
        initial_params,
        method='lm',
        args=(x_data, y_data),
        verbose=0
    )

    return result, initial_params

# Run the optimization
result, initial_params = run_optimization(x_data, y_data)

# Print results
print("True parameters:", true_params)
print("Optimal parameters:", result.x)
print("Cost:", result.cost)
print("Success:", result.success)

# Plot the optimization results for SciPy implementation
plt.figure(figsize=(10, 6))

# Plot the noisy data points
plt.scatter(x_data, y_data, color='blue', alpha=0.6, label='Noisy data')

# Plot the true function
plt.plot(x_data, y_true, 'g-', linewidth=2, label='True function')

# Plot the initial guess
plt.plot(spline_x_data, initial_params, 'r--', linewidth=2, label='Initial guess')

# Plot the optimized function
fitted_spline = CubicSpline(spline_x_data, result.x)
y_opt = fitted_spline(x_data)
plt.plot(x_data, y_opt, 'k-', linewidth=2, label='Optimized fit')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('SciPy Levenberg-Marquardt Optimization Results')
plt.legend()
plt.grid(True, alpha=0.3)

# Add text box with optimized parameters
param_text = 'Optimized parameters:\n'
plt.text(6, 4, param_text, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Implementation with AADC
#
# Note that objective function is using y_data internally and derivatives must be calculated with respect to y_data.
# AADC.least_squares() function doesn't require Jacobian w.r.t. y_data, it will be calculated automatically. See Automatic IFT publication for more details.

# %%
# AADC version

func = aadc.Kernel()
func.start_recording()

# Define the residual function with AADC inputs
iy_data = aadc.ndarray.AADCArray([aadc.idouble(y) for y in y_data])
iy_data_arg = [y.mark_as_input() for y in iy_data]

def objective(params):
    return spline_residual_func(params, x_data, iy_data)

initial_params = aadc.ndarray.AADCArray([aadc.idouble(y) for y in initial_params])
aadc_result = aadc.least_squares(objective, initial_params)

print("AADC Optimal parameters:", aadc_result.x)

x_res = [a.mark_as_output() for a in aadc_result.x]

func.stop_recording()

func.print_passive_extract_locations()

inputs = {y_arg: y for y_arg, y in zip(iy_data_arg, y_data)}
request = {x_res[i]: iy_data_arg for i in range(len(x_res))}

res = aadc.evaluate(func, request, inputs)

print("True parameters:", true_params)
print("Scipy Optimal parameters:", result.x)
print("AADC Kernel Optimal parameters:", [res[0][x_res[i]] for i in range(len(x_res))])

# %% [markdown]
# ## 3. Comparing Derivatives
#
# Let's compare the derivatives computed by both methods:

# %%
print("AADC Kernel Derivatives param 0:", res[1][x_res[0]])
print("AADC Kernel Derivatives param 1:", res[1][x_res[1]])
print("AADC Kernel Derivatives param 2:", res[1][x_res[2]])

# %% [markdown]
# Now let's verify the derivatives using the finite difference method:

# %%
# Define the function to calculate derivatives using finite differences
def calculate_finite_diff_derivatives(params, x_data, y_data, epsilon=1e-6):
    derivatives = []

    for i in range(len(params)):
        param_derivatives = []

        for j in range(len(y_data)):
            # Create modified y_data with a bump in one element
            y_bumped = y_data.copy()
            y_bumped[j] += epsilon

            # Run optimization with bumped data
            bumped_result, _ = run_optimization(x_data, y_bumped)

            # Calculate derivative (change in parameter / change in y)
            derivative = (bumped_result.x[i] - params[i]) / epsilon
            param_derivatives.append(derivative)

        derivatives.append(param_derivatives)

    return derivatives

# Calculate finite difference derivatives for all parameters
print("Calculating finite difference derivatives (this may take some time)...")
fd_derivatives = calculate_finite_diff_derivatives(result.x, x_data, y_data)

# Extract AADC derivatives in a format suitable for comparison
aadc_derivatives = []
for i in range(len(result.x)):
    # Extract derivatives from the AADC result
    aadc_deriv_i = np.array([res[1][x_res[i]][y_arg][0] for y_arg in iy_data_arg])
    aadc_derivatives.append(aadc_deriv_i)

    print(f"Parameter {i} derivatives:")
    print(f"  AADC:            {aadc_deriv_i}")
    print(f"  Finite diff:     {fd_derivatives[i]}")
    print(f"  Mean difference: {np.mean(np.abs(aadc_deriv_i - np.array(fd_derivatives[i])))}")
    print()

# %% [markdown]
# ## 4. Combined Visualization
#
# Let's visualize both the optimization results and parameter sensitivities in a single figure:

# %%
# Create a combined figure with optimization results and parameter sensitivities
fig = plt.figure(figsize=(15, 12))

# Plot 1: Optimization Results (Top)
ax1 = plt.subplot2grid((initial_params.size+1, 1), (0, 0), rowspan=1)

# Plot the noisy data points
ax1.scatter(x_data, y_data, color='blue', alpha=0.6, label='Noisy data')

# Plot the true function
ax1.plot(x_data, y_true, 'g-', linewidth=2, label='True function')

# Plot the initial guess
fitted_spline = CubicSpline(spline_x_data, initial_params)
initial_guess = fitted_spline(x_data)
ax1.plot(x_data, initial_guess, 'r--', linewidth=2, label='Initial guess')

# Plot the optimized function
fitted_spline = CubicSpline(spline_x_data, result.x)
y_opt = fitted_spline(x_data)
ax1.plot(x_data, y_opt, 'k-', linewidth=2, label='Optimized fit')

# Add labels and legend
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Levenberg-Marquardt Optimization Results')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Add text box with optimized parameters
param_text = 'Optimized parameters:\n'
ax1.text(0.05, 0.5, param_text, transform=ax1.transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

# Plots 2-4: Parameter Sensitivities (Bottom rows)
for i in range(len(result.x)):
    ax = plt.subplot2grid((initial_params.size+1, 1), (i+1, 0), rowspan=1)

    ax.plot(x_data, aadc_derivatives[i], 'bo-', label='AADC derivative')
    ax.plot(x_data, fd_derivatives[i], 'ro--', label='Finite diff derivative')

    ax.set_title(f'Parameter {i} sensitivity to data points')
    ax.set_xlabel('x')
    ax.set_ylabel(f'd{i}/dy')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]

# # Automatic Implicit Function Theorem - Summary

# ## Paper Details
# - **Title**: Automatic Implicit Function Theorem
# - **Authors**: Dmitri Goloubentsev, Evgeny Lakshtanov, Vladimir Piterbarg
# - **Affiliation**: Matlogica, Universidade de Aveiro, NatWest Markets, Imperial College London
# - **Published**: December 14, 2021 (Revised: May 31, 2022)
# - **SSRN ID**: 3984964
# - **URL**: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3984964](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3984964)
