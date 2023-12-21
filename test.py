from scipy.optimize import fsolve
import numpy as np

# Define your equation with the max function
def equation(x):
    return x**2 + max(10, x*10)

# Rewrite the equation using a piecewise function
def rewritten_equation(x):
    return x**2 + np.piecewise(x, [x < 1, x >= 1], [lambda x: 10, lambda x: x*10])

# Initial guess
initial_guess = 1.0

# Solve the rewritten equation
solution = fsolve(rewritten_equation, initial_guess)

print("Solution:", solution)