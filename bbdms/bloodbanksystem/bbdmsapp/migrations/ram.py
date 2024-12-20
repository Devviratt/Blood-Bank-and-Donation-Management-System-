import numpy as np
import matplotlib.pyplot as plt

# Function definition for the ODE dy/dx = x + y
def dydx(x, y):
    return x + y

# Analytical solution of the differential equation
def analytical_solution(x):
    return -x - 1 + 2 * np.exp(x)

# Adams-Moulton Method (Implicit Trapezoidal Rule)
def adams_moulton_method(f, x0, y0, h, x_end):
    n = int((x_end - x0) / h) + 1  # Number of steps
    x = np.linspace(x0, x_end, n)
    y = np.zeros(n)
    y[0] = y0  # Initial condition
   
    print(f"Step size (h): {h}")
    print(f"Initial condition: y(0) = {y0}\n")
   
    # Step 1: Use RK4 for the first step since AM requires two points
    print("Using RK4 for the first step:")
    k1 = f(x[0], y[0])
    k2 = f(x[0] + h / 2, y[0] + h * k1 / 2)
    k3 = f(x[0] + h / 2, y[0] + h * k2 / 2)
    k4 = f(x[0] + h, y[0] + h * k3)
    y[1] = y[0] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
   
    print(f"  k1 = {k1:.4f}, k2 = {k2:.4f}, k3 = {k3:.4f}, k4 = {k4:.4f}")
    print(f"  y[1] = {y[1]:.4f} at x = {x[1]:.1f}\n")
   
    # Adams-Moulton formula for subsequent steps
    print("Using Adams-Moulton Predictor-Corrector:")
    for i in range(1, n - 1):
        y_predictor = y[i] + h * f(x[i], y[i])  # Predictor step (Explicit Euler)
        y_corrector = y[i] + (h / 2) * (f(x[i], y[i]) + f(x[i + 1], y_predictor))  # Corrector step
        y[i + 1] = y_corrector
       
        print(f"Step {i}:")
        print(f"  Predictor (y_predictor) = {y_predictor:.4f}")
        print(f"  Corrector (y[{i+1}]) = {y_corrector:.4f} at x = {x[i+1]:.1f}")
    print()
    return x, y

# Parameters
x0 = 0
y0 = 0
h = 0.2
x_end = 2

# Solve analytically
x_analytical = np.linspace(x0, x_end, 100)
y_analytical = analytical_solution(x_analytical)

# Solve numerically using Adams-Moulton method
x_adams, y_adams = adams_moulton_method(dydx, x0, y0, h, x_end)

# Plot the solutions
plt.figure(figsize=(10, 6))
plt.plot(x_analytical, y_analytical, label='Analytical Solution', color='blue', linestyle='--')
plt.plot(x_adams, y_adams, 'o-', label='Adams-Moulton Method (h=0.2)', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Analytical and Adams-Moulton Methods')
plt.legend()
plt.grid()
plt.show()
