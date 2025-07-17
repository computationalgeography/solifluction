import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Your data
T_data = np.array([0, 5])
mu_data = np.array([2e13, 2e11])


# Initial guess for A=7.955 and B=0.479 ----> p0=(7.955, 0.479)
# Define the model
def viscosity_model(tpm, A, B):
    return A * np.exp(-abs(tpm) * B)


# Fit it
popt, pcov = curve_fit(viscosity_model, T_data, mu_data, p0=(2e13, 0.460))

A_fit, B_fit = popt
print("mu = A exp(-T B)")
print(f"Fitted A = {A_fit:.4e}")
print(f"Fitted B = {B_fit:.4f}")

# Plot
T_fit = np.linspace(T_data[0], T_data[-1], 100)
mu_fit = viscosity_model(T_fit, A_fit, B_fit)

plt.plot(T_data, mu_data, "o", label="Data")
plt.plot(T_fit, mu_fit, "-", label="Fit")
plt.yscale("log")
plt.xlabel("Temperature (°C)")
plt.ylabel("Viscosity (Pa·s)")
plt.legend()
plt.show()
