import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Schwarzschild parameters
# ----------------------------
M = 1.0             # Mass of the black hole
R0 = 4.0            # Initial radius (R0 > 2*M to stay outside horizon)

# Perturbation amplitude
epsilon = 0.01

# Spherical harmonic Y20
def Y20(theta, phi):
    return 0.5*(3*np.cos(theta)**2 - 1)

# ----------------------------
# Grid (offset theta to avoid poles)
# ----------------------------
Ntheta = 64
Nphi = 128
delta = 1e-6  # tiny offset to avoid theta=0 or pi
theta = np.linspace(delta, np.pi - delta, Ntheta)
phi = np.linspace(0, 2*np.pi, Nphi, endpoint=False)
TH, PH = np.meshgrid(theta, phi, indexing='ij')

# Initial perturbed surface: radial graph
r = R0*(1 + epsilon * Y20(TH, PH))

# ----------------------------
# Helper functions
# ----------------------------
def mean_curvature(r):
    """
    Approximate mean curvature for nearly radial surfaces in a Schwarzschild slice.
    """
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    # First derivatives
    r_theta = np.gradient(r, axis=0)/dtheta
    r_phi = np.gradient(r, axis=1)/dphi

    # Second derivatives
    r_theta2 = np.gradient(r_theta, axis=0)/dtheta
    r_phi2 = np.gradient(r_phi, axis=1)/dphi

    # Spherical Laplacian (avoid singularities at poles)
    sinTH = np.sin(TH)
    sinTH[sinTH < 1e-6] = 1e-6  # regularize very small values
    lap_sphere = r_theta2 + (np.cos(TH)/sinTH)*r_theta + r_phi2/(sinTH**2)

    # Leading order mean curvature
    H = 2/r - lap_sphere/(r**2)
    return H

def surface_area(r):
    dtheta = theta[1]-theta[0]
    dphi = phi[1]-phi[0]
    return np.sum((r**2)*np.sin(TH)*dtheta*dphi)

def hawking_mass(r):
    """Hawking mass for a radial surface in Schwarzschild slice."""
    H = mean_curvature(r)
    dA = r**2 * np.sin(TH) * (theta[1]-theta[0]) * (phi[1]-phi[0])
    integral_H2 = np.sum(H**2 * dA)
    A = surface_area(r)
    mH = np.sqrt(A/(16*np.pi))*(1 - (1/(16*np.pi))*integral_H2)
    return mH

# ----------------------------
# In-slice flow (IMCF-type)
# ----------------------------
dt = 0.01
nsteps = 50
mH_list = []

for step in range(nsteps):
    H = mean_curvature(r)
    drdt = 1.0 / np.maximum(H, 1e-6)  # regularize small H to avoid division by zero
    r += drdt * dt
    mH_list.append(hawking_mass(r))

# ----------------------------
# Output results
# ----------------------------
print(f"Initial m_H: {mH_list[0]}")
print(f"Final m_H: {mH_list[-1]}")

# Plot Hawking mass evolution
plt.figure(figsize=(6,4))
plt.plot(np.arange(nsteps)*dt, mH_list, marker='o')
plt.xlabel('Time')
plt.ylabel('Hawking mass m_H')
plt.title('In-slice flow for perturbed sphere in Schwarzschild (Y20)')
plt.grid(True)
plt.show()
