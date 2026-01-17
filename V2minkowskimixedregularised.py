import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Grid on S^2
# -----------------------------
Ntheta = 64
Nphi   = 128

theta = np.linspace(1e-6, np.pi-1e-6, Ntheta)
phi   = np.linspace(0, 2*np.pi, Nphi, endpoint=False)
TH, PH = np.meshgrid(theta, phi, indexing='ij')

dtheta = theta[1] - theta[0]
dphi   = phi[1] - phi[0]

sinTH = np.sin(TH)

# -----------------------------
# Spherical harmonics (real)
# -----------------------------
def Y20(th):
    return 0.25*np.sqrt(5/np.pi)*(3*np.cos(th)**2 - 1)

def Y30(th):
    return 0.25*np.sqrt(7/np.pi)*(5*np.cos(th)**3 - 3*np.cos(th))

# -----------------------------
# Laplacian on S^2 (finite-diff)
# -----------------------------
def laplacian_s2(f):
    fth = np.gradient(f, dtheta, axis=0)
    term1 = np.gradient(sinTH * fth, dtheta, axis=0) / sinTH
    fphph = np.gradient(np.gradient(f, dphi, axis=1), dphi, axis=1)
    return term1 + fphph / sinTH**2

# -----------------------------
# Area + Hawking mass
# -----------------------------
def area_element(r):
    rth = np.gradient(r, dtheta, axis=0)
    rph = np.gradient(r, dphi, axis=1)
    return r**2 * sinTH * np.sqrt(
        1 + (rth/r)**2 + (rph/(r*sinTH))**2
    )

def hawking_mass(r):
    dA = area_element(r)
    A  = np.sum(dA)*dtheta*dphi

    lap = laplacian_s2(r)
    H   = 2/r - lap/(r**2)

    integral_H2 = np.sum(H**2 * dA)*dtheta*dphi
    return np.sqrt(A/(16*np.pi)) * (1 - integral_H2/(16*np.pi))

# -----------------------------
# Initial surface: mixed mode
# -----------------------------
R0  = 1.0
eps = 0.01

r = R0*(1 + eps*(Y20(TH) + Y30(TH)))

# -----------------------------
# Flow parameters
# -----------------------------
dt    = 5e-4
nstep = 200
H_min = 0.1   # curvature floor

mH_history = []

# -----------------------------
# Time evolution
# -----------------------------
for n in range(nstep):
    lap = laplacian_s2(r)
    H   = 2/r - lap/(r**2)

    # regularization
    Hreg = np.where(H > H_min, H, H_min)

    r = r + dt*(1/Hreg)

    mH = hawking_mass(r)
    mH_history.append(mH)

# -----------------------------
# Diagnostics
# -----------------------------
print("Initial Hawking mass:", mH_history[0])
print("Final Hawking mass  :", mH_history[-1])

plt.plot(mH_history)
plt.xlabel("time step")
plt.ylabel("Hawking mass")
plt.title("Regularized mixed-mode IMCF (Minkowski)")
plt.grid()
plt.show()
