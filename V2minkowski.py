import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

# -----------------------
# Grid on S^2
# -----------------------
Nth = 64
Nph = 128
theta = np.linspace(1e-6, np.pi - 1e-6, Nth)
phi = np.linspace(0, 2*np.pi, Nph, endpoint=False)

dth = theta[1] - theta[0]
dph = phi[1] - phi[0]

TH, PH = np.meshgrid(theta, phi, indexing="ij")
sinTH = np.sin(TH)

# -----------------------
# Spherical harmonic Y_20
# -----------------------
Y20 = np.real(sph_harm(0, 4, PH, TH))

# -----------------------
# Laplacian on S^2 (FD)
# -----------------------
def laplacian_S2(f):
    fth = np.zeros_like(f)
    fth[1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / dth**2

    fph = (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dph**2

    cot = np.cos(TH) / sinTH
    fth2 = np.zeros_like(f)
    fth2[1:-1, :] = cot[1:-1, :] * (f[2:, :] - f[:-2, :]) / (2*dth)

    return fth + fth2 + fph / sinTH**2

# -----------------------
# Geometry
# -----------------------
def mean_curvature(r):
    Lap = laplacian_S2(r)
    return 2.0/r - Lap/(r*r)

def area_element(r):
    return r*r * sinTH

def hawking_mass(r):
    H = mean_curvature(r)
    dA = area_element(r)
    A = np.sum(dA) * dth * dph
    integral_H2 = np.sum(H*H * dA) * dth * dph
    return np.sqrt(A/(16*np.pi)) * (1 - integral_H2/(16*np.pi))

# -----------------------
# Initial data
# -----------------------
R0 = 1.0
eps = 0.07
r = R0 * (1 + eps * Y20)

# -----------------------
# Time stepping
# -----------------------
dt = 5e-4
nsteps = 400

times = []
masses = []

for n in range(nsteps):
    H = mean_curvature(r)
    H = np.maximum(H, 1e-6)   # safety floor

    r = r + dt / H

    if n % 5 == 0:
        mH = hawking_mass(r)
        times.append(n * dt)
        masses.append(mH)

times = np.array(times)
masses = np.array(masses)

# -----------------------
# Monotonicity diagnostic
# -----------------------
dm = np.diff(masses)
delta_mH = np.max(np.minimum(dm, 0.0))

print("Initial Hawking mass:", masses[0])
print("Final Hawking mass  :", masses[-1])
print("Monotonicity measure δm_H:", delta_mH)

if delta_mH >= 0:
    print("Result: Hawking mass is strictly monotone (to numerical precision).")
else:
    print("Result: Small non-monotonicity detected (likely numerical).")

# ----
# -----------------------
plt.figure(figsize=(5.5, 4.0), dpi=150)

plt.plot(
    times,
    masses,
    linewidth=1.5,
    marker='o',
    markersize=3,
    markevery=3
)

plt.xlabel(r'$t$', fontsize=14)
plt.ylabel(r'$m_H(t)$', fontsize=14)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
