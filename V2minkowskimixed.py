import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
R0 = 1.0
eps = 0.05
dt = 1e-3
nsteps = 300

Nth = 64
Nph = 128

# -----------------------------
# Spherical grid
# -----------------------------
theta = np.linspace(1e-6, np.pi-1e-6, Nth)
phi   = np.linspace(0, 2*np.pi, Nph, endpoint=False)
TH, PH = np.meshgrid(theta, phi, indexing='ij')

dth = theta[1] - theta[0]
dph = phi[1] - phi[0]

# -----------------------------
# Spherical harmonics: mixed mode
# -----------------------------
Y20 = sph_harm(0, 2, PH, TH).real
Y30 = sph_harm(0, 3, PH, TH).real

Y = Y20 + Y30
Y = Y / np.max(np.abs(Y))   # normalize

# -----------------------------
# Initial surface
# -----------------------------
r = R0 * (1 + eps * Y)

# -----------------------------
# Geometry helpers (Minkowski)
# -----------------------------
def laplacian_s2(f):
    f_th = np.gradient(f, dth, axis=0)
    f_thth = np.gradient(f_th, dth, axis=0)
    f_phph = np.gradient(np.gradient(f, dph, axis=1), dph, axis=1)
    return f_thth + np.cos(TH)/np.sin(TH)*f_th + f_phph/(np.sin(TH)**2)

def mean_curvature(r):
    return 2/r - laplacian_s2(r)/r**2

def area_element(r):
    return r**2 * np.sin(TH)

# -----------------------------
# Hawking mass (Minkowski)
# -----------------------------
def hawking_mass(r):
    H = mean_curvature(r)
    dA = area_element(r)
    area = np.sum(dA)*dth*dph
    intH2 = np.sum(H**2 * dA)*dth*dph
    return np.sqrt(area/(16*np.pi))*(1 - intH2/(16*np.pi))

# -----------------------------
# Evolution loop
# -----------------------------
mH_vals = []
area_vals = []

for step in range(nsteps):
    H = mean_curvature(r)
    r = r + dt*(1/H)

    mH_vals.append(hawking_mass(r))
    area_vals.append(np.sum(area_element(r))*dth*dph)

# -----------------------------
# Diagnostics
# -----------------------------
print("Initial Hawking mass:", mH_vals[0])
print("Final Hawking mass  :", mH_vals[-1])

# -----------------------------
# Plot
# -----------------------------
t = dt*np.arange(len(mH_vals))

plt.figure(figsize=(6,4))
plt.plot(t, mH_vals, lw=2)
plt.xlabel("t")
plt.ylabel("Hawking mass $m_H$")
plt.title("V2: Mixed-mode ($Y_{20}+Y_{30}$) Minkowski evolution")
plt.grid(True)
plt.tight_layout()
plt.show()
