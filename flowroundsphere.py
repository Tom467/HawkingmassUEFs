import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Grid on sphere
# -------------------------
Ntheta = 64
Nphi   = 128

theta = np.linspace(1e-6, np.pi - 1e-6, Ntheta)
phi   = np.linspace(0, 2*np.pi, Nphi, endpoint=False)

TH, PH = np.meshgrid(theta, phi, indexing='ij')

dtheta = theta[1] - theta[0]
dphi   = phi[1] - phi[0]

# -------------------------
# Initial round sphere
# -------------------------
R0 = 1.0
r = R0 * np.ones_like(TH)

# -------------------------
# Helpers
# -------------------------
def laplacian_sphere(f):
    f_theta  = np.gradient(f, dtheta, axis=0)
    f_theta2 = np.gradient(f_theta, dtheta, axis=0)
    f_phi2   = np.gradient(np.gradient(f, dphi, axis=1), dphi, axis=1)
    return (
        f_theta2
        + np.cos(TH)/np.sin(TH) * f_theta
        + f_phi2 / np.sin(TH)**2
    )

def mean_curvature_radial_graph(r):
    # Small-gradient expansion (exact for round sphere)
    return 2.0/r - laplacian_sphere(r)/r**2

def area_element(r):
    return r**2 * np.sin(TH)

def hawking_mass(r, H):
    dA = area_element(r)
    A = np.sum(dA) * dtheta * dphi
    integral_H2 = np.sum(H**2 * dA) * dtheta * dphi
    return np.sqrt(A/(16*np.pi)) * (1 - integral_H2/(16*np.pi))

# -------------------------
# Time stepping (IMCF)
# -------------------------
dt = 0.01
nsteps = 200

times = []
masses = []

for n in range(nsteps):
    H = mean_curvature_radial_graph(r)
    mH = hawking_mass(r, H)

    times.append(n * dt)
    masses.append(mH)

    # IMCF update
    r = r + dt * (1.0 / H)

times  = np.array(times)
masses = np.array(masses)

# -------------------------
# Δm_H(t)
# -------------------------
delta_mH = masses - masses[0]


plt.figure(figsize=(5.5, 4.0), dpi=150)

plt.plot(
    times,
    delta_mH,
    linewidth=1.5
)

plt.xlabel(r'$t$', fontsize=14)
plt.ylabel(r'$\Delta m_H(t)$', fontsize=14)


plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------
# Diagnostics
# -------------------------
print("Initial Hawking mass:", masses[0])
print("Final Hawking mass  :", masses[-1])
print("Max |Δm_H|          :", np.max(np.abs(delta_mH)))

