import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# ============================================================
# Grid setup
# ============================================================

Ntheta = 80
Nphi   = 160

theta = np.linspace(1e-6, np.pi - 1e-6, Ntheta)
phi   = np.linspace(0, 2*np.pi, Nphi, endpoint=False)

dtheta = theta[1] - theta[0]
dphi   = phi[1] - phi[0]

TH, PH = np.meshgrid(theta, phi, indexing='ij')

# ============================================================
# Parameters
# ============================================================

R0 = 1.0

# Choose spherical harmonic mode
l = 2
m = 0

# ============================================================
# Real spherical harmonic Y_{20}
# ============================================================

Ylm = np.real(sph_harm(m, l, PH, TH))

# Normalize harmonic so rms amplitude = 1
Ylm /= np.sqrt(np.mean(Ylm**2))

# ============================================================
# Angular derivatives
# ============================================================

def angular_derivatives(f):

    f_theta = np.gradient(f, dtheta, axis=0)
    f_phi   = np.gradient(f, dphi, axis=1)

    f_thetatheta = np.gradient(f_theta, dtheta, axis=0)
    f_phiphi     = np.gradient(f_phi, dphi, axis=1)

    return f_theta, f_phi, f_thetatheta, f_phiphi

# ============================================================
# Spherical Laplacian
# ============================================================

def sphere_laplacian(f):

    f_theta, f_phi, f_tt, f_pp = angular_derivatives(f)

    sinTH = np.sin(TH)
    cosTH = np.cos(TH)

    lap = (
        f_tt
        + (cosTH / sinTH) * f_theta
        + (1.0 / sinTH**2) * f_pp
    )

    return lap

# ============================================================
# Geometry for radial graph
# ============================================================

def compute_geometry(r):

    r_theta, r_phi, _, _ = angular_derivatives(r)

    grad_sq = (
        r_theta**2
        + (r_phi**2 / np.sin(TH)**2)
    )

    # Area element
    dA = r**2 * np.sin(TH) * np.sqrt(
        1.0 + grad_sq / r**2
    )

    area = np.sum(dA) * dtheta * dphi

    # Intermediate-order curvature approximation
    Delta_r = sphere_laplacian(r)

    H = (
        2.0 / r
        - Delta_r / r**2
    ) / np.sqrt(1.0 + grad_sq / r**2)

    return area, dA, H

# ============================================================
# Hawking mass
# ============================================================

def hawking_mass(r):

    area, dA, H = compute_geometry(r)

    H2_int = np.sum(H**2 * dA) * dtheta * dphi

    mH = np.sqrt(area / (16.0*np.pi)) * (
        1.0 - H2_int / (16.0*np.pi)
    )

    return mH, area, H, H2_int

# ============================================================
# Round sphere diagnostics
# ============================================================

print("=== Round sphere ===")

r_round = R0 * np.ones_like(TH)

mH0, A0, H0, H2_0 = hawking_mass(r_round)

print(f"Area             = {A0:.6e} (expected {4*np.pi:.6e})")
print(f"Mean(H)          = {np.mean(H0):.6e} (expected 2)")
print(f"Integral H^2 dA  = {H2_0:.6e} (expected {16*np.pi:.6e})")
print(f"Hawking mass     = {mH0:.6e} (expected 0)")
print()

# ============================================================
# epsilon^2 scaling test
# ============================================================

eps_list = np.array([
    0.01,
    0.02,
    0.03,
    0.05,
    0.07,
    0.10
])

mH_list = []

print("=== epsilon scaling ===")

for eps in eps_list:

    r = R0 * (1.0 + eps * Ylm)

    mH, A, H, H2 = hawking_mass(r)

    mH_list.append(mH)

    print(
        f"eps={eps:.3e} | "
        f"m_H={mH:.6e} | "
        f"m_H/eps^2={mH/(eps**2):.6e}"
    )

mH_list = np.array(mH_list)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7,5))

plt.plot(
    eps_list**2,
    mH_list,
    'o-'
)

plt.xlabel(r'$\epsilon^2$')
plt.ylabel(r'$m_H$')



plt.grid(True)

plt.tight_layout()
plt.show()
