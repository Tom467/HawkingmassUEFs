import numpy as np
from scipy.special import sph_harm

# ----------------------------
# Grid
# ----------------------------
Ntheta = 96
Nphi   = 192

theta = np.linspace(1e-6, np.pi-1e-6, Ntheta)
phi   = np.linspace(0, 2*np.pi, Nphi, endpoint=False)

dtheta = theta[1] - theta[0]
dphi   = phi[1] - phi[0]

TH, PH = np.meshgrid(theta, phi, indexing='ij')

# ----------------------------
# Parameters
# ----------------------------
R0 = 1.0
l = 2
m = 0

# Real Y_{20}
Y20 = np.real(sph_harm(m, l, PH, TH))

# Normalize so that <Y^2> ~ 1
Y20 /= np.sqrt(np.mean(Y20**2))

# ----------------------------
# Spherical Laplacian eigenvalue
# ----------------------------
lambda_l = l * (l + 1)

# ----------------------------
# Hawking mass function
# ----------------------------
def hawking_mass(r):
    """
    Hawking mass for a radial graph r(theta,phi)
    using the small-gradient expansion.
    """

    # Area element (to sufficient order)
    dA = r**2 * np.sin(TH)

    area = np.sum(dA) * dtheta * dphi

    # Laplacian on S^2 acting on r
    Delta_r = -lambda_l * (r - R0)

    # Mean curvature expansion
    H = 2.0 / r - Delta_r / r**2

    H2_int = np.sum(H**2 * dA) * dtheta * dphi

    mH = np.sqrt(area / (16*np.pi)) * (1.0 - H2_int / (16*np.pi))

    return mH, area, H, H2_int

# ----------------------------
# Diagnostics: round sphere
# ----------------------------
r_round = R0 * np.ones_like(TH)
mH0, A0, H0, H2_0 = hawking_mass(r_round)

print("=== Round sphere ===")
print(f"Area             = {A0:.6e} (expected {4*np.pi:.6e})")
print(f"Mean(H)          = {np.mean(H0):.6e} (expected {2.0:.6e})")
print(f"Integral H^2 dA  = {H2_0:.6e} (expected {16*np.pi:.6e})")
print(f"Hawking mass     = {mH0:.6e} (expected 0)")
print()

# ----------------------------
# V1 scaling test
# ----------------------------
eps_list = [1e-1, 5e-2, 2e-2, 1e-2]

print("=== V1: epsilon scaling ===")
for eps in eps_list:
    r = R0 * (1 + eps * Y20)
    mH, A, H, H2 = hawking_mass(r)
    print(f"eps={eps:.3e} | m_H={mH:.6e} | m_H/eps^2={mH/eps**2:.6e}")

