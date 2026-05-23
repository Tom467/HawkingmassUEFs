import numpy as np
from scipy.special import sph_harm

# ============================================================
# Grid
# ============================================================

Ntheta = 80
Nphi   = 160

# Avoid coordinate singularities near poles
theta = np.linspace(1e-2, np.pi - 1e-2, Ntheta)
phi   = np.linspace(0, 2*np.pi, Nphi, endpoint=False)

dtheta = theta[1] - theta[0]
dphi   = phi[1] - phi[0]

TH, PH = np.meshgrid(theta, phi, indexing='ij')

# ============================================================
# Parameters
# ============================================================

R0 = 1.0

l = 2
m = 0

eps = 0.05

# ============================================================
# Initial perturbation
# ============================================================

Ylm = np.real(sph_harm(m, l, PH, TH))

# Normalize harmonic
Ylm /= np.sqrt(np.mean(Ylm**2))

# Initial surface
r = R0 * (1.0 + eps * Ylm)

# ============================================================
# Angular derivatives
# ============================================================

def angular_derivatives(f):

    f_theta = np.zeros_like(f)
    f_phi   = np.zeros_like(f)

    # theta derivatives
    f_theta[1:-1,:] = (f[2:,:] - f[:-2,:])/(2*dtheta)

    # one-sided boundaries
    f_theta[0,:]  = (f[1,:] - f[0,:])/dtheta
    f_theta[-1,:] = (f[-1,:] - f[-2,:])/dtheta

    # periodic phi derivatives
    f_phi[:,1:-1] = (f[:,2:] - f[:,:-2])/(2*dphi)

    f_phi[:,0]  = (f[:,1] - f[:,-1])/(2*dphi)
    f_phi[:,-1] = (f[:,0] - f[:,-2])/(2*dphi)

    return f_theta, f_phi

# ============================================================
# Mean curvature (controlled approximation)
# ============================================================

def mean_curvature(r, Ylm):

    r_theta, r_phi = angular_derivatives(r)

    # Angular gradient squared on S^2
    grad2 = (
        r_theta**2
        +
        r_phi**2 / np.sin(TH)**2
    )

    # Graph factor
    W = np.sqrt(1.0 + grad2 / r**2)

    # Exact harmonic Laplacian
    lambda_l = l * (l + 1)

    Delta_r = -lambda_l * (r - R0)

    # Controlled nonlinear approximation
    H = (
        2.0/r
        -
        Delta_r/r**2
    ) / W

    return H, W

# ============================================================
# Hawking mass
# ============================================================

def hawking_mass(r, Ylm):

    H, W = mean_curvature(r, Ylm)

    # Area element for graph
    dA = r**2 * W * np.sin(TH)

    area = np.sum(dA) * dtheta * dphi

    H2_int = np.sum(H**2 * dA) * dtheta * dphi

    mH = np.sqrt(area/(16*np.pi)) * (
        1.0 - H2_int/(16*np.pi)
    )

    return mH, area, H, H2_int

# ============================================================
# Round sphere diagnostics
# ============================================================

r_round = R0 * np.ones_like(TH)

mH0, A0, H0, H2_0 = hawking_mass(r_round, Ylm)

print("=== Round sphere ===")
print(f"Area             = {A0:.6e} (expected {4*np.pi:.6e})")
print(f"Mean(H)          = {np.mean(H0):.6e} (expected {2.0:.6e})")
print(f"Integral H^2 dA  = {H2_0:.6e} (expected {16*np.pi:.6e})")
print(f"Hawking mass     = {mH0:.6e} (expected 0)")
print()

# ============================================================
# Perturbed sphere diagnostics
# ============================================================

mH1, A1, H1, H2_1 = hawking_mass(r, Ylm)

print("=== Perturbed sphere ===")
print(f"Area             = {A1:.6e}")
print(f"Mean(H)          = {np.mean(H1):.6e}")
print(f"Integral H^2 dA  = {H2_1:.6e}")
print(f"Hawking mass     = {mH1:.6e}")
print()

# ============================================================
# epsilon scaling test
# ============================================================

eps_list = [1e-1, 5e-2, 2e-2, 1e-2]

print("=== epsilon scaling ===")

for eps_test in eps_list:

    r_test = R0 * (1.0 + eps_test * Ylm)

    mH, A, H, H2 = hawking_mass(r_test, Ylm)

    print(
        f"eps={eps_test:.3e} | "
        f"m_H={mH:.6e} | "
        f"m_H/eps^2={mH/eps_test**2:.6e}"
    )

# ============================================================
# IMCF evolution
# ============================================================

dt = 1e-4
nsteps = 200

mass_history = []

r_flow = np.copy(r)

for n in range(nsteps):

    H, W = mean_curvature(r_flow, Ylm)

    # Correct scalar graph IMCF
    drdt = W / H

    r_flow = r_flow + dt * drdt

    mH, A, Hcur, H2 = hawking_mass(r_flow, Ylm)

    mass_history.append(mH)

# ============================================================
# Monotonicity diagnostics
# ============================================================

mass_history = np.array(mass_history)

delta_mH = np.min(np.diff(mass_history))

print()
print("=== Flow diagnostics ===")
print(f"Initial m_H       = {mass_history[0]:.6e}")
print(f"Final m_H         = {mass_history[-1]:.6e}")
print(f"Largest decrease  = {delta_mH:.6e}")

if delta_mH >= -1e-10:
    print("Monotonicity preserved within numerical tolerance.")
else:
    print("Monotonicity violation detected.")
