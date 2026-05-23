import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

# ============================================================
# Grid
# ============================================================

Ntheta = 80
Nphi   = 160

theta = np.linspace(1e-2, np.pi-1e-2, Ntheta)
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

eps = 0.02

# viscosity coefficient
eta = 1e-5

# ============================================================
# Initial perturbation
# ============================================================

Ylm = np.real(sph_harm(m, l, PH, TH))

# normalize rms amplitude
Ylm /= np.sqrt(np.mean(Ylm**2))

r = R0 * (1.0 + eps * Ylm)

# ============================================================
# Angular derivatives
# ============================================================

def angular_derivatives(f):

    f_theta = np.zeros_like(f)
    f_phi   = np.zeros_like(f)

    # theta derivatives
    f_theta[1:-1,:] = (
        f[2:,:] - f[:-2,:]
    )/(2*dtheta)

    f_theta[0,:] = (
        f[1,:] - f[0,:]
    )/dtheta

    f_theta[-1,:] = (
        f[-1,:] - f[-2,:]
    )/dtheta

    # periodic phi derivatives
    f_phi[:,1:-1] = (
        f[:,2:] - f[:,:-2]
    )/(2*dphi)

    f_phi[:,0] = (
        f[:,1] - f[:,-1]
    )/(2*dphi)

    f_phi[:,-1] = (
        f[:,0] - f[:,-2]
    )/(2*dphi)

    return f_theta, f_phi

# ============================================================
# Spherical Laplacian
# ============================================================

def sphere_laplacian(f):

    f_theta, f_phi = angular_derivatives(f)

    # second theta derivative
    f_tt = np.zeros_like(f)

    f_tt[1:-1,:] = (
        f[2:,:]
        - 2*f[1:-1,:]
        + f[:-2,:]
    )/(dtheta**2)

    f_tt[0,:] = f_tt[1,:]
    f_tt[-1,:] = f_tt[-2,:]

    # second phi derivative
    f_pp = (
        np.roll(f,-1,axis=1)
        - 2*f
        + np.roll(f,1,axis=1)
    )/(dphi**2)

    sinTH = np.sin(TH)
    cosTH = np.cos(TH)

    lap = (
        f_tt
        +
        (cosTH/sinTH)*f_theta
        +
        f_pp/(sinTH**2)
    )

    return lap

# ============================================================
# Mean curvature
# ============================================================

def mean_curvature(r):

    r_theta, r_phi = angular_derivatives(r)

    grad2 = (
        r_theta**2
        +
        r_phi**2/(np.sin(TH)**2)
    )

    W = np.sqrt(
        1.0 + grad2/r**2
    )

    Delta_r = sphere_laplacian(r)

    H = (
        2.0/r
        -
        Delta_r/r**2
    )/W

    return H, W, Delta_r

# ============================================================
# Hawking mass
# ============================================================

def hawking_mass(r):

    H, W, Delta_r = mean_curvature(r)

    dA = (
        r**2
        * W
        * np.sin(TH)
    )

    area = np.sum(dA)*dtheta*dphi

    H2_int = np.sum(
        H**2 * dA
    )*dtheta*dphi

    mH = np.sqrt(
        area/(16*np.pi)
    )*(
        1.0 - H2_int/(16*np.pi)
    )

    return mH

# ============================================================
# Initial diagnostics
# ============================================================

mH0 = hawking_mass(r)

print("Initial Hawking mass =", mH0)

# ============================================================
# Flow evolution
# ============================================================

dt = 1e-6
nsteps = 100

mass_history = []

r_flow = np.copy(r)

for n in range(nsteps):

    H, W, Delta_r = mean_curvature(r_flow)

    # avoid division by tiny H
    H_safe = np.maximum(H, 1e-6)

    # stabilized IMCF
    drdt = (
        W/H_safe
        +
        eta*Delta_r
    )

    r_flow = r_flow + dt*drdt

    mH = hawking_mass(r_flow)

    mass_history.append(mH)

mass_history = np.array(mass_history)

# ============================================================
# Diagnostics
# ============================================================

delta_mH = np.min(np.diff(mass_history))

print()
print("=== Flow diagnostics ===")
print(f"Initial m_H       = {mass_history[0]:.6e}")
print(f"Final m_H         = {mass_history[-1]:.6e}")
print(f"Largest decrease  = {delta_mH:.6e}")

if delta_mH >= -1e-10:
    print("Monotonicity preserved within tolerance.")
else:
    print("Small monotonicity violations detected.")

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7,5))

plt.plot(
    np.arange(nsteps)*dt,
    mass_history
)

plt.xlabel(r'$\lambda$')
plt.ylabel(r"$m_H$")

#plt.title(
   # "Hawking mass under stabilized IMCF"
#)

plt.grid(True)

plt.tight_layout()
plt.show()
