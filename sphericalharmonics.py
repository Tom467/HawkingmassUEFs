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

eps = 0.02
alpha = 0.02

eta = 1e-5

dt = 1e-6
nsteps = 400

# modes to test
mode_list = [1, 2, 3, 4]

# ============================================================
# Angular derivatives
# ============================================================

def angular_derivatives(f):

    f_theta = np.zeros_like(f)
    f_phi   = np.zeros_like(f)

    # theta derivative
    f_theta[1:-1,:] = (
        f[2:,:] - f[:-2,:]
    )/(2*dtheta)

    f_theta[0,:] = (
        f[1,:] - f[0,:]
    )/dtheta

    f_theta[-1,:] = (
        f[-1,:] - f[-2,:]
    )/dtheta

    # phi derivative
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
# Laplacian on S^2
# ============================================================

def sphere_laplacian(f):

    f_theta, f_phi = angular_derivatives(f)

    # theta second derivative
    f_tt = np.zeros_like(f)

    f_tt[1:-1,:] = (
        f[2:,:]
        - 2.0*f[1:-1,:]
        + f[:-2,:]
    )/(dtheta**2)

    f_tt[0,:]  = f_tt[1,:]
    f_tt[-1,:] = f_tt[-2,:]

    # phi second derivative
    f_pp = (
        np.roll(f,-1,axis=1)
        - 2.0*f
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
# Extrinsic curvature trace
# ============================================================

def extrinsic_trace(Ylm):

    return 2.0 * alpha * Ylm

# ============================================================
# Spacetime Hawking mass
# ============================================================

def hawking_mass_spacetime(r, Ylm):

    H, W, Delta_r = mean_curvature(r)

    P = extrinsic_trace(Ylm)

    H_spacetime_sq = H**2 - P**2

    dA = (
        r**2
        * W
        * np.sin(TH)
    )

    area = np.sum(dA)*dtheta*dphi

    integrand = (
        H_spacetime_sq * dA
    )

    H2_int = np.sum(integrand)*dtheta*dphi

    mH = np.sqrt(
        area/(16*np.pi)
    )*(
        1.0 - H2_int/(16*np.pi)
    )

    return mH

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7,5))

for l in mode_list:

    m = 0

    Ylm = np.real(
        sph_harm(m, l, PH, TH)
    )

    Ylm /= np.sqrt(np.mean(Ylm**2))

    r0 = R0 * (
        1.0 + eps*Ylm
    )

    r_flow = np.copy(r0)

    mass_history = []
    time_history = []

    for n in range(nsteps):

        H, W, Delta_r = mean_curvature(r_flow)

        H_safe = np.maximum(H, 1e-6)

        drdt = (
            W/H_safe
            +
            eta*Delta_r
        )

        r_flow = r_flow + dt*drdt

        mH = hawking_mass_spacetime(
            r_flow,
            Ylm
        )

        mass_history.append(mH)
        time_history.append(n*dt)

    mass_history = np.array(mass_history)

    plt.plot(
        time_history,
        mass_history,
        linewidth=2,
        label=rf"$Y_{{{l}0}}$"
    )

    largest_decrease = np.min(
        np.diff(mass_history)
    )

    print()
    print(f"Mode l={l}")
    print(
        f"Largest decrease = "
        f"{largest_decrease:.6e}"
    )

# ============================================================
# Final plot
# ============================================================

plt.xlabel(r'$\lambda$')
plt.ylabel(r'$m_H$')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
