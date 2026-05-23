import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

# ============================================================
# Grid
# ============================================================

Ntheta = 80
Nphi   = 160

theta = np.linspace(1e-2, np.pi - 1e-2, Ntheta)
phi   = np.linspace(0, 2*np.pi, Nphi, endpoint=False)

dtheta = theta[1] - theta[0]
dphi   = phi[1] - phi[0]

TH, PH = np.meshgrid(theta, phi, indexing='ij')

# ============================================================
# Parameters
# ============================================================

R0 = 1.0

# spherical harmonic mode
l = 2
m = 0

# fixed spacetime perturbation
alpha = 0.02

# artificial viscosity
eta = 1e-5

# flow parameters
dt = 1e-6
nsteps = 400

# perturbation amplitudes
epsilon_values = [0.01, 0.02, 0.05, 0.50]

# ============================================================
# Initial harmonic
# ============================================================

Ylm = np.real(sph_harm(m, l, PH, TH))

# normalize rms amplitude
Ylm /= np.sqrt(np.mean(Ylm**2))

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
# Extrinsic curvature perturbation
# ============================================================

def spacetime_correction(r):

    return alpha * (r - R0)

# ============================================================
# Hawking mass
# ============================================================

def hawking_mass(r):

    H, W, Delta_r = mean_curvature(r)

    P = spacetime_correction(r)

    H_spacetime_sq = H**2 - P**2

    # numerical safeguard
    H_spacetime_sq = np.maximum(
        H_spacetime_sq,
        1e-12
    )

    dA = (
        r**2
        * W
        * np.sin(TH)
    )

    area = np.sum(dA)*dtheta*dphi

    H2_int = np.sum(
        H_spacetime_sq * dA
    )*dtheta*dphi

    mH = np.sqrt(
        area/(16*np.pi)
    )*(
        1.0 - H2_int/(16*np.pi)
    )

    return mH

# ============================================================
# Plot setup
# ============================================================

plt.figure(figsize=(8,5))

# ============================================================
# Run flows for different epsilon
# ============================================================

for eps in epsilon_values:

    print()
    print("========================================")
    print(f"Running flow for epsilon = {eps:.4f}")
    print("========================================")

    # perturbed initial surface
    r0 = R0 * (1.0 + eps * Ylm)

    r_flow = np.copy(r0)

    mass_history = []
    time_history = []

    for n in range(nsteps):

        H, W, Delta_r = mean_curvature(r_flow)

        P = spacetime_correction(r_flow)

        H_eff_sq = H**2 - P**2

        # numerical safeguard
        H_eff_sq = np.maximum(
            H_eff_sq,
            1e-12
        )

        H_eff = np.sqrt(H_eff_sq)

        # stabilized spacetime flow
        drdt = (
            W/H_eff
            +
            eta*Delta_r
        )

        r_flow = r_flow + dt*drdt

        mH = hawking_mass(r_flow)

        mass_history.append(mH)
        time_history.append(n*dt)

    mass_history = np.array(mass_history)

    # diagnostics
    largest_decrease = np.min(
        np.diff(mass_history)
    )

    print(
        f"Initial m_H      = {mass_history[0]:.6e}"
    )

    print(
        f"Final m_H        = {mass_history[-1]:.6e}"
    )

    print(
        f"Largest decrease = {largest_decrease:.6e}"
    )

    if largest_decrease >= -1e-10:
        print(
            "Monotonicity preserved within tolerance."
        )
    else:
        print(
            "Small monotonicity violations detected."
        )

    # plot
    plt.plot(
        time_history,
        mass_history,
        linewidth=2,
        label=rf'$\epsilon={eps}$'
    )

# ============================================================
# Final figure
# ============================================================

plt.xlabel(r'$\lambda$')
plt.ylabel(r'$m_H$')

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()
