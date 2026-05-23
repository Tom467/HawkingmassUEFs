import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

# ============================================================
# Physical parameters
# ============================================================

R0 = 1.0

# surface perturbation
eps = 0.02

# spacetime perturbation
alpha = 0.02

# spherical harmonic mode
l = 2
m = 0

# viscosity
eta = 1e-5

# flow parameters
dt = 1e-6
nsteps = 400

# ============================================================
# Resolutions to test
# ============================================================

resolution_list = [
    (32, 64),
    (48, 96),
    (64, 128),
    (80, 160),
    (96, 192)
]

# storage
resolution_measure = []
largest_decrease_list = []

# ============================================================
# Loop over resolutions
# ============================================================

for Ntheta, Nphi in resolution_list:

    print()
    print("========================================")
    print(f"Running resolution Ntheta={Ntheta}, Nphi={Nphi}")
    print("========================================")

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    theta = np.linspace(
        1e-2,
        np.pi - 1e-2,
        Ntheta
    )

    phi = np.linspace(
        0,
        2*np.pi,
        Nphi,
        endpoint=False
    )

    dtheta = theta[1] - theta[0]
    dphi   = phi[1] - phi[0]

    TH, PH = np.meshgrid(
        theta,
        phi,
        indexing='ij'
    )

    # --------------------------------------------------------
    # Initial perturbation
    # --------------------------------------------------------

    Ylm = np.real(
        sph_harm(m, l, PH, TH)
    )

    Ylm /= np.sqrt(
        np.mean(Ylm**2)
    )

    r0 = R0 * (
        1.0 + eps*Ylm
    )

    # --------------------------------------------------------
    # Angular derivatives
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Sphere Laplacian
    # --------------------------------------------------------

    def sphere_laplacian(f):

        f_theta, f_phi = angular_derivatives(f)

        f_tt = np.zeros_like(f)

        f_tt[1:-1,:] = (
            f[2:,:]
            - 2.0*f[1:-1,:]
            + f[:-2,:]
        )/(dtheta**2)

        f_tt[0,:]  = f_tt[1,:]
        f_tt[-1,:] = f_tt[-2,:]

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

    # --------------------------------------------------------
    # Mean curvature
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Effective spacetime correction
    # --------------------------------------------------------

    def spacetime_correction(r):

        return alpha * (r - R0)

    # --------------------------------------------------------
    # Hawking mass
    # --------------------------------------------------------

    def hawking_mass(r):

        H, W, Delta_r = mean_curvature(r)

        P = spacetime_correction(r)

        H_eff = H + P

        dA = (
            r**2
            * W
            * np.sin(TH)
        )

        area = np.sum(dA)*dtheta*dphi

        H2_int = np.sum(
            H_eff**2 * dA
        )*dtheta*dphi

        mH = np.sqrt(
            area/(16*np.pi)
        )*(
            1.0 - H2_int/(16*np.pi)
        )

        return mH

    # --------------------------------------------------------
    # Flow evolution
    # --------------------------------------------------------

    r_flow = np.copy(r0)

    mass_history = []

    for n in range(nsteps):

        H, W, Delta_r = mean_curvature(r_flow)

        P = spacetime_correction(r_flow)

        H_eff = H + P

        H_safe = np.maximum(
            H_eff,
            1e-6
        )

        drdt = (
            W/H_safe
            +
            eta*Delta_r
        )

        r_flow = r_flow + dt*drdt

        mH = hawking_mass(r_flow)

        mass_history.append(mH)

    mass_history = np.array(mass_history)

    # --------------------------------------------------------
    # Convergence diagnostic
    # --------------------------------------------------------

    largest_decrease = np.min(
        np.diff(mass_history)
    )

    print(
        f"Largest decrease = "
        f"{largest_decrease:.6e}"
    )

    resolution_measure.append(Ntheta)
    largest_decrease_list.append(
        abs(largest_decrease)
    )

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7,5))

plt.plot(
    resolution_measure,
    largest_decrease_list,
    marker='o',
    linewidth=2
)

plt.yscale('log')

plt.xlabel(r'$N_\theta$')
plt.ylabel(r'$\left|\min(\Delta m_H)\right|$')
plt.grid(True)

plt.tight_layout()

plt.show()
