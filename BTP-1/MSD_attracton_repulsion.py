"""
MSD of Two Solid Magbots with Soft Repulsion + Magnetic Interactions
--------------------------------------------------------------------
- Same physics as your two-magbot animation code.
- Runs a long simulation (no animation) and computes MSD.

MSD here = time-averaged MSD, averaged over both bots:
  MSD(τ) = < |r_i(t+τ) - r_i(t)|^2 >_{t, i=1,2}
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
class P:
    # Geometry
    R_bot = 1.2
    R_ring = 0.95
    r_mag  = 0.18
    Nmag   = 6

    # ABP motion
    v0  = 0.5
    Dt  = 0.02
    Dr  = 0.08
    mu_t = 1.0
    mu_r = 1.0

    # Internal magnet dynamics
    mu_phi = 1.0
    Dphi   = 0.02
    k_m  = 2.5
    r0   = 1.0

    # Body repulsion
    k_rep = 10.0

    # Box (reflecting walls)
    box  = 8.0

    # Numerics
    dt    = 0.04
    steps = 20000   # increase for smoother MSD
    seed  = 7

# Fixed anchor angles of magnets
ALPHA = np.linspace(0, 2*np.pi, P.Nmag, endpoint=False)

# --------------- Helper functions ---------------
def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s,  c]])

def mag_centers(pos, theta):
    """
    pos: (2,) COM position
    theta: scalar body angle
    return: (Nmag, 2) centers of magnets
    """
    R = rot(theta)
    local = np.stack([P.R_ring*np.cos(ALPHA), P.R_ring*np.sin(ALPHA)], axis=0)  # (2,Nmag)
    return (R @ local).T + pos  # (Nmag,2)

def reflect(p, v):
    """
    Reflect position p and velocity v at square walls (size ±box),
    keeping the disk of radius R_bot inside.
    """
    x, y = p
    vx, vy = v

    if x < -P.box + P.R_bot:
        x  = -2*(P.box-P.R_bot) - x
        vx = abs(vx)
    if x >  P.box - P.R_bot:
        x  =  2*(P.box-P.R_bot) - x
        vx = -abs(vx)
    if y < -P.box + P.R_bot:
        y  = -2*(P.box-P.R_bot) - y
        vy = abs(vy)
    if y >  P.box - P.R_bot:
        y  =  2*(P.box-P.R_bot) - y
        vy = -abs(vy)

    return np.array([x,y]), np.array([vx,vy])

# --------------- Physics: forces & torques ---------------
def repulsive_force(p1, p2):
    """
    Linear spring repulsion if bot surfaces overlap.
    Returns forces (F_on_1, F_on_2).
    """
    r = p2 - p1
    d = np.linalg.norm(r)
    if d == 0:
        return np.zeros(2), np.zeros(2)
    u = r / d
    overlap = 2*P.R_bot - d
    if overlap > 0:
        F = -P.k_rep * overlap * u
        return F, -F
    else:
        return np.zeros(2), np.zeros(2)

def magnetic_torques(p1, th1, phi1, p2, th2, phi2):
    """
    Same magnet-magnet torque model as your code:
      T ~ k_m * exp(-(d/r0)^2) * sin(angle_difference)
    Returns (Tphi1, Tphi2) arrays over magnets.
    """
    Tphi1 = np.zeros(P.Nmag)
    Tphi2 = np.zeros(P.Nmag)

    c1 = mag_centers(p1, th1)  # (Nmag,2)
    c2 = mag_centers(p2, th2)

    ang1 = th1 + ALPHA + phi1
    ang2 = th2 + ALPHA + phi2

    for i in range(P.Nmag):
        for j in range(P.Nmag):
            r = c2[j] - c1[i]
            d = np.linalg.norm(r)
            if d < 1e-6 or d > 2*P.r0:
                continue
            weight = np.exp(-(d/P.r0)**2)
            delta = (ang2[j] - ang1[i])
            torque = P.k_m * weight * np.sin(delta)
            Tphi1[i] += torque
            Tphi2[j] -= torque
    return Tphi1, Tphi2

# --------------- Simulation ---------------
def simulate_two_magbots(return_traj=True):
    """
    Run one long simulation of two interacting Magbots.
    Returns:
      traj: array (steps, 2, 2) of positions if return_traj=True
    """
    rng = np.random.default_rng(P.seed)

    # initial positions, orientations, magnet phases
    pos = [np.array([-3.0, 0.0], dtype=float),
           np.array([ 3.0, 0.0], dtype=float)]
    theta = [0.0, np.pi]
    phi   = [np.zeros(P.Nmag, dtype=float),
             np.zeros(P.Nmag, dtype=float)]

    if return_traj:
        traj = np.zeros((P.steps, 2, 2), dtype=float)
        traj[0,0,:] = pos[0]
        traj[0,1,:] = pos[1]

    for step in range(1, P.steps):
        # ---- Magnetic torques ----
        Tphi1, Tphi2 = magnetic_torques(pos[0], theta[0], phi[0],
                                        pos[1], theta[1], phi[1])

        # ---- Magnet spin updates ----
        phi[0] += P.mu_phi * Tphi1 * P.dt + np.sqrt(2*P.Dphi*P.dt)*rng.normal(size=P.Nmag)
        phi[1] += P.mu_phi * Tphi2 * P.dt + np.sqrt(2*P.Dphi*P.dt)*rng.normal(size=P.Nmag)

        # ---- Repulsive body-body forces ----
        F1, F2 = repulsive_force(pos[0], pos[1])

        # ---- Body motion (ABP + force + noise) ----
        for i, F in enumerate([F1, F2]):
            n = np.array([np.cos(theta[i]), np.sin(theta[i])])
            v_det = P.mu_t*F + P.v0*n
            v_sto = np.sqrt(2*P.Dt*P.dt)*rng.normal(size=2)
            v = v_det + v_sto
            pos[i] += v * P.dt
            theta[i] += np.sqrt(2*P.Dr*P.dt)*rng.normal()
            pos[i], _ = reflect(pos[i], v)

        if return_traj:
            traj[step,0,:] = pos[0]
            traj[step,1,:] = pos[1]

    if return_traj:
        return traj
    else:
        return None

# --------------- MSD computation ---------------
def msd_time_averaged(traj, dt, max_lag_fraction=0.5):
    """
    Time-averaged MSD, averaged over both bots.

    traj: (T, 2, 2) positions
    """
    T = traj.shape[0]
    max_lag = int(T * max_lag_fraction)
    taus = np.arange(1, max_lag+1)
    msd  = np.zeros_like(taus, dtype=float)

    # average over bots i=0,1 and time t
    for idx, k in enumerate(taus):
        diffs = traj[k:, :, :] - traj[:-k, :, :]     # shape (T-k, 2, 2)
        sq = np.sum(diffs*diffs, axis=2)            # (T-k, 2)
        msd[idx] = np.mean(sq)                      # average over time and bots
    return taus*dt, msd

# --------------- Main: run and plot ---------------
if __name__ == "__main__":
    print("Running simulation...")
    traj = simulate_two_magbots(return_traj=True)
    print("Computing MSD...")
    tau, msd = msd_time_averaged(traj, P.dt, max_lag_fraction=0.5)

    # Plot MSD
    plt.figure(figsize=(6,4))
    plt.plot(tau, msd, label="MSD (2 Magbots with repulsion + magnets)")
    plt.xlabel(r"Lag time $\tau$")
    plt.ylabel(r"$\langle \Delta r^2(\tau)\rangle$")
    plt.title("MSD of Interacting Magbots (Simulation)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

