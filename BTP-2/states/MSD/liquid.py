"""
100-Magbot MSD Only Code
------------------------

This version uses your current simulation parameters but does NOT show animation.

It silently runs the dynamics for 1600 steps and then plots:

    Time-origin averaged MSD

Outputs:
    1. magbot_trajectory.npy
    2. magbot_msd.csv
    3. magbot_msd_plot.png
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import csv


# ============================================================
# Parameters
# ============================================================

@dataclass
class Params:
    # System
    Nbots: int = 100
    Nmag: int = 6

    # Geometry
    R_bot: float = 1.0
    R_ring: float = 0.78
    r_mag: float = 0.105

    # Box
    box: float = 20.0

    # Translational motion
    v_base: float = 8.00
    v_light_gain: float = 0.55

    # Light-gradient drift
    light_drift_strength: float = 1.20

    # Chiral rotation
    omega_base: float = 7.00
    omega_light_gain: float = 2.20

    # Noise
    Dt: float = 0.012
    Dr: float = 0.025
    Dphi: float = 0.012

    # Mobilities
    mu_t: float = 1.0
    mu_r: float = 0.50
    mu_phi: float = 1.0

    # Magnetic interaction
    k_m: float = 1.15
    r0: float = 1.40
    r_cut: float = 1.45

    # Steric repulsion
    k_rep: float = 200.0

    # Time
    dt: float = 0.015
    steps: int = 1600
    seed: int = 7

    # Chirality: "all_cw", "all_ccw", "mixed"
    chirality_mode: str = "all_cw"

    # Light field: "moving_spot", "center_spot", "two_spots", "homogeneous"
    light_mode: str = "moving_spot"
    light_base: float = 0.15
    light_amp: float = 0.85
    light_sigma: float = 5.5
    light_period: float = 10.0

    # MSD options
    max_lag_fraction: float = 0.5

    # Output files
    trajectory_file: str = "magbot_trajectory.npy"
    msd_csv_file: str = "magbot_msd.csv"
    msd_plot_file: str = "magbot_msd_plot.png"


P = Params()
rng = np.random.default_rng(P.seed)

ALPHA = np.linspace(0.0, 2.0 * np.pi, P.Nmag, endpoint=False)

LOCAL_ANCHORS = np.column_stack((
    P.R_ring * np.cos(ALPHA),
    P.R_ring * np.sin(ALPHA)
))


# ============================================================
# Utility functions
# ============================================================

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def rotate_points(points, theta):
    c = np.cos(theta)
    s = np.sin(theta)

    R = np.array([
        [c, -s],
        [s,  c]
    ])

    return points @ R.T


# ============================================================
# Light field
# ============================================================

def moving_light_center(t):
    radius = 10.0
    ang = 2.0 * np.pi * t / P.light_period

    return np.array([
        radius * np.cos(ang),
        radius * np.sin(ang)
    ])


def second_light_center(t):
    radius = 10.0
    ang = 2.0 * np.pi * t / P.light_period + np.pi

    return np.array([
        radius * np.cos(ang),
        radius * np.sin(ang)
    ])


def gaussian_value_and_gradient(pos, center, amp):
    """
    Gaussian light value and gradient.

    I = amp * exp(-|x-c|^2 / 2 sigma^2)
    grad I = I * (c - x) / sigma^2
    """

    diff = center[None, :] - pos
    d2 = np.sum(diff * diff, axis=1)

    val = amp * np.exp(-d2 / (2.0 * P.light_sigma ** 2))
    grad = val[:, None] * diff / (P.light_sigma ** 2)

    return val, grad


def light_intensity_and_gradient(pos, t):
    I = np.full(len(pos), P.light_base)
    grad = np.zeros_like(pos)

    if P.light_mode == "homogeneous":
        I += P.light_amp

    elif P.light_mode == "center_spot":
        val, g = gaussian_value_and_gradient(
            pos,
            np.array([0.0, 0.0]),
            P.light_amp
        )
        I += val
        grad += g

    elif P.light_mode == "moving_spot":
        val, g = gaussian_value_and_gradient(
            pos,
            moving_light_center(t),
            P.light_amp
        )
        I += val
        grad += g

    elif P.light_mode == "two_spots":
        val1, g1 = gaussian_value_and_gradient(
            pos,
            moving_light_center(t),
            0.5 * P.light_amp
        )

        val2, g2 = gaussian_value_and_gradient(
            pos,
            second_light_center(t),
            0.5 * P.light_amp
        )

        I += val1 + val2
        grad += g1 + g2

    else:
        raise ValueError(f"Unknown light_mode: {P.light_mode}")

    return np.clip(I, 0.0, 1.0), grad


# ============================================================
# Cell-list neighbor search
# ============================================================

def build_cell_list(pos, cell_size):
    cells = {}
    shift = P.box

    for i in range(len(pos)):
        cx = int((pos[i, 0] + shift) // cell_size)
        cy = int((pos[i, 1] + shift) // cell_size)

        cells.setdefault((cx, cy), []).append(i)

    return cells


def neighbor_pairs(pos, cutoff):
    cells = build_cell_list(pos, cutoff)
    pairs = []

    for (cx, cy), inds in cells.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (cx + dx, cy + dy)

                if nb not in cells:
                    continue

                for i in inds:
                    for j in cells[nb]:
                        if j <= i:
                            continue

                        rij = pos[j] - pos[i]

                        if np.dot(rij, rij) <= cutoff * cutoff:
                            pairs.append((i, j))

    return pairs


# ============================================================
# Initialization
# ============================================================

def initialize_positions():
    pos = np.zeros((P.Nbots, 2))

    count = 0
    max_tries = 400000

    low = -P.box + P.R_bot
    high = P.box - P.R_bot

    for _ in range(max_tries):
        if count >= P.Nbots:
            break

        candidate = rng.uniform(low=low, high=high, size=2)

        ok = True

        for j in range(count):
            if np.linalg.norm(candidate - pos[j]) < 2.0 * P.R_bot:
                ok = False
                break

        if ok:
            pos[count] = candidate
            count += 1

    if count < P.Nbots:
        raise RuntimeError("Could not place all bots. Increase box or reduce Nbots.")

    return pos


def initialize_chirality():
    if P.chirality_mode == "all_cw":
        return -np.ones(P.Nbots)

    if P.chirality_mode == "all_ccw":
        return np.ones(P.Nbots)

    if P.chirality_mode == "mixed":
        return rng.choice([-1.0, 1.0], size=P.Nbots)

    raise ValueError(f"Unknown chirality_mode: {P.chirality_mode}")


# ============================================================
# Wall handling
# ============================================================

def reflect_bot(pos_i, theta_i):
    x, y = pos_i

    xmin = -P.box + P.R_bot
    xmax =  P.box - P.R_bot
    ymin = -P.box + P.R_bot
    ymax =  P.box - P.R_bot

    hit_x = False
    hit_y = False

    if x < xmin:
        x = 2.0 * xmin - x
        hit_x = True

    elif x > xmax:
        x = 2.0 * xmax - x
        hit_x = True

    if y < ymin:
        y = 2.0 * ymin - y
        hit_y = True

    elif y > ymax:
        y = 2.0 * ymax - y
        hit_y = True

    if hit_x:
        theta_i = np.pi - theta_i

    if hit_y:
        theta_i = -theta_i

    return np.array([x, y]), wrap_angle(theta_i)


# ============================================================
# Magnet geometry
# ============================================================

def get_bot_magnet_geometry(pos, theta, phi):
    centers = np.zeros((P.Nbots, P.Nmag, 2))
    levers = np.zeros((P.Nbots, P.Nmag, 2))

    ang = theta[:, None] + ALPHA[None, :] + phi

    for i in range(P.Nbots):
        levers[i] = rotate_points(LOCAL_ANCHORS, theta[i])
        centers[i] = pos[i] + levers[i]

    return centers, levers, ang


# ============================================================
# Forces and torques
# ============================================================

def compute_forces_and_torques(pos, theta, phi):
    F = np.zeros((P.Nbots, 2))
    T_body = np.zeros(P.Nbots)
    T_phi = np.zeros((P.Nbots, P.Nmag))

    pair_cut = max(
        2.0 * P.R_bot,
        2.0 * P.R_ring + P.r_cut
    )

    pairs = neighbor_pairs(pos, pair_cut)

    centers, levers, ang = get_bot_magnet_geometry(pos, theta, phi)

    for i, j in pairs:
        rij = pos[j] - pos[i]
        d = np.linalg.norm(rij)

        # --------------------------
        # Steric repulsion
        # --------------------------
        if 1e-12 < d < 2.0 * P.R_bot:
            u = rij / d
            overlap = 2.0 * P.R_bot - d

            f_i = -P.k_rep * overlap * u

            F[i] += f_i
            F[j] -= f_i

        # --------------------------
        # Magnetic interactions
        # --------------------------
        ci = centers[i]
        cj = centers[j]

        li = levers[i]
        lj = levers[j]

        ai = ang[i]
        aj = ang[j]

        r = cj[None, :, :] - ci[:, None, :]
        d2 = np.sum(r * r, axis=-1)

        mask = (d2 > 1e-12) & (d2 < P.r_cut ** 2)

        if not np.any(mask):
            continue

        weight = np.exp(-d2 / (P.r0 ** 2))
        delta = aj[None, :] - ai[:, None]

        # Internal magnet phase torque
        tau = P.k_m * weight * np.sin(delta)
        tau *= mask

        T_phi[i] += np.sum(tau, axis=1)
        T_phi[j] -= np.sum(tau, axis=0)

        # Magnetic force
        coeff = (2.0 * P.k_m / (P.r0 ** 2)) * weight * np.cos(delta)
        coeff *= mask

        f_pair_on_i = coeff[..., None] * r

        F_i_mag = np.sum(f_pair_on_i, axis=(0, 1))

        F[i] += F_i_mag
        F[j] -= F_i_mag

        # Body torque
        T_body[i] += np.sum(cross2d(li[:, None, :], f_pair_on_i))
        T_body[j] += np.sum(cross2d(lj[None, :, :], -f_pair_on_i))

    return F, T_body, T_phi


# ============================================================
# Overlap correction
# ============================================================

def resolve_overlaps(pos, n_sweeps=3):
    min_d = 2.0 * P.R_bot

    for _ in range(n_sweeps):
        pairs = neighbor_pairs(pos, min_d + 1e-9)

        if not pairs:
            break

        for i, j in pairs:
            rij = pos[j] - pos[i]
            d = np.linalg.norm(rij)

            if d < 1e-12:
                u = rng.normal(size=2)
                u /= np.linalg.norm(u)
                d = 0.0

            else:
                u = rij / d

            if d < min_d:
                overlap = min_d - d
                shift = 0.5 * overlap * u

                pos[i] -= shift
                pos[j] += shift

                low = -P.box + P.R_bot
                high = P.box - P.R_bot

                pos[i] = np.clip(pos[i], low, high)
                pos[j] = np.clip(pos[j], low, high)

    return pos


# ============================================================
# Dynamics
# ============================================================

def step_dynamics(pos, theta, phi, chirality, t):
    F, T_body, T_phi = compute_forces_and_torques(pos, theta, phi)

    I, gradI = light_intensity_and_gradient(pos, t)

    # Rotation speed increases under stronger light
    omega_intrinsic = chirality * (
        P.omega_base + P.omega_light_gain * I
    )

    # Translational active motion
    v0 = P.v_base + P.v_light_gain * I

    n = np.column_stack((
        np.cos(theta),
        np.sin(theta)
    ))

    # Light-gradient drift toward bright region
    F_light = P.light_drift_strength * gradI

    # Magnet internal phase dynamics
    phi = phi + P.mu_phi * T_phi * P.dt
    phi += np.sqrt(2.0 * P.Dphi * P.dt) * rng.normal(size=phi.shape)
    phi = wrap_angle(phi)

    # Position update
    pos = pos + (
        P.mu_t * F
        + F_light
        + v0[:, None] * n
    ) * P.dt

    pos += np.sqrt(2.0 * P.Dt * P.dt) * rng.normal(size=pos.shape)

    # Body rotation update
    theta = theta + (
        omega_intrinsic
        + P.mu_r * T_body
    ) * P.dt

    theta += np.sqrt(2.0 * P.Dr * P.dt) * rng.normal(size=theta.shape)
    theta = wrap_angle(theta)

    # Walls
    for i in range(P.Nbots):
        pos[i], theta[i] = reflect_bot(pos[i], theta[i])

    # Hard-core correction
    pos = resolve_overlaps(pos, n_sweeps=3)

    return pos, theta, phi


# ============================================================
# Run simulation silently for MSD
# ============================================================

def run_simulation_for_msd():
    pos = initialize_positions()
    theta = rng.uniform(-np.pi, np.pi, size=P.Nbots)
    phi = np.zeros((P.Nbots, P.Nmag))
    chirality = initialize_chirality()

    trajectory = np.zeros((P.steps + 1, P.Nbots, 2))
    trajectory[0] = pos.copy()

    print("Running simulation for MSD only...")

    for step in range(1, P.steps + 1):
        t = (step - 1) * P.dt

        pos, theta, phi = step_dynamics(
            pos,
            theta,
            phi,
            chirality,
            t
        )

        trajectory[step] = pos.copy()

        if step % 100 == 0:
            print(f"Step {step}/{P.steps} completed")

    print("Simulation completed.")

    return trajectory


# ============================================================
# MSD calculation
# ============================================================

def calculate_msd_time_origin_average(trajectory, dt, max_lag=None):
    """
    Time-origin averaged MSD:

        MSD(tau) = < |r_i(t + tau) - r_i(t)|^2 >

    Average is taken over:
        1. all bots
        2. all possible time origins
    """

    n_frames = trajectory.shape[0]

    if max_lag is None:
        max_lag = int(P.max_lag_fraction * (n_frames - 1))

    lags = np.arange(1, max_lag + 1)
    times = lags * dt

    msd = np.zeros(len(lags))

    for index, lag in enumerate(lags):
        displacement = trajectory[lag:] - trajectory[:-lag]

        squared_displacement = np.sum(displacement ** 2, axis=2)

        msd[index] = np.mean(squared_displacement)

    return times, msd


# ============================================================
# Save MSD CSV
# ============================================================

def save_msd_csv(time_lag, msd):
    with open(P.msd_csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "lag_time",
            "time_origin_averaged_MSD"
        ])

        for t, m in zip(time_lag, msd):
            writer.writerow([t, m])

    print(f"MSD CSV saved as: {P.msd_csv_file}")


# ============================================================
# Plot MSD
# ============================================================

def plot_msd(time_lag, msd):
    plt.figure(figsize=(7, 5))

    plt.plot(
        time_lag,
        msd,
        linewidth=2,
        label="Time-origin averaged MSD"
    )

    plt.xlabel("Lag time")
    plt.ylabel("MSD")
    plt.title("Mean Squared Displacement of 100 Magbots")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(P.msd_plot_file, dpi=300)
    plt.show()

    print(f"MSD plot saved as: {P.msd_plot_file}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    trajectory = run_simulation_for_msd()

    np.save(P.trajectory_file, trajectory)
    print(f"Trajectory saved as: {P.trajectory_file}")

    time_lag, msd = calculate_msd_time_origin_average(
        trajectory,
        P.dt
    )

    save_msd_csv(
        time_lag,
        msd
    )

    plot_msd(
        time_lag,
        msd
    )

    print("MSD-only calculation completed.")