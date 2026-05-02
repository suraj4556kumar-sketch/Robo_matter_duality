"""
100-Magbot Moving Robo-Matter Simulation
with Order-Parameter Analysis Plot
----------------------------------------

This code keeps the same model and parameters from your uploaded code.

It does NOT run animation.
It runs the simulation and draws/saves:

1. local_topological_order_r6.png
2. neighbor_change_rate_nc.png
3. magbot_order_metrics.csv

Run:
    python magbot_order_parameter_full_code.py
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.animation import FuncAnimation
import shutil
import csv


# ============================================================
# Parameters
# ============================================================

@dataclass
class Params:
    # System
    Nbots: int = 120
    Nmag: int = 6

    # Geometry
    R_bot: float = 1.0
    R_ring: float = 0.78
    r_mag: float = 0.105

    # Larger box -> more open space for movement
    box: float = 17.0

    # Translational motion
    # Increased compared with previous code
    v_base: float = 1.10
    v_light_gain: float = 0.10

    # Light-gradient drift strength
    # This makes bots move toward the bright spot, not only rotate faster there.
    light_drift_strength: float = 1.20

    # Chiral rotation
    omega_base: float = 1.00
    omega_light_gain: float = 0.20

    # Noise
    Dt: float = 0.012
    Dr: float = 0.025
    Dphi: float = 0.012

    # Mobilities
    mu_t: float = 1.0
    mu_r: float = 0.50
    mu_phi: float = 1.0

    # Magnetic interaction
    k_m: float = 10.00
    r0: float = 1.50
    r_cut: float = 1.80

    # Steric repulsion
    k_rep: float = 200.0

    # Time
    dt: float = 0.003
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

    # Display
    interval_ms: int = 25
    show_heading_lines: bool = True
    show_magnets: bool = True
    show_interaction_lines: bool = True
    max_interaction_lines: int = 260
    interaction_strength_threshold: float = 0.06
    heading_line_scale: float = 0.82

    # Saving
    save_video: bool = False
    video_path: str = "magbot_100_spacious_moving_with_lines.mp4"
    gif_path: str = "magbot_100_spacious_moving_with_lines.gif"


P = Params()
rng = np.random.default_rng(P.seed)

ALPHA = np.linspace(0.0, 2.0 * np.pi, P.Nmag, endpoint=False)
LOCAL_ANCHORS = np.column_stack((
    P.R_ring * np.cos(ALPHA),
    P.R_ring * np.sin(ALPHA)
))


# ============================================================
# Utilities
# ============================================================

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def rotate_points(points, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    return points @ R.T


# ============================================================
# Light field
# ============================================================

def moving_light_center(t):
    radius = 10.0
    ang = 2.0 * np.pi * t / P.light_period
    return np.array([radius * np.cos(ang), radius * np.sin(ang)])


def second_light_center(t):
    radius = 10.0
    ang = 2.0 * np.pi * t / P.light_period + np.pi
    return np.array([radius * np.cos(ang), radius * np.sin(ang)])


def gaussian_value_and_gradient(pos, center, amp):
    """
    Gaussian light value and its spatial gradient.

    I = amp exp(-|x-c|^2 / 2 sigma^2)
    grad I = I * (c - x) / sigma^2
    """
    diff = center[None, :] - pos
    d2 = np.sum(diff * diff, axis=1)
    val = amp * np.exp(-d2 / (2.0 * P.light_sigma ** 2))
    grad = val[:, None] * diff / (P.light_sigma ** 2)
    return val, grad


def light_intensity_and_gradient(pos, t):
    """
    Returns:
        I    : local light intensity, shape (N,)
        grad : gradient of light intensity, shape (N,2)
    """
    I = np.full(len(pos), P.light_base)
    grad = np.zeros_like(pos)

    if P.light_mode == "homogeneous":
        I += P.light_amp

    elif P.light_mode == "center_spot":
        val, g = gaussian_value_and_gradient(pos, np.array([0.0, 0.0]), P.light_amp)
        I += val
        grad += g

    elif P.light_mode == "moving_spot":
        val, g = gaussian_value_and_gradient(pos, moving_light_center(t), P.light_amp)
        I += val
        grad += g

    elif P.light_mode == "two_spots":
        val1, g1 = gaussian_value_and_gradient(pos, moving_light_center(t), 0.5 * P.light_amp)
        val2, g2 = gaussian_value_and_gradient(pos, second_light_center(t), 0.5 * P.light_amp)
        I += val1 + val2
        grad += g1 + g2

    else:
        raise ValueError(f"Unknown light_mode: {P.light_mode}")

    return np.clip(I, 0.0, 1.0), grad


def light_intensity(pos, t):
    I, _ = light_intensity_and_gradient(pos, t)
    return I


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


pos = initialize_positions()
theta = rng.uniform(-np.pi, np.pi, size=P.Nbots)
phi = np.zeros((P.Nbots, P.Nmag))
chirality = initialize_chirality()


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

    pair_cut = max(2.0 * P.R_bot, 2.0 * P.R_ring + P.r_cut)
    pairs = neighbor_pairs(pos, pair_cut)

    centers, levers, ang = get_bot_magnet_geometry(pos, theta, phi)

    for i, j in pairs:
        rij = pos[j] - pos[i]
        d = np.linalg.norm(rij)

        # Steric repulsion
        if 1e-12 < d < 2.0 * P.R_bot:
            u = rij / d
            overlap = 2.0 * P.R_bot - d
            f_i = -P.k_rep * overlap * u
            F[i] += f_i
            F[j] -= f_i

        # Magnetic interactions
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
# Display interaction segments
# ============================================================

def compute_interaction_segments(pos, theta, phi, max_lines):
    pair_cut = max(2.0 * P.R_bot, 2.0 * P.R_ring + P.r_cut)
    pairs = neighbor_pairs(pos, pair_cut)
    centers, _, ang = get_bot_magnet_geometry(pos, theta, phi)

    segments = []

    for i, j in pairs:
        ci = centers[i]
        cj = centers[j]
        ai = ang[i]
        aj = ang[j]

        r = cj[None, :, :] - ci[:, None, :]
        d2 = np.sum(r * r, axis=-1)
        mask = (d2 > 1e-12) & (d2 < P.r_cut ** 2)

        if not np.any(mask):
            continue

        weight = np.exp(-d2 / (P.r0 ** 2))
        delta = aj[None, :] - ai[:, None]
        strength = weight * np.abs(np.cos(delta))

        ia, jb = np.where(mask)
        for a, b in zip(ia, jb):
            s = float(strength[a, b])
            if s < P.interaction_strength_threshold:
                continue

            x1, y1 = ci[a]
            x2, y2 = cj[b]
            segments.append((x1, y1, x2, y2, s))

    if not segments:
        return []

    segments.sort(key=lambda x: x[4], reverse=True)
    return segments[:max_lines]


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

def step_dynamics(pos, theta, phi, t):
    F, T_body, T_phi = compute_forces_and_torques(pos, theta, phi)

    I, gradI = light_intensity_and_gradient(pos, t)

    # Rotation speed increases under stronger light
    omega_intrinsic = chirality * (P.omega_base + P.omega_light_gain * I)

    # Translational active motion
    v0 = P.v_base + P.v_light_gain * I
    n = np.column_stack((np.cos(theta), np.sin(theta)))

    # Light-gradient drift toward bright region
    F_light = P.light_drift_strength * gradI

    # Magnet internal phase dynamics
    phi = phi + P.mu_phi * T_phi * P.dt
    phi += np.sqrt(2.0 * P.Dphi * P.dt) * rng.normal(size=phi.shape)
    phi = wrap_angle(phi)

    # Position update
    pos = pos + (P.mu_t * F + F_light + v0[:, None] * n) * P.dt
    pos += np.sqrt(2.0 * P.Dt * P.dt) * rng.normal(size=pos.shape)

    # Body rotation update
    theta = theta + (omega_intrinsic + P.mu_r * T_body) * P.dt
    theta += np.sqrt(2.0 * P.Dr * P.dt) * rng.normal(size=theta.shape)
    theta = wrap_angle(theta)

    # Walls
    for i in range(P.Nbots):
        pos[i], theta[i] = reflect_bot(pos[i], theta[i])

    # Hard-core correction
    pos = resolve_overlaps(pos, n_sweeps=3)

    return pos, theta, phi


# ============================================================
# Order-parameter tools
# ============================================================

def metric_neighbor_sets(pos, cutoff):
    sets = [set() for _ in range(P.Nbots)]
    for i, j in neighbor_pairs(pos, cutoff):
        sets[i].add(j)
        sets[j].add(i)
    return sets


def local_topological_order_r6(pos, cutoff=2.35):
    """
    Local sixfold/topological order parameter.

    psi6_i = (1/N_i) sum_j exp(i*6*theta_ij)
    r6 = mean_i |psi6_i|

    High r6  -> crystal-like local hexagonal order
    Low r6   -> liquid/gas/glass disorder
    """
    nsets = metric_neighbor_sets(pos, cutoff)
    values = []

    for i, neigh in enumerate(nsets):
        if len(neigh) < 2:
            values.append(0.0)
            continue

        psi = 0.0 + 0.0j

        for j in neigh:
            dx, dy = pos[j] - pos[i]
            angle = np.arctan2(dy, dx)
            psi += np.exp(1j * 6.0 * angle)

        psi /= len(neigh)
        values.append(abs(psi))

    return float(np.mean(values)), nsets


def neighbor_change_rate(current_sets, previous_sets, dt_between_measurements):
    """
    Measures how rapidly neighbors change.

    High value -> liquid-like rearrangement
    Low value  -> solid/glass/crystal stable contacts
    """
    if previous_sets is None:
        return 0.0

    changes = []
    for cur, prev in zip(current_sets, previous_sets):
        changes.append(len(cur.symmetric_difference(prev)))

    return float(np.mean(changes) / dt_between_measurements)


def run_order_parameter_analysis():
    global pos, theta, phi

    times = []
    r6_values = []
    nc_values = []
    mean_light_values = []

    previous_sets = None

    metric_every = 5
    neighbor_cutoff = 2.35
    dt_metric = P.dt * metric_every

    for frame in range(P.steps):
        t = frame * P.dt
        pos, theta, phi = step_dynamics(pos, theta, phi, t)

        if frame % metric_every == 0:
            r6, current_sets = local_topological_order_r6(pos, neighbor_cutoff)
            nc = neighbor_change_rate(current_sets, previous_sets, dt_metric)
            I = light_intensity(pos, t)

            times.append(t)
            r6_values.append(r6)
            nc_values.append(nc)
            mean_light_values.append(float(np.mean(I)))

            previous_sets = current_sets

    # Save CSV
    with open("magbot_order_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "r6", "neighbor_change_rate", "mean_light"])
        writer.writerows(zip(times, r6_values, nc_values, mean_light_values))

    # Combined paper-style plot
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(times, r6_values, linewidth=2)
    axes[0].axhline(0.30, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Local topological\norder parameter $r_6$")
    axes[0].set_title("Magbot structural order and neighbor change rate")

    axes[1].plot(times, nc_values, linewidth=2)
    axes[1].axhline(5.0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Nearest-neighbor\nchange rate $n_c$")

    plt.tight_layout()
    plt.savefig("magbot_order_parameter_combined.png", dpi=200)
    plt.show()

    # Separate r6 plot
    plt.figure(figsize=(8, 4))
    plt.plot(times, r6_values, linewidth=2)
    plt.axhline(0.30, linestyle="--", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Local topological order parameter r6")
    plt.title("Local topological order vs time")
    plt.tight_layout()
    plt.savefig("local_topological_order_r6.png", dpi=200)
    plt.show()

    # Separate neighbor change plot
    plt.figure(figsize=(8, 4))
    plt.plot(times, nc_values, linewidth=2)
    plt.axhline(5.0, linestyle="--", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Nearest-neighbor change rate")
    plt.title("Nearest-neighbor change rate vs time")
    plt.tight_layout()
    plt.savefig("neighbor_change_rate_nc.png", dpi=200)
    plt.show()

    print("Saved:")
    print("magbot_order_metrics.csv")
    print("magbot_order_parameter_combined.png")
    print("local_topological_order_r6.png")
    print("neighbor_change_rate_nc.png")


# ============================================================
# Main
# ============================================================

run_order_parameter_analysis()
