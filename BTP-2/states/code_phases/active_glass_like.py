"""
100-Magbot Moving Robo-Matter Simulation
----------------------------------------

This version gives the Magbots more space to move while keeping visible translation and interactions.

What is changed:
1. Stronger translational motion.
2. Light-gradient drift: Magbots move slightly toward the bright light region.
3. More frequent collisions/interactions by using a slightly smaller box.
4. Chiral rotation is still present.
5. Rotation/heading lines are visible.
6. Magnet interaction lines are visible.
7. 100 Magbots, each with 6 magnetic sites.

Run:
    python magbot_100_spacious_moving_with_lines.py
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.animation import FuncAnimation
import shutil


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
    v_base: float = 0.70
    v_light_gain: float = 0.10

    # Light-gradient drift strength
    # This makes bots move toward the bright spot, not only rotate faster there.
    light_drift_strength: float = 1.20

    # Chiral rotation
    omega_base: float = 0.60
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
    r0: float = 2.00
    r_cut: float = 2.80

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
    light_base: float = 0.25
    light_amp: float = 0.20
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
# Animation setup
# ============================================================

fig, ax = plt.subplots(figsize=(8.2, 8.2))
ax.set_xlim(-P.box, P.box)
ax.set_ylim(-P.box, P.box)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

title = ax.set_title("100 Spacious Moving Magbots | rotation + interaction lines")

light1 = Circle((0, 0), P.light_sigma, facecolor="yellow",
                edgecolor="gold", alpha=0.16, zorder=0)
light2 = Circle((0, 0), P.light_sigma, facecolor="yellow",
                edgecolor="gold", alpha=0.10, zorder=0)
ax.add_patch(light1)
ax.add_patch(light2)

interaction_lines = []
for _ in range(P.max_interaction_lines):
    ln, = ax.plot([], [], lw=0.8, color=(0.0, 0.55, 0.0, 0.35), zorder=1.5)
    interaction_lines.append(ln)

bodies = []
norths = []
souths = []
heading_lines = []

for i in range(P.Nbots):
    edge_color = "tab:blue" if chirality[i] > 0 else "black"

    body = Circle(
        tuple(pos[i]),
        P.R_bot,
        facecolor="#E8E8FF",
        edgecolor=edge_color,
        lw=0.8,
        zorder=2
    )
    ax.add_patch(body)
    bodies.append(body)

    line_color = "tab:blue" if chirality[i] > 0 else "black"
    line, = ax.plot([], [], lw=1.4, color=line_color, alpha=0.95, zorder=3)
    heading_lines.append(line)

    bot_north = []
    bot_south = []

    if P.show_magnets:
        levers_i = rotate_points(LOCAL_ANCHORS, theta[i])
        for k in range(P.Nmag):
            center = pos[i] + levers_i[k]
            ang = theta[i] + ALPHA[k] + phi[i, k]
            deg = np.degrees(ang)

            n_patch = Wedge(
                tuple(center), P.r_mag, deg, deg + 180,
                facecolor="royalblue", edgecolor="none", zorder=4
            )
            s_patch = Wedge(
                tuple(center), P.r_mag, deg + 180, deg + 360,
                facecolor="tomato", edgecolor="none", zorder=4
            )

            ax.add_patch(n_patch)
            ax.add_patch(s_patch)
            bot_north.append(n_patch)
            bot_south.append(s_patch)

    norths.append(bot_north)
    souths.append(bot_south)

ax.plot([], [], lw=5, color="royalblue", label="North pole")
ax.plot([], [], lw=5, color="tomato", label="South pole")
ax.plot([], [], lw=2, color="black", label="CW body line")
ax.plot([], [], lw=2, color="tab:blue", label="CCW body line")
ax.plot([], [], lw=2, color="green", label="Magnet interaction")
ax.legend(loc="upper right", frameon=False, fontsize=8)


def update_light_artists(t):
    if P.light_mode == "homogeneous":
        light1.set_visible(False)
        light2.set_visible(False)

    elif P.light_mode == "center_spot":
        light1.center = (0, 0)
        light1.set_visible(True)
        light2.set_visible(False)

    elif P.light_mode == "moving_spot":
        light1.center = tuple(moving_light_center(t))
        light1.set_visible(True)
        light2.set_visible(False)

    elif P.light_mode == "two_spots":
        light1.center = tuple(moving_light_center(t))
        light2.center = tuple(second_light_center(t))
        light1.set_visible(True)
        light2.set_visible(True)


def update(frame):
    global pos, theta, phi

    t = frame * P.dt
    pos, theta, phi = step_dynamics(pos, theta, phi, t)

    I = light_intensity(pos, t)
    update_light_artists(t)

    title.set_text(
        f"100 Spacious Moving Magbots | frame={frame} | light={P.light_mode} | "
        f"mean I={np.mean(I):.2f}"
    )

    artists = [title, light1, light2]

    # interaction lines
    if P.show_interaction_lines:
        segs = compute_interaction_segments(pos, theta, phi, P.max_interaction_lines)
        for idx, ln in enumerate(interaction_lines):
            if idx < len(segs):
                x1, y1, x2, y2, s = segs[idx]
                alpha = min(0.90, 0.18 + 0.72 * s)
                lw = 0.5 + 1.6 * s
                ln.set_data([x1, x2], [y1, y2])
                ln.set_alpha(alpha)
                ln.set_linewidth(lw)
                ln.set_color((0.0, 0.55, 0.0, alpha))
            else:
                ln.set_data([], [])
            artists.append(ln)
    else:
        for ln in interaction_lines:
            ln.set_data([], [])
            artists.append(ln)

    # bots
    for i in range(P.Nbots):
        bodies[i].center = tuple(pos[i])
        artists.append(bodies[i])

        if P.show_heading_lines:
            head = pos[i] + P.heading_line_scale * P.R_bot * np.array(
                [np.cos(theta[i]), np.sin(theta[i])]
            )
            heading_lines[i].set_data([pos[i, 0], head[0]], [pos[i, 1], head[1]])
        else:
            heading_lines[i].set_data([], [])
        artists.append(heading_lines[i])

        if P.show_magnets:
            levers_i = rotate_points(LOCAL_ANCHORS, theta[i])
            for k in range(P.Nmag):
                center = pos[i] + levers_i[k]
                ang = theta[i] + ALPHA[k] + phi[i, k]
                deg = np.degrees(ang)

                norths[i][k].set_center(tuple(center))
                norths[i][k].set_theta1(deg)
                norths[i][k].set_theta2(deg + 180)

                souths[i][k].set_center(tuple(center))
                souths[i][k].set_theta1(deg + 180)
                souths[i][k].set_theta2(deg + 360)

                artists.append(norths[i][k])
                artists.append(souths[i][k])

    return artists


ani = FuncAnimation(
    fig,
    update,
    frames=P.steps,
    interval=P.interval_ms,
    blit=False
)


# ============================================================
# Optional saving
# ============================================================

if P.save_video:
    has_ffmpeg = shutil.which("ffmpeg") is not None

    try:
        fps = max(1, int(1.0 / P.dt))

        if has_ffmpeg:
            from matplotlib.animation import FFMpegWriter
            ani.save(P.video_path, writer=FFMpegWriter(fps=fps))
            print(f"[Saved video] {P.video_path}")

        else:
            from matplotlib.animation import PillowWriter
            ani.save(P.gif_path, writer=PillowWriter(fps=min(fps, 30)))
            print(f"[Saved GIF] {P.gif_path}")

    except Exception as e:
        print("Could not save animation:", e)

plt.show()

