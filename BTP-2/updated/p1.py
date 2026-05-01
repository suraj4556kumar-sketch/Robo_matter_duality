"""
Updated Many-Magbot Robo-Matter Simulation
------------------------------------------
This version is updated to be closer to the Robo-Matter / Magbot paper idea.

Main updates compared with previous code:
1. Adds intrinsic chiral rotation of each Magbot body.
   - CW Magbot  : chirality = -1
   - CCW Magbot : chirality = +1

2. Adds light-field-controlled activation.
   - Local light intensity controls rotation speed.
   - This imitates the paper idea: LED field -> photoresistor -> motor speed -> rotation speed.

3. Keeps six internal freely rotating magnets per Magbot.
   - Magnetic interaction gives translational force, body torque, and internal magnet phase torque.

4. Fixes the magnetic force sign so aligned magnets attract under the chosen energy model:
       U = -k_m * exp(-r^2/r0^2) * cos(delta)

5. Uses cell-list neighbor search for faster nearby-pair calculation.

6. Keeps hard-disk non-overlap correction and wall reflection.

Dependencies:
    numpy
    matplotlib
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
    # system size
    Nbots: int = 40
    Nmag: int = 6

    # geometry
    R_bot: float = 1.2
    R_ring: float = 0.95
    r_mag: float = 0.15

    # active translational motion
    # In the real paper, Magbots mainly rotate because of vibration + tilted brushes.
    # We keep a small ABP-like drift so the simulation remains visually active.
    v_base: float = 1.20
    v_light_gain: float = 0.15

    # chiral body rotation
    # This is the main paper-inspired update.
    omega_base: float = 0.8       # base angular speed
    omega_light_gain: float = 2.0 # extra angular speed due to light intensity

    # diffusion / noise
    Dt: float = 0.015
    Dr: float = 0.02
    Dphi: float = 0.015

    # mobilities
    mu_t: float = 1.0
    mu_r: float = 0.55
    mu_phi: float = 1.0

    # magnetic interaction
    k_m: float = 1.2
    r0: float = 1.0
    r_cut: float = 2.4

    # steric body repulsion
    k_rep: float = 35.0

    # simulation box half-size
    box: float = 15.0

    # time stepping
    dt: float = 0.01
    steps: int = 900
    seed: int = 7

    # chirality mode:
    # "all_cw", "all_ccw", or "mixed"
    chirality_mode: str = "all_cw"

    # light field mode:
    # "homogeneous", "moving_spot", or "center_spot"
    light_mode: str = "moving_spot"
    light_base: float = 0.25
    light_amp: float = 0.75
    light_sigma: float = 4.0
    light_period: float = 8.0

    # animation
    interval_ms: int = 30
    save_video: bool = False
    video_path: str = "updated_many_magbots_light_chiral.mp4"
    gif_path: str = "updated_many_magbots_light_chiral.gif"


P = Params()
rng = np.random.default_rng(P.seed)

ALPHA = np.linspace(0, 2 * np.pi, P.Nmag, endpoint=False)


# ============================================================
# Basic utilities
# ============================================================

def wrap_angle(a):
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def cross2d(a, b):
    """2D scalar cross product: a_x b_y - a_y b_x."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def rotate_points(points, theta):
    """
    Rotate local points by angle theta.

    points: (M, 2)
    theta : scalar
    return: (M, 2)
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    return points @ R.T


LOCAL_ANCHORS = np.column_stack((
    P.R_ring * np.cos(ALPHA),
    P.R_ring * np.sin(ALPHA)
))


# ============================================================
# Light field
# ============================================================

def light_center(t):
    """
    Position of a moving light spot.
    The light spot moves in a circle.
    """
    radius = 7.0
    ang = 2 * np.pi * t / P.light_period
    return np.array([radius * np.cos(ang), radius * np.sin(ang)])


def light_intensity(pos, t):
    """
    Return local light intensity at each Magbot position.

    pos: (N, 2)
    t  : simulation time

    The returned intensity is dimensionless and clipped to [0, 1].
    """
    if P.light_mode == "homogeneous":
        I = np.full(len(pos), P.light_base + P.light_amp)

    elif P.light_mode == "center_spot":
        c = np.array([0.0, 0.0])
        d2 = np.sum((pos - c) ** 2, axis=1)
        I = P.light_base + P.light_amp * np.exp(-d2 / (2 * P.light_sigma ** 2))

    elif P.light_mode == "moving_spot":
        c = light_center(t)
        d2 = np.sum((pos - c) ** 2, axis=1)
        I = P.light_base + P.light_amp * np.exp(-d2 / (2 * P.light_sigma ** 2))

    else:
        raise ValueError(f"Unknown light mode: {P.light_mode}")

    return np.clip(I, 0.0, 1.0)


# ============================================================
# Wall handling
# ============================================================

def reflect_bot(pos_i, theta_i):
    """
    Reflect a bot from walls.

    The translational heading is tied to body orientation theta_i,
    so wall collision changes theta_i like specular reflection.
    """
    x, y = pos_i
    hit_x = False
    hit_y = False

    xmin = -P.box + P.R_bot
    xmax =  P.box - P.R_bot
    ymin = -P.box + P.R_bot
    ymax =  P.box - P.R_bot

    if x < xmin:
        x = 2 * xmin - x
        hit_x = True
    elif x > xmax:
        x = 2 * xmax - x
        hit_x = True

    if y < ymin:
        y = 2 * ymin - y
        hit_y = True
    elif y > ymax:
        y = 2 * ymax - y
        hit_y = True

    if hit_x:
        theta_i = np.pi - theta_i
    if hit_y:
        theta_i = -theta_i

    return np.array([x, y]), wrap_angle(theta_i)


# ============================================================
# Cell-list neighbor search
# ============================================================

def build_cell_list(pos, cell_size):
    cells = {}
    shift = P.box

    for i in range(len(pos)):
        cx = int((pos[i, 0] + shift) // cell_size)
        cy = int((pos[i, 1] + shift) // cell_size)
        key = (cx, cy)
        cells.setdefault(key, []).append(i)

    return cells


def neighbor_pairs(pos, cutoff):
    """
    Generate unique nearby pairs using a cell list.
    """
    cell_size = cutoff
    cells = build_cell_list(pos, cell_size)
    pairs = []
    seen = set()

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
                        if (i, j) in seen:
                            continue

                        rij = pos[j] - pos[i]
                        if np.dot(rij, rij) <= cutoff * cutoff:
                            pairs.append((i, j))
                            seen.add((i, j))

    return pairs


# ============================================================
# Initialization
# ============================================================

def initialize_positions():
    """
    Rejection-sampling initialization with no overlaps.
    """
    pos = np.zeros((P.Nbots, 2))
    count = 0
    max_tries = 100000

    for _ in range(max_tries):
        if count >= P.Nbots:
            break

        candidate = rng.uniform(
            low=-P.box + P.R_bot,
            high=P.box - P.R_bot,
            size=2
        )

        ok = True
        for j in range(count):
            if np.linalg.norm(candidate - pos[j]) < 2 * P.R_bot:
                ok = False
                break

        if ok:
            pos[count] = candidate
            count += 1

    if count < P.Nbots:
        raise RuntimeError(
            "Could not place all bots without overlap. "
            "Increase P.box or reduce P.Nbots."
        )

    return pos


def initialize_chirality():
    """
    Return chirality array:
        +1 = CCW
        -1 = CW
    """
    if P.chirality_mode == "all_cw":
        return -np.ones(P.Nbots)
    if P.chirality_mode == "all_ccw":
        return np.ones(P.Nbots)
    if P.chirality_mode == "mixed":
        return rng.choice([-1.0, 1.0], size=P.Nbots)

    raise ValueError(f"Unknown chirality mode: {P.chirality_mode}")


pos = initialize_positions()
theta = rng.uniform(-np.pi, np.pi, size=P.Nbots)
phi = np.zeros((P.Nbots, P.Nmag))
chirality = initialize_chirality()


# ============================================================
# Geometry of magnetic sites
# ============================================================

def get_bot_magnet_geometry(pos, theta, phi):
    """
    centers : (N, M, 2) magnet centers
    levers  : (N, M, 2) vectors from body center to magnet center
    ang     : (N, M) magnet orientation angles
    """
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
    """
    Computes:
    1. hard-body steric repulsion force
    2. magnetic translational force
    3. magnetic body torque
    4. internal magnet phase torque
    """
    F = np.zeros((P.Nbots, 2))
    T_body = np.zeros(P.Nbots)
    T_phi = np.zeros((P.Nbots, P.Nmag))

    pair_cut = max(2 * P.R_bot, 2 * P.R_ring + P.r_cut)
    pairs = neighbor_pairs(pos, pair_cut)

    centers, levers, ang = get_bot_magnet_geometry(pos, theta, phi)

    for i, j in pairs:
        rij = pos[j] - pos[i]
        d = np.linalg.norm(rij)

        # ---------------- Steric body repulsion ----------------
        if 1e-12 < d < 2 * P.R_bot:
            u = rij / d
            overlap = 2 * P.R_bot - d

            # force on i points away from j
            f_rep_i = -P.k_rep * overlap * u
            F[i] += f_rep_i
            F[j] -= f_rep_i

        # ---------------- Magnetic interactions ----------------
        ci = centers[i]
        cj = centers[j]
        li = levers[i]
        lj = levers[j]
        ai = ang[i]
        aj = ang[j]

        # r[a,b] = magnet_j[b] - magnet_i[a]
        r = cj[None, :, :] - ci[:, None, :]
        d2 = np.sum(r * r, axis=-1)

        mask = (d2 > 1e-12) & (d2 < P.r_cut ** 2)
        if not np.any(mask):
            continue

        weight = np.exp(-d2 / (P.r0 ** 2))
        delta = aj[None, :] - ai[:, None]

        # Energy model:
        # U = -k_m * w(r) * cos(delta)
        #
        # Phase torque:
        # tau = -dU/d(delta) with opposite signs on the two magnets.
        tau = P.k_m * weight * np.sin(delta)
        tau *= mask

        T_phi[i] += np.sum(tau, axis=1)
        T_phi[j] -= np.sum(tau, axis=0)

        # Force sign:
        # For U = -k_m * exp(-r^2/r0^2) * cos(delta),
        # aligned magnets should attract.
        # Since r = x_j - x_i, force on i is positive along r when cos(delta) > 0.
        coeff = (2.0 * P.k_m / (P.r0 ** 2)) * weight * np.cos(delta)
        coeff *= mask

        f_pair_on_i = coeff[..., None] * r

        F_i_mag = np.sum(f_pair_on_i, axis=(0, 1))
        F[i] += F_i_mag
        F[j] -= F_i_mag

        # Body torque from magnetic forces at magnet anchors.
        T_body[i] += np.sum(cross2d(li[:, None, :], f_pair_on_i))
        T_body[j] += np.sum(cross2d(lj[None, :, :], -f_pair_on_i))

    return F, T_body, T_phi


# ============================================================
# Hard overlap correction
# ============================================================

def resolve_overlaps(pos, n_sweeps=2):
    """
    Correct remaining body overlaps using local neighbor pairs only.
    """
    min_d = 2 * P.R_bot

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
    """
    One Euler-Maruyama update step.
    """
    F, T_body, T_phi = compute_forces_and_torques(pos, theta, phi)

    I = light_intensity(pos, t)

    # Light controls angular speed, matching the idea that light changes motor strength.
    omega_intrinsic = chirality * (P.omega_base + P.omega_light_gain * I)

    # Small translational drift also increases with light.
    v0 = P.v_base + P.v_light_gain * I
    n = np.column_stack((np.cos(theta), np.sin(theta)))

    # Internal magnet phase dynamics
    phi = phi + P.mu_phi * T_phi * P.dt
    phi += np.sqrt(2 * P.Dphi * P.dt) * rng.normal(size=phi.shape)
    phi = wrap_angle(phi)

    # Translational dynamics
    pos = pos + (P.mu_t * F + v0[:, None] * n) * P.dt
    pos += np.sqrt(2 * P.Dt * P.dt) * rng.normal(size=pos.shape)

    # Rotational dynamics:
    # intrinsic chiral rotation + magnetic body torque + rotational noise
    theta = theta + (omega_intrinsic + P.mu_r * T_body) * P.dt
    theta += np.sqrt(2 * P.Dr * P.dt) * rng.normal(size=theta.shape)
    theta = wrap_angle(theta)

    # Wall reflection
    for i in range(P.Nbots):
        pos[i], theta[i] = reflect_bot(pos[i], theta[i])

    # Hard overlap correction
    pos = resolve_overlaps(pos, n_sweeps=2)

    return pos, theta, phi


# ============================================================
# Animation setup
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-P.box, P.box)
ax.set_ylim(-P.box, P.box)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

title = ax.set_title("Updated Magbot Robo-Matter Simulation")

# optional light spot visualization
light_artist = Circle((0, 0), P.light_sigma, facecolor="yellow", edgecolor="gold",
                      alpha=0.18, zorder=0)
ax.add_patch(light_artist)

bodies = []
norths = []
souths = []
heading_lines = []

for i in range(P.Nbots):
    edge_color = "tab:blue" if chirality[i] > 0 else "k"

    body = Circle(tuple(pos[i]), P.R_bot, facecolor="#E8E8FF",
                  edgecolor=edge_color, lw=1.0, zorder=2)
    ax.add_patch(body)
    bodies.append(body)

    # heading line
    line, = ax.plot([], [], lw=1.0, color="k", alpha=0.45, zorder=3)
    heading_lines.append(line)

    bot_north = []
    bot_south = []

    for k in range(P.Nmag):
        center = pos[i] + rotate_points(LOCAL_ANCHORS, theta[i])[k]
        ang = theta[i] + ALPHA[k] + phi[i, k]
        deg = np.degrees(ang)

        n_patch = Wedge(tuple(center), P.r_mag, deg, deg + 180,
                        facecolor="royalblue", edgecolor="none", zorder=4)
        s_patch = Wedge(tuple(center), P.r_mag, deg + 180, deg + 360,
                        facecolor="tomato", edgecolor="none", zorder=4)

        ax.add_patch(n_patch)
        ax.add_patch(s_patch)
        bot_north.append(n_patch)
        bot_south.append(s_patch)

    norths.append(bot_north)
    souths.append(bot_south)

ax.plot([], [], lw=6, color="royalblue", label="North pole")
ax.plot([], [], lw=6, color="tomato", label="South pole")
ax.plot([], [], lw=2, color="k", label="CW body")
ax.plot([], [], lw=2, color="tab:blue", label="CCW body")
ax.legend(loc="upper right", frameon=False)


def update(frame):
    global pos, theta, phi

    t = frame * P.dt
    pos, theta, phi = step_dynamics(pos, theta, phi, t)

    I = light_intensity(pos, t)

    if P.light_mode == "moving_spot":
        c = light_center(t)
        light_artist.center = tuple(c)
        light_artist.set_visible(True)
    elif P.light_mode == "center_spot":
        light_artist.center = (0, 0)
        light_artist.set_visible(True)
    else:
        light_artist.set_visible(False)

    title.set_text(
        f"Updated Magbot Simulation | frame={frame} | "
        f"light={P.light_mode} | mean I={np.mean(I):.2f}"
    )

    artists = [title, light_artist]

    for i in range(P.Nbots):
        bodies[i].center = tuple(pos[i])

        # heading line
        head = pos[i] + 0.9 * P.R_bot * np.array([np.cos(theta[i]), np.sin(theta[i])])
        heading_lines[i].set_data([pos[i, 0], head[0]], [pos[i, 1], head[1]])

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

        artists.append(bodies[i])
        artists.append(heading_lines[i])
        artists.extend(norths[i])
        artists.extend(souths[i])

    return artists


ani = FuncAnimation(
    fig,
    update,
    frames=P.steps,
    interval=P.interval_ms,
    blit=False
)


# ============================================================
# Optional save
# ============================================================

if P.save_video:
    has_ffmpeg = shutil.which("ffmpeg") is not None
    try:
        fps = max(1, int(1 / P.dt))
        if has_ffmpeg:
            from matplotlib.animation import FFMpegWriter
            ani.save(P.video_path, writer=FFMpegWriter(fps=fps))
            print(f"[Saved video] {P.video_path}")
        else:
            from matplotlib.animation import PillowWriter
            ani.save(P.gif_path, writer=PillowWriter(fps=fps))
            print(f"[Saved GIF] {P.gif_path}")
    except Exception as e:
        print("Could not save animation:", e)

plt.show()
