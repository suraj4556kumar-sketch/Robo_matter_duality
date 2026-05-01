"""
Two Solid Magbots with Rotating Internal Magnets (No Overlap)
Physics:
- Each Magbot is a rigid disk that cannot overlap with another.
- Each Magbot has 6 internal magnets rotating with interactions.
- COM motion follows Active Brownian Particle dynamics with reflection.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.animation import FuncAnimation
import shutil  # to detect ffmpeg

# ---------------- Parameters ----------------
class P:
    R_bot = 1.2
    R_ring = 0.95
    r_mag  = 0.18
    Nmag   = 6

    v0  = 0.5
    Dt  = 0.02
    Dr  = 0.08
    mu_t = 1.0
    mu_r = 1.0

    mu_phi = 1.0
    Dphi   = 0.02
    k_m  = 2.5
    r0   = 1.0

    k_rep = 10.0
    box  = 8.0
    dt   = 0.04
    steps = 600
    seed = 7

    # Output
    video_path = "two_magbots.mp4"
    gif_path   = "two_magbots.gif"

# Fixed anchor angles of magnets
ALPHA = np.linspace(0, 2*np.pi, P.Nmag, endpoint=False)

# --------------- Helpers ---------------
def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s,  c]])

def mag_centers(pos, theta):
    R = rot(theta)
    local = np.stack([P.R_ring*np.cos(ALPHA), P.R_ring*np.sin(ALPHA)], axis=0)
    return (R @ local).T + pos

def reflect(p, v):
    x, y = p
    vx, vy = v
    if x < -P.box + P.R_bot: x = -2*(P.box-P.R_bot)-x; vx = abs(vx)
    if x >  P.box - P.R_bot: x =  2*(P.box-P.R_bot)-x; vx = -abs(vx)
    if y < -P.box + P.R_bot: y = -2*(P.box-P.R_bot)-y; vy = abs(vy)
    if y >  P.box - P.R_bot: y =  2*(P.box-P.R_bot)-y; vy = -abs(vy)
    return np.array([x,y]), np.array([vx,vy])

# --------------- Physics Functions ---------------
def repulsive_force(p1, p2):
    r = p2 - p1
    d = np.linalg.norm(r)
    if d == 0: return np.zeros(2), np.zeros(2)
    u = r / d
    overlap = 2*P.R_bot - d
    if overlap > 0:
        F = -P.k_rep * overlap * u
        return F, -F
    else:
        return np.zeros(2), np.zeros(2)

def magnetic_torques(p1, th1, phi1, p2, th2, phi2):
    Tphi1 = np.zeros(P.Nmag)
    Tphi2 = np.zeros(P.Nmag)

    c1 = mag_centers(p1, th1)
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

def draw_bot(ax, pos, theta, phi, face):
    body = Circle(pos, P.R_bot, facecolor=face, edgecolor='k', lw=1.6, zorder=2)
    ax.add_patch(body)
    for k in range(P.Nmag):
        ang = ALPHA[k] + theta + phi[k]
        c = pos + P.R_ring * np.array([np.cos(ALPHA[k] + theta), np.sin(ALPHA[k] + theta)])
        ax.add_patch(Wedge(c, P.r_mag, np.degrees(ang), np.degrees(ang)+180,
                           facecolor='royalblue', edgecolor='none', zorder=4))
        ax.add_patch(Wedge(c, P.r_mag, np.degrees(ang)+180, np.degrees(ang)+360,
                           facecolor='tomato', edgecolor='none', zorder=4))
    return body

# --------------- Initialization ---------------
rng = np.random.default_rng(P.seed)
pos = [np.array([-3.0, 0.0]), np.array([3.0, 0.0])]
theta = [0.0, np.pi]
phi = [np.zeros(P.Nmag), np.zeros(P.Nmag)]
trail1_x, trail1_y = [pos[0][0]], [pos[0][1]]
trail2_x, trail2_y = [pos[1][0]], [pos[1][1]]

# --------------- Animation ---------------
fig, ax = plt.subplots(figsize=(7,7))
ax.set_xlim(-P.box, P.box)
ax.set_ylim(-P.box, P.box)
ax.set_aspect('equal')
ax.set_title("Two Solid Magbots — Rotating Magnets Interaction")

def update(frame):
    global pos, theta, phi

    ax.clear()
    ax.set_xlim(-P.box, P.box)
    ax.set_ylim(-P.box, P.box)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Two Solid Magbots — Rotating Magnets Interaction")

    # ---- Magnetic torques ----
    Tphi1, Tphi2 = magnetic_torques(pos[0], theta[0], phi[0],
                                    pos[1], theta[1], phi[1])

    # ---- Magnet spin updates ----
    phi[0] += P.mu_phi * Tphi1 * P.dt + np.sqrt(2*P.Dphi*P.dt)*rng.normal(size=P.Nmag)
    phi[1] += P.mu_phi * Tphi2 * P.dt + np.sqrt(2*P.Dphi*P.dt)*rng.normal(size=P.Nmag)

    # ---- Repulsive body-body forces ----
    F1, F2 = repulsive_force(pos[0], pos[1])

    # ---- Body motion ----
    for i, F in enumerate([F1, F2]):
        n = np.array([np.cos(theta[i]), np.sin(theta[i])])
        v_det = P.mu_t*F + P.v0*n
        v_sto = np.sqrt(2*P.Dt*P.dt)*rng.normal(size=2)
        v = v_det + v_sto
        pos[i] += v * P.dt
        theta[i] += np.sqrt(2*P.Dr*P.dt)*rng.normal()
        pos[i], _ = reflect(pos[i], v)

    # ---- Draw bots ----
    draw_bot(ax, pos[0], theta[0], phi[0], '#E6E8FF')
    draw_bot(ax, pos[1], theta[1], phi[1], '#FFE8E6')

    # trails
    trail1_x.append(pos[0][0]); trail1_y.append(pos[0][1])
    trail2_x.append(pos[1][0]); trail2_y.append(pos[1][1])
    ax.plot(trail1_x, trail1_y, lw=1, alpha=0.4, color='#3B5BDB')
    ax.plot(trail2_x, trail2_y, lw=1, alpha=0.4, color='#D94841')

    ax.plot([], [], lw=6, color='royalblue', label='North (blue)')
    ax.plot([], [], lw=6, color='tomato', label='South (red)')
    ax.legend(loc='upper right', frameon=False)
    return []

ani = FuncAnimation(fig, update, frames=P.steps, interval=40, blit=False)

# --------------- Auto-Save Animation ---------------
has_ffmpeg = shutil.which("ffmpeg") is not None
try:
    if has_ffmpeg:
        from matplotlib.animation import FFMpegWriter
        ani.save(P.video_path, writer=FFMpegWriter(fps=25))
        print(f"[Saved video] {P.video_path}")
    else:
        from matplotlib.animation import PillowWriter
        ani.save(P.gif_path, writer=PillowWriter(fps=25))
        print(f"[Saved GIF] {P.gif_path}")
except Exception as e:
    print("❌ Could not save animation:", e)

plt.show()
