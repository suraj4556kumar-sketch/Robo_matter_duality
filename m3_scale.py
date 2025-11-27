"""
Many-Magbot Simulation (50 Bots)
--------------------------------
• Each bot = rigid disk with 6 fixed magnets (blue/red halves).
• 50 bots move as Active Brownian Particles (ABPs).
• Non-overlap via soft repulsion (no attraction).
• Magnets rotate locally due to nearby magnet interactions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.animation import FuncAnimation

# ---------------- Parameters ----------------
class P:
    Nbots = 40
    R_bot = 1.2
    R_ring = 0.95
    r_mag = 0.15
    Nmag = 6
    # motion
    v0, Dt, Dr = 2.45, 0.02, 0.08
    mu_t, mu_r = 1.0, 1.0
    # magnet rotation
    mu_phi, Dphi = 1.0, 0.02
    k_m, r0 = 2.0, 1.0
    # body repulsion
    k_rep = 10.0
    # arena
    box = 15.0
    # numerics
    dt = 0.04
    steps = 400
    seed = 7

ALPHA = np.linspace(0, 2*np.pi, P.Nmag, endpoint=False)
rng = np.random.default_rng(P.seed)

# ---------------- Utilities ----------------
def mag_centers(pos, theta):
    """Return (Nbots, Nmag, 2) centers of magnets for each bot."""
    c, s = np.cos(theta), np.sin(theta)
    local_x = P.R_ring*np.cos(ALPHA)
    local_y = P.R_ring*np.sin(ALPHA)
    centers = np.zeros((len(theta), P.Nmag, 2))
    for k in range(P.Nmag):
        centers[:,k,0] = pos[:,0] + local_x[k]*c - local_y[k]*s
        centers[:,k,1] = pos[:,1] + local_x[k]*s + local_y[k]*c
    return centers

def reflect(pos):
    """Keep all bots inside box."""
    for i in range(P.Nbots):
        for d in range(2):
            if pos[i,d] < -P.box + P.R_bot:
                pos[i,d] = -2*(P.box-P.R_bot) - pos[i,d]
            elif pos[i,d] > P.box - P.R_bot:
                pos[i,d] = 2*(P.box-P.R_bot) - pos[i,d]
    return pos

# ---------------- Interactions ----------------
def compute_repulsive_forces(pos):
    F = np.zeros_like(pos)
    for i in range(P.Nbots):
        for j in range(i+1, P.Nbots):
            r = pos[j]-pos[i]
            d = np.linalg.norm(r)
            if d == 0 or d > 2*P.R_bot: continue
            u = r/d
            overlap = 2*P.R_bot - d
            f = -P.k_rep * overlap * u
            F[i] += f
            F[j] -= f
    return F

def compute_magnet_torques(pos, theta, phi):
    Tphi = np.zeros((P.Nbots, P.Nmag))
    centers = mag_centers(pos, theta)
    # magnet directions
    ang = theta[:,None] + ALPHA[None,:] + phi
    mdir = np.stack((np.cos(ang), np.sin(ang)), axis=-1)
    # pairwise magnet torque interactions (short-range)
    for i in range(P.Nbots):
        for j in range(i+1, P.Nbots):
            rij = pos[j]-pos[i]
            if np.linalg.norm(rij) > 2*P.r0 + 2*P.R_bot:
                continue
            for a in range(P.Nmag):
                for b in range(P.Nmag):
                    r = centers[j,b]-centers[i,a]
                    d = np.linalg.norm(r)
                    if d < 1e-6 or d > 2*P.r0: continue
                    w = np.exp(-(d/P.r0)**2)
                    delta = ang[j,b]-ang[i,a]
                    t = P.k_m*w*np.sin(delta)
                    Tphi[i,a] += t
                    Tphi[j,b] -= t
    return Tphi

# ---------------- Drawing ----------------
def draw_bot(ax, pos, theta, phi, color):
    for i in range(P.Nbots):
        body = Circle(pos[i], P.R_bot, facecolor=color, edgecolor='k', lw=1.0, zorder=2)
        ax.add_patch(body)
        for k in range(P.Nmag):
            ang = ALPHA[k] + theta[i] + phi[i,k]
            c = pos[i] + P.R_ring*np.array([np.cos(ALPHA[k]+theta[i]), np.sin(ALPHA[k]+theta[i])])
            ax.add_patch(Wedge(c, P.r_mag, np.degrees(ang), np.degrees(ang)+180,
                               facecolor='royalblue', edgecolor='none', zorder=4))
            ax.add_patch(Wedge(c, P.r_mag, np.degrees(ang)+180, np.degrees(ang)+360,
                               facecolor='tomato', edgecolor='none', zorder=4))

# ---------------- Initialization ----------------
pos = rng.uniform(-P.box*0.8, P.box*0.8, size=(P.Nbots,2))
theta = rng.uniform(0,2*np.pi,size=P.Nbots)
phi = np.zeros((P.Nbots,P.Nmag))

# ---------------- Animation ----------------
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-P.box, P.box); ax.set_ylim(-P.box, P.box)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"{P.Nbots} Solid Magbots — Rotating Internal Magnets")

def update(frame):
    global pos, theta, phi
    ax.clear()
    ax.set_xlim(-P.box, P.box); ax.set_ylim(-P.box, P.box)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{P.Nbots} Solid Magbots — Rotating Internal Magnets")

    # --- Interactions ---
    F_rep = compute_repulsive_forces(pos)
    Tphi = compute_magnet_torques(pos, theta, phi)

    # --- Magnet rotation ---
    phi += P.mu_phi*Tphi*P.dt + np.sqrt(2*P.Dphi*P.dt)*rng.normal(size=phi.shape)

    # --- ABP body motion ---
    n = np.column_stack((np.cos(theta), np.sin(theta)))
    v_det = P.mu_t*F_rep + P.v0*n
    v_sto = np.sqrt(2*P.Dt*P.dt)*rng.normal(size=pos.shape)
    v = v_det + v_sto
    pos += v*P.dt
    theta += np.sqrt(2*P.Dr*P.dt)*rng.normal(size=theta.shape)
    pos = reflect(pos)

    # --- Draw ---
    draw_bot(ax, pos, theta, phi, '#E8E8FF')
    ax.plot([], [], lw=6, color='royalblue', label='North (blue)')
    ax.plot([], [], lw=6, color='tomato', label='South (red)')
    ax.legend(loc='upper right', frameon=False)
    return []

ani = FuncAnimation(fig, update, frames=P.steps, interval=40, blit=False)
plt.show()
