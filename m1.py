"""
Magbot Simulation + MSD with Walls (Rigid-Body Version)
-------------------------------------------------------

✅ Auto-detects video writer:
   - If FFmpeg available → saves .mp4 video
   - Else → saves .gif animation
✅ Always saves:
   - MSD plot (.png)
   - MSD data (.csv)
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Wedge
import pandas as pd
import shutil  # to check if ffmpeg exists


# ---------------------------- Configuration ----------------------------

@dataclass
class SimParams:
    bot_radius: float = 1.2
    n_magnets: int = 6
    magnet_radius: float = 0.18
    rim_inset: float = 0.25
    omega_body: float = 1.2
    v0: float = 0.9
    Dr: float = 0.4
    Dt: float = 0.05
    box_half: float = 6.0
    dt: float = 0.05
    n_frames_anim: int = 280
    n_frames_msd: int = 4000
    save_video: bool = True
    save_csv: bool = True
    save_plot: bool = True
    video_path: str = "magbot_sim.mp4"
    gif_path: str = "magbot_sim.gif"
    csv_path: str = "msd_with_walls.csv"
    plot_path: str = "msd_with_walls.png"
    seed: int = 7


# ---------------------------- ABP utilities ----------------------------

def reflect_if_outside(next_pos, heading, box_half, radius):
    """Specular reflection keeping the body fully inside the box."""
    x, y = next_pos
    if x < -box_half + radius:
        x = -2*(box_half - radius) - x
        heading = np.pi - heading
    elif x > box_half - radius:
        x = 2*(box_half - radius) - x
        heading = np.pi - heading

    if y < -box_half + radius:
        y = -2*(box_half - radius) - y
        heading = -heading
    elif y > box_half - radius:
        y = 2*(box_half - radius) - y
        heading = -heading

    heading = (heading + np.pi) % (2*np.pi) - np.pi
    return np.array([x, y]), heading


def simulate_abp_with_walls(n_steps, dt, v0, Dr, Dt, box_half, body_radius, rng):
    """Simulate ABP COM motion in a box with reflections."""
    pos = np.zeros((n_steps, 2))
    r = np.array([0.0, 0.0])
    psi = rng.uniform(0, 2*np.pi)

    for t in range(1, n_steps):
        psi += np.sqrt(2*Dr*dt) * rng.normal()
        step = v0*dt*np.array([np.cos(psi), np.sin(psi)]) + np.sqrt(2*Dt*dt)*rng.normal(size=2)
        r_next = r + step
        r, psi = reflect_if_outside(r_next, psi, box_half, body_radius)
        pos[t] = r
    return pos


def time_averaged_msd(pos, dt, max_lag_fraction=0.5):
    """Time-averaged MSD from one trajectory."""
    n = len(pos)
    max_lag = int(n * max_lag_fraction)
    taus = np.arange(1, max_lag + 1)
    msd = np.empty_like(taus, dtype=float)

    for i, k in enumerate(taus):
        diffs = pos[k:] - pos[:-k]
        msd[i] = np.mean(np.sum(diffs*diffs, axis=1))
    return taus*dt, msd


# ---------------------------- Animation ----------------------------

def animate_magbot(params: SimParams):
    """Animate a single rigid-body Magbot (body + magnets move together)."""

    anchor_radius = params.bot_radius - params.rim_inset
    rng = np.random.default_rng(params.seed)

    theta0 = np.linspace(0, 2*np.pi, params.n_magnets, endpoint=False)

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.set_aspect('equal')
    margin = 0.3
    ax.set_xlim(-params.box_half - margin, params.box_half + margin)
    ax.set_ylim(-params.box_half - margin, params.box_half + margin)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")

    # Box outline
    ax.plot([-params.box_half, params.box_half, params.box_half, -params.box_half, -params.box_half],
            [-params.box_half, -params.box_half, params.box_half, params.box_half, -params.box_half],
            color="#999999", lw=1.5)

    # Body + guide ring
    body = Circle((0, 0), params.bot_radius, facecolor='#dddddd', edgecolor='k', linewidth=1.6, zorder=2)
    guide = Circle((0, 0), anchor_radius, facecolor='none', edgecolor='#888888',
                   linestyle='--', linewidth=1.0, alpha=0.5, zorder=2)
    ax.add_patch(body); ax.add_patch(guide)

    # Magnets
    mag_circles, north_wedges, south_wedges = [], [], []
    for _ in range(params.n_magnets):
        c = Circle((0, 0), params.magnet_radius, facecolor='white', edgecolor='k', lw=1.0, zorder=3)
        n = Wedge((0, 0), params.magnet_radius, 0, 180, facecolor='blue', edgecolor='none', zorder=4)
        s = Wedge((0, 0), params.magnet_radius, 180, 360, facecolor='red', edgecolor='none', zorder=4)
        ax.add_patch(c); ax.add_patch(n); ax.add_patch(s)
        mag_circles.append(c); north_wedges.append(n); south_wedges.append(s)

    ax.legend([plt.Line2D([], [], color='blue', lw=6), plt.Line2D([], [], color='red', lw=6)],
              ['North (blue)', 'South (red)'], loc='upper right', frameon=False)

    # Initial state
    body_angle = 0.0
    rng_abp = np.random.default_rng(params.seed + 1)
    pos = np.array([0.0, 0.0])
    psi = rng_abp.uniform(0, 2*np.pi)

    def update(frame):
        nonlocal body_angle, pos, psi

        # Body rotation
        body_angle += params.omega_body * params.dt

        # ABP step
        psi += np.sqrt(2*params.Dr*params.dt) * rng_abp.normal()
        step = params.v0*params.dt*np.array([np.cos(psi), np.sin(psi)]) \
             + np.sqrt(2*params.Dt*params.dt) * rng_abp.normal(size=2)
        pos_next = pos + step
        pos, psi = reflect_if_outside(pos_next, psi, params.box_half, params.bot_radius)

        # Update visuals
        body.center = pos
        guide.center = pos

        for i in range(params.n_magnets):
            theta = theta0[i] + body_angle
            center = pos + anchor_radius * np.array([np.cos(theta), np.sin(theta)])
            mag_circles[i].center = tuple(center)
            deg = np.degrees(theta)
            north_wedges[i].set_center(tuple(center))
            south_wedges[i].set_center(tuple(center))
            north_wedges[i].set_theta1(deg - 90)
            north_wedges[i].set_theta2(deg + 90)
            south_wedges[i].set_theta1(deg + 90)
            south_wedges[i].set_theta2(deg + 270)

        return [body, guide, *mag_circles, *north_wedges, *south_wedges]

    anim = animation.FuncAnimation(fig, update, frames=params.n_frames_anim, interval=30, blit=True)

    # Auto-detect FFmpeg
    has_ffmpeg = shutil.which("ffmpeg") is not None

    try:
        if params.save_video:
            if has_ffmpeg:
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=int(1/params.dt))
                anim.save(params.video_path, writer=writer)
                print(f"[Saved video] {params.video_path}")
            else:
                from matplotlib.animation import PillowWriter
                anim.save(params.gif_path, writer=PillowWriter(fps=int(1/params.dt)))
                print(f"[Saved GIF] {params.gif_path}")
    except Exception as e:
        print("❌ Could not save animation:", e)

    plt.show()


# ---------------------------- MSD computation ----------------------------

def run_msd_with_walls(params: SimParams):
    rng = np.random.default_rng(params.seed + 2)
    pos = simulate_abp_with_walls(
        n_steps=params.n_frames_msd,
        dt=params.dt,
        v0=params.v0,
        Dr=params.Dr,
        Dt=params.Dt,
        box_half=params.box_half,
        body_radius=params.bot_radius,
        rng=rng
    )
    tau, msd = time_averaged_msd(pos, params.dt)

    if params.save_csv:
        pd.DataFrame({"tau": tau, "TAMSD_with_walls": msd}).to_csv(params.csv_path, index=False)
        print(f"[Saved MSD data] {params.csv_path}")

    plt.figure(figsize=(7, 5))
    plt.plot(tau, msd, label="TAMSD (with walls)", color='k')
    plt.xlabel("Lag time τ")
    plt.ylabel("MSD(τ)")
    plt.title("MSD with Wall Reflections (Rigid Magbot ABP)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if params.save_plot:
        plt.savefig(params.plot_path, dpi=300, bbox_inches='tight')
        print(f"[Saved plot] {params.plot_path}")

    plt.show()


# ---------------------------- Main ----------------------------

if __name__ == "__main__":
    P = SimParams()
    animate_magbot(P)
    run_msd_with_walls(P)
