import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

_L          = 0.3
_PROP_W     = 0.12
_PROP_H     = 0.03
_QUIVER_SCALE      = 20
_QUIVER_SCALE_ZOOM = 8
_QUIVER_NORM       = 3e-4


def _arm_endpoints(x, y, phi, l=_L):
    c, s = np.cos(phi), np.sin(phi)
    return (np.array([x - l/2*c, y - l/2*s]),
            np.array([x + l/2*c, y + l/2*s]))


def _make_propellers(ax, color='cyan', alpha=1.0):
    kw = dict(width=_PROP_W, height=_PROP_H, color=color, zorder=6, alpha=alpha)
    rl = Rectangle((0, 0), **kw)
    rr = Rectangle((0, 0), **kw)
    ax.add_patch(rl)
    ax.add_patch(rr)
    return rl, rr


def _update_propellers(rl, rr, left, right, phi):
    rl.set_xy((left[0]  - _PROP_W/2, left[1]  - _PROP_H/2))
    rr.set_xy((right[0] - _PROP_W/2, right[1] - _PROP_H/2))
    rl.set_angle(np.degrees(phi))
    rr.set_angle(np.degrees(phi))


def _make_quiver(ax, scale):
    return ax.quiver(
        [0, 0], [0, 0], [0, 0], [0, 0],
        color='orangered', scale=scale, width=0.008,
        zorder=7, alpha=0.95,
    )


def _update_quiver(q, left, right, phi, u_s, u_d):
    F_left  = (u_s - u_d) / 2.0
    F_right = (u_s + u_d) / 2.0
    perp    = np.array([-np.sin(phi), np.cos(phi)])
    U = np.array([perp[0]*F_left, perp[0]*F_right]) * _QUIVER_NORM
    V = np.array([perp[1]*F_left, perp[1]*F_right]) * _QUIVER_NORM
    q.set_offsets([[left[0], left[1]], [right[0], right[1]]])
    q.set_UVC(U, V)


def _draw_bicopter_static(ax, x, y, phi, u_s, u_d, color='steelblue'):
    left, right = _arm_endpoints(x, y, phi)
    ax.plot([left[0], right[0]], [left[1], right[1]],
            '-', color=color, lw=5, solid_capstyle='round', zorder=4)
    ax.plot(x, y, 'o', color=color, ms=10, zorder=5)
    for tip in (left, right):
        r = Rectangle((tip[0]-_PROP_W/2, tip[1]-_PROP_H/2),
                      _PROP_W, _PROP_H, color='cyan', zorder=6,
                      angle=np.degrees(phi))
        ax.add_patch(r)
    F_left  = (u_s - u_d) / 2.0
    F_right = (u_s + u_d) / 2.0
    perp    = np.array([-np.sin(phi), np.cos(phi)])
    vis_scale = 0.06
    for tip, F in [(left, F_left), (right, F_right)]:
        dv = perp * np.sign(F) * vis_scale
        ax.annotate('', xy=(tip[0]+dv[0], tip[1]+dv[1]), xytext=(tip[0], tip[1]),
                    arrowprops=dict(arrowstyle='->', color='orangered', lw=2))


def animate(t, x, y, phi, u_s, u_d, title=None,
            fps=30, real_time_factor=1.5, gif_dpi=90,
            color='steelblue', zoom_half=0.55):
    """
    Animate a bicopter trajectory and save output files.

    Parameters
    ----------
    t, x, y, phi, u_s, u_d : array-like, shape (N,)
        Time vector and state/control trajectories. phi is the body angle in
        radians, u_s is the sum thrust [N], u_d is the differential thrust [N].
    title : str, optional
        Figure suptitle and stem for all saved files. Spaces and special
        characters are sanitised automatically. Defaults to 'bicopter'.
    fps : int
        Frames per second of the output GIF. Default 30.
    real_time_factor : float
        Playback speed relative to real time (>1 faster, <1 slower). Default 1.5.
    gif_dpi : int
        DPI of the saved GIF and the animation figure. Default 90.
    color : str
        Matplotlib colour for the bicopter arm. Default 'steelblue'.
    zoom_half : float
        Half-side length [m] of the zoomed inset window. Default 0.55.

    Returns
    -------
    gif_path : str
        Path to the saved GIF file (<title>.gif).
    figA : Figure
        Trajectory snapshot with the bicopter drawn at its final position.
    figB : Figure
        Time-series of states x, y, theta.
    figC : Figure
        Time-series of controls u_s, u_d.

    All four files are also saved to disk as:
        <title>.gif
        <title>_trajectory.png
        <title>_states.png
        <title>_controls.png

    Example
    -------
    gif, fA, fB, fC = animate(
        t, out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4],
        title='Nonlinear Model', fps=30, real_time_factor=1.5,
    )
    """
    t     = np.asarray(t,     dtype=float)
    x     = np.asarray(x,     dtype=float)
    y     = np.asarray(y,     dtype=float)
    phi   = np.asarray(phi,   dtype=float)
    u_s   = np.asarray(u_s,   dtype=float)
    u_d   = np.asarray(u_d,   dtype=float)
    N   = len(t)
    T   = t[-1]
    stem = (title or 'bicopter').replace(' ', '_').replace(':', '').replace('/', '_')

    total_gif_frames = max(1, int(T * fps / real_time_factor))
    skip   = max(1, N // total_gif_frames)
    frames = np.arange(0, N, skip)

    pad  = 0.4
    x_lo, x_hi = x.min() - pad, x.max() + pad
    y_lo, y_hi = y.min() - pad, y.max() + pad
    half = max(x_hi - x_lo, y_hi - y_lo) / 2 + 0.1
    cx   = (x_lo + x_hi) / 2
    cy   = (y_lo + y_hi) / 2
    traj_xlim = (cx - half, cx + half)
    traj_ylim = (cy - half, cy + half)

    traj_px  = 340
    zoom_px  = 220
    sig_w_px = 300
    gap_px   = 50
    mar_l    = 55
    mar_r    = 18
    mar_t    = 38
    mar_b    = 42

    fig_w_px = mar_l + traj_px + gap_px + zoom_px + gap_px + sig_w_px + mar_r
    fig_h_px = mar_t + traj_px + mar_b

    dpi = gif_dpi
    fig_w_in = fig_w_px / dpi
    fig_h_in = fig_h_px / dpi

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    fig.patch.set_facecolor('white')

    def px2fr(x_px, y_px, w_px, h_px):
        return (x_px/fig_w_px, y_px/fig_h_px, w_px/fig_w_px, h_px/fig_h_px)

    ax_traj = fig.add_axes(px2fr(mar_l, mar_b, traj_px, traj_px))

    x0_zoom = mar_l + traj_px + gap_px
    ax_zoom = fig.add_axes(px2fr(x0_zoom, mar_b, zoom_px, zoom_px))

    sig_mar_l = 48
    x0_sig    = x0_zoom + zoom_px + gap_px
    n_sig   = 5
    sig_gap = 6
    row_h   = (traj_px - (n_sig-1)*sig_gap) // n_sig
    sig_axes = []
    for k in range(n_sig):
        y0 = mar_b + k * (row_h + sig_gap)
        ax = fig.add_axes(px2fr(x0_sig + sig_mar_l, y0,
                                sig_w_px - sig_mar_l, row_h))
        sig_axes.append(ax)

    def _style_ax(ax):
        ax.set_facecolor('#F5F5F5')
        ax.tick_params(colors='#333333', labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor('#BBBBBB')
        ax.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')

    _style_ax(ax_traj)
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.set_xlim(*traj_xlim)
    ax_traj.set_ylim(*traj_ylim)
    ax_traj.set_xlabel('x [m]', fontsize=8, color='#333333', labelpad=2)
    ax_traj.set_ylabel('y [m]', fontsize=8, color='#333333', labelpad=2)
    ax_traj.set_title('Trajectory', fontsize=9, fontweight='bold',
                      color='#111111', pad=3)

    _style_ax(ax_zoom)
    ax_zoom.set_aspect('equal', adjustable='box')
    ax_zoom.set_xlabel('x [m]', fontsize=8, color='#333333', labelpad=2)
    ax_zoom.set_ylabel('y [m]', fontsize=8, color='#333333', labelpad=2)
    ax_zoom.set_title(f'Zoomed  ±{zoom_half:.2f} m', fontsize=9,
                      fontweight='bold', color='#111111', pad=3)

    sig_data = [
        (x,   'x [m]',   'C0'),
        (y,   'y [m]',   'C1'),
        (phi, 'θ [rad]', 'C2'),
        (u_s, 'u_s [N]', '#8822BB'),
        (u_d, 'u_d [N]', '#CC6600'),
    ]
    cursors = []
    for k, (data, ylabel, c) in enumerate(sig_data):
        ax = sig_axes[n_sig - 1 - k]
        _style_ax(ax)
        ax.plot(t, data, color=c, lw=1.1)
        ax.set_ylabel(ylabel, fontsize=7, color='#333333', labelpad=2)
        ax.set_xlim(t[0], t[-1])
        vl = ax.axvline(x=t[0], color='crimson', lw=1.1, ls='--')
        cursors.append(vl)
        if k == n_sig - 1:
            ax.set_xlabel('Time [s]', fontsize=7, color='#333333', labelpad=2)

    if title:
        fig.text(0.5, 1 - 6/fig_h_px, title, ha='center', va='top',
                 fontsize=9, fontweight='bold', color='#111111')

    traj_line, = ax_traj.plot([], [], '-', color=color, lw=1.2, alpha=0.5)
    body_main,  = ax_traj.plot([], [], 'o-', color=color, lw=4,
                               markersize=7, zorder=4, solid_capstyle='round')
    prop_main_l, prop_main_r = _make_propellers(ax_traj, color='cyan')
    thrust_main = _make_quiver(ax_traj, scale=_QUIVER_SCALE)
    time_txt = ax_traj.text(
        0.03, 0.97, '', transform=ax_traj.transAxes, fontsize=7,
        va='top', color='#111111',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#BBBBBB', alpha=0.85),
    )

    zoom_traj, = ax_zoom.plot([], [], '-', color=color, lw=2, alpha=0.55)
    body_zoom,  = ax_zoom.plot([], [], 'o-', color=color, lw=6,
                               markersize=10, zorder=4, solid_capstyle='round')
    prop_zoom_l, prop_zoom_r = _make_propellers(ax_zoom, color='cyan')
    thrust_zoom = _make_quiver(ax_zoom, scale=_QUIVER_SCALE_ZOOM)

    def _update(fi):
        i   = int(fi)
        xi, yi, thi = x[i], y[i], phi[i]
        usi, udi     = u_s[i], u_d[i]
        ti           = t[i]
        lft, rgt = _arm_endpoints(xi, yi, thi)

        traj_line.set_data(x[:i+1], y[:i+1])
        body_main.set_data([lft[0], rgt[0]], [lft[1], rgt[1]])
        _update_propellers(prop_main_l, prop_main_r, lft, rgt, thi)
        _update_quiver(thrust_main, lft, rgt, thi, usi, udi)
        time_txt.set_text(f't = {ti:.2f} s')

        ax_zoom.set_xlim(xi - zoom_half, xi + zoom_half)
        ax_zoom.set_ylim(yi - zoom_half, yi + zoom_half)
        i0 = max(0, i - 80)
        zoom_traj.set_data(x[i0:i+1], y[i0:i+1])
        body_zoom.set_data([lft[0], rgt[0]], [lft[1], rgt[1]])
        _update_propellers(prop_zoom_l, prop_zoom_r, lft, rgt, thi)
        _update_quiver(thrust_zoom, lft, rgt, thi, usi, udi)

        for vl in cursors:
            vl.set_xdata([ti, ti])

        return (traj_line, body_main, prop_main_l, prop_main_r, thrust_main,
                time_txt,
                zoom_traj, body_zoom, prop_zoom_l, prop_zoom_r, thrust_zoom,
                *cursors)

    anim = animation.FuncAnimation(
        fig, _update,
        frames=frames,
        interval=int(1000 / fps),
        blit=True, repeat=True,
    )

    gif_path = f'{stem}.gif'
    print(f'Saving GIF → {gif_path}  ({len(frames)} frames @ {fps} fps, '
          f'{fig_w_px}×{fig_h_px} px)…')
    anim.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
    print('Done.')
    plt.close(fig)

    figA, axA = plt.subplots(figsize=(6, 6))
    figA.patch.set_facecolor('white')
    axA.set_facecolor('#F5F5F5')
    axA.set_aspect('equal', adjustable='box')
    axA.set_xlim(*traj_xlim)
    axA.set_ylim(*traj_ylim)
    axA.plot(x, y, '-', color=color, lw=1.5, alpha=0.6)
    _draw_bicopter_static(axA, x[-1], y[-1], phi[-1], u_s[-1], u_d[-1], color)
    axA.set_xlabel('x [m]', fontsize=11)
    axA.set_ylabel('y [m]', fontsize=11)
    axA.set_title((title or 'Bicopter') + ' — Trajectory',
                  fontsize=12, fontweight='bold')
    axA.grid(True, ls='--', alpha=0.4)
    for sp in axA.spines.values(): sp.set_edgecolor('#BBBBBB')
    figA.tight_layout()
    figA.savefig(f'{stem}_trajectory.png', dpi=150, bbox_inches='tight')

    figB, axsB = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    figB.patch.set_facecolor('white')
    figB.suptitle((title or 'Bicopter') + ' — States',
                  fontsize=13, fontweight='bold')
    for ax, data, lbl, c in zip(axsB, [x, y, phi],
                                ['x [m]', 'y [m]', 'θ [rad]'],
                                ['C0', 'C1', 'C2']):
        ax.plot(t, data, color=c, lw=1.8)
        ax.set_ylabel(lbl, fontsize=11)
        ax.set_facecolor('#F5F5F5')
        ax.grid(True, alpha=0.4)
        for sp in ax.spines.values(): sp.set_edgecolor('#BBBBBB')
    axsB[-1].set_xlabel('Time [s]', fontsize=11)
    figB.tight_layout()
    figB.savefig(f'{stem}_states.png', dpi=150, bbox_inches='tight')

    figC, axsC = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    figC.patch.set_facecolor('white')
    figC.suptitle((title or 'Bicopter') + ' — Controls',
                  fontsize=13, fontweight='bold')
    for ax, data, lbl, c in zip(axsC, [u_s, u_d],
                                ['u_s [N]  (sum thrust)', 'u_d [N]  (diff thrust)'],
                                ['#8822BB', '#CC6600']):
        ax.plot(t, data, color=c, lw=1.8)
        ax.set_ylabel(lbl, fontsize=11)
        ax.set_facecolor('#F5F5F5')
        ax.grid(True, alpha=0.4)
        for sp in ax.spines.values(): sp.set_edgecolor('#BBBBBB')
    axsC[-1].set_xlabel('Time [s]', fontsize=11)
    figC.tight_layout()
    figC.savefig(f'{stem}_controls.png', dpi=150, bbox_inches='tight')

    plt.close('all')
    return gif_path, figA, figB, figC