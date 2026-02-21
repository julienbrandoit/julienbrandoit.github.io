import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def setup_plotting_style():
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'figure.titlesize': 18,
        'axes.linewidth': 1.2,
        'grid.linewidth': 1.0,
        'grid.color': '#4a4a4a',
        'grid.alpha': 0.5,
        'grid.linestyle': '-.',
        'lines.linewidth': 2,
        'patch.linewidth': 0.8,
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'text.usetex': True,
    })

setup_plotting_style()

def f1(x, y):
    return -(np.tanh(0.7*x**3 + 2*x**2 + 0.1*x - 1.4) - y)

def f2(x, y):
    return -(1.3*x**2 + 2*x - 1 - y)

x_range = (-5, 5.0)
y_range = (-5, 5.0)

N_null = 400
N_vec  = 20

x_null = np.linspace(*x_range, N_null)
y_null = np.linspace(*y_range, N_null)
X_null, Y_null = np.meshgrid(x_null, y_null)
F1 = f1(X_null, Y_null)
F2 = f2(X_null, Y_null)

x_vec = np.linspace(*x_range, N_vec)
y_vec = np.linspace(*y_range, N_vec)
X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
U = f1(X_vec, Y_vec)
V = f2(X_vec, Y_vec)
M = np.hypot(U, V)
M[M == 0] = 1
U_norm, V_norm = U / M, V / M

def system(xy):
    return [f1(xy[0], xy[1]), f2(xy[0], xy[1])]

seeds_x = np.linspace(*x_range, 15)
seeds_y = np.linspace(*y_range, 15)
fixed_points = []
for sx in seeds_x:
    for sy in seeds_y:
        sol, info, ier, _ = fsolve(system, [sx, sy], full_output=True)
        if ier == 1:
            residual = np.max(np.abs(info["fvec"]))
            in_domain = (x_range[0] <= sol[0] <= x_range[1] and
                         y_range[0] <= sol[1] <= y_range[1])
            if residual < 1e-8 and in_domain:
                if not any(np.hypot(sol[0]-p[0], sol[1]-p[1]) < 1e-4 for p in fixed_points):
                    fixed_points.append(tuple(sol))

def classify(x, y, h=1e-5):
    a = (f1(x+h, y) - f1(x-h, y)) / (2*h)
    b = (f1(x, y+h) - f1(x, y-h)) / (2*h)
    c = (f2(x+h, y) - f2(x-h, y)) / (2*h)
    d = (f2(x, y+h) - f2(x, y-h)) / (2*h)
    tr, det = a + d, a*d - b*c
    disc = tr**2 - 4*det
    if det < 0:
        return "saddle"
    if disc > 1e-8:
        return "stable node" if tr < 0 else "unstable node"
    if disc < -1e-8:
        if tr < 0:
            return "stable spiral"
        if tr > 1e-6:
            return "unstable spiral"
        return "center"
    return "star node"

fig, ax = plt.subplots(figsize=(8, 7))

ax.quiver(X_vec, Y_vec, U_norm, V_norm,
          M, cmap="coolwarm", alpha=0.55,
          scale=38, width=0.003, headwidth=2, headlength=3)

ax.contour(X_null, Y_null, F1, levels=[0], colors="#4da6ff", linewidths=2)
ax.contour(X_null, Y_null, F2, levels=[0], colors="#4caf50", linewidths=2)
ax.plot([], [], color="#4da6ff", linewidth=2, label=r"$x$-nullcline ($f_1(x,y) = 0$)")
ax.plot([], [], color="#4caf50", linewidth=2, label=r"$y$-nullcline ($f_2(x,y) = 0$)")

for px, py in fixed_points:
    label = classify(px, py)
    ax.plot(px, py, "o", color="#ff7043", markersize=9,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate(
        f"{label}",
        xy=(px, py), xytext=(10, 10),
        textcoords="offset points", fontsize=11,
        color="#ff7043",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff7043", alpha=0.9)
    )

ax.set_xlim(*x_range)
ax.set_ylim(*y_range)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title(r"Phase Plane $\quad \dot{x} = f_1(x,y), \quad \dot{y} = f_2(x,y)$")
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.savefig("phase_plane.png", dpi=150)
plt.show()