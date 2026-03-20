import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

A = 1.0
T = 2.0
omega = 2 * np.pi / T

t = np.linspace(-T, T, 4000)

def square_wave(t, A, T):
    return A * np.sign(np.sin(2 * np.pi * t / T))

def fourier_reconstruction(t, A, T, n_terms):
    result = np.zeros_like(t)
    for k in range(1, n_terms + 1):
        n = 2 * k - 1
        result += (4 * A) / (np.pi * n) * np.sin(n * omega * t)
    return result

n_terms_list = [1, 3, 7, 20, 50, 200]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
})

fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

target = square_wave(t, A, T)
accent = "#2166ac"

for idx, n in enumerate(n_terms_list):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    ax.set_facecolor("white")

    recon = fourier_reconstruction(t, A, T, n)

    ax.plot(t, target, color="#aaaaaa", linewidth=1.5, linestyle="--", zorder=1)
    ax.plot(t, recon, color=accent, linewidth=2.0, zorder=2)

    ax.set_xlim(-T, T)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("$x(t)$")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-1, 0, 1])
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
    ax.set_title(f"$N = {n}$", pad=8)

plt.show()