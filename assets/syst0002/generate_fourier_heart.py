import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch

n_terms = 50
n_frames = 200
fps = 30

def heart_function(t):
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    return x, y

t_vals = np.linspace(0, 2*np.pi, 1000)
x_heart, y_heart = heart_function(t_vals)

def compute_fourier_coefficients(x_vals, y_vals, n_terms):
    N = len(x_vals)
    c = []
    for n in range(-n_terms, n_terms + 1):
        cx = np.sum(x_vals * np.exp(-1j * n * 2 * np.pi * np.arange(N) / N)) / N
        cy = np.sum(y_vals * np.exp(-1j * n * 2 * np.pi * np.arange(N) / N)) / N
        c.append((n, cx + 1j * cy))
    c.sort(key=lambda x: abs(x[1]), reverse=True)
    return c[:n_terms]

coeffs = compute_fourier_coefficients(x_heart, y_heart, n_terms)

plt.style.use('default')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
fig = plt.figure(figsize=(16, 8), facecolor='white')
ax = fig.add_subplot(111, aspect='equal')

colors = ['#FF6B6B', '#FFA500', '#FFD93D', '#6BCF7F', '#4ECDC4', 
          '#45B7D1', '#5B7CFF', '#A78BFA', '#EC4899']

max_radius = sum([abs(c[1]) for c in coeffs])
heart_center_x = -max_radius * 0.3
ax.set_xlim([-max_radius * 1.5, max_radius * 1.5])
ax.set_ylim([-max_radius * 1.2, max_radius * 1.2])
ax.set_facecolor('white')
ax.axis('off')

trail_x = []
trail_y = []

def animate(frame):
    ax.clear()
    ax.set_xlim([-max_radius * 1.5, max_radius * 1.5])
    ax.set_ylim([-max_radius * 1.2, max_radius * 1.2])
    ax.set_facecolor('white')
    ax.axis('off')
    
    arrow_x = FancyArrowPatch((heart_center_x - max_radius * 1.0, 0), (heart_center_x + max_radius * 1.0, 0),
                             arrowstyle='->', mutation_scale=15, linewidth=1.5, 
                             color='#999999', alpha=0.3)
    ax.add_patch(arrow_x)
    arrow_y = FancyArrowPatch((heart_center_x, -max_radius * 1.0), (heart_center_x, max_radius * 1.0),
                             arrowstyle='->', mutation_scale=15, linewidth=1.5, 
                             color='#999999', alpha=0.3)
    ax.add_patch(arrow_y)
    
    t = 2 * np.pi * frame / n_frames
    
    x, y = heart_center_x, 0
    
    for i, (freq, coeff) in enumerate(coeffs):
        radius = abs(coeff)
        angle = np.angle(coeff) + freq * t
        color = colors[i % len(colors)]
        
        circle = Circle((x, y), radius, fill=False, color=color, 
                       linewidth=1.5, alpha=0.4)
        ax.add_patch(circle)
        
        circle_glow = Circle((x, y), radius, fill=False, color=color, 
                            linewidth=3, alpha=0.15)
        ax.add_patch(circle_glow)
        
        new_x = x + radius * np.cos(angle)
        new_y = y + radius * np.sin(angle)
        
        ax.plot([x, new_x], [y, new_y], color=color, 
               linewidth=2, alpha=0.8, solid_capstyle='round')
        
        ax.plot(new_x, new_y, 'o', color=color, 
               markersize=5, alpha=0.9, markeredgewidth=0)
        ax.plot(new_x, new_y, 'o', color=color, 
               markersize=10, alpha=0.2, markeredgewidth=0)
        
        x, y = new_x, new_y
    
    ax.plot(x, y, 'o', color='#FF3366', markersize=8, 
           alpha=1, markeredgewidth=0, zorder=10)
    ax.plot(x, y, 'o', color='#FF3366', markersize=15, 
           alpha=0.3, markeredgewidth=0, zorder=9)
    
    trail_x.append(x)
    trail_y.append(y)
    
    if len(trail_x) > n_frames:
        trail_x.pop(0)
        trail_y.pop(0)
    
    if len(trail_x) > 1:
        for i in range(len(trail_x) - 1):
            alpha = 0.3 + (i / len(trail_x)) * 0.7
            ax.plot(trail_x[i:i+2], trail_y[i:i+2], color='#FF3366', 
                   linewidth=3.5, alpha=alpha, solid_capstyle='round')
    
    text_x = max_radius * 2
    text_y_top = max_radius * 0.08
    text_y_bottom = max_radius * -0.12
    
    ax.text(text_x, text_y_top, r'\textbf{S}ignals \& \textbf{S}ystems', fontsize=34, 
           color='#444444', ha='center', va='center', alpha=0.85)
    ax.text(text_x, text_y_bottom, r'by \textbf{J}ulien \& \textbf{J}ulien', fontsize=26, 
           color='#666666', ha='center', va='center', alpha=0.75)
    
    return []

anim = FuncAnimation(fig, animate, frames=n_frames, 
                     interval=1000/fps, blit=False, repeat=True)

output_path = 'fourier_heart_animation.gif'
writer = PillowWriter(fps=fps)
anim.save(output_path, writer=writer, dpi=110)

plt.close()
print('Done!')
