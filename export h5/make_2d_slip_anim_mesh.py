import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import os

L = 50
R = 17
D_jump = 10

# High resolution grid for deformation
x = np.linspace(-L/2, L/2, 41)
y = np.linspace(-L/2, L/2, 41)
X, Y = np.meshgrid(x, y)
is_center = (X**2 + Y**2) <= R**2

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-L/2 - 2, L/2 + D_jump + 2)
ax.set_ylim(-L/2 - 2, L/2 + 2)
ax.set_aspect('equal')
ax.axis('off')

# Fixed outline for the sample boundary and nucleation zone
rect_outline = Rectangle((-L/2, -L/2), L, L, linewidth=2, edgecolor='black', facecolor='none', zorder=3)
ax.add_patch(rect_outline)

# Nucleation zone outline
theta = np.linspace(0, 2*np.pi, 100)
cx = R * np.cos(theta)
cy = R * np.sin(theta)
ax.plot(cx, cy, color='#1b3b5c', lw=2, zorder=3, linestyle='--')

# Plot grid lines
v_lines = [ax.plot([], [], color='#3498db', lw=1.5, zorder=1)[0] for _ in range(len(x))]
h_lines = [ax.plot([], [], color='#3498db', lw=1.5, zorder=1)[0] for _ in range(len(y))]

title_text = ax.text(D_jump/2, L/2 + 2.5, '', ha='center', va='bottom', fontsize=18, fontfamily='sans-serif', fontweight='bold')

def init():
    for line in v_lines + h_lines:
        line.set_data([], [])
    title_text.set_text('')
    return v_lines + h_lines + [title_text]

def update(frame):
    U = np.zeros_like(X).astype(float)
    
    if frame < 15:
        title_text.set_text("Initial State")
    elif frame < 25:
        title_text.set_text("Event (Slip)")
        progress = (frame - 15) / 10.0
        progress = np.sin(progress * np.pi / 2) # Ease out
        U[is_center] = D_jump * progress
    elif frame < 95:
        title_text.set_text("Interseismic (Creep)")
        U[is_center] = D_jump
        progress = (frame - 25) / 70.0
        U[~is_center] = D_jump * progress
    else:
        title_text.set_text("Cycle Complete")
        U[:] = D_jump
        
    X_def = X + U
    Y_def = Y
    
    for i in range(len(x)):
        v_lines[i].set_data(X_def[:, i], Y_def[:, i])
    for j in range(len(y)):
        h_lines[j].set_data(X_def[j, :], Y_def[j, :])
        
    return v_lines + h_lines + [title_text]

ani = animation.FuncAnimation(fig, update, frames=120, init_func=init, blit=False, interval=50)
output_path = '2D_Slip_Deformation.gif'
ani.save(output_path, writer='pillow', dpi=120)
print(f"Animation saved to {output_path}")
