import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
# ============================================================
# Parameters
# ============================================================
SAMPLE_SIZE = 50.0     # cm x cm
R = 10.0               # seismogenic zone radius (cm)
# h* = G * Dc / (sigma_n * (b - a))
G = 2000.0             # MPa
Dc = 0.35e-4           # cm (0.35 um)
b_minus_a = 0.005
sigma_range = np.linspace(8.0, 2.0, 60)  # 8 -> 2 MPa
h_star_values = G * Dc / (sigma_range * b_minus_a)
# ============================================================
# Figure setup
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))
center_x = SAMPLE_SIZE / 2.0
center_y = SAMPLE_SIZE / 2.0
ax.set_xlim(0, SAMPLE_SIZE)
ax.set_ylim(0, SAMPLE_SIZE)
ax.set_aspect('equal')
ax.axis('off')
# ── Static elements ──
# Sample outline (white fill, black border)
sample_rect = patches.Rectangle(
    (0, 0), SAMPLE_SIZE, SAMPLE_SIZE,
    facecolor='#F5F5F5', edgecolor='black', linewidth=3, zorder=1
)
ax.add_patch(sample_rect)
# Seismogenic zone (light blue circle)
seis_circle = patches.Circle(
    (center_x, center_y), R,
    facecolor='#DCEAF7', edgecolor='black', linewidth=1.8, zorder=2, alpha=0.9
)
ax.add_patch(seis_circle)
# R label (horizontal right)
# ax.annotate('', xy=(center_x, center_y), xytext=(center_x + R, center_y),
#             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5), zorder=4)
# ax.text(center_x + R/2.0, center_y - 1.5, r'$R$ = 10 cm', ha='center', va='top',
#         fontsize=16, color='black', fontweight='bold', zorder=4)
# ── Dynamic elements (h* region) ──
# h* circle (red/orange, will be updated)
h_star_circle = patches.Circle(
    (center_x, center_y), 0.1,
    facecolor='#FF6B6B', edgecolor='#D32F2F', linewidth=1.5,
    zorder=3, alpha=0.7
)
ax.add_patch(h_star_circle)
# h* arrow (pointing upwards)
h_arrow = ax.annotate('', xy=(center_x, center_y), xytext=(center_x, center_y),
                       arrowprops=dict(arrowstyle='<->', color='#D32F2F', lw=1.5), zorder=5)
h_label = ax.text(center_x, center_y + R + 2.0, '', ha='center', va='bottom',
                  fontsize=20, color='#D32F2F', fontweight='bold', zorder=5)
# Sigma label (top right)
sigma_text = ax.text(SAMPLE_SIZE - 2, SAMPLE_SIZE - 2, '', ha='right', va='top',
                     fontsize=20,
                     zorder=5)
def update(frame_idx):
    sigma = sigma_range[frame_idx]
    h_star = h_star_values[frame_idx]
    # Update h* circle
    h_star_circle.set_radius(h_star)
    # Update h* arrow (upwards from center)
    h_arrow.xy = (center_x, center_y)
    h_arrow.xyann = (center_x, center_y + h_star)
    # Update h* label
    h_label.set_text(r'$\boldsymbol{h}^{\boldsymbol{*}}$ = ' + f'{h_star:.2f} cm')
    h_label.set_position((center_x, center_y + max(h_star, R) + 1.0))
    # Update sigma text
    sigma_text.set_text(rf'$\sigma_n$ = {sigma:.1f} MPa')
    return [h_star_circle, h_label, sigma_text, h_arrow]
ani = animation.FuncAnimation(fig, update, frames=len(sigma_range),
                              blit=False, interval=80, repeat=True)
save_dir = r'c:\experiment\labquake_explorer_re\export h5'
save_path = os.path.join(save_dir, 'h_star_2d_animation.gif')
print(f'Saving animation to {save_path} ...')
ani.save(save_path, writer='pillow', fps=15, dpi=150)
print('Done.')
plt.close()