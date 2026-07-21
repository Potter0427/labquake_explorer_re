import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os

# ============================================================
# Parameters
# ============================================================
SAMPLE_LENGTH = 50.0   # cm
SAMPLE_HEIGHT = 10.0    # cm
L = 5.0                # seismogenic zone width (cm)

# h* = G * Dc / (sigma_n * (b - a))
G = 2000.0             # MPa
Dc = 0.35e-4           # cm (0.35 um)
b_minus_a = 0.005

sigma_range = np.linspace(32, 8, 60)  # 32 -> 8 MPa
h_star_values = G * Dc / (sigma_range * b_minus_a)

# ============================================================
# Figure setup
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))

# Drawing coordinates: sample centered at x=25, y=2.5
sample_x0 = 0
sample_y0 = 0
center_x = SAMPLE_LENGTH / 2.0
center_y = SAMPLE_HEIGHT / 2.0

# Seismogenic zone boundaries
L_x0 = center_x - L / 2.0
L_x1 = center_x + L / 2.0

ax.set_xlim(-3, SAMPLE_LENGTH + 3)
ax.set_ylim(-3, SAMPLE_HEIGHT + 6)
ax.set_aspect('equal')
ax.axis('off')

# ── Static elements ──

# Sample outline (white fill, black border)
sample_rect = patches.FancyBboxPatch(
    (sample_x0, sample_y0), SAMPLE_LENGTH, SAMPLE_HEIGHT,
    boxstyle="round,pad=0.05",
    facecolor='#F5F5F5', edgecolor='black', linewidth=1.8, zorder=1
)
ax.add_patch(sample_rect)

# Seismogenic zone (light blue)
seis_rect = patches.Rectangle(
    (L_x0, sample_y0), L, SAMPLE_HEIGHT,
    facecolor='#DCEAF7', edgecolor='black', linewidth=1.8,
    linestyle='-', zorder=2, alpha=0.6
)
ax.add_patch(seis_rect)

# L label (below sample)
# ax.annotate('', xy=(L_x0, -1.0), xytext=(L_x1, -1.0),
#             arrowprops=dict(arrowstyle='<->', color='#2196F3', lw=1.5))
# ax.text(center_x, -1.6, r'$L$ = 5 cm', ha='center', va='top',
#         fontsize=12, color='#2196F3', fontweight='bold')

# Sample length label (below L label)
# ax.annotate('', xy=(sample_x0, -2.5), xytext=(sample_x0 + SAMPLE_LENGTH, -2.5),
#             arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
# ax.text(center_x, -2.9, '50 cm', ha='center', va='top', fontsize=10, color='black')

# Height label (right side)
# ax.annotate('', xy=(SAMPLE_LENGTH + 1.0, sample_y0), xytext=(SAMPLE_LENGTH + 1.0, sample_y0 + SAMPLE_HEIGHT),
#             arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
# ax.text(SAMPLE_LENGTH + 1.5, center_y, '5 cm', ha='left', va='center', fontsize=10, color='black')

# ── Dynamic elements (h* region) ──

# h* patch (red/orange, will be updated)
h_star_patch = patches.Rectangle(
    (center_x, sample_y0), 0, SAMPLE_HEIGHT,
    facecolor='#FF6B6B', edgecolor='#D32F2F', linewidth=1.5,
    zorder=3, alpha=0.7
)
ax.add_patch(h_star_patch)

# h* arrow and label (above sample)
h_arrow = ax.annotate('', xy=(center_x, SAMPLE_HEIGHT + 0.5), xytext=(center_x, SAMPLE_HEIGHT + 0.5),
                       arrowprops=dict(arrowstyle='<->', color='#D32F2F', lw=1.5), zorder=5)

h_label = ax.text(center_x, SAMPLE_HEIGHT + 1.2, '', ha='center', va='bottom',
                  fontsize=20, color='#D32F2F', fontweight='bold', zorder=5)

# Sigma label (top right)
sigma_text = ax.text(SAMPLE_LENGTH - 1, SAMPLE_HEIGHT + 1.2, '', ha='right', va='bottom',
                     fontsize=20,
                     zorder=5)

# Title
# ax.text(center_x, SAMPLE_HEIGHT + 5.0,
#         r'Critical Nucleation Length $\boldsymbol{h}^*$ vs Normal Stress $\boldsymbol{\sigma}_n$',
#         ha='center', va='bottom', fontsize=14, fontweight='bold')

# Legend patches
# legend_seis = patches.Patch(facecolor='#A8D8EA', edgecolor='#2196F3', linestyle='--',
#                             label=f'Seismogenic Zone ($L$ = {L:.0f} cm)')
# legend_hstar = patches.Patch(facecolor='#FF6B6B', edgecolor='#D32F2F',
#                              label=r'$h^*$ (Critical Nucleation Length)')
# ax.legend(handles=[legend_seis, legend_hstar], loc='upper left', fontsize=10,
#           framealpha=0.9, edgecolor='gray')


def update(frame_idx):
    sigma = sigma_range[frame_idx]
    h_star = h_star_values[frame_idx]

    # Update h* patch (centered)
    hx0 = center_x - h_star / 2.0
    h_star_patch.set_x(hx0)
    h_star_patch.set_width(h_star)

    # Update h* arrow
    h_arrow.xy = (hx0, SAMPLE_HEIGHT + 0.5)
    h_arrow.xyann = (hx0 + h_star, SAMPLE_HEIGHT + 0.5)

    # Update h* label
    h_label.set_position((center_x, SAMPLE_HEIGHT + 1.2))
    h_label.set_text(r'$\boldsymbol{h}^{\boldsymbol{*}}$ = ' + f'{h_star:.2f} cm')

    # Update sigma text
    sigma_text.set_text(rf'$\sigma_n$ = {sigma:.1f} MPa')

    return [h_star_patch, h_label, sigma_text]


ani = animation.FuncAnimation(fig, update, frames=len(sigma_range),
                              blit=False, interval=80, repeat=True)

save_dir = r'c:\experiment\labquake_explorer_re\export h5'
save_path = os.path.join(save_dir, 'h_star_animation.gif')
print(f'Saving animation to {save_path} ...')
ani.save(save_path, writer='pillow', fps=15, dpi=150)
print('Done.')
plt.close()
