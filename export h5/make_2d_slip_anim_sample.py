import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

L = 50
R = 17
D_jump = 8

# 高解析度網格，用於呈現連續變形 (Continuum deformation)
N = 150
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
r2 = X**2 + Y**2

# 利用平滑過渡函數模擬剪切帶 (Shear band)，避免非物理的絕對斷層斷裂
shear_band_width = 1.5
transition = 0.5 * (1 - np.tanh((np.sqrt(r2) - R) / shear_band_width))

fig, ax = plt.subplots(figsize=(7, 6))

# 事先建立 colorbar 的對應物件
dummy_mesh = ax.pcolormesh(X, Y, np.zeros_like(X), cmap='plasma', vmin=0, vmax=D_jump)
cbar = fig.colorbar(dummy_mesh, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Displacement Magnitude ($\mu m$)', rotation=270, labelpad=15)

def update(frame):
    ax.clear()
    ax.set_xlim(-L/2 - 2, L/2 + D_jump + 2)
    ax.set_ylim(-L/2 - 2, L/2 + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if frame < 15:
        title = "Initial State"
        U_center = 0; U_outer = 0
    elif frame < 25:
        title = "Event (Slip)"
        progress = np.sin((frame - 15) / 10.0 * np.pi / 2)
        U_center = D_jump * progress; U_outer = 0
    elif frame < 95:
        title = "Interseismic (Creep)"
        U_center = D_jump
        progress = (frame - 25) / 70.0
        U_outer = D_jump * progress
    else:
        title = "Cycle Complete"
        U_center = D_jump; U_outer = D_jump
        
    # 計算全區連續位移場
    U = U_outer + (U_center - U_outer) * transition
    X_def = X + U
    Y_def = Y
    
    # 繪製位移熱力圖 (Heatmap)
    ax.pcolormesh(X_def, Y_def, U, shading='gouraud', cmap='plasma', vmin=0, vmax=D_jump)
    
    # 繪製試體外框
    ox = np.concatenate([X_def[0, :], X_def[:, -1], X_def[-1, ::-1], X_def[::-1, 0]])
    oy = np.concatenate([Y_def[0, :], Y_def[:, -1], Y_def[-1, ::-1], Y_def[::-1, 0]])
    ax.plot(ox, oy, 'k-', lw=2)
    
    # 繪製發震區邊界參考線
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R * np.cos(theta), R * np.sin(theta), color='white', lw=1.5, linestyle='--', alpha=0.7)
    
    ax.text(D_jump/2, L/2 + 1.5, title, ha='center', va='bottom', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    return ax,

ani = animation.FuncAnimation(fig, update, frames=120, blit=False, interval=50)
output_path = '2D_Slip_Sample_Continuum.gif'
ani.save(output_path, writer='pillow', dpi=120)
print(f"Animation saved to {output_path}")
