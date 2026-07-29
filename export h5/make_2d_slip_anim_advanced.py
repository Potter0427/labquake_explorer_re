import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os

L = 50
R = 17

# 設定 5 個循環，單次錯動位移調小
D_jump_per_cycle = 0.5
total_cycles = 5
slip_frames = 10
creep_frames = 60
frames_per_cycle = slip_frames + creep_frames
total_frames = total_cycles * frames_per_cycle

# 高解析度網格，用於呈現連續變形 (Continuum deformation)
# 適度調降 N 可大幅加速渲染 (從 200 降至 120)
N = 50
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
r_dist = np.sqrt(X**2 + Y**2)

# 平滑過渡函數 (對應物質內部與外部)
shear_band_width = 1.5
transition = 0.5 * (1 - np.tanh((r_dist - R) / shear_band_width))
# 定義基底標量場 (0: 發震區, 1: 外圍區)
C_base = 1.0 - transition

fig, ax = plt.subplots(figsize=(8, 6))

import matplotlib.colors as mcolors

color_inner = np.array(mcolors.to_rgb('#DCEAF7'))
color_outer = np.array(mcolors.to_rgb('#F2F2F2'))
color_red = np.array(mcolors.to_rgb('#FFE4E1'))

# 發震區外框參考點
theta = np.linspace(0, 2*np.pi, 200)
cx_ref = R * np.cos(theta)
cy_ref = R * np.sin(theta)

def update(frame):
    ax.clear()
    
    # 確保 X 軸範圍涵蓋 5 次循環的總位移
    max_D = D_jump_per_cycle * total_cycles
    ax.set_xlim(-L/2 - 2, L/2 + max_D + 2)
    ax.set_ylim(-L/2 - 2, L/2 + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    cycle_idx = frame // frames_per_cycle
    local_frame = frame % frames_per_cycle
    base_U = cycle_idx * D_jump_per_cycle
    
    if local_frame < slip_frames:
        title = "Event (Slip)"
        # Sine ease-out for quick slip
        progress = np.sin(local_frame / float(slip_frames - 1) * np.pi / 2)
        U_center = base_U + D_jump_per_cycle * progress
        U_outer = base_U
    else:
        title = "Interseismic (Creep)"
        U_center = base_U + D_jump_per_cycle
        # Linear progress for slow creep
        creep_progress = (local_frame - slip_frames) / float(creep_frames - 1)
        U_outer = base_U + D_jump_per_cycle * creep_progress
        
    # 計算該 frame 下每個點的位移
    U = U_outer + (U_center - U_outer) * transition
    X_def = X + U
    Y_def = Y
    
    # 1. 繪製基底顏色 (使用動態生成的 Colormap，將拉扯程度轉化為紅色)
    shear_amount = (U_center - U_outer) / D_jump_per_cycle
    mid_gray = 0.5 * color_inner + 0.5 * color_outer
    mid_color = (1 - shear_amount) * mid_gray + shear_amount * color_red
    
    dynamic_cmap = mcolors.LinearSegmentedColormap.from_list("dynamic", [
        (0.0, color_inner),
        (0.5, mid_color),
        (1.0, color_outer)
    ])
    
    ax.pcolormesh(X_def, Y_def, C_base, shading='gouraud', cmap=dynamic_cmap, vmin=0, vmax=1)
    
    # 3. 繪製試體外框
    ox = np.concatenate([X_def[0, :], X_def[:, -1], X_def[-1, ::-1], X_def[::-1, 0]])
    oy = np.concatenate([Y_def[0, :], Y_def[:, -1], Y_def[-1, ::-1], Y_def[::-1, 0]])
    ax.plot(ox, oy, 'k-', lw=2)
    
    # 4. 繪製發震區外框 (黑色虛線)
    # 發震區虛線嚴格跟隨 U_center，因此只有在 slip 階段才會移動
    cx_def = cx_ref + U_center
    cy_def = cy_ref
    ax.plot(cx_def, cy_def, color='black', lw=1.5, linestyle='--') 
    
    # 加入狀態文字
    ax.text(max_D/2, L/2 + 1.5, title, ha='center', va='bottom', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    
    return ax,

ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=False, interval=40)
output_path = '2D_Slip_Multi_Cycle.gif'
ani.save(output_path, writer='pillow', dpi=120)
print(f"Animation saved to {output_path}")
