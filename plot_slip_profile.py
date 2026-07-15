"""
plot_slip_profile.py
--------------------
畫單一 event 的 slip profile（5 個 eddy sensor 的位移分佈），
並將速度弱化區（VW zone）用黃色標記。

用法：
    python plot_slip_profile.py

執行後會彈出互動式對話框讓你選擇：
    1. .h5 檔案
    2. Run index（0-based）
    3. Event index（1-based，符合 explorer 的慣例）

資料結構假設（與 labquake_explorer HDF5 格式相同）：
    data['name']                    → 實驗名稱，如 "t0145_200PC"（用來解析 VW zone 大小）
    data['runs'][run_idx]['events'][event_idx]['delta']['E1_value' ~ 'E5_value']
                                    → 各 sensor 的 slip drop（μm）
    data['runs'][run_idx]['events'][event_idx]['skipped']
                                    → 是否為跳過的 event
"""

import re
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tkinter import Tk, filedialog, simpledialog, messagebox


# ─────────────────────────────────────────────
# 常數設定
# ─────────────────────────────────────────────
from labquake_explorer.utils.config import LabquakeExplorerConfig
SENSOR_POSITIONS_MM = LabquakeExplorerConfig.EDDY_POSITIONS_5CH_MM   # legacy default
VW_CENTER_MM = 250.0                              # VW zone 中心（fault 中心）
VW_COLOR = '#FFD700'                              # 黃色（與 colored_lines_view 一致）
VW_ALPHA = 0.35


# ─────────────────────────────────────────────
# HDF5 載入（完整重現 data_manager._load_hdf5 的核心邏輯）
# ─────────────────────────────────────────────
def load_dataset(item):
    """將 HDF5 dataset 轉換為 Python scalar 或 numpy array。"""
    try:
        data = np.array(item)
        if data.dtype.kind in ('S', 'O'):
            if isinstance(data.flat[0], bytes):
                if data.size == 1:
                    return data.flat[0].decode('utf-8')
                return [x.decode('utf-8') for x in data.flat]
        if data.size == 1:
            return data.item()
        return data
    except Exception as e:
        print(f"Dataset loading error: {e}")
        return None


def load_group(group, name=""):
    """
    遞迴載入 HDF5 group。
    若 group 的所有 key 都是數字 → 轉為 list（events / runs 的索引結構）。
    例外：'per_event_windows', 'config', 'analysis' 不做此轉換（避免 List Bug）。
    """
    result = {}
    keys = list(group.keys())

    EXEMPT_NAMES = {'per_event_windows', 'config', 'analysis'}
    if all(k.isdigit() for k in keys) and name not in EXEMPT_NAMES:
        try:
            num_keys = max(int(k) for k in keys) + 1
            array_data = []
            for i in range(num_keys):
                matching = [k for k in keys if int(k) == i]
                if matching:
                    best_key = matching[0]
                    if len(matching) > 1:
                        best_key = max(
                            matching,
                            key=lambda k: len(group[k].keys())
                            if isinstance(group[k], h5py.Group) else 0
                        )
                    item = group[best_key]
                    if isinstance(item, h5py.Group):
                        array_data.append(load_group(item, best_key))
                    else:
                        array_data.append(load_dataset(item))
                else:
                    array_data.append({"name": "(No Data)"})
            return array_data
        except ValueError:
            pass

    for key in keys:
        try:
            item = group[key]
            if isinstance(item, h5py.Group):
                result[key] = load_group(item, key)
            else:
                result[key] = load_dataset(item)
        except Exception as e:
            print(f"Error loading {key}: {e}")

    return result


def load_h5(path: str) -> dict:
    """載入整個 HDF5 檔案為 Python dict。"""
    with h5py.File(path, 'r') as f:
        return load_group(f)


# ─────────────────────────────────────────────
# VW zone 解析
# ─────────────────────────────────────────────
def parse_vw_zone(name_str: str):
    """
    從實驗名稱（如 't0145_200PC' 或 't0145_200PC_run1'）解析 VW zone 半徑。
    回傳 (vw_start_mm, vw_end_mm) 或 None（若無法解析）。
    """
    if isinstance(name_str, bytes):
        name_str = name_str.decode()
    if hasattr(name_str, 'item'):
        name_str = name_str.item()
    if isinstance(name_str, bytes):
        name_str = name_str.decode()

    m = re.search(r'(\d+)(P|PC)', str(name_str).upper())
    if m:
        diameter_mm = float(m.group(1))
        vw_start = VW_CENTER_MM - diameter_mm / 2.0
        vw_end   = VW_CENTER_MM + diameter_mm / 2.0
        return vw_start, vw_end
    return None


# ─────────────────────────────────────────────
# 取得單一 event 的 slip profile
# ─────────────────────────────────────────────
def get_slip_profile(events: list, event_idx: int, n_sensors: int = 5):
    """
    從 events[event_idx]['delta'] 讀取 E1_value ~ En_value。
    回傳 numpy array，長度 = n_sensors；無資料者填 NaN。
    """
    ev = events[event_idx]
    if not isinstance(ev, dict):
        return np.full(n_sensors, np.nan)

    delta = ev.get('delta', {})
    if not isinstance(delta, dict):
        return np.full(n_sensors, np.nan)

    slips = []
    for i in range(1, n_sensors + 1):
        val = delta.get(f'E{i}_value', np.nan)
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = np.nan
        slips.append(v)
    return np.array(slips)


# ─────────────────────────────────────────────
# 主繪圖函式
# ─────────────────────────────────────────────
def plot_slip_profile(data: dict, run_idx: int, event_idx: int):
    """
    繪製單一 event 的 slip profile。

    Parameters
    ----------
    data      : 從 load_h5() 取得的完整資料 dict
    run_idx   : Run 編號（0-based）
    event_idx : Event 編號（1-based，對應 HDF5 的 '001', '002'...）
    """
    # ── 取得 runs ──
    runs = data.get('runs')
    if runs is None:
        raise ValueError("data 中找不到 'runs' 欄位")

    if isinstance(runs, list):
        if run_idx >= len(runs):
            raise IndexError(f"run_idx={run_idx} 超出範圍（共 {len(runs)} 個 run）")
        run = runs[run_idx]
    elif isinstance(runs, dict):
        key = str(run_idx)
        if key not in runs:
            raise KeyError(f"找不到 runs['{key}']")
        run = runs[key]
    else:
        raise TypeError("runs 格式錯誤")

    # ── 取得 events ──
    events = run.get('events')
    if not events:
        raise ValueError(f"Run {run_idx} 中找不到 events")

    if event_idx <= 0 or event_idx >= len(events):
        raise IndexError(
            f"event_idx={event_idx} 超出範圍（此 run 共 {len(events)-1} 個 events，1-based）"
        )

    ev = events[event_idx]
    if not isinstance(ev, dict):
        raise ValueError(f"events[{event_idx}] 不是有效的 dict")

    # ── 檢查是否 skipped ──
    skipped = ev.get('skipped', False)
    if skipped == 'YES' or skipped is True or skipped == 1:
        print(f"警告：Event {event_idx} 被標記為 skipped，仍繼續繪圖。")

    # ── 自動偵測 sensor 數量 ──
    delta = ev.get('delta', {})
    if isinstance(delta, dict):
        n_sensors = sum(1 for k in delta if re.match(r'^E\d+_value$', k))
    else:
        n_sensors = 0

    if n_sensors == 0:
        # Fallback：若未能自 delta 偵測到數量，預設為 8 個
        n_sensors = 8

    positions = LabquakeExplorerConfig.get_eddy_positions(n_sensors)
    slips = get_slip_profile(events, event_idx, n_sensors)

    # ── 取得實驗名稱（用來解析 VW zone）──
    name_str = data.get('name', '')
    vw_zone = parse_vw_zone(name_str)

    # ── 取得 run 名稱 ──
    run_name = run.get('name', f'Run {run_idx}')
    if isinstance(run_name, bytes):
        run_name = run_name.decode()

    # ── 取得 event_time ──
    event_time = ev.get('event_time', np.nan)
    try:
        event_time = float(event_time)
    except (TypeError, ValueError):
        event_time = np.nan

    # ─────────────────────────────
    # 繪圖
    # ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    # 速度弱化區（VW zone）背景
    legend_patches = []
    if vw_zone is not None:
        vw_start, vw_end = vw_zone
        ax.axvspan(vw_start, vw_end, color=VW_COLOR, alpha=VW_ALPHA, zorder=0)
        vw_patch = mpatches.Patch(
            facecolor=VW_COLOR, alpha=VW_ALPHA + 0.2,
            # label=f'VW Zone ({vw_end - vw_start:.0f} mm)'
        )
        legend_patches.append(vw_patch)

    # Slip profile 折線
    valid_mask = ~np.isnan(slips)
    if valid_mask.any():
        ax.plot(
            np.array(positions)[valid_mask],
            slips[valid_mask],
            'o-',
            color='#2C5F8A',
            linewidth=2.0,
            markersize=8,
            markerfacecolor='white',
            markeredgewidth=2,
            markeredgecolor='#2C5F8A',
            zorder=3
        )
    else:
        ax.text(
            0.5, 0.5, 'No slip data available',
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=13, color='gray'
        )

    # 垂直參考線（各 sensor 位置）
    # for pos in positions:
    #     ax.axvline(pos, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=1)

    # 座標軸設定
    ax.set_xlabel('Distance along fault [mm]', fontsize=12)
    ax.set_ylabel('Slip [μm]', fontsize=12)

    t_str = f'  (t = {event_time:.3f} s)' if not np.isnan(event_time) else ''
    skip_str = '  [SKIPPED]' if (skipped == 'YES' or skipped is True or skipped == 1) else ''
    # ax.set_title(
    #     f'Slip Profile — {run_name} | Event {event_idx}{t_str}{skip_str}',
    #     fontsize=12
    # )

    ax.set_xticks(positions)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Y 軸下限至 0（slip 不為負）
    y_min = -2
    y_max_val = np.nanmax(slips) if valid_mask.any() else 1.0
    ax.set_ylim(y_min, y_max_val * 1.1 if y_max_val > 0 else 1.0)

    # ax.set_xlim(positions[0] - 60, positions[-1] + 60)
    # ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Legend
    # handles, labels = ax.get_legend_handles_labels()
    # all_handles = legend_patches + handles
    # all_labels  = [p.get_label() for p in legend_patches] + labels
    # if all_handles:
    #     ax.legend(all_handles, all_labels, fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 互動式選擇介面
# ─────────────────────────────────────────────
def select_file_and_params() -> tuple:
    """
    用 tkinter 對話框讓使用者選擇 .h5 檔案、run index 及 event index。
    回傳 (h5_path, run_idx, event_idx)。
    """
    root = Tk()
    root.withdraw()

    # 選擇 .h5 檔案
    h5_path = filedialog.askopenfilename(
        title="選擇 .h5 檔案",
        filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
    )
    if not h5_path:
        messagebox.showinfo("取消", "未選擇檔案，程式結束。")
        root.destroy()
        sys.exit(0)

    # 輸入 run index（0-based）
    run_idx_str = simpledialog.askstring(
        "Run Index",
        "請輸入 Run Index（0-based，第一個 run = 0）：",
        initialvalue="0"
    )
    if run_idx_str is None:
        messagebox.showinfo("取消", "未輸入 Run Index，程式結束。")
        root.destroy()
        sys.exit(0)

    try:
        run_idx = int(run_idx_str.strip())
    except ValueError:
        messagebox.showerror("錯誤", f"無效的 Run Index：{run_idx_str}")
        root.destroy()
        sys.exit(1)

    # 輸入 event index（1-based）
    event_idx_str = simpledialog.askstring(
        "Event Index",
        "請輸入 Event Index（1-based，第一個 event = 1）：",
        initialvalue="1"
    )
    if event_idx_str is None:
        messagebox.showinfo("取消", "未輸入 Event Index，程式結束。")
        root.destroy()
        sys.exit(0)

    try:
        event_idx = int(event_idx_str.strip())
    except ValueError:
        messagebox.showerror("錯誤", f"無效的 Event Index：{event_idx_str}")
        root.destroy()
        sys.exit(1)

    root.destroy()
    return h5_path, run_idx, event_idx


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # ── 互動式取得參數 ──
    h5_path, run_idx, event_idx = select_file_and_params()

    print(f"載入檔案：{h5_path}")
    print(f"Run index (0-based): {run_idx}")
    print(f"Event index (1-based): {event_idx}")

    # ── 載入資料 ──
    try:
        data = load_h5(h5_path)
    except Exception as e:
        print(f"載入失敗：{e}")
        sys.exit(1)

    # ── 繪圖 ──
    try:
        plot_slip_profile(data, run_idx, event_idx)
    except (IndexError, KeyError, ValueError) as e:
        print(f"繪圖失敗：{e}")

        root = Tk()
        root.withdraw()
        messagebox.showerror("繪圖失敗", str(e))
        root.destroy()
        sys.exit(1)
