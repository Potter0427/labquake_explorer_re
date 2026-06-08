# Labquake Explorer - Agent Context & Architecture Guide

This document is optimized for LLM Agents. It provides the technical context, data schema, and core business logic of the `labquake_explorer` repository to prevent repetitive explanations.

## 1. System Overview & Project Goal
A Python-based application for processing and visualizing laboratory earthquake (labquake) data. It reads from NPZ/HDF5 formats, extracts trigger events, computes stress drops (`tau`), slip drops (`delta`, `lvdt`), inter-event displacements (`D`), and stiffness (`k`), then saves the analysis back to HDF5.

## 2. Core Data Structures (HDF5 / In-Memory)
The data is managed by `data_manager.py` and structured as runs and events.

### 2.1 Top-Level Access Path
```
data['name']                                      # str, e.g. "t0145_200PC" — experiment identifier
data['runs'][run_idx]                             # dict — one run (0-based index)
data['runs'][run_idx]['name']                     # str, e.g. "run1_8MPa"
data['runs'][run_idx]['time_history']             # dict — continuous time-series
data['runs'][run_idx]['config']                   # dict — per-run configuration & skip list
data['runs'][run_idx]['events'][event_idx]        # dict — single event (1-based; index 0 is placeholder)
```

### 2.2 `time_history` Fields
- `time` — absolute timestamps (s)
- `shear_pressure` — raw shear (MPa)
- `tau_local` — area-corrected shear (MPa); may not exist in older files
- `LP_displacement` — LVDT displacement (μm)
- `eddy_ch8` … `eddy_ch12` — Eddy current sensor channels (μm); 5 channels total

### 2.3 `events[event_idx]` Fields
- `event_time` — trigger timestamp (s)
- `skipped` — `'YES'` / `True` / `1` if this event is excluded from analysis
- `tau` — dict: `{value, pre_start, pre_end, post_start, post_end, smooth_w}`
- `delta` — dict with eddy slip drops:
  - `E1_value` … `E5_value` — slip drop (μm) for each Eddy sensor (1-based index, corresponding to `eddy_ch8` … `eddy_ch12` in order)
  - `pre_start`, `pre_end`, `post_start`, `post_end` — window boundaries used for the 2-pt trend calculation
- `lvdt` — dict: `{value, pre_start, pre_end, post_start, post_end, smooth_w}`
- `D_Push`, `D_max`, `D_E3` — inter-event displacement metrics (μm)
- `k` — dict: `{value, start, end}` — pre-slip loading stiffness

## 3. Physical Sensor Layout & VW Zone
These spatial constants are required for any visualization (Slip Profile, Colored Lines, Heatmap).

### 3.1 Eddy Sensor Positions
The 5 Eddy current sensors are fixed along the fault at:
```
[50, 150, 250, 350, 450]  mm   (E1 → E5, i.e. eddy_ch8 → eddy_ch12)
```

### 3.2 Velocity-Weakening (VW) Zone
- **Center**: 250 mm (fault center)
- **Size**: parsed from `data['name']` using the pattern `{N}PC` (e.g. `200PC` → diameter = 200 mm)
- **Range**: `[250 - N/2, 250 + N/2]` mm
- **Parsing snippet**:
  ```python
  import re
  m = re.search(r'(\d+)(P|PC)', str(name).upper())
  if m:
      diameter = float(m.group(1))
      vw_start = 250 - diameter / 2
      vw_end   = 250 + diameter / 2
  ```
- **Display color**: `khaki` / `#FFD700` (yellow), used consistently across `colored_lines_view.py` and standalone scripts.

## 4. Key Modules & Algorithms

### `analysis/event_drop_analyzer.py` (Drop Analysis)
- **Algorithm**: Does NOT simply subtract peak-to-peak. It uses a **2-point linear trend extrapolation** (`calculate_2pt_trend_drop`). It fits a line through 2 pre-trigger points and 2 post-trigger points, extrapolates both to $t=0$, and calculates the delta.
- **D (Inter-event Displacement)**: Uses `event_idx - 1` as the reference point to calculate displacement since the last event. **Critical Logic**: Skipped events (e.g., noise triggers) are STILL counted as physical displacement steps. Do NOT skip over them when calculating `D`.
- **`compute_flags`**: A dictionary `{tau: bool, slip: bool, lvdt: bool, D: bool}` passed to `analyze_single_event`. It allows partial recomputation. Unchecked metrics are omitted from the returned dict to prevent overwriting existing data with NaNs.

### `analysis/k_stiffness_analyzer.py` (Stiffness Analysis)
- Calculates the pre-slip loading stiffness (slope).
- Supports standard linear regression and RANSAC (to filter outliers).

### `data/data_manager.py` (Data Persistence)
- `fast_save_event_analysis(run_idx, event_idx, category, data)`: Uses `dict.update()` to merge new analysis into the in-memory event dict, and writes directly to the specific HDF5 path (`runs/{run}/events/{event}`) to avoid full-file rewrites.

### `ui/views/*.py` (GUI & State)
- The UI is built with `tkinter`.
- Configurations (like pre/post windows, smooth window, `compute_flags`) are saved persistently across sessions using `UserPrefs.set('Domain', 'key', val)`.
- When users click "Apply & Save" in editors, the system silently updates `status_label` (no blocking `messagebox.showinfo`) to allow rapid manual processing.

## 5. Known Gotchas & Business Logic Rules
1. **`tau_local` vs `shear_pressure`**: The codebase prioritizes `tau_local` (which is `shear_pressure * (fault_area / sliding_area)`) if it exists in `time_history`. Otherwise, it falls back to `shear_pressure`.
2. **Partial Updates**: When updating an event's analysis, always ensure keys that are NOT being recomputed are `pop()`ped from the result dictionary before calling `fast_save_event_analysis`. Otherwise, `dict.update()` will overwrite valid historical data with `NaN`.
3. **Data Filtering in Jupyter**: In Jupyter notebooks (e.g., `2D_Summary_Plots.ipynb`), events are filtered out if `skipped == 'YES'` or if the `k` value is negative/null (`df[pd.to_numeric(df['k'], errors='coerce') >= 0]`). Note that `k` filtering must strictly apply ONLY when plotting `k`, not when plotting `tau` or `slip`.
4. **HDF5 Numeric-Key → List Transformation**: `data_manager._load_hdf5` automatically converts any HDF5 Group whose keys are all digits into a Python `list` (0-based). This affects `runs` and `events`. **Exceptions** — groups named `per_event_windows`, `config`, or `analysis` are always kept as `dict` even if their keys are digits. When writing a standalone script that reads HDF5 directly (without `DataManager`), you must replicate this transformation; otherwise accessing `runs[0]` or `events[1]` will raise a `KeyError`.
