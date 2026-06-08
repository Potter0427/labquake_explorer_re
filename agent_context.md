# Labquake Explorer - Agent Context & Architecture Guide

This document is optimized for LLM Agents. It provides the technical context, data schema, and core business logic of the `labquake_explorer` repository to prevent repetitive explanations.

## 1. System Overview & Project Goal
A Python-based application for processing and visualizing laboratory earthquake (labquake) data. It reads from NPZ/HDF5 formats, extracts trigger events, computes stress drops (`tau`), slip drops (`delta`, `lvdt`), inter-event displacements (`D`), and stiffness (`k`), then saves the analysis back to HDF5.

## 2. Core Data Structures (HDF5 / In-Memory)
The data is managed by `data_manager.py` and structured as runs and events:
- **`time_history`**: A dictionary containing full continuous time-series data.
  - Key fields: `time`, `shear_pressure`, `tau_local` (area-corrected shear), `LP_displacement` (LVDT), `eddy_chX` (Eddy current sensors).
- **`events`**: A list of dictionaries (1-indexed, index 0 is usually placeholder/continuous data).
  - Represents individual trigger events.
  - Key fields: `trigger_time`, `tau` (dict with value and windows), `delta` (dict with eddy slips), `lvdt`, `D_Push`, `D_max`, `D_E3`, `k` (stiffness).
  - `skipped`: Boolean flag indicating if the event should be excluded from summary analysis.

## 3. Key Modules & Algorithms

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

## 4. Known Gotchas & Business Logic Rules
1. **`tau_local` vs `shear_pressure`**: The codebase prioritizes `tau_local` (which is `shear_pressure * (fault_area / sliding_area)`) if it exists in `time_history`. Otherwise, it falls back to `shear_pressure`.
2. **Partial Updates**: When updating an event's analysis, always ensure keys that are NOT being recomputed are `pop()`ped from the result dictionary before calling `fast_save_event_analysis`. Otherwise, `dict.update()` will overwrite valid historical data with `NaN`.
3. **Data Filtering in Jupyter**: In Jupyter notebooks (e.g., `2D_Summary_Plots.ipynb`), events are filtered out if `skipped == 'YES'` or if the `k` value is negative/null (`df[pd.to_numeric(df['k'], errors='coerce') >= 0]`). Note that `k` filtering must strictly apply ONLY when plotting `k`, not when plotting `tau` or `slip`.
