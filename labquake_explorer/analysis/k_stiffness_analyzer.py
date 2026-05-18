"""
K Stiffness Analyzer – compute per-event system stiffness k from the
pre-rupture (locking phase) slope of shear stress vs. LVDT displacement.

k is defined as the slope of the linear regression of smoothed/filtered
tau vs. smoothed/filtered LVDT in a user-defined pre-trigger window.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.signal import butter, filtfilt

from labquake_explorer.analysis.event_drop_analyzer import (
    moving_average,
    _get_t_trig,
)


# ------------------------------------------------------------------
# Default K analysis configuration
# ------------------------------------------------------------------

DEFAULT_K_CONFIG = {
    'k_pre_start': -3.0,       # locking window start (s, relative to trigger)
    'k_pre_end': -0.5,         # locking window end (s, relative to trigger)
    'k_smooth_w': 100,         # moving average window
    'k_highpass_freq': 0.0,    # high-pass filter cutoff frequency (Hz), 0 = off
    'k_window_sec': 3.5,       # half-window around trigger for data extraction
    'skip_events': [],
}


# ------------------------------------------------------------------
# Signal processing helper
# ------------------------------------------------------------------

def _apply_highpass(data: np.ndarray, cutoff_freq: float, fs: float) -> np.ndarray:
    """Apply a 4th-order Butterworth high-pass filter."""
    if cutoff_freq <= 0 or fs <= 0:
        return data
    nyq = fs / 2.0
    if cutoff_freq >= nyq:
        return data
    b, a = butter(4, cutoff_freq / nyq, btype='high')
    return filtfilt(b, a, data)


def _process_signal(raw: np.ndarray, w: int, hp_freq: float, fs: float) -> np.ndarray:
    """Apply moving average then optional high-pass filter."""
    out = moving_average(raw, w)
    # Pad if moving_average shortened the array
    if len(out) < len(raw):
        out = np.pad(out, (0, len(raw) - len(out)), 'edge')
    if hp_freq > 0 and fs > 0:
        out = _apply_highpass(out, hp_freq, fs)
    return out


# ------------------------------------------------------------------
# Single-event K analysis
# ------------------------------------------------------------------

def analyze_single_k(
    time_history: Dict[str, np.ndarray],
    events: List[Dict],
    event_idx: int,
    config: dict,
) -> Dict[str, Any]:
    """
    Compute stiffness k for one event.

    Returns a dict with keys:
        trigger_time, k, k_coeffs, skipped
    """
    skip_list = config.get('skip_events', [])
    k_pre_start = config.get('k_pre_start', -3.0)
    k_pre_end = config.get('k_pre_end', -0.5)
    w = config.get('k_smooth_w', 100)
    hp_freq = config.get('k_highpass_freq', 0.0)
    
    # Ensure window is wide enough for pre_start
    half_win = max(config.get('k_window_sec', 3.5), abs(k_pre_start) + 0.5)

    row = {
        'event_idx': event_idx,
        'skipped': event_idx in skip_list,
        'trigger_time': np.nan,
        'k': np.nan,
    }

    t_trig = _get_t_trig(events[event_idx])
    if t_trig is None:
        row['skipped'] = True
        return row

    row['trigger_time'] = t_trig

    if row['skipped']:
        return row

    t_all = time_history['time']

    # Extract window around trigger
    mask = (t_all >= t_trig - half_win) & (t_all <= t_trig + half_win)
    t_rel = t_all[mask] - t_trig

    if len(t_rel) < 20:
        row['skipped'] = True
        return row

    # Estimate sampling rate
    dt = np.median(np.diff(t_all[mask]))
    fs = 1.0 / dt if dt > 0 else 0

    # Process tau
    tau_key = 'tau_local' if 'tau_local' in time_history else 'shear_pressure'
    tau_raw = time_history[tau_key][mask]
    tau_proc = _process_signal(tau_raw, w, hp_freq, fs)
    tau_proc = tau_proc - tau_proc[0]

    # Process LVDT
    lvdt_raw = time_history['LP_displacement'][mask]
    lvdt_proc = _process_signal(lvdt_raw, w, hp_freq, fs)
    lvdt_proc = lvdt_proc - lvdt_proc[0]

    # Extract locking window
    pre_mask = (t_rel >= k_pre_start) & (t_rel <= k_pre_end)
    if np.sum(pre_mask) > 5:
        tau_pre = tau_proc[pre_mask]
        lvdt_pre = lvdt_proc[pre_mask]
        try:
            coeffs = np.polyfit(lvdt_pre, tau_pre, 1)
            row['k'] = coeffs[0]  # slope = stiffness
            row['k_coeffs'] = coeffs.tolist()
        except Exception:
            pass

    return row


# ------------------------------------------------------------------
# Batch analysis
# ------------------------------------------------------------------

def analyze_all_k(
    time_history: Dict[str, np.ndarray],
    events: List[Dict],
    config: dict,
) -> List[Dict[str, Any]]:
    """Run K analysis on all events. Returns list of result dicts."""
    results = []
    for i in range(len(events)):
        row = analyze_single_k(time_history, events, i, config)
        results.append(row)
    return results


def k_results_to_arrays(results: List[Dict]) -> Dict[str, np.ndarray]:
    """Convert list-of-dicts to dict-of-arrays for HDF5 storage."""
    if not results:
        return {}

    out = {}
    keys = ['trigger_time', 'k']
    for k in keys:
        out[k] = np.array([r.get(k, np.nan) for r in results], dtype=float)
    out['skipped'] = np.array([r.get('skipped', False) for r in results], dtype=bool)
    return out


# ------------------------------------------------------------------
# Diagnostic plot generation
# ------------------------------------------------------------------

def generate_k_diagnostic_plot(
    time_history: Dict[str, np.ndarray],
    events: List[Dict],
    event_idx: int,
    result: Dict[str, Any],
    config: dict,
    save_path: Optional[str] = None,
):
    """
    Generate a 3-panel diagnostic figure for K analysis of one event.

    Subplot 1: LVDT vs time (raw C0 + processed red)
    Subplot 2: Shear stress vs time (raw C0 + processed red)
    Subplot 3: Processed tau (Y) vs processed LVDT (X) with fit line
    """
    import matplotlib.pyplot as plt

    ev = events[event_idx]
    t_trig = _get_t_trig(ev)
    if t_trig is None:
        return

    k_pre_start = config.get('k_pre_start', -3.0)
    k_pre_end = config.get('k_pre_end', -0.5)
    w = config.get('k_smooth_w', 100)
    hp_freq = config.get('k_highpass_freq', 0.0)
    half_win = max(config.get('k_window_sec', 3.5), abs(k_pre_start) + 0.5)

    t_all = time_history['time']
    mask = (t_all >= t_trig - half_win) & (t_all <= t_trig + half_win)
    t_rel = t_all[mask] - t_trig

    if len(t_rel) < 20:
        return

    dt = np.median(np.diff(t_all[mask]))
    fs = 1.0 / dt if dt > 0 else 0

    # Raw and processed signals
    tau_key = 'tau_local' if 'tau_local' in time_history else 'shear_pressure'
    tau_raw = time_history[tau_key][mask]
    tau_proc = _process_signal(tau_raw, w, hp_freq, fs)
    tau_raw_z = tau_raw - tau_raw[0]
    tau_proc_z = tau_proc - tau_proc[0]

    lvdt_raw = time_history['LP_displacement'][mask]
    lvdt_proc = _process_signal(lvdt_raw, w, hp_freq, fs)
    lvdt_raw_z = lvdt_raw - lvdt_raw[0]
    lvdt_proc_z = lvdt_proc - lvdt_proc[0]

    # Display time range: k_pre_start to abs(k_pre_start) (doubled forward)
    t_disp_start = k_pre_start
    t_disp_end = abs(k_pre_start) - 1.0  # e.g. -3 -> +2
    disp_mask = (t_rel >= t_disp_start) & (t_rel <= t_disp_end)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 10),
        gridspec_kw={'height_ratios': [1, 1, 1.2]}
    )

    # --- Subplot 1: LVDT vs time ---
    ax1.plot(t_rel[disp_mask], lvdt_raw_z[disp_mask], color='C0', alpha=0.5, lw=0.8, label='Raw')
    ax1.plot(t_rel[disp_mask], lvdt_proc_z[disp_mask], color='red', alpha=0.6, lw=1.5, label='Processed')
    ax1.axvline(x=k_pre_start, color='blue', ls='--', alpha=0.5, lw=1)
    ax1.axvline(x=k_pre_end, color='blue', ls='--', alpha=0.5, lw=1)
    ax1.axvspan(k_pre_start, k_pre_end, alpha=0.08, color='blue')
    ax1.set_ylabel('LVDT slip [\u03bcm]')
    ax1.set_title(f'Event {event_idx + 1} - K Stiffness Analysis')
    ax1.legend(loc='upper left', fontsize='small')
    ax1.grid(True)

    # --- Subplot 2: Tau vs time ---
    ax2.plot(t_rel[disp_mask], tau_raw_z[disp_mask], color='C0', alpha=0.5, lw=0.8, label='Raw')
    ax2.plot(t_rel[disp_mask], tau_proc_z[disp_mask], color='red', alpha=0.6, lw=1.5, label='Processed')
    ax2.axvline(x=k_pre_start, color='blue', ls='--', alpha=0.5, lw=1)
    ax2.axvline(x=k_pre_end, color='blue', ls='--', alpha=0.5, lw=1)
    ax2.axvspan(k_pre_start, k_pre_end, alpha=0.08, color='blue')
    ax2.set_ylabel(r'rel. $\tau$ [MPa]')
    ax2.set_xlabel('time relative [s]')
    ax2.legend(loc='upper left', fontsize='small')
    ax2.grid(True)

    # --- Subplot 3: Tau vs LVDT (processed, locking window only) ---
    pre_mask = (t_rel >= k_pre_start) & (t_rel <= k_pre_end)
    if np.sum(pre_mask) > 5:
        tau_pre = tau_proc_z[pre_mask]
        lvdt_pre = lvdt_proc_z[pre_mask]

        ax3.plot(lvdt_pre, tau_pre, 'o', color='teal', markersize=3, alpha=0.5, label='Data')

        k_val = result.get('k', np.nan)
        if not np.isnan(k_val):
            k_coeffs = result.get('k_coeffs', None)
            if k_coeffs is not None:
                fit_y = k_coeffs[0] * lvdt_pre + k_coeffs[1]
                ax3.plot(lvdt_pre, fit_y, 'r-', lw=2,
                         label=fr'Fit: $k$ = {k_val:.4f} MPa/$\mu$m')

        ax3.legend(loc='best', fontsize='small')

    ax3.set_xlabel('LVDT slip [\u03bcm]')
    ax3.set_ylabel(r'rel. $\tau$ [MPa]')
    ax3.set_title('Pre-Rupture Stiffness')
    ax3.grid(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
