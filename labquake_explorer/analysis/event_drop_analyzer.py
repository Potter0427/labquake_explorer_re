"""
Event Drop Analyzer – compute delta_tau, delta_slip, delta_lvdt, and D values.

Ported from taudrop_check_4.py logic. Pure computation module with no UI
dependencies.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


# ------------------------------------------------------------------
# Signal utilities (self-contained, no external dependency)
# ------------------------------------------------------------------

def moving_average(x, w, pad_mode='reflect'):
    """Moving-average filter with edge-padding."""
    if w <= 1:
        return np.array(x, dtype=float)
    x = np.ravel(x).astype(float)
    r = w // 2
    xpad = np.pad(x, (r, w - 1 - r), mode=pad_mode)
    k = np.ones(w, dtype=float) / w
    return np.convolve(xpad, k, mode='valid')


def calculate_2pt_trend_drop(
    t_rel: np.ndarray,
    y: np.ndarray,
    pts: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """
    Compute drop using exactly 2 points before and 2 points after the trigger.
    pts = (x_pre1, x_pre2, x_post1, x_post2)
    """
    result: Dict[str, Any] = {'valid': False}
    
    x_pre1, x_pre2, x_post1, x_post2 = pts
    
    # Ensure x values are within the time array bounds to avoid errors, or just find nearest
    try:
        idx_pre1 = np.argmin(np.abs(t_rel - x_pre1))
        idx_pre2 = np.argmin(np.abs(t_rel - x_pre2))
        idx_post1 = np.argmin(np.abs(t_rel - x_post1))
        idx_post2 = np.argmin(np.abs(t_rel - x_post2))
        
        y_pre1 = y[idx_pre1]
        y_pre2 = y[idx_pre2]
        y_post1 = y[idx_post1]
        y_post2 = y[idx_post2]
        
        # Real x values found in the array (closer to what's plotted)
        rx_pre1, rx_pre2 = t_rel[idx_pre1], t_rel[idx_pre2]
        rx_post1, rx_post2 = t_rel[idx_post1], t_rel[idx_post2]
        
        # Avoid division by zero
        if rx_pre2 == rx_pre1:
            m_pre, b_pre = 0.0, y_pre1
        else:
            m_pre = (y_pre2 - y_pre1) / (rx_pre2 - rx_pre1)
            b_pre = y_pre1 - m_pre * rx_pre1
            
        if rx_post2 == rx_post1:
            m_post, b_post = 0.0, y_post1
        else:
            m_post = (y_post2 - y_post1) / (rx_post2 - rx_post1)
            b_post = y_post1 - m_post * rx_post1
            
        result.update({
            'valid': True,
            'coeff_pre': [m_pre, b_pre],
            'coeff_post': [m_post, b_post],
            'delta': b_pre - b_post,
            'val_pre_0': b_pre,
            'val_post_0': b_post,
        })
    except Exception:
        pass
        
    return result

def calculate_trend_drop(
    t_rel: np.ndarray,
    y: np.ndarray,
    pre_win: Tuple[float, float],
    post_win: Tuple[float, float],
) -> Dict[str, Any]:
    """
    Fit a linear trend on the pre-trigger window and the post-trigger
    window, then compute the jump (delta) at t=0.

    Parameters
    ----------
    t_rel : 1-D array of time relative to trigger (trigger = 0).
    y : 1-D signal array (same length as t_rel).
    pre_win : (start, end) for the pre-trigger fitting segment.
    post_win : (start, end) for the post-trigger fitting segment.

    Returns
    -------
    dict with keys:
        valid      – bool
        coeff_pre  – (slope, intercept) of pre-window fit
        coeff_post – (slope, intercept) of post-window fit
        delta      – val_pre_at_0 - val_post_at_0  (signed)
        val_pre_0  – pre-fit value at t=0
        val_post_0 – post-fit value at t=0
    """
    mask_pre = (t_rel >= pre_win[0]) & (t_rel <= pre_win[1])
    mask_post = (t_rel >= post_win[0]) & (t_rel <= post_win[1])
    result: Dict[str, Any] = {'valid': False}

    if np.sum(mask_pre) > 5 and np.sum(mask_post) > 5:
        try:
            coeff_pre = np.polyfit(t_rel[mask_pre], y[mask_pre], 1)
            coeff_post = np.polyfit(t_rel[mask_post], y[mask_post], 1)
            val_pre_0 = np.polyval(coeff_pre, 0)
            val_post_0 = np.polyval(coeff_post, 0)
            result.update({
                'valid': True,
                'coeff_pre': coeff_pre,
                'coeff_post': coeff_post,
                'delta': val_pre_0 - val_post_0,
                'val_pre_0': val_pre_0,
                'val_post_0': val_post_0,
            })
        except Exception:
            pass
    return result


# ------------------------------------------------------------------
# Default analysis configuration
# ------------------------------------------------------------------

DEFAULT_CONFIG = {
    'pre_win': (-1.0, -0.5),
    'post_win': (0.5, 1.0),
    'tau_smooth_w': 100,
    'lvdt_smooth_w': 100,
    'push_speed': 3.508,       # um/s
    'delay_sec': 0.05,
    'window_sec': 1.5,         # half-window around trigger for extraction
    'skip_events': [],         # list of event indices to skip
    'per_event_windows': {},   # {event_idx: {'pre_win': (...), 'post_win': (...)}}
}


def _get_event_windows(event_idx: int, config: dict):
    """Return tau_pts, slip_pts, lvdt_pts for a specific event. Default to config pre/post wins."""
    pre = config['pre_win']
    post = config['post_win']
    default_pts = (pre[0], pre[1], post[0], post[1])
    
    per = config.get('per_event_windows', {})
    if event_idx in per:
        ew = per[event_idx]
        # Legacy support
        if 'pre_win' in ew and 'post_win' in ew:
            legacy_pts = (ew['pre_win'][0], ew['pre_win'][1], ew['post_win'][0], ew['post_win'][1])
            return legacy_pts, legacy_pts, legacy_pts
            
        tau_pts = ew.get('tau_pts', default_pts)
        slip_pts = ew.get('slip_pts', default_pts)
        lvdt_pts = ew.get('lvdt_pts', default_pts)
        return tau_pts, slip_pts, lvdt_pts
        
    return default_pts, default_pts, default_pts


def _get_t_trig(ev: Any) -> Optional[float]:
    """Safely extract trigger time from an event object (dict or float)."""
    if isinstance(ev, dict):
        val = ev.get('event_time')
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None
    try:
        return float(ev)
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------
# Single-event analysis
# ------------------------------------------------------------------

def analyze_single_event(
    time_history: Dict[str, np.ndarray],
    events: List[Dict],
    event_idx: int,
    config: dict,
) -> Dict[str, Any]:
    """
    Analyze one event and return computed drop values.

    Returns a dict with keys:
        trigger_time, delta_tau, delta_slip_chXX (per eddy channel),
        delta_lvdt, D_Push, D_max, D_E3, skipped
    """
    tau_pts, slip_pts, lvdt_pts = _get_event_windows(event_idx, config)
    skip_list = config.get('skip_events', [])
    half_win = config.get('window_sec', 1.5)

    eddy_keys = sorted([k for k in time_history.keys() if 'eddy' in k.lower()])
    row = {
        'event_idx': event_idx,
        'skipped': event_idx in skip_list,
        'trigger_time': np.nan,
        'delta_tau': np.nan,
        'delta_lvdt': np.nan,
        'D_Push': np.nan,
        'D_max': np.nan,
        'D_E3': np.nan,
    }
    for i in range(len(eddy_keys)):
        row[f'delta_E{i+1}'] = np.nan

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

    # --- delta_tau ---
    tau_key = 'tau_local' if 'tau_local' in time_history else 'shear_pressure'
    tau_raw = time_history[tau_key][mask]
    tau_sm = moving_average(tau_raw, config.get('tau_smooth_w', 100))
    if len(tau_sm) < len(t_rel):
        tau_sm = np.pad(tau_sm, (0, len(t_rel) - len(tau_sm)), 'edge')
    tau_sm = tau_sm - tau_sm[0]

    res_tau = calculate_2pt_trend_drop(t_rel, tau_sm, tau_pts)
    row['delta_tau'] = abs(res_tau['delta']) if res_tau['valid'] else np.nan
    row['tau_res'] = res_tau  # keep full result for plotting

    # --- delta_slip (each eddy channel) ---
    eddy_keys = sorted([k for k in time_history.keys() if 'eddy' in k.lower()])
    for i, k in enumerate(eddy_keys):
        d = time_history[k][mask] - time_history[k][mask][0]
        res_slip = calculate_2pt_trend_drop(t_rel, d, slip_pts)
        label = f'delta_E{i+1}'
        row[label] = abs(res_slip['delta']) if res_slip['valid'] else np.nan
        row[f'{label}_res'] = res_slip

    # --- delta_lvdt ---
    lvdt_raw = time_history['LP_displacement'][mask]
    lvdt_sm = moving_average(lvdt_raw, config.get('lvdt_smooth_w', 100))
    if len(lvdt_sm) < len(t_rel):
        lvdt_sm = np.pad(lvdt_sm, (0, len(t_rel) - len(lvdt_sm)), 'edge')
    lvdt_0 = lvdt_sm - lvdt_sm[0]

    res_lvdt = calculate_2pt_trend_drop(t_rel, lvdt_0, lvdt_pts)
    row['delta_lvdt'] = abs(res_lvdt['delta']) if res_lvdt['valid'] else np.nan
    row['lvdt_res'] = res_lvdt

    # --- k (stiffness): slope of tau vs. LVDT in pre-trigger window ---
    # k represents the effective system stiffness during the locking phase
    # before each rupture event.
    row['k'] = np.nan
    pre_win = config.get('pre_win', (-1.0, -0.5))
    pre_mask = (t_rel >= pre_win[0]) & (t_rel <= pre_win[1])
    if np.sum(pre_mask) > 5:
        tau_pre = tau_sm[pre_mask]
        lvdt_pre = lvdt_0[pre_mask]
        try:
            # Linear regression: tau = k * lvdt + b
            coeffs = np.polyfit(lvdt_pre, tau_pre, 1)
            row['k'] = coeffs[0]  # slope = stiffness
        except Exception:
            pass

    # --- D values ---
    row['D_Push'] = np.nan
    row['D_max'] = np.nan
    row['D_E3'] = np.nan

    push_speed = config.get('push_speed', 3.508)
    delay_sec = config.get('delay_sec', 0.05)

    # Need previous non-skipped and valid event
    if event_idx > 0:
        prev_idx = event_idx - 1
        t_trig_prev = None
        while prev_idx >= 0:
            if prev_idx in skip_list:
                prev_idx -= 1
                continue
            
            pe = events[prev_idx]
            t_trig_prev = _get_t_trig(pe)
            if t_trig_prev is not None:
                break
            prev_idx -= 1

        if t_trig_prev is not None:
            # D_Push
            dt = t_trig - t_trig_prev
            row['D_Push'] = dt * push_speed

            # D_max (LVDT) and D_E3 (Eddy)
            lvdt_all = time_history['LP_displacement']
            lvdt_all_sm = moving_average(lvdt_all, config.get('lvdt_smooth_w', 100))

            idx_curr = np.argmin(np.abs(t_all - (t_trig + delay_sec)))
            idx_prev = np.argmin(np.abs(t_all - (t_trig_prev + delay_sec)))
            row['D_max'] = abs(lvdt_all_sm[idx_curr] - lvdt_all_sm[idx_prev])

            # D_E3: use 3rd eddy channel if available
            if len(eddy_keys) >= 3:
                target_key = eddy_keys[2]
            else:
                target_key = eddy_keys[0] if eddy_keys else None

            if target_key:
                eddy_data = time_history[target_key]
                row['D_E3'] = abs(eddy_data[idx_curr] - eddy_data[idx_prev])

    return row


# ------------------------------------------------------------------
# Batch analysis
# ------------------------------------------------------------------

def analyze_all_events(
    time_history: Dict[str, np.ndarray],
    events: List[Dict],
    config: dict,
) -> List[Dict[str, Any]]:
    """Run analysis on all events. Returns list of result dicts."""
    results = []
    for i in range(len(events)):
        row = analyze_single_event(time_history, events, i, config)
        results.append(row)
    return results


def results_to_arrays(results: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Convert list-of-dicts results into dict-of-arrays for HDF5 storage.
    Excludes internal keys (*_res).
    """
    if not results:
        return {}

    # Collect all numeric keys present in ANY result row
    all_keys = set()
    for r in results:
        for k in r.keys():
            if k.endswith('_res') or k in ['skipped', 'event_idx']:
                continue
            all_keys.add(k)
    keys = sorted(list(all_keys))


    out: Dict[str, np.ndarray] = {}
    for k in keys:
        vals = []
        for r in results:
            v = r.get(k, np.nan)
            if isinstance(v, (int, float, np.number)):
                vals.append(float(v))
            else:
                vals.append(np.nan)
        out[k] = np.array(vals)

    # skipped mask
    out['skipped'] = np.array([r.get('skipped', False) for r in results], dtype=bool)

    return out
