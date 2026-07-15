"""
Event Drop Analyzer – compute delta_tau, delta_slip, delta_lvdt, and D values.

Ported from taudrop_check_4.py logic. Pure computation module with no UI
dependencies.
"""
import math
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


def compute_half_win(config: dict) -> float:
    """Compute the half-window for data extraction around each trigger.

    The total span (2 * half_win) is rounded up to the nearest multiple of 3
    so that the diagnostic-plot arrows always have clean integer labels.
    """
    import math
    pre_win = config.get('pre_win', (-1.0, -0.5))
    post_win = config.get('post_win', (0.5, 1.0))
    base = max(abs(pre_win[0]), abs(post_win[1]))
    # Round up so that total span = 2*half_win is a multiple of 3
    return math.ceil(2 * base / 3) * 1.5

def calculate_2pt_trend_drop(
    t_rel: np.ndarray,
    y: np.ndarray,
    pts: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """
    Deprecated: previously computed two-point extrapolation.
    Now redirects to calculate_trend_drop to enforce two-window linear least-squares extrapolation (雙區間線性外插法).
    """
    return calculate_trend_drop(t_rel, y, pts)


def calculate_trend_drop(
    t_rel: np.ndarray,
    y: np.ndarray,
    *args,
) -> Dict[str, Any]:
    """
    Fit a linear trend on the pre-trigger window and the post-trigger
    window using least-squares linear regression (雙區間線性外插法), then compute the jump (delta) at t=0.

    Parameters
    ----------
    t_rel : 1-D array of time relative to trigger (trigger = 0).
    y : 1-D signal array (same length as t_rel).
    args : either a single 4-tuple pts = (pre_start, pre_end, post_start, post_end)
           or two 2-tuples (pre_win, post_win).

    Returns
    -------
    dict with keys:
        valid      – bool
        coeff_pre  – (slope, intercept) of pre-window fit
        coeff_post – (slope, intercept) of post-window fit
        delta      – val_pre_0 - val_post_0  (signed)
        val_pre_0  – pre-fit value at t=0
        val_post_0  – post-fit value at t=0
    """
    result: Dict[str, Any] = {'valid': False}

    if len(args) == 1 and len(args[0]) == 4:
        pre_win = (args[0][0], args[0][1])
        post_win = (args[0][2], args[0][3])
    elif len(args) == 2:
        pre_win, post_win = args[0], args[1]
    else:
        return result

    pre_start, pre_end = min(pre_win[0], pre_win[1]), max(pre_win[0], pre_win[1])
    post_start, post_end = min(post_win[0], post_win[1]), max(post_win[0], post_win[1])

    mask_pre = (t_rel >= pre_start) & (t_rel <= pre_end)
    mask_post = (t_rel >= post_start) & (t_rel <= post_end)

    if np.sum(mask_pre) >= 2 and np.sum(mask_post) >= 2:
        try:
            coeff_pre = np.polyfit(t_rel[mask_pre], y[mask_pre], 1)
            coeff_post = np.polyfit(t_rel[mask_post], y[mask_post], 1)
            val_pre_0 = float(np.polyval(coeff_pre, 0))
            val_post_0 = float(np.polyval(coeff_post, 0))
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
    compute_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Analyze one event and return computed drop values.

    Parameters
    ----------
    compute_flags : dict, optional
        Keys: 'tau', 'slip', 'lvdt', 'D'.
        Set a key to False to skip computing that metric
        (its value stays NaN so existing data is not overwritten).
        Defaults to all True when not provided.

    Returns a nested dict matching the new schema.
    """
    if compute_flags is None:
        compute_flags = {}
    do_tau  = compute_flags.get('tau',  True)
    do_mu   = compute_flags.get('mu',   True)
    do_slip = compute_flags.get('slip', True)
    do_lvdt = compute_flags.get('lvdt', True)
    do_D    = compute_flags.get('D',    True)
    tau_pts, slip_pts, lvdt_pts = _get_event_windows(event_idx, config)
    skip_list = config.get('skip_events', [])
    half_win = compute_half_win(config)

    eddy_keys = sorted([k for k in time_history.keys() if 'eddy' in k.lower()])
    
    row = {
        'event_idx': event_idx,
        'skipped': event_idx in skip_list,
        'trigger_time': np.nan,
        'tau': {
            'value': np.nan,
            'pre_start': tau_pts[0] if tau_pts else np.nan,
            'pre_end': tau_pts[1] if tau_pts else np.nan,
            'post_start': tau_pts[2] if tau_pts else np.nan,
            'post_end': tau_pts[3] if tau_pts else np.nan,
            'smooth_w': config.get('tau_smooth_w', 100),
        },
        'delta': {
            'pre_start': slip_pts[0] if slip_pts else np.nan,
            'pre_end': slip_pts[1] if slip_pts else np.nan,
            'post_start': slip_pts[2] if slip_pts else np.nan,
            'post_end': slip_pts[3] if slip_pts else np.nan,
        },
        'lvdt': {
            'value': np.nan,
            'pre_start': lvdt_pts[0] if lvdt_pts else np.nan,
            'pre_end': lvdt_pts[1] if lvdt_pts else np.nan,
            'post_start': lvdt_pts[2] if lvdt_pts else np.nan,
            'post_end': lvdt_pts[3] if lvdt_pts else np.nan,
            'smooth_w': config.get('lvdt_smooth_w', 100),
        },
        'delta_mu': np.nan,
        'D_Push': np.nan,
        'D_max': np.nan,
        'D_E3': np.nan,
    }
    
    for i in range(len(eddy_keys)):
        row['delta'][f'E{i+1}_value'] = np.nan

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
    if do_tau:
        tau_key = 'tau_local' if 'tau_local' in time_history else 'shear_pressure'
        tau_raw = time_history[tau_key][mask]
        tau_sm = moving_average(tau_raw, config.get('tau_smooth_w', 100))
        if len(tau_sm) < len(t_rel):
            tau_sm = np.pad(tau_sm, (0, len(t_rel) - len(tau_sm)), 'edge')
        tau_sm = tau_sm - tau_sm[0]

        res_tau = calculate_trend_drop(t_rel, tau_sm, tau_pts)
        row['tau']['value'] = abs(res_tau['delta']) if res_tau['valid'] else np.nan
        row['tau_res'] = res_tau  # keep full result for plotting

    # --- delta_mu ---
    if do_mu and 'mu' in time_history:
        mu_raw = time_history['mu'][mask]
        mu_sm = moving_average(mu_raw, config.get('tau_smooth_w', 100))
        if len(mu_sm) < len(t_rel):
            mu_sm = np.pad(mu_sm, (0, len(t_rel) - len(mu_sm)), 'edge')
        mu_sm = mu_sm - mu_sm[0]

        res_mu = calculate_trend_drop(t_rel, mu_sm, tau_pts)
        row['delta_mu'] = abs(res_mu['delta']) if res_mu['valid'] else np.nan
        row['mu_res'] = res_mu  # keep full result for plotting

    # --- delta_slip (each eddy channel) ---
    if do_slip:
        eddy_keys = sorted([k for k in time_history.keys() if 'eddy' in k.lower()])
        for i, k in enumerate(eddy_keys):
            d = time_history[k][mask] - time_history[k][mask][0]
            res_slip = calculate_trend_drop(t_rel, d, slip_pts)
            row['delta'][f'E{i+1}_value'] = abs(res_slip['delta']) if res_slip['valid'] else np.nan
            row[f'delta_E{i+1}_res'] = res_slip

    # --- delta_lvdt ---
    is_1d = time_history.get('is_1d', False)

    if do_lvdt and not is_1d:
        lvdt_raw = time_history['LP_displacement'][mask]
        lvdt_sm = moving_average(lvdt_raw, config.get('lvdt_smooth_w', 100))
        if len(lvdt_sm) < len(t_rel):
            lvdt_sm = np.pad(lvdt_sm, (0, len(t_rel) - len(lvdt_sm)), 'edge')
        lvdt_0 = lvdt_sm - lvdt_sm[0]

        res_lvdt = calculate_trend_drop(t_rel, lvdt_0, lvdt_pts)
        row['lvdt']['value'] = abs(res_lvdt['delta']) if res_lvdt['valid'] else np.nan
        row['lvdt_res'] = res_lvdt

    if do_D and not is_1d:
        push_speed = config.get('push_speed', 3.508)
        delay_sec = config.get('delay_sec', 0.05)

        # Use the immediately previous event as reference for D calculation,
        # regardless of skip status (skipped events may be noise triggers but
        # still represent real elapsed time / displacement between events).
        # Only skip entries where the trigger time itself is invalid (None).
        if event_idx > 0:
            prev_idx = event_idx - 1
            t_trig_prev = None
            while prev_idx >= 0:
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
    compute_flags: Optional[Dict[str, bool]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Run analysis on all events. Returns dict of result dicts."""
    results = {}
    for i in range(1, len(events)):
        row = analyze_single_event(time_history, events, i, config, compute_flags)
        results[i] = row
    return results
