"""
Batch runner – execute drop analysis on all events, generate diagnostic
plots, and write results into in-memory data (ready for HDF5 save).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from labquake_explorer.analysis.event_drop_analyzer import (
    analyze_all_events,
    analyze_single_event,
    results_to_arrays,
    moving_average,
    _get_t_trig,
    _get_event_windows,
    DEFAULT_CONFIG,
)


def generate_diagnostic_plot(
    time_history: Dict[str, np.ndarray],
    events: List[Dict],
    event_idx: int,
    result: Dict[str, Any],
    config: dict,
    save_path: Optional[str] = None,
):
    """
    Generate a 3-panel diagnostic figure for one event, matching the preferred style.
    """
    ev = events[event_idx]
    t_trig = _get_t_trig(ev)
    if t_trig is None:
        return
    half_win = config.get('window_sec', 1.5)
    t_all = time_history['time']

    mask = (t_all >= t_trig - half_win) & (t_all <= t_trig + half_win)
    t_rel = t_all[mask] - t_trig

    if len(t_rel) < 20:
        return

    # Use the 4-point windows from analyzer
    tau_pts, slip_pts, lvdt_pts = _get_event_windows(event_idx, config)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 10), sharex=True,
        gridspec_kw={'height_ratios': [1.2, 1.5, 1.5]}
    )

    # Helper for drawing trend lines
    def _draw_trend_lines(ax, pts, res, color):
        if not res.get('valid'): return
        # Pre-drop line (pts[0] to 0)
        t_pre = np.linspace(pts[0], 0, 50)
        ax.plot(t_pre, res['coeff_pre'][0]*t_pre + res['coeff_pre'][1], '--', color=color, alpha=0.7, lw=1.5)
        # Post-drop line (0 to pts[3])
        t_post = np.linspace(0, pts[3], 50)
        ax.plot(t_post, res['coeff_post'][0]*t_post + res['coeff_post'][1], '--', color=color, alpha=0.7, lw=1.5)

    # Helper for drawing delta arrows (tau3, tau0.5)
    def _add_window_delta(ax, t_arr, y_arr, offset, color, lbl, symbol='\\tau'):
        try:
            i_n = np.argmin(np.abs(t_arr - (-offset)))
            i_p = np.argmin(np.abs(t_arr - offset))
            val = abs(y_arr[i_n] - y_arr[i_p])
            x_pos = -offset if color == 'orange' else offset
            # Both orange and green texts should be on the right side of the arrow (ha='left')
            ha = 'left'
            # Force symbols to be correct in LaTeX with bold math font
            if symbol == '\\delta':
                label = fr"$\boldsymbol{{\delta}}_{{\boldsymbol{{{lbl}}}}}$=$\boldsymbol{{{val:.1f}}}$ $\boldsymbol{{\mu m}}$"
            else:
                sym_clean = symbol.replace('\\', '')
                label = fr"$\boldsymbol{{\Delta}}\boldsymbol{{{sym_clean}}}_{{\boldsymbol{{{lbl}}}}}$=$\boldsymbol{{{val:.3f}}}$"
                
            _plot_arrow(ax, x_pos, y_arr[i_n], y_arr[i_p], color, label, ha=ha)
            ax.hlines(y=y_arr[i_p], xmin=-offset, xmax=offset, colors=color, linestyles='--', alpha=0.6, lw=1.2)
        except Exception:
            pass

    # --- (1) Stress Drop ---
    tau_key = 'tau_local' if 'tau_local' in time_history else 'shear_pressure'
    tau_raw = time_history[tau_key][mask]
    tau_sm = moving_average(tau_raw, config.get('tau_smooth_w', 100))
    if len(tau_sm) < len(t_rel):
        tau_sm = np.pad(tau_sm, (0, len(t_rel) - len(tau_sm)), 'edge')
    tau_sm = tau_sm - tau_sm[0]

    ax1.plot(t_rel, tau_sm, 'k', alpha=0.8)
    tau_res = result.get('tau_res', {})
    if tau_res.get('valid'):
        _draw_trend_lines(ax1, tau_pts, tau_res, 'darkslateblue')
        val = abs(tau_res['delta'])
        label = fr"$\boldsymbol{{\Delta}}\boldsymbol{{\tau}}$=$\boldsymbol{{{val:.4f}}}$"
        _plot_arrow(ax1, 0, tau_res['val_pre_0'], tau_res['val_post_0'],
                    'darkslateblue', label, ha='right')

    # Add specific window deltas
    _add_window_delta(ax1, t_rel, tau_sm, 1.5, 'orange', '3', symbol='\\tau')
    _add_window_delta(ax1, t_rel, tau_sm, 0.25, 'green', '0.5', symbol='\\tau')

    ax1.set_ylabel(r'rel. $\tau$ [MPa]')
    ax1.set_title(f"Event {event_idx}")
    ax1.grid(True)

    # --- (2) Slip ---
    eddy_keys = sorted([k for k in time_history.keys() if 'eddy' in k.lower()])
    eddy_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                   'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    # Target only E3 (3rd channel) for arrows
    target_idx = 2 if len(eddy_keys) >= 3 else (0 if len(eddy_keys) > 0 else -1)
    
    for i, k in enumerate(eddy_keys):
        d = time_history[k][mask] - time_history[k][mask][0]
        label = f'E{i+1}'
        color = eddy_colors[i % len(eddy_colors)]
        ax2.plot(t_rel, d, alpha=0.8, label=label, color=color)

        res_key = f'delta_E{i+1}_res'
        res_slip = result.get(res_key, {})
        
        if i == target_idx and res_slip.get('valid'):
            _draw_trend_lines(ax2, slip_pts, res_slip, 'darkslateblue')
            val = abs(res_slip['delta'])
            arr_label = fr"$\boldsymbol{{\delta}}$=$\boldsymbol{{{val:.1f}}}$ $\boldsymbol{{\mu m}}$"
            _plot_arrow(ax2, 0, res_slip['val_pre_0'], res_slip['val_post_0'],
                        'darkslateblue', arr_label, ha='right')
            
            # Add specific window deltas for E3
            _add_window_delta(ax2, t_rel, d, 1.5, 'orange', '3', symbol='\\delta')
            _add_window_delta(ax2, t_rel, d, 0.25, 'green', '0.5', symbol='\\delta')

    ax2.set_ylabel('rel. slip [\u03bcm]')
    ax2.legend(loc='upper left', fontsize='small')
    ax2.grid(True)

    # --- (3) LVDT ---
    lvdt_raw = time_history['LP_displacement'][mask]
    lvdt_sm = moving_average(lvdt_raw, config.get('lvdt_smooth_w', 100))
    if len(lvdt_sm) < len(t_rel):
        lvdt_sm = np.pad(lvdt_sm, (0, len(t_rel) - len(lvdt_sm)), 'edge')
    lvdt_0 = lvdt_sm - lvdt_sm[0]

    ax3.plot(t_rel, lvdt_0, alpha=0.7, color='slategrey')
    lvdt_res = result.get('lvdt_res', {})
    if lvdt_res.get('valid'):
        _draw_trend_lines(ax3, lvdt_pts, lvdt_res, 'darkslateblue')
        val = abs(lvdt_res['delta'])
        label = fr"$\boldsymbol{{\delta}}_{{\boldsymbol{{LVDT}}}}$=$\boldsymbol{{{val:.1f}}}$ $\boldsymbol{{\mu m}}$"
        _plot_arrow(ax3, 0, lvdt_res['val_pre_0'], lvdt_res['val_post_0'],
                    'darkslateblue', label, ha='right')

    ax3.set_xlabel('time relative [s]')
    ax3.set_ylabel('LVDT slip [\u03bcm]')
    ax3.grid(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_arrow(ax, x_pos, y_start, y_end, color, label_text, ha='right'):
    """Draw a double-headed arrow with label."""
    ax.annotate('', xy=(x_pos, y_end), xytext=(x_pos, y_start),
                arrowprops=dict(arrowstyle='<->', color=color, lw=2.5))
    if label_text:
        # Move text slightly away from arrow
        text_x = x_pos - 0.05 if ha == 'right' else x_pos + 0.05
        ax.text(text_x, (y_start + y_end) / 2, label_text,
                ha=ha, va='center', color=color, fontweight='bold', fontsize=10)


def run_batch_analysis(
    time_history: Dict[str, np.ndarray],
    events: List[Dict],
    config: dict,
    output_dir: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, np.ndarray]:
    """
    Run analysis on all events, generate diagnostic plots.
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []
    n = len(events)
    skip_list = config.get('skip_events', [])

    for i in range(n):
        row = analyze_single_event(time_history, events, i, config)
        results.append(row)

        # Generate diagnostic plot (skip if event is skipped)
        if output_dir and i not in skip_list and not row.get('skipped', False):
            save_path = os.path.join(output_dir, f"Event_{i:03d}.png")
            try:
                generate_diagnostic_plot(
                    time_history, events, i, row, config, save_path
                )
            except Exception as e:
                print(f"Warning: failed to generate plot for event {i}: {e}")

        if progress_callback:
            progress_callback(i + 1, n)

    return results_to_arrays(results)


def run_batch_k_analysis(
    time_history: Dict[str, np.ndarray],
    events: List[Dict],
    config: dict,
    output_dir: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, np.ndarray]:
    """
    Run K stiffness analysis on all events, generate diagnostic plots.
    """
    from labquake_explorer.analysis.k_stiffness_analyzer import (
        analyze_single_k,
        k_results_to_arrays,
        generate_k_diagnostic_plot,
    )

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []
    n = len(events)
    skip_list = config.get('skip_events', [])

    for i in range(n):
        row = analyze_single_k(time_history, events, i, config)
        results.append(row)

        if output_dir and i not in skip_list and not row.get('skipped', False):
            save_path = os.path.join(output_dir, f"Event_{i:03d}_k.png")
            try:
                generate_k_diagnostic_plot(
                    time_history, events, i, row, config, save_path
                )
            except Exception as e:
                print(f"Warning: failed to generate K plot for event {i}: {e}")

        if progress_callback:
            progress_callback(i + 1, n)

    return k_results_to_arrays(results)
