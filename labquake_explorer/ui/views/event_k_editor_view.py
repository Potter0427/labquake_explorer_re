"""
Event K Stiffness Editor – allows the user to adjust the locking window,
smoothing, and low-pass filter parameters for a single event, then
recompute and save the updated k value.

Performance notes
-----------------
Drawing is split into two layers:

  _draw_static(config)
      Clears ax1 and ax2, redraws fixed time-series lines (raw + processed
      LVDT and tau).  tight_layout() runs here.  Called once per event load
      and whenever a parameter that changes the signal shape is updated.

  _draw_dynamic(result, config)
      Only updates mutable Artists: vline positions (set_xdata), axvspan,
      and ax3 (scatter + fit line) which is always cleared and redrawn
      because it depends on the locking window bounds.  Uses draw_idle().

Signal computation cache
------------------------
_process_signal (moving average + Butterworth high-pass filter) is
expensive.  Results are cached in self._signal_cache keyed by a tuple of
(event_idx, w, hp_freq, half_win).  Cache is invalidated automatically on
key mismatch and holds at most one entry (single-event editor context).
"""
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from labquake_explorer.analysis.k_stiffness_analyzer import (
    analyze_single_k,
    DEFAULT_K_CONFIG,
    _process_signal,
    _get_t_trig,
)


class EventKEditorView(tk.Toplevel):
    def __init__(self, parent, run_idx, event_idx):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"Event K Editor - Run {run_idx + 1}")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.run_idx = run_idx
        self.event_idx = event_idx
        self.data_manager = self.parent.data_manager

        # Load data
        self.time_history = self.data_manager.get_data(
            f"runs/[{self.run_idx}]/time history"
        )
        self.events = self.data_manager.get_data(
            f"runs/[{self.run_idx}]/events"
        )

        if not self.time_history or not self.events:
            messagebox.showerror("Error", "Cannot load time history or events")
            self.destroy()
            return

        if event_idx >= len(self.events):
            messagebox.showerror("Error", f"Event {event_idx} out of range")
            self.destroy()
            return

        self._load_config()

        self.figure = None
        self.canvas = None
        self.toolbar = None

        # Signal computation cache: holds result of last _process_signal call.
        # Key: (event_idx, w, hp_freq, half_win)
        self._signal_cache: dict = {}

        # Dynamic Artist handles
        self._vlines_start: list = []
        self._vlines_end: list = []
        self._axvspan = None
        self._drag_target = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Control frame ---
        ctrl = ttk.Frame(self)
        ctrl.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Event selector
        ttk.Label(ctrl, text="Event:").grid(row=0, column=0, padx=5)
        self.event_combo = ttk.Combobox(ctrl, state="readonly", width=8)
        self.event_combo.grid(row=0, column=1, padx=5)
        self.event_combo['values'] = [str(i) for i in range(len(self.events))]
        self.event_combo.current(self.event_idx)
        self.event_combo.bind("<<ComboboxSelected>>", self._on_event_change)

        # Pre start / end
        ttk.Label(ctrl, text="Pre start:").grid(row=0, column=2, padx=5)
        self.pre_start_var = tk.StringVar(value=str(self.config['k_pre_start']))
        ttk.Entry(ctrl, textvariable=self.pre_start_var, width=8).grid(row=0, column=3, padx=3)

        ttk.Label(ctrl, text="Pre end:").grid(row=0, column=4, padx=5)
        self.pre_end_var = tk.StringVar(value=str(self.config['k_pre_end']))
        ttk.Entry(ctrl, textvariable=self.pre_end_var, width=8).grid(row=0, column=5, padx=3)

        # Smooth w
        ttk.Label(ctrl, text="Smooth w:").grid(row=0, column=6, padx=5)
        self.smooth_w_var = tk.StringVar(value=str(self.config['k_smooth_w']))
        ttk.Entry(ctrl, textvariable=self.smooth_w_var, width=6).grid(row=0, column=7, padx=3)

        # Highpass / Lowpass freq
        ttk.Label(ctrl, text="HP (Hz):").grid(row=0, column=8, padx=5)
        self.hp_freq_var = tk.StringVar(value=str(self.config.get('k_highpass_freq', 0.0)))
        ttk.Entry(ctrl, textvariable=self.hp_freq_var, width=6).grid(row=0, column=9, padx=3)

        ttk.Label(ctrl, text="LP (Hz):").grid(row=0, column=10, padx=5)
        self.lp_freq_var = tk.StringVar(value=str(self.config.get('k_lowpass_freq', 0.0)))
        ttk.Entry(ctrl, textvariable=self.lp_freq_var, width=6).grid(row=0, column=11, padx=3)

        # Buttons
        ttk.Button(ctrl, text="Recompute", command=self._recompute).grid(
            row=0, column=12, padx=10
        )
        ttk.Button(ctrl, text="Apply & Save", command=self._apply_and_save).grid(
            row=0, column=13, padx=5
        )

        self.focus_y_var = tk.BooleanVar(value=True)
        self.focus_y_check = ttk.Checkbutton(
            ctrl, text="Focus Y", variable=self.focus_y_var, command=self._on_focus_y_changed
        )
        self.focus_y_check.grid(row=0, column=14, padx=5)

        # Build figure once – Canvas is never destroyed after this point
        self._build_figure()
        self._draw_static(self._get_current_config())
        self._recompute()

    # ------------------------------------------------------------------
    # Event switching
    # ------------------------------------------------------------------

    def _on_event_change(self, event=None):
        self.event_idx = int(self.event_combo.get())
        self._load_config()
        self._signal_cache.clear()   # invalidate cache on event switch
        self.pre_start_var.set(str(self.config['k_pre_start']))
        self.pre_end_var.set(str(self.config['k_pre_end']))
        self.smooth_w_var.set(str(self.config['k_smooth_w']))
        self.hp_freq_var.set(str(self.config.get('k_highpass_freq', 0.0)))
        self.lp_freq_var.set(str(self.config.get('k_lowpass_freq', 0.0)))
        self._draw_static(self._get_current_config())
        self._recompute()

    def _on_focus_y_changed(self):
        self._draw_static(self._get_current_config())
        self._recompute()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self):
        """Load K analysis config, merging saved per-event overrides."""
        self.config = dict(DEFAULT_K_CONFIG)
        try:
            k_analysis = self.data_manager.get_data(
                f"runs/[{self.run_idx}]/k_analysis"
            )
            if isinstance(k_analysis, dict):
                cfg = k_analysis.get('config', {})
                if isinstance(cfg, dict):
                    for k in ['k_pre_start', 'k_pre_end', 'k_smooth_w',
                              'k_highpass_freq', 'k_lowpass_freq', 'k_window_sec']:
                        if k in cfg:
                            self.config[k] = (
                                float(cfg[k])
                                if 'freq' in k or 'start' in k or 'end' in k or 'sec' in k
                                else int(cfg[k])
                            )

                per_event = k_analysis.get('per_event_config', {})
                if isinstance(per_event, dict):
                    ev_key = str(self.event_idx)
                    if ev_key in per_event and isinstance(per_event[ev_key], dict):
                        ev_cfg = per_event[ev_key]
                        for k in ['k_pre_start', 'k_pre_end', 'k_smooth_w', 'k_highpass_freq', 'k_lowpass_freq']:
                            if k in ev_cfg:
                                val = ev_cfg[k]
                                if hasattr(val, 'item'):
                                    val = val.item()
                                self.config[k] = val
        except Exception:
            pass

    def _get_current_config(self):
        """Read current parameter values from UI."""
        cfg = dict(self.config)
        try:
            cfg['k_pre_start'] = float(self.pre_start_var.get())
            cfg['k_pre_end'] = float(self.pre_end_var.get())
            cfg['k_smooth_w'] = int(self.smooth_w_var.get())
            cfg['k_highpass_freq'] = float(self.hp_freq_var.get())
            cfg['k_lowpass_freq'] = float(self.lp_freq_var.get())
        except ValueError:
            pass
        return cfg

    # ------------------------------------------------------------------
    # Figure / Canvas – created once, never destroyed
    # ------------------------------------------------------------------

    def _build_figure(self):
        self.figure = Figure(figsize=(10, 10), dpi=100)
        gs = self.figure.add_gridspec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.3)
        self.ax1 = self.figure.add_subplot(gs[0])
        self.ax2 = self.figure.add_subplot(gs[1], sharex=self.ax1)
        self.ax3 = self.figure.add_subplot(gs[2])

        self.ax1.set_ylabel('LVDT slip [μm]')
        self.ax2.set_ylabel(r'$\tau$ [MPa]')
        self.ax2.set_xlabel('time relative [s]')
        self.ax3.set_ylabel(r'$\tau$ [MPa]')
        self.ax3.set_xlabel('LVDT slip [μm]')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self._setup_drag_events()

    # ------------------------------------------------------------------
    # Signal cache helper
    # ------------------------------------------------------------------

    def _get_processed_signals(self, config):
        """Return processed tau and LVDT arrays, using cache when possible.

        Cache key: (event_idx, w, hp_freq, half_win).
        On a cache hit the expensive _process_signal / filter step is skipped.
        """
        ev = self.events[self.event_idx]
        t_trig = _get_t_trig(ev)
        if t_trig is None:
            return None

        k_pre_start = config.get('k_pre_start', -3.0)
        w = config.get('k_smooth_w', 100)
        hp_freq = config.get('k_highpass_freq', 0.0)
        lp_freq = config.get('k_lowpass_freq', 0.0)
        half_win = max(config.get('k_window_sec', 3.5), abs(k_pre_start) + 0.5)

        cache_key = (self.event_idx, w, hp_freq, lp_freq, half_win)

        if cache_key in self._signal_cache:
            return self._signal_cache[cache_key]

        t_all = self.time_history['time']
        mask = (t_all >= t_trig - half_win) & (t_all <= t_trig + half_win)
        t_rel = t_all[mask] - t_trig

        if len(t_rel) < 20:
            return None

        dt = np.median(np.diff(t_all[mask]))
        fs = 1.0 / dt if dt > 0 else 0

        tau_key = 'tau_local' if 'tau_local' in self.time_history else 'shear_pressure'
        tau_raw = self.time_history[tau_key][mask]
        tau_proc = _process_signal(tau_raw, w, hp_freq, lp_freq, fs)

        lvdt_raw = self.time_history['LP_displacement'][mask]
        lvdt_proc = _process_signal(lvdt_raw, w, hp_freq, lp_freq, fs)

        payload = {
            't_trig': t_trig,
            't_rel': t_rel,
            'mask': mask,
            'fs': fs,
            'tau_raw': tau_raw,
            'tau_proc': tau_proc,
            'lvdt_raw': lvdt_raw,
            'lvdt_proc': lvdt_proc,
        }
        # Keep only the latest cache entry to bound memory usage
        self._signal_cache = {cache_key: payload}
        return payload

    # ------------------------------------------------------------------
    # Static layer – slow path, called once per event / signal-param change
    # ------------------------------------------------------------------

    def _draw_static(self, config):
        """Clear ax1 and ax2, draw fixed time-series lines.

        Leaves ax3 empty (it will be populated by _draw_dynamic).
        Installs vline handles into self._vlines_start / _end so that
        _draw_dynamic can reposition them cheaply.
        """
        signals = self._get_processed_signals(config)
        if signals is None:
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.clear()
            self.canvas.draw()
            return

        k_pre_start = config.get('k_pre_start', -3.0)
        k_pre_end = config.get('k_pre_end', -0.5)

        t_rel = signals['t_rel']
        tau_raw = signals['tau_raw']
        tau_proc = signals['tau_proc']
        lvdt_raw = signals['lvdt_raw']
        lvdt_proc = signals['lvdt_proc']

        tau_raw_z = tau_raw - tau_raw[0]
        tau_proc_z = tau_proc - tau_proc[0]
        lvdt_raw_z = lvdt_raw - lvdt_raw[0]
        lvdt_proc_z = lvdt_proc - lvdt_proc[0]

        # Cache zero-referenced processed signals for dynamic layer
        self._tau_proc_z = tau_proc_z
        self._lvdt_proc_z = lvdt_proc_z
        self._t_rel_static = t_rel

        t_disp_start = k_pre_start
        t_disp_end = abs(k_pre_start) - 1.0
        disp_mask = (t_rel >= t_disp_start) & (t_rel <= t_disp_end)

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()

        # Reset dynamic Artist references
        self._vlines_start = []
        self._vlines_end = []
        self._axvspan = None

        # ---- ax1: LVDT vs time ----
        self.ax1.plot(t_rel[disp_mask], lvdt_raw_z[disp_mask],
                      color='C0', alpha=0.5, lw=0.8, label='Raw')
        self.ax1.plot(t_rel[disp_mask], lvdt_proc_z[disp_mask],
                      color='red', alpha=0.6, lw=1.5, label='Processed')
        l1s = self.ax1.axvline(x=k_pre_start, color='blue', ls='--', alpha=0.5, lw=1)
        l1e = self.ax1.axvline(x=k_pre_end, color='blue', ls='--', alpha=0.5, lw=1)
        self.ax1.axvspan(k_pre_start, k_pre_end, alpha=0.08, color='blue')
        self.ax1.set_ylabel('LVDT slip [μm]')
        self.ax1.set_title(f'Event {self.event_idx} - K Stiffness')
        self.ax1.legend(loc='upper left', fontsize='small')
        self.ax1.grid(True)

        if hasattr(self, 'focus_y_var') and self.focus_y_var.get():
            y1_data = lvdt_proc_z[disp_mask]
            if len(y1_data) > 0:
                y1_min, y1_max = np.min(y1_data), np.max(y1_data)
                y1_range = y1_max - y1_min
                margin1 = max(0.5, y1_range * 0.1)
                self.ax1.set_ylim(y1_min - margin1, y1_max + margin1)

        # ---- ax2: Tau vs time ----
        self.ax2.plot(t_rel[disp_mask], tau_raw_z[disp_mask],
                      color='C0', alpha=0.5, lw=0.8, label='Raw')
        self.ax2.plot(t_rel[disp_mask], tau_proc_z[disp_mask],
                      color='red', alpha=0.6, lw=1.5, label='Processed')
        l2s = self.ax2.axvline(x=k_pre_start, color='blue', ls='--', alpha=0.5, lw=1)
        l2e = self.ax2.axvline(x=k_pre_end, color='blue', ls='--', alpha=0.5, lw=1)
        self.ax2.axvspan(k_pre_start, k_pre_end, alpha=0.08, color='blue')
        self.ax2.set_ylabel(r'$\tau$ [MPa]')
        self.ax2.set_xlabel('time relative [s]')
        self.ax2.legend(loc='upper left', fontsize='small')
        self.ax2.grid(True)

        if hasattr(self, 'focus_y_var') and self.focus_y_var.get():
            y2_data = tau_proc_z[disp_mask]
            if len(y2_data) > 0:
                y2_min, y2_max = np.min(y2_data), np.max(y2_data)
                y2_range = y2_max - y2_min
                margin2 = max(0.01, y2_range * 0.1)
                self.ax2.set_ylim(y2_min - margin2, y2_max + margin2)

        self._vlines_start = [l1s, l2s]
        self._vlines_end = [l1e, l2e]

        # ax3 stays clear; _draw_dynamic will populate it
        self.ax3.set_xlabel('LVDT slip [μm]')
        self.ax3.set_ylabel(r'$\tau$ [MPa]')
        self.ax3.set_title('Pre-Rupture Stiffness')
        self.ax3.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Dynamic layer – fast path, called after every recompute
    # ------------------------------------------------------------------

    def _draw_dynamic(self, result, config):
        """Update vline positions and redraw ax3 without clearing ax1/ax2.

        ax3 (scatter + fit) is always cleared because its content is fully
        determined by k_pre_start / k_pre_end; it is cheap since it only
        contains scatter points, not long time-series.
        """
        k_pre_start = config.get('k_pre_start', -3.0)
        k_pre_end = config.get('k_pre_end', -0.5)

        # ---- Update vline positions ----
        for line in self._vlines_start:
            line.set_xdata([k_pre_start, k_pre_start])
        for line in self._vlines_end:
            line.set_xdata([k_pre_end, k_pre_end])

        # ---- Update axvspan (remove old, add new) ----
        if self._axvspan is not None:
            try:
                self._axvspan.remove()
            except Exception:
                pass
        # axvspan affects both ax1 and ax2; add to ax1 only (ax2 shares same limits)
        # For simplicity we re-add to both axes independently
        for ax in [self.ax1, self.ax2]:
            for coll in ax.collections:
                try:
                    coll.remove()
                except Exception:
                    pass
            ax.axvspan(k_pre_start, k_pre_end, alpha=0.08, color='blue')
        self._axvspan = True   # sentinel; actual objects managed per-axis

        # ---- Redraw ax3 (cheap: scatter data only) ----
        self.ax3.clear()

        t_rel = getattr(self, '_t_rel_static', None)
        tau_proc_z = getattr(self, '_tau_proc_z', None)
        lvdt_proc_z = getattr(self, '_lvdt_proc_z', None)

        if t_rel is not None and tau_proc_z is not None and lvdt_proc_z is not None:
            pre_mask = (t_rel >= k_pre_start) & (t_rel <= k_pre_end)
            if np.sum(pre_mask) > 5:
                tau_pre = tau_proc_z[pre_mask]
                lvdt_pre = lvdt_proc_z[pre_mask]

                self.ax3.plot(lvdt_pre, tau_pre, 'o', color='teal',
                              markersize=3, alpha=0.5, label='Data')

                k_val = result.get('k', np.nan)
                if not np.isnan(k_val):
                    k_coeffs = result.get('k_coeffs', None)
                    if k_coeffs is not None:
                        fit_y = k_coeffs[0] * lvdt_pre + k_coeffs[1]
                        self.ax3.plot(lvdt_pre, fit_y, 'r-', lw=2,
                                      label=fr'Fit: $k$ = {k_val:.4f} MPa/$\mu$m')

                self.ax3.legend(loc='best', fontsize='small')

        self.ax3.set_xlabel('LVDT slip [μm]')
        self.ax3.set_ylabel(r'$\tau$ [MPa]')
        self.ax3.set_title('Pre-Rupture Stiffness')
        self.ax3.grid(True)

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Drag interaction
    # ------------------------------------------------------------------

    def _setup_drag_events(self):
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

    def is_navigation_active(self):
        """Check if pan or zoom tools are currently active."""
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            return self.toolbar.mode in ['pan/zoom', 'zoom rect']
        return False

    def _on_press(self, event):
        if self.is_navigation_active():
            return
        if event.inaxes not in (self.ax1, self.ax2) or event.button != 1:
            return

        x = event.xdata
        if not self._vlines_start or not self._vlines_end:
            return

        start_x = self._vlines_start[0].get_xdata()[0]
        end_x = self._vlines_end[0].get_xdata()[0]

        dist_start = abs(x - start_x)
        dist_end = abs(x - end_x)

        threshold = 0.05 * (self.ax1.get_xlim()[1] - self.ax1.get_xlim()[0])

        if dist_start < threshold and dist_start < dist_end:
            self._drag_target = 'start'
        elif dist_end < threshold:
            self._drag_target = 'end'

    def _on_motion(self, event):
        if self._drag_target is None or event.inaxes not in (self.ax1, self.ax2):
            return

        x = event.xdata
        if self._drag_target == 'start':
            for line in self._vlines_start:
                line.set_xdata([x, x])
        else:
            for line in self._vlines_end:
                line.set_xdata([x, x])

        self.canvas.draw_idle()

    def _on_release(self, event):
        if self._drag_target is None:
            return

        x = event.xdata if event.xdata is not None else (
            self._vlines_start[0].get_xdata()[0]
            if self._drag_target == 'start'
            else self._vlines_end[0].get_xdata()[0]
        )

        if self._drag_target == 'start':
            self.pre_start_var.set(f"{x:.3f}")
        else:
            self.pre_end_var.set(f"{x:.3f}")

        self._drag_target = None
        self._recompute()

    # ------------------------------------------------------------------
    # Recompute
    # ------------------------------------------------------------------

    def _recompute(self):
        cfg = self._get_current_config()

        # Check whether signal-shaping parameters changed; if so, invalidate
        # the static layer and redraw it (this triggers _draw_static which also
        # repopulates the cache with the new w / hp_freq / half_win).
        k_pre_start = cfg.get('k_pre_start', -3.0)
        w = cfg.get('k_smooth_w', 100)
        hp_freq = cfg.get('k_highpass_freq', 0.0)
        lp_freq = cfg.get('k_lowpass_freq', 0.0)
        half_win = max(cfg.get('k_window_sec', 3.5), abs(k_pre_start) + 0.5)
        new_key = (self.event_idx, w, hp_freq, lp_freq, half_win)

        static_needs_redraw = new_key not in self._signal_cache

        if static_needs_redraw:
            # Invalidate stale cache entries before _draw_static refills it
            self._signal_cache.clear()
            self._draw_static(cfg)

        result = analyze_single_k(
            self.time_history, self.events, self.event_idx, cfg
        )
        self._result = result
        self._current_cfg = cfg
        self._draw_dynamic(result, cfg)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _apply_and_save(self):
        if not hasattr(self, '_result'):
            messagebox.showwarning("Warning", "Please run Recompute first")
            return

        cfg = self._current_cfg

        try:
            k_analysis = self.data_manager.get_data(f"runs/[{self.run_idx}]/k_analysis")
        except (KeyError, TypeError, ValueError):
            k_analysis = None

        if k_analysis is None or not isinstance(k_analysis, dict):
            k_analysis = {
                'config': {
                    'k_pre_start': cfg['k_pre_start'],
                    'k_pre_end': cfg['k_pre_end'],
                    'k_smooth_w': cfg['k_smooth_w'],
                    'k_highpass_freq': cfg['k_highpass_freq'],
                    'k_lowpass_freq': cfg.get('k_lowpass_freq', 0.0),
                },
                'per_event_config': {},
                'results': {},
            }

        if 'per_event_config' not in k_analysis or not isinstance(k_analysis['per_event_config'], dict):
            k_analysis['per_event_config'] = {}
        if 'results' not in k_analysis or not isinstance(k_analysis['results'], dict):
            k_analysis['results'] = {}

        k_analysis['per_event_config'][str(self.event_idx)] = {
            'k_pre_start': cfg['k_pre_start'],
            'k_pre_end': cfg['k_pre_end'],
            'k_smooth_w': cfg['k_smooth_w'],
            'k_highpass_freq': cfg['k_highpass_freq'],
            'k_lowpass_freq': cfg.get('k_lowpass_freq', 0.0),
        }

        r = self._result
        results = k_analysis['results']
        n_events = len(self.events)

        def _ensure_array(key, default_val=np.nan):
            if key not in results or not isinstance(results[key], np.ndarray):
                results[key] = np.full(n_events, default_val)
            return results[key]

        _ensure_array('trigger_times')[self.event_idx] = r.get('trigger_time', np.nan)
        _ensure_array('k')[self.event_idx] = r.get('k', np.nan)

        run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
        run_data['k_analysis'] = k_analysis

        try:
            self.data_manager.fast_save_analysis(self.run_idx, k_analysis, group_name='k_analysis')
        except TypeError:
            self.data_manager.fast_save_analysis(self.run_idx, k_analysis)

        # Overwrite K diagnostic plot if the directory exists
        import os
        from labquake_explorer.analysis.k_stiffness_analyzer import generate_k_diagnostic_plot
        if self.data_manager.data_path:
            h5_path = self.data_manager.data_path
            h5_dir = str(h5_path.parent)
            h5_stem = h5_path.stem
            try:
                run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
                run_name_raw = run_data.get('name', f'run{self.run_idx+1}')
                run_part = run_name_raw.split('_')[0] if '_' in run_name_raw else run_name_raw
                output_dir = os.path.join(h5_dir, f"{h5_stem}_{run_part}_k")
            except Exception:
                output_dir = os.path.join(h5_dir, f"{h5_stem}_run{self.run_idx+1}_k")

            if os.path.exists(output_dir):
                save_path = os.path.join(output_dir, f"Event_{self.event_idx:03d}_k.png")
                try:
                    generate_k_diagnostic_plot(
                        self.time_history, self.events, self.event_idx, r, cfg, save_path
                    )
                    print(f"Overwrote K diagnostic plot at {save_path}")
                except Exception as e:
                    print(f"Warning: failed to overwrite K diagnostic plot: {e}")

        msg = f"Event {self.event_idx} k value updated and saved."
        messagebox.showinfo("Applied & Saved", msg)

    def on_close(self):
        if self.figure:
            plt.close(self.figure)
        self.destroy()
