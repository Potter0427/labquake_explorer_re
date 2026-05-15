"""
Event K Stiffness Editor – allows the user to adjust the locking window,
smoothing, and low-pass filter parameters for a single event, then
recompute and save the updated k value.
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
        self.title(f"Event K Editor - Run {run_idx}")
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

        # Highpass freq
        ttk.Label(ctrl, text="HP freq (Hz):").grid(row=0, column=8, padx=5)
        self.hp_freq_var = tk.StringVar(value=str(self.config['k_highpass_freq']))
        ttk.Entry(ctrl, textvariable=self.hp_freq_var, width=6).grid(row=0, column=9, padx=3)

        # Buttons
        ttk.Button(ctrl, text="Recompute", command=self._recompute).grid(
            row=0, column=10, padx=10
        )
        ttk.Button(ctrl, text="Apply & Save", command=self._apply_and_save).grid(
            row=0, column=11, padx=5
        )

        # Build figure
        self._build_figure()
        self._recompute()

    def _on_event_change(self, event=None):
        self.event_idx = int(self.event_combo.get())
        self._load_config()
        # Update UI from config
        self.pre_start_var.set(str(self.config['k_pre_start']))
        self.pre_end_var.set(str(self.config['k_pre_end']))
        self.smooth_w_var.set(str(self.config['k_smooth_w']))
        self.hp_freq_var.set(str(self.config['k_highpass_freq']))
        self._build_figure()
        self._recompute()

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
                              'k_highpass_freq', 'k_window_sec']:
                        if k in cfg:
                            self.config[k] = float(cfg[k]) if 'freq' in k or 'start' in k or 'end' in k or 'sec' in k else int(cfg[k])

                # Check per-event overrides
                per_event = k_analysis.get('per_event_config', {})
                if isinstance(per_event, dict):
                    ev_key = str(self.event_idx)
                    if ev_key in per_event and isinstance(per_event[ev_key], dict):
                        ev_cfg = per_event[ev_key]
                        for k in ['k_pre_start', 'k_pre_end', 'k_smooth_w', 'k_highpass_freq']:
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
        except ValueError:
            pass
        return cfg

    def _build_figure(self):
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().destroy()
        if hasattr(self, 'toolbar_frame') and self.toolbar_frame:
            self.toolbar_frame.destroy()

        self.figure = Figure(figsize=(10, 10), dpi=100)
        gs = self.figure.add_gridspec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.3)
        self.ax1 = self.figure.add_subplot(gs[0])
        self.ax2 = self.figure.add_subplot(gs[1], sharex=self.ax1)
        self.ax3 = self.figure.add_subplot(gs[2])

        self.ax1.set_ylabel('LVDT slip [\u03bcm]')
        self.ax2.set_ylabel(r'rel. $\tau$ [MPa]')
        self.ax2.set_xlabel('time relative [s]')
        self.ax3.set_ylabel(r'rel. $\tau$ [MPa]')
        self.ax3.set_xlabel('LVDT slip [\u03bcm]')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

    def _recompute(self):
        cfg = self._get_current_config()
        result = analyze_single_k(
            self.time_history, self.events, self.event_idx, cfg
        )
        self._result = result
        self._current_cfg = cfg
        self._draw(result, cfg)

    def _draw(self, result, config):
        ev = self.events[self.event_idx]
        t_trig = _get_t_trig(ev)
        if t_trig is None:
            return

        k_pre_start = config.get('k_pre_start', -3.0)
        k_pre_end = config.get('k_pre_end', -0.5)
        w = config.get('k_smooth_w', 100)
        hp_freq = config.get('k_highpass_freq', 0.0)
        half_win = max(config.get('k_window_sec', 3.5), abs(k_pre_start) + 0.5)

        t_all = self.time_history['time']
        mask = (t_all >= t_trig - half_win) & (t_all <= t_trig + half_win)
        t_rel = t_all[mask] - t_trig

        if len(t_rel) < 20:
            return

        dt = np.median(np.diff(t_all[mask]))
        fs = 1.0 / dt if dt > 0 else 0

        # Raw and processed
        tau_key = 'tau_local' if 'tau_local' in self.time_history else 'shear_pressure'
        tau_raw = self.time_history[tau_key][mask]
        tau_proc = _process_signal(tau_raw, w, hp_freq, fs)
        tau_raw_z = tau_raw - tau_raw[0]
        tau_proc_z = tau_proc - tau_proc[0]

        lvdt_raw = self.time_history['LP_displacement'][mask]
        lvdt_proc = _process_signal(lvdt_raw, w, hp_freq, fs)
        lvdt_raw_z = lvdt_raw - lvdt_raw[0]
        lvdt_proc_z = lvdt_proc - lvdt_proc[0]

        # Display range
        t_disp_start = k_pre_start
        t_disp_end = abs(k_pre_start) - 1.0
        disp_mask = (t_rel >= t_disp_start) & (t_rel <= t_disp_end)

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()

        # --- Subplot 1: LVDT vs time ---
        self.ax1.plot(t_rel[disp_mask], lvdt_raw_z[disp_mask], color='C0', alpha=0.5, lw=0.8, label='Raw')
        self.ax1.plot(t_rel[disp_mask], lvdt_proc_z[disp_mask], color='red', alpha=0.6, lw=1.5, label='Processed')
        self.ax1.axvline(x=k_pre_start, color='blue', ls='--', alpha=0.5, lw=1)
        self.ax1.axvline(x=k_pre_end, color='blue', ls='--', alpha=0.5, lw=1)
        self.ax1.axvspan(k_pre_start, k_pre_end, alpha=0.08, color='blue')
        self.ax1.set_ylabel('LVDT slip [\u03bcm]')
        self.ax1.set_title(f'Event {self.event_idx} - K Stiffness')
        self.ax1.legend(loc='upper left', fontsize='small')
        self.ax1.grid(True)

        # --- Subplot 2: Tau vs time ---
        self.ax2.plot(t_rel[disp_mask], tau_raw_z[disp_mask], color='C0', alpha=0.5, lw=0.8, label='Raw')
        self.ax2.plot(t_rel[disp_mask], tau_proc_z[disp_mask], color='red', alpha=0.6, lw=1.5, label='Processed')
        self.ax2.axvline(x=k_pre_start, color='blue', ls='--', alpha=0.5, lw=1)
        self.ax2.axvline(x=k_pre_end, color='blue', ls='--', alpha=0.5, lw=1)
        self.ax2.axvspan(k_pre_start, k_pre_end, alpha=0.08, color='blue')
        self.ax2.set_ylabel(r'rel. $\tau$ [MPa]')
        self.ax2.set_xlabel('time relative [s]')
        self.ax2.legend(loc='upper left', fontsize='small')
        self.ax2.grid(True)

        # --- Subplot 3: Tau vs LVDT ---
        pre_mask = (t_rel >= k_pre_start) & (t_rel <= k_pre_end)
        if np.sum(pre_mask) > 5:
            tau_pre = tau_proc_z[pre_mask]
            lvdt_pre = lvdt_proc_z[pre_mask]

            self.ax3.plot(lvdt_pre, tau_pre, 'o', color='teal', markersize=3, alpha=0.5, label='Data')

            k_val = result.get('k', np.nan)
            if not np.isnan(k_val):
                k_coeffs = result.get('k_coeffs', None)
                if k_coeffs is not None:
                    fit_y = k_coeffs[0] * lvdt_pre + k_coeffs[1]
                    self.ax3.plot(lvdt_pre, fit_y, 'r-', lw=2,
                                 label=fr'Fit: $k$ = {k_val:.4f} MPa/$\mu$m')

            self.ax3.legend(loc='best', fontsize='small')

        self.ax3.set_xlabel('LVDT slip [\u03bcm]')
        self.ax3.set_ylabel(r'rel. $\tau$ [MPa]')
        self.ax3.set_title('Pre-Rupture Stiffness')
        self.ax3.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

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
                },
                'per_event_config': {},
                'results': {},
            }

        if 'per_event_config' not in k_analysis or not isinstance(k_analysis['per_event_config'], dict):
            k_analysis['per_event_config'] = {}
        if 'results' not in k_analysis or not isinstance(k_analysis['results'], dict):
            k_analysis['results'] = {}

        # Save per-event config override
        k_analysis['per_event_config'][str(self.event_idx)] = {
            'k_pre_start': cfg['k_pre_start'],
            'k_pre_end': cfg['k_pre_end'],
            'k_smooth_w': cfg['k_smooth_w'],
            'k_highpass_freq': cfg['k_highpass_freq'],
        }

        # Save result
        r = self._result
        results = k_analysis['results']
        n_events = len(self.events)

        def _ensure_array(key, default_val=np.nan):
            if key not in results or not isinstance(results[key], np.ndarray):
                results[key] = np.full(n_events, default_val)
            return results[key]

        _ensure_array('trigger_times')[self.event_idx] = r.get('trigger_time', np.nan)
        _ensure_array('k')[self.event_idx] = r.get('k', np.nan)

        # Store in memory
        run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
        run_data['k_analysis'] = k_analysis

        # Save to HDF5
        try:
            self.data_manager.fast_save_analysis(self.run_idx, k_analysis, group_name='k_analysis')
        except TypeError:
            # Fallback if fast_save_analysis doesn't support group_name
            self.data_manager.fast_save_analysis(self.run_idx, k_analysis)

        msg = f"Event {self.event_idx} k value updated and saved."
        messagebox.showinfo("Applied & Saved", msg)

    def on_close(self):
        if self.figure:
            plt.close(self.figure)
        self.destroy()
