"""
Interactive Event Drop Editor – allows the user to visually adjust the
pre/post fitting windows for a single event by dragging vertical lines,
then recompute and save the updated values.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from labquake_explorer.analysis.event_drop_analyzer import (
    analyze_single_event,
    moving_average,
    DEFAULT_CONFIG,
)


class EventDropEditorView(tk.Toplevel):
    def __init__(self, parent, run_idx, event_idx):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"Event Drop Editor - Run {run_idx}")
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
        self._drag_line = None  # (ax_name, point_idx)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Control frame
        ctrl = ttk.Frame(self)
        ctrl.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        ttk.Label(ctrl, text="Event:").grid(row=0, column=0, padx=5)
        self.event_combo = ttk.Combobox(ctrl, state="readonly", width=8)
        self.event_combo.grid(row=0, column=1, padx=5)
        self.event_combo['values'] = [str(i) for i in range(len(self.events))]
        self.event_combo.current(self.event_idx)
        self.event_combo.bind("<<ComboboxSelected>>", self._on_event_change)

        ttk.Label(ctrl, text="Drag the 4 vertical lines on each plot to set the 2 pre-drop and 2 post-drop points.").grid(row=0, column=2, padx=15, sticky='w')

        ttk.Button(ctrl, text="Recompute", command=self._recompute).grid(
            row=0, column=7, padx=5
        )
        ttk.Button(ctrl, text="Apply & Save", command=self._apply_and_save).grid(
            row=0, column=8, padx=5
        )

        # Build figure
        self._build_figure()
        self._recompute()

    def _on_event_change(self, event=None):
        self.event_idx = int(self.event_combo.get())
        self._load_config()
        self._build_figure()
        self._recompute()

    def _load_config(self):
        self.config = dict(DEFAULT_CONFIG)
        pew_resolved = {}  # int-keyed dict of per-event window dicts
        try:
            analysis = self.data_manager.get_data(
                f"runs/[{self.run_idx}]/analysis"
            )
            if isinstance(analysis, dict):
                cfg = analysis.get('config', {})
                if isinstance(cfg, dict):
                    for k in ['pre_win', 'post_win', 'tau_smooth_w', 'lvdt_smooth_w',
                              'push_speed', 'delay_sec', 'window_sec']:
                        if k in cfg:
                            self.config[k] = cfg[k]

                raw_pew = analysis.get('per_event_windows', {})
                # HDF5 reload may turn numeric-keyed groups into a list
                if isinstance(raw_pew, list):
                    for i, v in enumerate(raw_pew):
                        if isinstance(v, dict):
                            pew_resolved[i] = v
                elif isinstance(raw_pew, dict):
                    for k, v in raw_pew.items():
                        if isinstance(v, dict):
                            try:
                                pew_resolved[int(k)] = v
                            except (ValueError, TypeError):
                                pass
        except Exception:
            pass

        self.config['per_event_windows'] = pew_resolved

        pre = self.config['pre_win']
        post = self.config['post_win']
        default_pts = [pre[0], pre[1], post[0], post[1]]
        
        self.pts = {
            'tau': list(default_pts),
            'slip': list(default_pts),
            'lvdt': list(default_pts)
        }

        if self.event_idx in pew_resolved:
            ew = pew_resolved[self.event_idx]
            if 'pre_win' in ew and 'post_win' in ew:
                pw = ew['pre_win']
                qw = ew['post_win']
                # Handle ndarray or list
                if hasattr(pw, 'tolist'): pw = pw.tolist()
                if hasattr(qw, 'tolist'): qw = qw.tolist()
                legacy_pts = [pw[0], pw[1], qw[0], qw[1]]
                self.pts = {'tau': list(legacy_pts), 'slip': list(legacy_pts), 'lvdt': list(legacy_pts)}
            else:
                for ch in ('tau', 'slip', 'lvdt'):
                    key = f'{ch}_pts'
                    if key in ew:
                        val = ew[key]
                        if hasattr(val, 'tolist'):
                            val = val.tolist()
                        elif isinstance(val, tuple):
                            val = list(val)
                        self.pts[ch] = list(val)

    def _build_figure(self):
        # Cleanup if exists (for event switching)
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().destroy()
        if hasattr(self, 'toolbar') and self.toolbar:
            # Finding the toolbar frame is tricky if we don't store it
            pass

        self.figure = Figure(figsize=(10, 10), dpi=100)
        gs = self.figure.add_gridspec(3, 1, height_ratios=[1.2, 1.5, 1.5], hspace=0.15)
        self.ax1 = self.figure.add_subplot(gs[0])
        self.ax2 = self.figure.add_subplot(gs[1], sharex=self.ax1)
        self.ax3 = self.figure.add_subplot(gs[2], sharex=self.ax1)
        
        self.ax_map = {'tau': self.ax1, 'slip': self.ax2, 'lvdt': self.ax3}
        self.ax_inv_map = {self.ax1: 'tau', self.ax2: 'slip', self.ax3: 'lvdt'}

        self.ax1.set_ylabel(r'rel. $\tau$ [MPa]')
        self.ax1.set_title(f"Event {self.event_idx}")
        self.ax2.set_ylabel('rel. slip [\u03bcm]')
        self.ax3.set_ylabel('LVDT slip [\u03bcm]')
        self.ax3.set_xlabel('time relative [s]')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Reuse or recreate toolbar frame
        if hasattr(self, 'toolbar_frame') and self.toolbar_frame:
            self.toolbar_frame.destroy()
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self._vlines = {'tau': [], 'slip': [], 'lvdt': []}
        self._setup_drag_events()

    def _setup_drag_events(self):
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

    def _on_press(self, event):
        if event.inaxes is None or event.button != 1:
            return
        
        ax_name = self.ax_inv_map.get(event.inaxes)
        if not ax_name: return
        
        threshold = 0.05
        # Find closest line in THIS axis
        closest_idx = -1
        min_dist = float('inf')
        
        for i, line in enumerate(self._vlines[ax_name]):
            x = line.get_xdata()[0]
            dist = abs(event.xdata - x)
            if dist < threshold and dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        if closest_idx != -1:
            self._drag_line = (ax_name, closest_idx)

    def _on_motion(self, event):
        if self._drag_line is None or event.inaxes is None:
            return
            
        ax_name, idx = self._drag_line
        # Allow dragging across axes? No, restrict to original axis
        if self.ax_inv_map.get(event.inaxes) != ax_name:
            return
            
        x = event.xdata
        self._vlines[ax_name][idx].set_xdata([x, x])
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self._drag_line is None:
            return
            
        ax_name, idx = self._drag_line
        x = self._vlines[ax_name][idx].get_xdata()[0]
        self.pts[ax_name][idx] = x
        self._drag_line = None
        
        # Optionally recompute on drop
        self._recompute()

    def _recompute(self):
        cfg = dict(self.config)
        cfg['per_event_windows'] = {
            self.event_idx: {
                'tau_pts': tuple(self.pts['tau']),
                'slip_pts': tuple(self.pts['slip']),
                'lvdt_pts': tuple(self.pts['lvdt']),
            }
        }

        result = analyze_single_event(
            self.time_history, self.events, self.event_idx, cfg
        )
        self._result = result
        self._draw(result, cfg)

    def _draw(self, result, config):
        ev = self.events[self.event_idx]
        t_trig = ev['event_time'] if isinstance(ev, dict) else float(ev)
        half_win = config.get('window_sec', 1.5)
        t_all = self.time_history['time']
        mask = (t_all >= t_trig - half_win) & (t_all <= t_trig + half_win)
        t_rel = t_all[mask] - t_trig

        if len(t_rel) < 20:
            return

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()

        # Helper to draw extrapolating lines
        def _draw_lines(ax, pts, res, color):
            if not res.get('valid'): return
            t_pre = np.linspace(pts[0], 0, 50)
            ax.plot(t_pre, res['coeff_pre'][0]*t_pre + res['coeff_pre'][1], '--', color=color, alpha=0.7, lw=1.5)
            t_post = np.linspace(0, pts[3], 50)
            ax.plot(t_post, res['coeff_post'][0]*t_post + res['coeff_post'][1], '--', color=color, alpha=0.7, lw=1.5)

        # --- (1) Tau ---
        tau_key = 'tau_local' if 'tau_local' in self.time_history else 'shear_pressure'
        tau_raw = self.time_history[tau_key][mask]
        tau_sm = moving_average(tau_raw, config.get('tau_smooth_w', 100))
        if len(tau_sm) < len(t_rel):
            tau_sm = np.pad(tau_sm, (0, len(t_rel) - len(tau_sm)), 'edge')
        tau_sm = tau_sm - tau_sm[0]

        self.ax1.plot(t_rel, tau_sm, 'k', alpha=0.8)
        tau_res = result.get('tau_res', {})
        if tau_res.get('valid'):
            _draw_lines(self.ax1, self.pts['tau'], tau_res, 'darkslateblue')
            val = abs(tau_res['delta'])
            self.ax1.annotate('', xy=(0, tau_res['val_post_0']),
                              xytext=(0, tau_res['val_pre_0']),
                              arrowprops=dict(arrowstyle='<->', color='darkslateblue', lw=2.5))
            self.ax1.text(-0.1, (tau_res['val_pre_0'] + tau_res['val_post_0']) / 2,
                          r"$\Delta\tau$=%.4f" % val, ha='right', va='center',
                          color='darkslateblue', fontweight='bold', fontsize=10)
        self.ax1.set_ylabel(r'rel. $\tau$ [MPa]')
        self.ax1.set_title(f"Event {self.event_idx}")
        self.ax1.grid(True)

        # --- (2) Slip ---
        eddy_keys = sorted([k for k in self.time_history.keys() if 'eddy' in k.lower()])
        eddy_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
                       
        target_idx = 2 if len(eddy_keys) >= 3 else (0 if len(eddy_keys) > 0 else -1)
        for i, k in enumerate(eddy_keys):
            d = self.time_history[k][mask] - self.time_history[k][mask][0]
            color = eddy_colors[i % len(eddy_colors)]
            self.ax2.plot(t_rel, d, alpha=0.8, label=f'E{i+1}', color=color)

            res_key = f'delta_E{i+1}_res'
            res_slip = result.get(res_key, {})
            if i == target_idx and res_slip.get('valid'):
                _draw_lines(self.ax2, self.pts['slip'], res_slip, color)
                val = abs(res_slip['delta'])
                self.ax2.annotate('', xy=(0, res_slip['val_post_0']),
                                  xytext=(0, res_slip['val_pre_0']),
                                  arrowprops=dict(arrowstyle='<->', color=color, lw=2))
                self.ax2.text(-0.1, (res_slip['val_pre_0'] + res_slip['val_post_0']) / 2,
                              fr"$\delta_{{E{i+1}}}$={val:.1f}", ha='right', va='center',
                              color=color, fontweight='bold', fontsize=8)

        self.ax2.set_ylabel('rel. slip [μm]')
        self.ax2.legend(loc='upper left', fontsize='small')
        self.ax2.grid(True)

        # --- (3) LVDT ---
        lvdt_raw = self.time_history['LP_displacement'][mask]
        lvdt_sm = moving_average(lvdt_raw, config.get('lvdt_smooth_w', 100))
        if len(lvdt_sm) < len(t_rel):
            lvdt_sm = np.pad(lvdt_sm, (0, len(t_rel) - len(lvdt_sm)), 'edge')
        lvdt_0 = lvdt_sm - lvdt_sm[0]

        self.ax3.plot(t_rel, lvdt_0, alpha=0.7, color='slategrey')
        lvdt_res = result.get('lvdt_res', {})
        if lvdt_res.get('valid'):
            _draw_lines(self.ax3, self.pts['lvdt'], lvdt_res, 'darkslateblue')
            val = abs(lvdt_res['delta'])
            self.ax3.annotate('', xy=(0, lvdt_res['val_post_0']),
                              xytext=(0, lvdt_res['val_pre_0']),
                              arrowprops=dict(arrowstyle='<->', color='darkslateblue', lw=2.5))
            self.ax3.text(-0.1, (lvdt_res['val_pre_0'] + lvdt_res['val_post_0']) / 2,
                          fr"$\delta_{{LVDT}}$={val:.1f} $\mu m$", ha='right', va='center',
                          color='darkslateblue', fontweight='bold', fontsize=10)
        self.ax3.set_xlabel('time relative [s]')
        self.ax3.set_ylabel('LVDT slip [μm]')
        self.ax3.grid(True)

        # --- Draw draggable vlines ---
        colors = ['blue', 'blue', 'red', 'red']
        styles = ['--', '-', '-', '--']
        
        self._vlines = {'tau': [], 'slip': [], 'lvdt': []}
        for ax_name, ax in [('tau', self.ax1), ('slip', self.ax2), ('lvdt', self.ax3)]:
            for i in range(4):
                line = ax.axvline(x=self.pts[ax_name][i], color=colors[i],
                                  linestyle=styles[i], alpha=0.6, lw=1.5,
                                  picker=5)
                self._vlines[ax_name].append(line)

        self.figure.tight_layout()
        self.canvas.draw()

    def _apply_and_save(self):
        if not hasattr(self, '_result'):
            messagebox.showwarning("Warning", "Please run Recompute first")
            return

        per_event = {
            'tau_pts': tuple(self.pts['tau']),
            'slip_pts': tuple(self.pts['slip']),
            'lvdt_pts': tuple(self.pts['lvdt']),
        }

        try:
            analysis = self.data_manager.get_data(f"runs/[{self.run_idx}]/analysis")
        except (KeyError, TypeError, ValueError):
            analysis = None

        if analysis is None or not isinstance(analysis, dict):
            analysis = {
                'config': dict(self.config),
                'per_event_windows': {},
                'results': {},
            }

        if 'per_event_windows' not in analysis or not isinstance(analysis['per_event_windows'], dict):
            analysis['per_event_windows'] = {}
        if 'results' not in analysis or not isinstance(analysis['results'], dict):
            analysis['results'] = {}

        analysis['per_event_windows'][str(self.event_idx)] = per_event

        r = self._result
        results = analysis['results']
        n_events = len(self.events)

        def _ensure_array(key, default_val=np.nan):
            if key not in results or not isinstance(results[key], np.ndarray):
                results[key] = np.full(n_events, default_val)
            return results[key]

        _ensure_array('trigger_times')[self.event_idx] = r.get('trigger_time', np.nan)
        _ensure_array('delta_tau')[self.event_idx] = r.get('delta_tau', np.nan)
        _ensure_array('delta_lvdt')[self.event_idx] = r.get('delta_lvdt', np.nan)
        _ensure_array('D_Push')[self.event_idx] = r.get('D_Push', np.nan)
        _ensure_array('D_max')[self.event_idx] = r.get('D_max', np.nan)
        _ensure_array('D_E3')[self.event_idx] = r.get('D_E3', np.nan)

        eddy_keys = sorted([k for k in self.time_history.keys() if 'eddy' in k.lower()])
        for i in range(len(eddy_keys)):
            label = f'delta_E{i+1}'
            _ensure_array(label)[self.event_idx] = r.get(label, np.nan)

        # Directly update HDF5 file (Partial Save)
        self.data_manager.fast_save_analysis(self.run_idx, analysis)

        msg = f"Event {self.event_idx} updated and saved to HDF5."
        messagebox.showinfo("Applied & Saved", msg)

    def on_close(self):
        if self.figure:
            plt.close(self.figure)
        self.destroy()
