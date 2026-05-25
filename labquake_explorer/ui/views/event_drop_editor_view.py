"""
Interactive Event Drop Editor – allows the user to visually adjust the
pre/post fitting windows for a single event by dragging vertical lines,
then recompute and save the updated values.

Performance notes
-----------------
Drawing is split into two layers to avoid full-redraw on every mouse release:

  _draw_static(config)
      Clears all axes and redraws the time-series data that never changes
      for a given event (raw signal lines, grid, labels).  Called once per
      event load / event switch.  tight_layout() runs here.

  _draw_dynamic(result, config)
      Only updates mutable Artists: vline positions, fit-line segments and
      annotation text.  Does NOT call ax.clear().  Uses canvas.draw_idle()
      for a non-blocking repaint.

The FigureCanvasTkAgg widget is created once and reused across event
switches; only the Figure artists are refreshed.
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
    _get_t_trig,
    DEFAULT_CONFIG,
)


class EventDropEditorView(tk.Toplevel):
    def __init__(self, parent, run_idx, event_idx):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"Event Drop Editor - Run {run_idx + 1}")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if event_idx <= 0:
            self.event_idx = 1
        else:
            self.event_idx = event_idx

        self.run_idx = run_idx
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

        if self.event_idx >= len(self.events):
            messagebox.showerror("Error", f"Event {self.event_idx} out of range")
            self.destroy()
            return

        self._load_config()

        self.figure = None
        self.canvas = None
        self.toolbar = None
        self._drag_line = None  # (ax_name, point_idx)
        self._preview_active = False  # True while recompute preview overrides skip state

        # Dynamic Artist handles – populated by _draw_static / _draw_dynamic
        self._vlines = {'tau': [], 'slip': [], 'lvdt': []}
        self._fit_lines = []        # list of Line2D fit-segment artists
        self._annotations = []      # list of Annotation artists

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Control frame
        ctrl = ttk.Frame(self)
        ctrl.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        ttk.Label(ctrl, text="Event:").grid(row=0, column=0, padx=5)
        ttk.Button(ctrl, text="< Prev", command=self._go_prev).grid(row=0, column=1, padx=2)
        self.event_combo = ttk.Combobox(ctrl, state="readonly", width=8)
        self.event_combo.grid(row=0, column=2, padx=2)
        valid_indices = [i for i in range(1, len(self.events))]
        self.event_combo['values'] = [str(i) for i in valid_indices]
        
        # Determine the index to set in the combobox
        idx_to_set = self.event_idx if self.event_idx in valid_indices else (valid_indices[0] if valid_indices else None)
        if idx_to_set is not None:
            try:
                cb_index = valid_indices.index(idx_to_set)
                self.event_combo.current(cb_index)
            except ValueError:
                pass
                
        self.event_combo.bind("<<ComboboxSelected>>", self._on_event_change)
        ttk.Button(ctrl, text="Next >", command=self._go_next).grid(row=0, column=3, padx=2)

        # Status Label
        self.status_label = ttk.Label(ctrl, text="", font=("TkDefaultFont", 10, "bold"))
        self.status_label.grid(row=0, column=4, padx=10)

        ttk.Label(ctrl, text="Drag the 4 vertical lines on each plot to set the 2 pre-drop and 2 post-drop points.").grid(row=0, column=5, padx=15, sticky='w')

        ttk.Button(ctrl, text="Recompute", command=self._recompute_preview).grid(
            row=0, column=6, padx=5
        )
        ttk.Button(ctrl, text="Apply & Save", command=self._apply_and_save).grid(
            row=0, column=7, padx=5
        )
        ttk.Button(ctrl, text="Delete Event", command=self._delete_event).grid(
            row=0, column=8, padx=5
        )

        # Build figure once – Canvas is never destroyed after this point
        self._build_figure()
        self._draw_static(self.config)
        self._recompute()

    # ------------------------------------------------------------------
    # Event switching
    # ------------------------------------------------------------------

    def _on_event_change(self, event=None):
        self.event_idx = int(self.event_combo.get())
        self._preview_active = False
        self._load_config()
        # Redraw static layer for the new event, then recompute dynamic layer
        self._draw_static(self.config)
        self._recompute()

    def _go_prev(self):
        valid_indices = [i for i in range(1, len(self.events))]
        if self.event_idx not in valid_indices:
            return
        current_cb_index = valid_indices.index(self.event_idx)
        if current_cb_index > 0:
            self.event_combo.current(current_cb_index - 1)
            self._on_event_change()

    def _go_next(self):
        valid_indices = [i for i in range(1, len(self.events))]
        if self.event_idx not in valid_indices:
            return
        current_cb_index = valid_indices.index(self.event_idx)
        if current_cb_index < len(valid_indices) - 1:
            self.event_combo.current(current_cb_index + 1)
            self._on_event_change()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self):
        self.config = dict(DEFAULT_CONFIG)
        pew_resolved = {}
        try:
            run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
            if isinstance(run_data, dict) and 'config' in run_data:
                cfg = run_data['config']
                for k in ['pre_win', 'post_win', 'tau_smooth_w', 'lvdt_smooth_w',
                          'push_speed', 'delay_sec', 'window_sec']:
                    if k in cfg:
                        self.config[k] = cfg[k]
        except Exception:
            pass

        # Load event-specific windows from self.events
        if self.event_idx < len(self.events):
            ev = self.events[self.event_idx]
            pew = {}
            if isinstance(ev, dict):
                import numpy as np
                def _valid_pts(d):
                    if 'pre_start' not in d: return False
                    pts = [d['pre_start'], d['pre_end'], d['post_start'], d['post_end']]
                    return all(x is not None and not np.isnan(x) for x in pts)

                if 'tau' in ev and _valid_pts(ev['tau']):
                    pew['tau_pts'] = [ev['tau']['pre_start'], ev['tau']['pre_end'], ev['tau']['post_start'], ev['tau']['post_end']]
                if 'delta' in ev and _valid_pts(ev['delta']):
                    pew['slip_pts'] = [ev['delta']['pre_start'], ev['delta']['pre_end'], ev['delta']['post_start'], ev['delta']['post_end']]
                if 'lvdt' in ev and _valid_pts(ev['lvdt']):
                    pew['lvdt_pts'] = [ev['lvdt']['pre_start'], ev['lvdt']['pre_end'], ev['lvdt']['post_start'], ev['lvdt']['post_end']]
            if pew:
                pew_resolved[self.event_idx] = pew

        # Always read skip_events from the shared run-level list.
        self.config['skip_events'] = self.data_manager.get_run_skip_events(self.run_idx)

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
            for ch in ('tau', 'slip', 'lvdt'):
                key = f'{ch}_pts'
                if key in ew:
                    val = ew[key]
                    if hasattr(val, 'tolist'):
                        val = val.tolist()
                    elif isinstance(val, tuple):
                        val = list(val)
                    self.pts[ch] = list(val)

    # ------------------------------------------------------------------
    # Figure / Canvas – created once, never destroyed
    # ------------------------------------------------------------------

    def _build_figure(self):
        self.figure = Figure(figsize=(10, 10), dpi=100)
        gs = self.figure.add_gridspec(3, 1, height_ratios=[1.2, 1.5, 1.5], hspace=0.15)
        self.ax1 = self.figure.add_subplot(gs[0])
        self.ax2 = self.figure.add_subplot(gs[1], sharex=self.ax1)
        self.ax3 = self.figure.add_subplot(gs[2], sharex=self.ax1)

        self.ax_map = {'tau': self.ax1, 'slip': self.ax2, 'lvdt': self.ax3}
        self.ax_inv_map = {self.ax1: 'tau', self.ax2: 'slip', self.ax3: 'lvdt'}

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self._setup_drag_events()

    # ------------------------------------------------------------------
    # Static layer – slow path, called once per event
    # ------------------------------------------------------------------

    def _draw_static(self, config):
        """Clear axes and draw the fixed time-series data for the current event.

        Populates self._vlines with axvline handles so that _draw_dynamic
        can update their positions without clearing the axes.
        """
        ev = self.events[self.event_idx]
        t_trig = _get_t_trig(ev)
        if t_trig is None:
            return
        half_win = config.get('window_sec', 1.5)
        t_all = self.time_history['time']
        mask = (t_all >= t_trig - half_win) & (t_all <= t_trig + half_win)
        t_rel = t_all[mask] - t_trig

        if len(t_rel) < 20:
            return

        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()

        # Reset dynamic Artist references
        self._fit_lines = []
        self._annotations = []

        # ---- (1) Tau ----
        tau_key = 'tau_local' if 'tau_local' in self.time_history else 'shear_pressure'
        tau_raw = self.time_history[tau_key][mask]
        tau_sm = moving_average(tau_raw, config.get('tau_smooth_w', 100))
        if len(tau_sm) < len(t_rel):
            tau_sm = np.pad(tau_sm, (0, len(t_rel) - len(tau_sm)), 'edge')
        self._tau_sm = tau_sm - tau_sm[0]   # cache for dynamic layer

        self.ax1.plot(t_rel, self._tau_sm, 'k', alpha=0.8)
        self.ax1.set_ylabel(r'rel. $\tau$ [MPa]')
        self.ax1.set_title(f"Event {self.event_idx}")
        self.ax1.grid(True)

        # ---- (2) Slip ----
        eddy_keys = sorted([k for k in self.time_history.keys() if 'eddy' in k.lower()])
        eddy_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        self._eddy_keys = eddy_keys  # cache for dynamic layer

        for i, k in enumerate(eddy_keys):
            d = self.time_history[k][mask] - self.time_history[k][mask][0]
            color = eddy_colors[i % len(eddy_colors)]
            self.ax2.plot(t_rel, d, alpha=0.8, label=f'E{i+1}', color=color)

        self.ax2.set_ylabel('rel. slip [μm]')
        self.ax2.legend(loc='upper left', fontsize='small')
        self.ax2.grid(True)

        # ---- (3) LVDT ----
        lvdt_raw = self.time_history['LP_displacement'][mask]
        lvdt_sm = moving_average(lvdt_raw, config.get('lvdt_smooth_w', 100))
        if len(lvdt_sm) < len(t_rel):
            lvdt_sm = np.pad(lvdt_sm, (0, len(t_rel) - len(lvdt_sm)), 'edge')
        self._lvdt_0 = lvdt_sm - lvdt_sm[0]  # cache for dynamic layer

        self.ax3.plot(t_rel, self._lvdt_0, alpha=0.7, color='slategrey')
        self.ax3.set_xlabel('time relative [s]')
        self.ax3.set_ylabel('LVDT slip [μm]')
        self.ax3.grid(True)

        # ---- Draw draggable vlines (initially at pts positions) ----
        colors = ['blue', 'blue', 'red', 'red']
        styles = ['--', '-', '-', '--']
        self._vlines = {'tau': [], 'slip': [], 'lvdt': []}
        for ax_name, ax in [('tau', self.ax1), ('slip', self.ax2), ('lvdt', self.ax3)]:
            for i in range(4):
                line = ax.axvline(x=self.pts[ax_name][i], color=colors[i],
                                  linestyle=styles[i], alpha=0.6, lw=1.5,
                                  picker=5)
                self._vlines[ax_name].append(line)

        # Cache time array slice for dynamic layer
        self._t_rel = t_rel
        self._mask = mask

        # Force full window on x-axis so the first event (with limited
        # post-trigger data) always displays the complete intended range.
        self.ax1.set_xlim(-half_win, half_win)

        self.figure.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Dynamic layer – fast path, called after every recompute
    # ------------------------------------------------------------------

    def _draw_dynamic(self, result, config):
        """Update only mutable Artists without clearing any axis.

        Removes old fit lines and annotations, then redraws them at their
        new positions.  Vline positions are updated via set_xdata().
        Ends with canvas.draw_idle() for a non-blocking repaint.
        """
        # Remove stale dynamic artists
        for artist in self._fit_lines + self._annotations:
            try:
                artist.remove()
            except Exception:
                pass
        self._fit_lines = []
        self._annotations = []

        def _add_fit_lines(ax, pts, res, color):
            if not res.get('valid'):
                return
            t_pre = np.linspace(pts[0], 0, 50)
            l1, = ax.plot(t_pre, res['coeff_pre'][0] * t_pre + res['coeff_pre'][1],
                          '--', color=color, alpha=0.7, lw=1.5)
            t_post = np.linspace(0, pts[3], 50)
            l2, = ax.plot(t_post, res['coeff_post'][0] * t_post + res['coeff_post'][1],
                          '--', color=color, alpha=0.7, lw=1.5)
            self._fit_lines.extend([l1, l2])

        def _add_annotation(ax, pts, res, label_fmt, color, fontsize=10):
            if not res.get('valid'):
                return
            val = abs(res['delta'])
            ann = ax.annotate('', xy=(0, res['val_post_0']),
                              xytext=(0, res['val_pre_0']),
                              arrowprops=dict(arrowstyle='<->', color=color, lw=2.5))
            txt = ax.text(0.1, (res['val_pre_0'] + res['val_post_0']) / 2,
                          label_fmt % val, ha='left', va='center',
                          color=color, fontweight='bold', fontsize=fontsize)
            self._annotations.extend([ann, txt])

        # ---- Tau vlines + fit ----
        tau_res = result.get('tau_res', {})
        for i, line in enumerate(self._vlines['tau']):
            line.set_xdata([self.pts['tau'][i], self.pts['tau'][i]])
        _add_fit_lines(self.ax1, self.pts['tau'], tau_res, 'darkslateblue')
        _add_annotation(self.ax1, self.pts['tau'], tau_res,
                        r'$\boldsymbol{\Delta}\boldsymbol{\tau}$=$\boldsymbol{%.4f}$', 'darkslateblue', fontsize=10)

        # ---- Slip vlines + fit ----
        eddy_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        target_idx = 2 if len(self._eddy_keys) >= 3 else (0 if self._eddy_keys else -1)
        for i, line in enumerate(self._vlines['slip']):
            line.set_xdata([self.pts['slip'][i], self.pts['slip'][i]])
        if target_idx >= 0:
            res_key = f'delta_E{target_idx + 1}_res'
            res_slip = result.get(res_key, {})
            slip_color = 'darkslateblue'
            _add_fit_lines(self.ax2, self.pts['slip'], res_slip, slip_color)
            if res_slip.get('valid'):
                val = abs(res_slip['delta'])
                ann = self.ax2.annotate('', xy=(0, res_slip['val_post_0']),
                                        xytext=(0, res_slip['val_pre_0']),
                                        arrowprops=dict(arrowstyle='<->', color=slip_color, lw=2))
                txt = self.ax2.text(0.1, (res_slip['val_pre_0'] + res_slip['val_post_0']) / 2,
                                    fr"$\boldsymbol{{\delta}}$=$\boldsymbol{{{val:.1f}}}$ $\boldsymbol{{\mu m}}$",
                                    ha='left', va='center',
                                    color=slip_color, fontweight='bold', fontsize=10)
                self._annotations.extend([ann, txt])

        # ---- LVDT vlines + fit ----
        lvdt_res = result.get('lvdt_res', {})
        for i, line in enumerate(self._vlines['lvdt']):
            line.set_xdata([self.pts['lvdt'][i], self.pts['lvdt'][i]])
        _add_fit_lines(self.ax3, self.pts['lvdt'], lvdt_res, 'darkslateblue')
        if lvdt_res.get('valid'):
            val = abs(lvdt_res['delta'])
            ann = self.ax3.annotate('', xy=(0, lvdt_res['val_post_0']),
                                    xytext=(0, lvdt_res['val_pre_0']),
                                    arrowprops=dict(arrowstyle='<->', color='darkslateblue', lw=2.5))
            txt = self.ax3.text(0.1, (lvdt_res['val_pre_0'] + lvdt_res['val_post_0']) / 2,
                                fr"$\boldsymbol{{\delta}}_{{\boldsymbol{{LVDT}}}}$=$\boldsymbol{{{val:.1f}}}$ $\boldsymbol{{\mu m}}$",
                                ha='left', va='center',
                                color='darkslateblue', fontweight='bold', fontsize=10)
            self._annotations.extend([ann, txt])

        self._update_status_label()

        is_skipped = (self.event_idx in self.config.get('skip_events', []))
        if is_skipped and not self._preview_active:
            txt = self.ax1.text(0.5, 0.5, "EVENT SKIPPED / DELETED", color='red', fontsize=16,
                                ha='center', va='center', transform=self.ax1.transAxes,
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
            self._annotations.append(txt)

        self.canvas.draw_idle()

    def _update_status_label(self):
        is_skipped = (self.event_idx in self.config.get('skip_events', []))
        if is_skipped and not self._preview_active:
            self.status_label.config(text="DELETED / SKIPPED", foreground="red")
        elif is_skipped and self._preview_active:
            self.status_label.config(text="PREVIEW (unsaved)", foreground="orange")
        else:
            self.status_label.config(text="ACTIVE", foreground="green")

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
        if event.inaxes is None or event.button != 1:
            return

        ax_name = self.ax_inv_map.get(event.inaxes)
        if not ax_name:
            return

        threshold = 0.05
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

        # Recompute triggers _draw_dynamic only (no ax.clear)
        self._recompute()

    # ------------------------------------------------------------------
    # Recompute
    # ------------------------------------------------------------------

    def _recompute_preview(self):
        """Called by the Recompute button: show computed lines even if event
        is skipped.  Sets _preview_active so the red overlay is suppressed."""
        if self.event_idx in self.config.get('skip_events', []):
            self._preview_active = True
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

        # Force computation if we are previewing a deleted event
        if self._preview_active and 'skip_events' in cfg:
            skip_list = list(cfg['skip_events'])
            if self.event_idx in skip_list:
                skip_list.remove(self.event_idx)
            cfg['skip_events'] = skip_list

        result = analyze_single_event(
            self.time_history, self.events, self.event_idx, cfg
        )
        self._result = result
        self._draw_dynamic(result, cfg)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------



    def _apply_and_save(self):
        if not hasattr(self, '_result'):
            messagebox.showwarning("Warning", "Please run Recompute first")
            return

        r = self._result
        
        # 移除畫圖用的殘差資料, 但不要修改 r 本身以免影響後續繪圖
        save_data = {key: val for key, val in r.items() if not key.endswith('_res') and key != 'event_idx'}

        # category=None 直接寫入根目錄，只覆寫 r 中有的項目
        self.data_manager.fast_save_event_analysis(self.run_idx, self.event_idx, None, save_data)

        # Overwrite Event Drop diagnostic plot if the directory exists
        import os
        from labquake_explorer.analysis.batch_runner import generate_diagnostic_plot
        if self.data_manager.data_path:
            h5_path = self.data_manager.data_path
            h5_dir = str(h5_path.parent)
            h5_stem = h5_path.stem
            try:
                run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
                run_name_raw = run_data.get('name', f'run{self.run_idx+1}')
                run_part = run_name_raw.split('_')[0] if '_' in run_name_raw else run_name_raw
                output_dir = os.path.join(h5_dir, f"{h5_stem}_{run_part}_drop")
            except Exception:
                output_dir = os.path.join(h5_dir, f"{h5_stem}_run{self.run_idx+1}")

            if os.path.exists(output_dir):
                save_path = os.path.join(output_dir, f"Event_{self.event_idx:03d}.png")
                try:
                    generate_diagnostic_plot(
                        self.time_history, self.events, self.event_idx, r, self.config, save_path
                    )
                    print(f"Overwrote Event Drop diagnostic plot at {save_path}")
                except Exception as e:
                    print(f"Warning: failed to overwrite Event Drop diagnostic plot: {e}")

        # Sync the restored state to shared list so both editors agree
        self._preview_active = False

        if hasattr(self.parent, 'refresh_tree'):
            self.parent.refresh_tree()

        msg = f"Event {self.event_idx} updated and saved to HDF5."
        messagebox.showinfo("Applied & Saved", msg)

    def _delete_event(self):
        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete/exclude Event {self.event_idx} from Event Drop analysis?"
        )
        if not confirm:
            return

        # Add this event to the shared skip list
        se = self.data_manager.get_run_skip_events(self.run_idx)
        if self.event_idx not in se:
            se.append(self.event_idx)
        self.data_manager.save_run_skip_events(self.run_idx, se)
        self.config['skip_events'] = se

        event_drop_data = {
            'skipped': True,
            'trigger_time': np.nan,
            'tau': {
                'value': np.nan,
                'pre_start': self.pts['tau'][0], 'pre_end': self.pts['tau'][1],
                'post_start': self.pts['tau'][2], 'post_end': self.pts['tau'][3]
            },
            'delta': {
                'pre_start': self.pts['slip'][0], 'pre_end': self.pts['slip'][1],
                'post_start': self.pts['slip'][2], 'post_end': self.pts['slip'][3]
            },
            'lvdt': {
                'value': np.nan,
                'pre_start': self.pts['lvdt'][0], 'pre_end': self.pts['lvdt'][1],
                'post_start': self.pts['lvdt'][2], 'post_end': self.pts['lvdt'][3]
            },
            'D_Push': np.nan,
            'D_max': np.nan,
            'D_E3': np.nan,
        }
        
        eddy_keys = sorted([k for k in self.time_history.keys() if 'eddy' in k.lower()])
        for i in range(len(eddy_keys)):
            event_drop_data['delta'][f'E{i+1}_value'] = np.nan

        self.data_manager.fast_save_event_analysis(self.run_idx, self.event_idx, None, event_drop_data)

        # Delete diagnostic plot if it exists
        import os
        if self.data_manager.data_path:
            h5_path = self.data_manager.data_path
            h5_dir = str(h5_path.parent)
            h5_stem = h5_path.stem
            try:
                run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
                run_name_raw = run_data.get('name', f'run{self.run_idx+1}')
                run_part = run_name_raw.split('_')[0] if '_' in run_name_raw else run_name_raw
                output_dir = os.path.join(h5_dir, f"{h5_stem}_{run_part}_drop")
            except Exception:
                output_dir = os.path.join(h5_dir, f"{h5_stem}_run{self.run_idx+1}")

            plot_path = os.path.join(output_dir, f"Event_{self.event_idx:03d}.png")
            if os.path.exists(plot_path):
                try:
                    os.remove(plot_path)
                    print(f"Deleted diagnostic plot at {plot_path}")
                except Exception as e:
                    print(f"Warning: failed to delete diagnostic plot: {e}")

        # Update UI: reload configuration (preview OFF so red overlay shows) and redraw
        self._preview_active = False
        self._load_config()
        self._recompute()
        
        if hasattr(self.parent, 'refresh_tree'):
            self.parent.refresh_tree()


    def on_close(self):
        if self.figure:
            plt.close(self.figure)
        self.destroy()
