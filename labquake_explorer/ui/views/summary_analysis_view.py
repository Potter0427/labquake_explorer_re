"""
Summary Analysis View – multi-row time-history plot showing per-event
analysis results (delta_tau, delta_slip, delta_lvdt, D values) alongside
continuous waveforms (slip, mu).
"""
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from labquake_explorer.analysis.event_drop_analyzer import _get_t_trig
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from labquake_explorer.analysis.event_drop_analyzer import moving_average
from labquake_explorer.utils.user_prefs import UserPrefs

# Available subplot definitions: (key, label)
SUMMARY_SUBPLOT_DEFS = [
    ('slip', 'Slip'),
    ('mu', 'Mu'),
    ('delta_tau', r'$\Delta\tau$'),
    ('delta_mu', r'$\Delta\mu$'),
    ('delta_slip', r'$\Delta$ Slip'),
    ('delta_lvdt', r'$\Delta$ LVDT'),
    ('d_values', 'D'),
    ('stiffness', 'k'),
    ('eddy_lvdt', 'Eddy-LVDT'),
    ('slip_rate', 'Slip Rate'),
    ('heatmap', 'Heatmap'),
]


class SummaryAnalysisView(tk.Toplevel):
    def __init__(self, parent, run_idx, path):
        self.parent = parent
        super().__init__(self.parent.root)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.run_idx = run_idx
        self.path = path
        self.data_manager = self.parent.data_manager
        run_data = self.data_manager.get_data(f"runs/[{run_idx}]")
        run_name = run_data.get("name", str(run_idx))
        self.title(f"Summary Analysis - {run_name}")

        # Load data
        self.time_history = self.data_manager.get_data(self.path)
        self.events = self.data_manager.get_data(f"runs/[{self.run_idx}]/events")

        if not self.time_history or 'time' not in self.time_history:
            messagebox.showerror("Error", "Invalid time history data")
            self.destroy()
            return

        if not self.events:
            messagebox.showerror("Error", "No events found for this run")
            self.destroy()
            return

        # Load analysis results from events directly
        self.analysis = None
        self.results = {}
        keys_to_extract = ['delta_tau', 'delta_mu', 'delta_lvdt', 'D_Push', 'D_max', 'D_E3', 'skipped', 'k']
        
        # Determine any dynamic keys (e.g., delta_E1) from the first available event
        for ev in self.events:
            if isinstance(ev, dict) and 'delta' in ev:
                keys_to_extract.extend([k for k in ev['delta'].keys() if k.endswith('_value')])
                break
                
        # Build arrays for each key
        for k in keys_to_extract:
            arr = []
            for ev in self.events:
                if not isinstance(ev, dict):
                    arr.append(np.nan)
                    continue
                if k == 'delta_tau':
                    arr.append(ev.get('tau', {}).get('value', np.nan))
                elif k == 'delta_lvdt':
                    arr.append(ev.get('lvdt', {}).get('value', np.nan))
                elif k == 'k':
                    k_obj = ev.get('k', np.nan)
                    arr.append(k_obj.get('value', np.nan) if isinstance(k_obj, dict) else k_obj)
                elif k.endswith('_value') and k.startswith('E'): # E1_value, E2_value
                    arr.append(ev.get('delta', {}).get(k, np.nan))
                else:
                    arr.append(ev.get(k, np.nan))
            
            # 轉換動態鍵名（E1_value -> delta_E1）以維持後續邏輯一致
            final_key = f'delta_{k.split("_")[0]}' if k.endswith('_value') else k
            self.results[final_key] = np.array(arr)
            
        # Convert skipped to boolean array explicitly
        if 'skipped' in self.results:
            self.results['skipped'] = np.array([bool(x) if not np.isnan(x) else False for x in self.results['skipped']])

        # Trigger times for vertical lines
        self.trigger_times = []
        for ev in self.events:
            t = _get_t_trig(ev)
            self.trigger_times.append(t if t is not None else np.nan)
        self.trigger_times = np.array(self.trigger_times)

        # Figure state
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.canvas_widget = None
        self.toolbar_frame = None

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Control Frame ---
        ctrl = ttk.Frame(self)
        ctrl.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        ttk.Label(ctrl, text="Start Event:").grid(row=0, column=0, padx=5, pady=5)
        self.start_combo = ttk.Combobox(ctrl, state="readonly", width=10)
        self.start_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(ctrl, text="End Event:").grid(row=0, column=2, padx=5, pady=5)
        self.end_combo = ttk.Combobox(ctrl, state="readonly", width=10)
        self.end_combo.grid(row=0, column=3, padx=5, pady=5)

        # Subplot checkboxes
        ttk.Label(ctrl, text="Show:").grid(row=0, column=4, padx=(15, 5), pady=5)
        self.subplot_vars = {}
        col = 5
        
        is_1d = self.time_history.get('is_1d', False)

        for key, label in SUMMARY_SUBPLOT_DEFS:
            if is_1d and key == 'd_values':
                continue
                
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(ctrl, text=label, variable=var, command=self.rebuild_figure)
            cb.grid(row=0, column=col, padx=3, pady=5)
            self.subplot_vars[key] = var
            col += 1

        # Smoothing controls
        ttk.Label(ctrl, text="Mu smooth:").grid(row=1, column=0, padx=5, pady=2)
        self.mu_smooth_var = tk.IntVar(value=500)
        ttk.Entry(ctrl, textvariable=self.mu_smooth_var, width=6).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(ctrl, text="LVDT smooth:").grid(row=1, column=2, padx=5, pady=2)
        self.lvdt_smooth_var = tk.IntVar(value=100)
        ttk.Entry(ctrl, textvariable=self.lvdt_smooth_var, width=6).grid(row=1, column=3, padx=5, pady=2)

        ttk.Label(ctrl, text="Heatmap vmin:").grid(row=1, column=4, padx=5, pady=2)
        self.vmin_var = tk.StringVar(value="1e-1")
        ttk.Entry(ctrl, textvariable=self.vmin_var, width=5).grid(row=1, column=5, padx=5, pady=2)

        ttk.Label(ctrl, text="vmax:").grid(row=1, column=6, padx=5, pady=2)
        self.vmax_var = tk.StringVar(value="1e4")
        ttk.Entry(ctrl, textvariable=self.vmax_var, width=5).grid(row=1, column=7, padx=5, pady=2)

        ttk.Button(ctrl, text="Refresh", command=self.update_plot).grid(row=1, column=8, padx=15, pady=2)

        # 讀取全域配置 (不再侷限於單一 HDF5)
        saved_config = UserPrefs.get('SummaryAnalysisView', 'config', {})

        # Event dropdowns
        n_events = len(self.events)
        options = [str(i) for i in range(1, n_events)]
        self.start_combo.config(values=options)
        self.end_combo.config(values=options)
        
        # 讀取該實驗專屬的記憶 (HDF5 內部)，只用於記錄事件區間
        local_config = {}
        try:
            run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
            if isinstance(run_data, dict) and 'config' in run_data:
                local_config = run_data['config'].get('summary_config', {})
        except Exception:
            pass
            
        # Load start/end selection (從檔案專屬設定讀取)
        start_idx = local_config.get('start_idx', 1) - 1  # 1-based to 0-based index
        end_idx = local_config.get('end_idx', n_events - 1) - 1
        
        # Validate indices
        cb_len = len(options)
        if cb_len > 0:
            start_idx = max(0, min(start_idx, cb_len - 1))
            end_idx = max(0, min(end_idx, cb_len - 1))
            self.start_combo.current(start_idx)
            self.end_combo.current(end_idx)

        # Load checkbox states
        if 'subplots' in saved_config:
            saved_active = saved_config['subplots']
            for key in self.subplot_vars:
                self.subplot_vars[key].set(key in saved_active)
        
        # Load numeric inputs
        if 'mu_smooth' in saved_config: self.mu_smooth_var.set(saved_config['mu_smooth'])
        if 'lvdt_smooth' in saved_config: self.lvdt_smooth_var.set(saved_config['lvdt_smooth'])
        if 'vmin' in saved_config: self.vmin_var.set(saved_config['vmin'])
        if 'vmax' in saved_config: self.vmax_var.set(saved_config['vmax'])

        self.start_combo.bind("<<ComboboxSelected>>", self.update_plot)
        self.end_combo.bind("<<ComboboxSelected>>", self.update_plot)

        self.rebuild_figure()

    def _get_active_subplots(self):
        return [key for key, _ in SUMMARY_SUBPLOT_DEFS if key in self.subplot_vars and self.subplot_vars[key].get()]

    def _get_event_time(self, ev_idx):
        if ev_idx < len(self.events):
            t = _get_t_trig(self.events[ev_idx])
            if t is not None:
                return t
        # Fallback if valid event time is completely missing
        return np.nan

    def rebuild_figure(self):
        active = self._get_active_subplots()
        n = len(active)

        if self.canvas_widget:
            self.canvas_widget.destroy()
        if self.toolbar_frame:
            self.toolbar_frame.destroy()
        if self.figure:
            plt.close(self.figure)

        if n == 0:
            self.figure = None
            self.canvas = None
            return

        # Limit window size to screen height
        screen_h = self.winfo_screenheight()
        self.maxsize(self.winfo_screenwidth(), screen_h - 80)

        self.figure = Figure(figsize=(12, max(4, 2 * n)), dpi=100)

        # Use a 2-column GridSpec: col 0 = main plots, col 1 = colorbar (static width).
        # This prevents colorbar from stealing layout space on every redraw.
        import matplotlib.gridspec as gridspec
        height_ratios = [1.5 if key == 'slip' else 1.0 for key in active]
        # Dynamic bottom margin: fewer subplots → taller per subplot → need more fraction for x-axis label
        bottom_margin = max(0.06, min(0.18, 0.18 / n))
        gs = gridspec.GridSpec(
            n, 2,
            figure=self.figure,
            width_ratios=[20, 1],
            height_ratios=height_ratios,
            left=0.08, right=0.93, top=0.95, bottom=bottom_margin,
            hspace=0.25, wspace=0.05,
        )

        self.axs_map = {}
        self.caxs_map = {}
        for i, key in enumerate(active):
            ax = self.figure.add_subplot(gs[i, 0])
            cax = self.figure.add_subplot(gs[i, 1])
            cax.axis('off')  # hidden by default; heatmap will turn it on
            if i > 0:
                ax.sharex(list(self.axs_map.values())[0])
            self.axs_map[key] = ax
            self.caxs_map[key] = cax

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.update_plot()

    def _add_trigger_lines(self, ax, t_start, t_end, t_offset, add_text=False):
        t_end_ext = t_end + (t_end - t_start) * 0.01
        for i, tr_time in enumerate(self.trigger_times):
            if t_start <= tr_time <= t_end_ext:
                tr = tr_time - t_offset
                ax.axvline(x=tr, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
                if add_text and i > 0 and i % 3 == 0:
                    ax.text(tr, 1.01, str(i), transform=ax.get_xaxis_transform(),
                            fontsize=9, color='dimgray', ha='center', va='bottom')

    def update_plot(self, event=None):
        if self.figure is None:
            return

        try:
            start_idx = int(self.start_combo.get())
            end_idx = int(self.end_combo.get())
        except ValueError:
            return

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        try:
            t_start = self._get_event_time(start_idx)
            t_end = self._get_event_time(end_idx)
        except Exception as e:
            print(f"Error getting event times: {e}")
            return

        t_all = self.time_history['time']
        
        t_plot_start = t_start - 1.0
        if end_idx + 1 < len(self.events):
            t_next = self._get_event_time(end_idx + 1)
            t_plot_end = t_next if not np.isnan(t_next) else t_end + 5.0
        else:
            t_plot_end = t_end + 5.0
            
        if t_plot_start >= t_plot_end:
            t_plot_start = t_start - 1.0
            t_plot_end = t_start + 1.0

        mask = (t_all >= t_plot_start) & (t_all <= t_plot_end)
        t_mask = t_all[mask]
        if len(t_mask) == 0:
            return

        t_offset = t_mask[0]
        t_plot = t_mask - t_offset

        # Trigger times in plot coords
        triggers_plot = self.trigger_times - t_offset

        for key, ax in self.axs_map.items():
            ax.clear()
            ax.grid(True, linestyle='-', alpha=0.3)
            # Reset colorbar axes
            if key in self.caxs_map:
                self.caxs_map[key].clear()
                self.caxs_map[key].axis('off')

        active = self._get_active_subplots()

        def _get_analysis_in_range(key):
            if key not in self.results:
                return None, None
            
            arr = self.results[key]
            trigs = self.trigger_times
            
            # Ignore event 0 (placeholder)
            valid_mask = np.ones(len(trigs), dtype=bool)
            valid_mask[0] = False
            
            # Mask by valid trigger times within the current viewport range
            range_mask = valid_mask & ~np.isnan(trigs) & (trigs >= t_plot_start - 1) & (trigs <= t_plot_end + 1)
            
            # 檢查 skipped
            if 'skipped' in self.results:
                range_mask &= ~self.results['skipped']
            
            return trigs[range_mask] - t_offset, arr[range_mask]

        # --- (1) Slip ---
        if 'slip' in self.axs_map:
            ax = self.axs_map['slip']
            # LVDT
            if 'LP_displacement' in self.time_history:
                try:
                    w_lvdt = max(1, self.lvdt_smooth_var.get())
                except tk.TclError:
                    w_lvdt = 1
                lvdt_raw = self.time_history['LP_displacement'][mask]
                lvdt_sm = moving_average(lvdt_raw, w_lvdt)
                if len(lvdt_sm) < len(t_plot):
                    lvdt_sm = np.pad(lvdt_sm, (0, len(t_plot) - len(lvdt_sm)), 'edge')
                lvdt_0 = lvdt_sm - lvdt_sm[0]
                ax.plot(t_plot, lvdt_0, 'gray', label='LVDT', alpha=0.5)
            # Eddy
            eddy_keys = sorted([k for k in self.time_history.keys() if 'eddy' in k.lower()])
            for i, k in enumerate(eddy_keys):
                e_0 = self.time_history[k][mask] - self.time_history[k][mask][0]
                ax.plot(t_plot, e_0, label=f'E{i+1}', alpha=0.7)
            ax.set_ylabel('slip [μm]')
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small', handletextpad=1.5, borderaxespad=1.0)
            self._add_trigger_lines(ax, t_plot_start, t_plot_end, t_offset, add_text=(active[0]=='slip'))
        # --- (2) Mu ---
        if 'mu' in self.axs_map:
            ax = self.axs_map['mu']
            if 'mu' in self.time_history:
                mu_data = self.time_history['mu'][mask]
                try:
                    w = self.mu_smooth_var.get()
                except tk.TclError:
                    w = 500
                mu_sm = moving_average(mu_data, max(1, w))
                if len(mu_sm) < len(t_plot):
                    mu_sm = np.pad(mu_sm, (0, len(t_plot) - len(mu_sm)), 'edge')
                ax.plot(t_plot, mu_sm, 'k')
            ax.set_ylabel(r'$\mu$')
            self._add_trigger_lines(ax, t_plot_start, t_plot_end, t_offset, add_text=(active[0]=='mu'))

        # --- (3) Delta Tau ---
        if 'delta_tau' in self.axs_map:
            ax = self.axs_map['delta_tau']
            t_r, vals = _get_analysis_in_range('delta_tau')
            if t_r is not None and len(t_r) > 0:
                valid = ~np.isnan(vals)
                ax.plot(t_r[valid], vals[valid], 'o-', markersize=3)
            else:
                ax.text(0.5, 0.5, 'Run "Run Drop Analysis" first',
                        ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_ylabel(r'$\Delta\tau$ [MPa]')
            self._add_trigger_lines(ax, t_plot_start, t_plot_end, t_offset, add_text=(active[0]=='delta_tau'))

        # --- (3b) Delta Mu ---
        if 'delta_mu' in self.axs_map:
            ax = self.axs_map['delta_mu']
            t_r, vals = _get_analysis_in_range('delta_mu')
            if t_r is not None and len(t_r) > 0:
                valid = ~np.isnan(vals)
                ax.plot(t_r[valid], vals[valid], 'o-', markersize=3, color='darkorange')
            else:
                ax.text(0.5, 0.5, 'Run "Run Drop Analysis" first',
                        ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_ylabel(r'$\Delta\mu$')
            self._add_trigger_lines(ax, t_plot_start, t_plot_end, t_offset, add_text=(active[0]=='delta_mu'))

        # --- (4) Delta Slip ---
        if 'delta_slip' in self.axs_map:
            ax = self.axs_map['delta_slip']
            if self.results and isinstance(self.results, dict):
                eddy_keys = sorted([k for k in self.time_history.keys() if 'eddy' in k.lower()])
                for i in range(len(eddy_keys)):
                    label = f'delta_E{i+1}'
                    t_r, vals = _get_analysis_in_range(label)
                    if t_r is not None and len(t_r) > 0:
                        valid = ~np.isnan(vals)
                        ax.plot(t_r[valid], vals[valid], 'o-', alpha=0.7, markersize=3, label=f'E{i+1}')
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small', handletextpad=1.5, borderaxespad=1.0)
            else:
                ax.text(0.5, 0.5, 'Run "Run Drop Analysis" first',
                        ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_ylabel(r'$\delta$ [μm]')
            self._add_trigger_lines(ax, t_plot_start, t_plot_end, t_offset, add_text=(active[0]=='delta_slip'))

        # --- (5) Delta LVDT ---
        if 'delta_lvdt' in self.axs_map:
            ax = self.axs_map['delta_lvdt']
            t_r, vals = _get_analysis_in_range('delta_lvdt')
            if t_r is not None and len(t_r) > 0:
                valid = ~np.isnan(vals)
                ax.plot(t_r[valid], vals[valid], 'o-', color='slategrey', markersize=3)
            else:
                ax.text(0.5, 0.5, 'Run "Run Drop Analysis" first',
                        ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_ylabel(r'$\Delta$ LVDT [μm]')
            self._add_trigger_lines(ax, t_plot_start, t_plot_end, t_offset, add_text=(active[0]=='delta_lvdt'))

        # --- (6) D values ---
        if 'd_values' in self.axs_map:
            ax = self.axs_map['d_values']
            has_data = False
            for key, color, label in [
                ('D_Push', 'teal', r'$D_{Push}$'),
                ('D_max', 'coral', r'$D_{max}$'),
                ('D_E3', None, r'$D_{E3}$'),
            ]:
                t_r, vals = _get_analysis_in_range(key)
                if t_r is not None and len(t_r) > 0:
                    valid = ~np.isnan(vals)
                    kwargs = {'markersize': 3, 'alpha': 0.8, 'label': label}
                    if color:
                        kwargs['color'] = color
                    ax.plot(t_r[valid], vals[valid], 'o-', **kwargs)
                    has_data = True
            if not has_data:
                ax.text(0.5, 0.5, 'Run "Run Drop Analysis" first',
                        ha='center', va='center', transform=ax.transAxes, color='gray')
            else:
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small', handletextpad=1.5, borderaxespad=1.0)
            ax.set_ylabel(r'D [μm]')
            self._add_trigger_lines(ax, t_plot_start, t_plot_end, t_offset, add_text=(active[0]=='d_values'))

        # --- (7) Stiffness k ---
        if 'stiffness' in self.axs_map:
            ax = self.axs_map['stiffness']
            has_data = False
            t_r, vals = _get_analysis_in_range('k')
            if t_r is not None and len(t_r) > 0:
                valid = ~np.isnan(vals)
                if np.any(valid):
                    ax.plot(t_r[valid], vals[valid], 'o-', color='teal', markersize=3)
                    has_data = True

            if not has_data:
                ax.text(0.5, 0.5, 'Run "Run K Analysis" first',
                        ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_ylabel('k [MPa/μm]')
            self._add_trigger_lines(ax, t_plot_start, t_plot_end, t_offset, add_text=(active[0]=='stiffness'))

        # --- (8) Eddy - LVDT ---
        if 'eddy_lvdt' in self.axs_map:
            ax = self.axs_map['eddy_lvdt']
            if 'LP_displacement' in self.time_history:
                try:
                    w_lvdt = max(1, self.lvdt_smooth_var.get())
                except tk.TclError:
                    w_lvdt = 1
                lvdt_raw = self.time_history['LP_displacement'][mask]
                lvdt_sm = moving_average(lvdt_raw, w_lvdt)
                if len(lvdt_sm) < len(t_plot):
                    lvdt_sm = np.pad(lvdt_sm, (0, len(t_plot) - len(lvdt_sm)), 'edge')
                lvdt_0 = lvdt_sm - lvdt_sm[0]

                eddy_keys = sorted([k for k in self.time_history.keys() if 'eddy' in k.lower()])
                for i, k in enumerate(eddy_keys):
                    e_0 = self.time_history[k][mask] - self.time_history[k][mask][0]
                    diff = e_0 - lvdt_0
                    ax.plot(t_plot, diff, f'C{i}', alpha=0.7)
            ax.set_ylabel('Eddy-LVDT [μm]')
            self._add_trigger_lines(ax, t_plot_start, t_plot_end, t_offset, add_text=(active[0]=='eddy_lvdt'))

        # --- (8) Slip Rate ---
        if 'slip_rate' in self.axs_map:
            ax = self.axs_map['slip_rate']
            eddy_keys = sorted([k for k in self.time_history.keys() if k.startswith('eddy_ch')])
            has_data = False
            
            # Sub-group for high-rate data if it exists
            hr_group = self.time_history.get('high_rate_sliprates', {})
            
            for i, k in enumerate(eddy_keys):
                ch_num = k.replace('eddy_ch', '')
                t_rate_key = f't_sliprate_ch{ch_num}'
                rate_key = f'sliprate_ch{ch_num}'
                
                all_t_parts = []
                all_r_parts = []
                
                # (1) Restore Block 1 Background Rate as "base"
                if t_rate_key in self.time_history and rate_key in self.time_history:
                    t_sr = self.time_history[t_rate_key]
                    r_sr = self.time_history[rate_key]
                    sr_mask = (t_sr >= t_plot_start) & (t_sr <= t_plot_end)
                    if np.sum(sr_mask) > 0:
                        all_t_parts.append(t_sr[sr_mask] - t_offset)
                        all_r_parts.append(r_sr[sr_mask])
                
                # (2) Collect High-rate event blocks from new sub-group or root (compatibility)
                blk_idx = 2
                while True:
                    hr_t_name = f't_high_sliprate_ch{ch_num}_blk{blk_idx}'
                    hr_r_name = f'high_sliprate_ch{ch_num}_blk{blk_idx}'
                    
                    # Check in sub-group first, then root
                    t_hr = hr_group.get(hr_t_name) if isinstance(hr_group, dict) else hr_group.get(hr_t_name)
                    r_hr = hr_group.get(hr_r_name) if isinstance(hr_group, dict) else hr_group.get(hr_r_name)
                    
                    if t_hr is None: # Fallback to root for old files
                        t_hr = self.time_history.get(hr_t_name)
                        r_hr = self.time_history.get(hr_r_name)
                        
                    if t_hr is None:
                        break
                        
                    hr_mask = (t_hr >= t_plot_start) & (t_hr <= t_plot_end)
                    if np.sum(hr_mask) > 0:
                        all_t_parts.append(t_hr[hr_mask] - t_offset)
                        all_r_parts.append(r_hr[hr_mask])
                    blk_idx += 1
                
                if all_t_parts:
                    combined_t = np.concatenate(all_t_parts)
                    combined_r = np.concatenate(all_r_parts)
                    idx_sort = np.argsort(combined_t)
                    # Plot with very thin line to keep background and spikes distinguishable
                    ax.plot(combined_t[idx_sort], combined_r[idx_sort], 
                            color=f'C{i}', alpha=0.7)
                    has_data = True
            
            ax.set_yscale('log')
            ax.set_ylabel('Rate [\u03bcm/s]')
            self._add_trigger_lines(ax, t_plot_start, t_plot_end, t_offset, add_text=(active[0]=='slip_rate'))

        # --- (9) Heatmap ---
        if 'heatmap' in self.axs_map:
            ax = self.axs_map['heatmap']
            eddy_keys = sorted([k for k in self.time_history.keys() if k.startswith('eddy_ch')])
            positions = np.array([50, 150, 250, 350, 450])
            
            if len(t_plot) > 1 and len(eddy_keys) == len(positions):
                # Build heatmap matrix by interpolating sparse slip rate data onto common time grid
                # Downsample the time grid to keep rendering fast
                num_steps = min(10000, len(t_plot))
                common_time = np.linspace(t_plot[0], t_plot[-1], num_steps)
                heatmap_mat = np.zeros((len(positions), num_steps))
                
                for i, k in enumerate(eddy_keys):
                    ch_num = k.replace('eddy_ch', '')
                    t_rate_key = f't_sliprate_ch{ch_num}'
                    rate_key = f'sliprate_ch{ch_num}'
                    
                    if t_rate_key in self.time_history and rate_key in self.time_history:
                        all_t_parts = []
                        all_r_parts = []
                        
                        t_sr = self.time_history[t_rate_key]
                        r_sr = self.time_history[rate_key]
                        sr_mask = (t_sr >= t_plot_start) & (t_sr <= t_plot_end)
                        if np.sum(sr_mask) > 0:
                            all_t_parts.append(t_sr[sr_mask] - t_offset)
                            all_r_parts.append(r_sr[sr_mask])
                        
                        hr_group = self.time_history.get('high_rate_sliprates', {})
                        blk_idx = 2
                        while True:
                            hr_t_key = f't_high_sliprate_ch{ch_num}_blk{blk_idx}'
                            hr_r_key = f'high_sliprate_ch{ch_num}_blk{blk_idx}'
                            
                            t_hr = hr_group.get(hr_t_key) if isinstance(hr_group, dict) else hr_group.get(hr_t_key)
                            r_hr = hr_group.get(hr_r_key) if isinstance(hr_group, dict) else hr_group.get(hr_r_key)
                            
                            if t_hr is None:
                                t_hr = self.time_history.get(hr_t_key)
                                r_hr = self.time_history.get(hr_r_key)
                                
                            if t_hr is None:
                                break
                                
                            hr_mask = (t_hr >= t_plot_start) & (t_hr <= t_plot_end)
                            if np.sum(hr_mask) > 0:
                                all_t_parts.append(t_hr[hr_mask] - t_offset)
                                all_r_parts.append(r_hr[hr_mask])
                            blk_idx += 1
                        
                        if all_t_parts:
                            combined_t = np.concatenate(all_t_parts)
                            combined_r = np.concatenate(all_r_parts)
                            idx_sort = np.argsort(combined_t)
                            heatmap_mat[i, :] = np.interp(common_time, combined_t[idx_sort], combined_r[idx_sort], left=0, right=0)
                
                import matplotlib.colors as mcolors
                try:
                    vmin = float(self.vmin_var.get())
                    vmax = float(self.vmax_var.get())
                except Exception:
                    vmin, vmax = 0.1, 10000.0
                    
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                c = ax.pcolormesh(common_time, positions, heatmap_mat, norm=norm, cmap='cividis', shading='nearest')
                
                cax = self.caxs_map['heatmap']
                cax.axis('on')
                ax.figure.colorbar(c, cax=cax, label='Rate [\u03bcm/s]')
                ax.set_yticks(positions)
                ax.set_ylabel('Distance along fault [mm]')
            
            self._add_trigger_lines(ax, t_plot_start, t_plot_end, t_offset, add_text=(active[0]=='heatmap'))

        # X label on bottom
        if active:
            for i, key in enumerate(active):
                if i < len(active) - 1:
                    self.axs_map[key].tick_params(labelbottom=False)
                else:
                    self.axs_map[key].tick_params(labelbottom=True)
                    self.axs_map[key].set_xlabel('Time [s]')
            self.axs_map[active[-1]].set_xlim([t_plot[0], t_plot[-1]])

        # Do NOT call tight_layout here – it shrinks subplots cumulatively
        # on every refresh. Layout is fixed statically in rebuild_figure via GridSpec.
        self.canvas.draw()

    def _save_summary_config(self):
        """Save current UI state to the analysis config in memory and globally."""
        # Global config: UI settings only (no event indices)
        global_config = {
            'subplots': self._get_active_subplots(),
            'mu_smooth': self.mu_smooth_var.get(),
            'lvdt_smooth': self.lvdt_smooth_var.get(),
            'vmin': self.vmin_var.get(),
            'vmax': self.vmax_var.get(),
        }
        
        # Local config: Includes event indices
        local_config = dict(global_config)
        try:
            local_config['start_idx'] = int(self.start_combo.get())
            local_config['end_idx'] = int(self.end_combo.get())
        except ValueError:
            local_config['start_idx'] = 1
            local_config['end_idx'] = len(self.events) - 1
        
        # Save globally using UserPrefs
        UserPrefs.set('SummaryAnalysisView', 'config', global_config)
        
        # Update analysis config in DataManager (for file persistence, local)
        try:
            run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
            config = run_data.setdefault('config', {})
            config['summary_config'] = local_config
        except Exception as e:
            print(f"Warning: failed to save summary config in memory: {e}")

    def on_close(self):
        try:
            self._save_summary_config()
            # Persist summary config to HDF5 file using fast_save
            run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
            if 'config' in run_data:
                # Custom fast save for config group
                import h5py
                data_path = self.data_manager.data_path
                if data_path and data_path.exists() and data_path.suffix.lower() in ['.h5', '.hdf5']:
                    with h5py.File(data_path, 'r+') as f:
                        config_path = f"runs/{self.run_idx}/config"
                        if config_path in f:
                            del f[config_path]
                        config_group = f.create_group(config_path)
                        def recursive_save(group, d):
                            for k, v in d.items():
                                if isinstance(v, dict):
                                    sub = group.create_group(k)
                                    recursive_save(sub, v)
                                elif isinstance(v, (list, tuple)):
                                    arr = np.array(v)
                                    if arr.dtype.kind == 'U':
                                        arr = np.array([x.encode() for x in arr.flat]).reshape(arr.shape)
                                    group.create_dataset(k, data=arr)
                                elif isinstance(v, str):
                                    group.create_dataset(k, data=v.encode())
                                else:
                                    group.create_dataset(k, data=v)
                        recursive_save(config_group, run_data['config'])
        except Exception as e:
            print(f"Warning: failed to save summary config to HDF5: {e}")
            
        if self.figure:
            plt.close(self.figure)
        self.destroy()
