import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from labquake_explorer.analysis.event_drop_analyzer import _get_t_trig
from labquake_explorer.utils.user_prefs import UserPrefs


class ColoredLinesView(tk.Toplevel):
    """
    Colored Lines visualisation – mirrors Notebook Figure 8.

    For each sampled time step in the selected event range, all eddy-current
    sensor readings (relative to their baseline at t_start) are plotted as a
    line coloured by time using the viridis_r colourmap.
    X-axis: sensor position along the fault (mm).
    Y-axis: slip / displacement relative to event-range start (μm).
    """

    # Sensor positions in mm (dynamically retrieved based on channel count)
    from labquake_explorer.utils.config import LabquakeExplorerConfig
    DEFAULT_POSITIONS = LabquakeExplorerConfig.EDDY_POSITIONS_5CH_MM

    def __init__(self, parent, run_idx, path):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"Colored Lines - Run {run_idx + 1}")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.run_idx = run_idx
        self.path = path
        self.data_manager = self.parent.data_manager

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

        # Discover eddy keys (sorted)
        self.eddy_keys = sorted(
            [k for k in self.time_history.keys() if 'eddy' in k.lower()]
        )
        if not self.eddy_keys:
            messagebox.showerror("Error", "No eddy-current channels found in time history")
            self.destroy()
            return

        # Assign sensor positions dynamically based on channel count (backward compatible)
        n_keys = len(self.eddy_keys)
        from labquake_explorer.utils.config import LabquakeExplorerConfig
        self.positions = np.array(LabquakeExplorerConfig.get_eddy_positions(n_keys))

        # Figure state
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.canvas_widget = None
        self.toolbar_frame = None
        self.ax = None        # main plot axes
        self.cax = None       # colorbar axes (fixed, never recreated)
        self._colorbar = None

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Control Frame ---
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Event range selectors
        ttk.Label(ctrl_frame, text="Start Event:").grid(row=0, column=0, padx=5, pady=5)
        self.start_combo = ttk.Combobox(ctrl_frame, state="readonly", width=10)
        self.start_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(ctrl_frame, text="End Event:").grid(row=0, column=2, padx=5, pady=5)
        self.end_combo = ttk.Combobox(ctrl_frame, state="readonly", width=10)
        self.end_combo.grid(row=0, column=3, padx=5, pady=5)

        # Sample density (looseness) slider
        ttk.Label(ctrl_frame, text="Samples:").grid(row=0, column=4, padx=(15, 5), pady=5)
        self.sample_var = tk.IntVar(value=2000)
        
        self.sample_slider = ttk.Scale(
            ctrl_frame, from_=50, to=2000,
            orient="horizontal", length=150,
            variable=self.sample_var,
            command=self._on_slider
        )
        self.sample_slider.grid(row=0, column=5, padx=5, pady=5)

        # Entry for direct input
        self.sample_entry = ttk.Entry(ctrl_frame, width=6)
        self.sample_entry.grid(row=0, column=6, padx=5, pady=5)
        self.sample_entry.bind("<Return>", self._on_entry_change)
        self.sample_entry.bind("<FocusOut>", self._on_entry_change)
        # 讀取全域配置 (不再侷限於單一 HDF5)
        saved_config = UserPrefs.get('ColoredLinesView', 'config', {})
        if 'samples' in saved_config:
            self.sample_var.set(saved_config['samples'])
            self.sample_entry.delete(0, tk.END)
            self.sample_entry.insert(0, str(saved_config['samples']))

        # 讀取該實驗專屬的記憶 (HDF5 內部)，只用於記錄事件區間
        local_config = {}
        summary_config = {}
        try:
            run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
            if isinstance(run_data, dict) and 'config' in run_data:
                local_config = run_data['config'].get('colored_lines_config', {})
                summary_config = run_data['config'].get('summary_config', {})
        except Exception:
            pass

        # Populate event dropdowns
        options = []
        for i in range(1, len(self.events)):
            if _get_t_trig(self.events[i]) is not None:
                options.append(str(i))
        
        self.start_combo.config(values=options)
        self.end_combo.config(values=options)
        
        if options:
            # Fallback chain: local colored lines config -> summary config -> default
            try:
                def_start = int(options[0])
                def_end = int(options[-1])
            except (ValueError, IndexError):
                def_start, def_end = 1, len(self.events) - 1

            start_idx = local_config.get('start_idx', summary_config.get('start_idx', def_start))
            end_idx = local_config.get('end_idx', summary_config.get('end_idx', def_end))
            
            # Map event ID to combobox index
            try:
                cb_start_idx = options.index(str(start_idx))
            except ValueError:
                cb_start_idx = 0
            try:
                cb_end_idx = options.index(str(end_idx))
            except ValueError:
                cb_end_idx = len(options) - 1
                
            self.start_combo.current(cb_start_idx)
            self.end_combo.current(cb_end_idx)

        self.start_combo.bind("<<ComboboxSelected>>", self.update_plot)
        self.end_combo.bind("<<ComboboxSelected>>", self.update_plot)

        # Build initial figure
        self._build_figure()
        self.update_plot()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _on_slider(self, value):
        n = int(float(value))
        self.sample_var.set(n)
        self.sample_entry.delete(0, tk.END)
        self.sample_entry.insert(0, str(n))
        self.update_plot()

    def _on_entry_change(self, event=None):
        try:
            val = int(self.sample_entry.get())
            if 2 <= val <= 10000: # Allow larger range for manual input if desired
                self.sample_var.set(val)
                self.update_plot()
            else:
                # Reset to current variable value if out of range
                self.sample_entry.delete(0, tk.END)
                self.sample_entry.insert(0, str(self.sample_var.get()))
        except ValueError:
            self.sample_entry.delete(0, tk.END)
            self.sample_entry.insert(0, str(self.sample_var.get()))

    def _build_figure(self):
        """Create the matplotlib figure with a fixed GridSpec layout.

        Key design: use GridSpec with an explicit colorbar column so that
        matplotlib never 'steals' space from the main axes when we call
        figure.colorbar(cax=self.cax).  This prevents the plot from
        shrinking on every redraw.
        """
        if self.canvas_widget:
            self.canvas_widget.destroy()
        if self.toolbar_frame:
            self.toolbar_frame.destroy()
        if self.figure:
            plt.close(self.figure)

        self.figure = Figure(figsize=(6, 8), dpi=100)

        # GridSpec: main plot + colorbar column
        # width_ratios=[15, 1] matches standard colorbar proportions better
        gs = self.figure.add_gridspec(
            1, 2,
            width_ratios=[15, 1],
            left=0.15, right=0.83,
            top=0.95, bottom=0.10,
            wspace=0.1
        )
        self.ax  = self.figure.add_subplot(gs[0, 0])
        self.cax = self.figure.add_subplot(gs[0, 1])

        self._colorbar = None   # will be created on first update_plot

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

    # ------------------------------------------------------------------
    # Main drawing logic
    # ------------------------------------------------------------------

    def update_plot(self, event=None):
        if self.figure is None:
            return

        try:
            start_idx = int(self.start_combo.get())
            end_idx   = int(self.end_combo.get())
        except ValueError:
            return

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
            self.start_combo.current(start_idx)
            self.end_combo.current(end_idx)

        try:
            t_start = _get_t_trig(self.events[start_idx])
            t_end   = _get_t_trig(self.events[end_idx])
            if t_start is None or t_end is None:
                return
        except IndexError:
            return

        t_all = self.time_history['time']

        t_plot_start = t_start - 1.0
        
        # Extend end time so the last event's slip is not cut off
        next_idx = end_idx + 1
        t_next = None
        if next_idx < len(self.events):
            t_next = _get_t_trig(self.events[next_idx])
            
        if t_next is not None and not np.isnan(t_next):
            t_plot_end = t_next
        else:
            t_plot_end = t_end + 5.0

        mask   = (t_all >= t_plot_start) & (t_all <= t_plot_end)
        t_mask = t_all[mask]

        if len(t_mask) == 0:
            return

        # Stack eddy displacement relative to start of window
        # Shape: (n_channels, n_time_points)
        all_disp = np.vstack([
            self.time_history[k][mask] - self.time_history[k][mask][0]
            for k in self.eddy_keys
        ])

        # Subsample
        n_samples = max(2, int(self.sample_var.get()))
        n_samples = min(n_samples, len(t_mask))
        idxs   = np.linspace(0, len(t_mask) - 1, n_samples, dtype=int)
        t_samp = t_mask[idxs] - t_mask[0]
        d_samp = all_disp[:, idxs]   # shape: (n_channels, n_samples)

        # Colourmap normalised to time
        cmap = cm.viridis_r
        norm = plt.Normalize(t_samp.min(), t_samp.max())

        # --- Redraw main axes only (cax layout stays fixed) ---
        self.ax.clear()

        for i in range(n_samples):
            self.ax.plot(
                self.positions,
                d_samp[:, i],
                color=cmap(norm(t_samp[i])),
                alpha=0.8,
                linewidth=0.8
            )

        # Update colorbar inside its dedicated fixed axes
        self.cax.clear()
        self._colorbar = self.figure.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=self.cax    # reuse fixed axes – no space stolen from self.ax
        )
        self._colorbar.set_label('Time [s]', fontsize=14)
        self._colorbar.ax.tick_params(labelsize=12)

        # Draw VW range in the middle
        try:
            vw_name = self.data_manager.data.get('name', '')
            if isinstance(vw_name, bytes):
                vw_name = vw_name.decode()
            elif hasattr(vw_name, 'item'):
                vw_name = vw_name.item()
            if isinstance(vw_name, bytes):
                vw_name = vw_name.decode()
                
            import re
            match = re.search(r'(\d+)(P|PC)', str(vw_name).upper())
            if match:
                vw_size = float(match.group(1))
                vw_start = 250 - vw_size / 2.0
                vw_end = 250 + vw_size / 2.0
                self.ax.axvspan(vw_start, vw_end, alpha=0.8, color='khaki', label='VW Zone', zorder=0)
        except Exception as e:
            pass

        # Axes labels / formatting
        self.ax.set_xlabel('Distance along fault [mm]', fontsize=14)
        self.ax.set_ylabel('Slip [\u03bcm]', fontsize=14)
        self.ax.grid(True, axis='x', linestyle='-', alpha=0.6) # Standard alpha, vertical only
        self.ax.set_xticks(self.positions)
        self.ax.tick_params(axis='both', which='major', labelsize=12)
        self.ax.margins(x=0)
        max_slip = np.max(d_samp) if d_samp.size > 0 else 1
        self.ax.set_ylim(0, max_slip)

        self.canvas.draw()

    def _save_config(self):
        """Save current UI state to the config in memory and globally."""
        # Global config: samples density
        global_config = {
            'samples': self.sample_var.get(),
        }
        UserPrefs.set('ColoredLinesView', 'config', global_config)
        
        # Local config: Includes event indices
        local_config = dict(global_config)
        try:
            local_config['start_idx'] = int(self.start_combo.get())
            local_config['end_idx'] = int(self.end_combo.get())
        except ValueError:
            pass
            
        try:
            run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
            config = run_data.setdefault('config', {})
            config['colored_lines_config'] = local_config
        except Exception as e:
            print(f"Warning: failed to save Colored Lines config in memory: {e}")

    def on_close(self):
        try:
            self._save_config()
            # Persist summary config to HDF5 file using fast_save logic
            run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
            if 'config' in run_data:
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
            print(f"Warning: failed to save Colored Lines config to HDF5: {e}")
            
        if self.figure:
            plt.close(self.figure)
        self.destroy()
