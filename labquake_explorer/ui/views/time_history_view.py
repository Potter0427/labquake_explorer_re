import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from labquake_explorer.analysis.event_drop_analyzer import moving_average
from labquake_explorer.utils.user_prefs import UserPrefs

# Defined groups
GROUPS = {
    '[Group] Eddy Current Sensors': 'eddy',
    '[Group] Pressure & Friction': 'pressure',
    '[Group] PZT Sensors': 'pzt'
}

class TimeHistoryView(tk.Toplevel):
    def __init__(self, parent, run_idx, path):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"Time History Preview - Run {run_idx + 1}")
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

        # Collect trigger times
        self.trigger_times = []
        for ev in self.events:
            if isinstance(ev, dict) and 'event_time' in ev:
                self.trigger_times.append(ev['event_time'])
        self.trigger_times = np.array(self.trigger_times)

        # Build list of plottable fields (same length as 'time')
        self.time_length = len(self.time_history['time'])
        self.plottable_fields = self._find_plottable_fields(self.time_history)

        # Twin axis reference
        self.pressure_twin_ax = None

        # Configure layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Control Frame ---
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Event range
        ttk.Label(ctrl_frame, text="Start Event:").grid(row=0, column=0, padx=5, pady=5)
        self.start_combo = ttk.Combobox(ctrl_frame, state="readonly", width=10)
        self.start_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(ctrl_frame, text="End Event:").grid(row=0, column=2, padx=5, pady=5)
        self.end_combo = ttk.Combobox(ctrl_frame, state="readonly", width=10)
        self.end_combo.grid(row=0, column=3, padx=5, pady=5)

        # X axis selector
        ttk.Label(ctrl_frame, text="X Axis:").grid(row=0, column=4, padx=(15, 5), pady=5)
        self.x_combo = ttk.Combobox(ctrl_frame, state="readonly", width=20)
        self.x_combo.grid(row=0, column=5, padx=5, pady=5)

        # Y axis selector (multi-select via Listbox)
        ttk.Label(ctrl_frame, text="Y Axis:").grid(row=1, column=0, padx=5, pady=5, sticky="nw")

        y_frame = ttk.Frame(ctrl_frame)
        y_frame.grid(row=1, column=1, columnspan=5, padx=5, pady=5, sticky="ew")

        self.y_listbox = tk.Listbox(y_frame, selectmode=tk.EXTENDED, height=6, exportselection=False)
        y_scroll = ttk.Scrollbar(y_frame, orient=tk.VERTICAL, command=self.y_listbox.yview)
        self.y_listbox.config(yscrollcommand=y_scroll.set)
        self.y_listbox.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        y_frame.grid_columnconfigure(0, weight=1)

        ttk.Button(ctrl_frame, text="Plot", command=self.update_plot).grid(row=1, column=6, padx=15, pady=5)

        # Populate X combobox
        x_options = ['time'] + [f for f in self.plottable_fields if f != 'time']
        self.x_combo.config(values=x_options)

        # Populate Y listbox
        y_items_all = []
        for group_name in GROUPS.keys():
            self.y_listbox.insert(tk.END, group_name)
            y_items_all.append(group_name)
            
        for f in self.plottable_fields:
            if f != 'time':
                self.y_listbox.insert(tk.END, f)
                y_items_all.append(f)

        # Init combobox values for events
        n_events = len(self.events)
        options = [str(i) for i in range(n_events)]
        self.start_combo.config(values=options)
        self.end_combo.config(values=options)

        # Load global user preferences
        saved_config = UserPrefs.get('TimeHistoryView', 'config', {})

        # Load X selection
        saved_x = saved_config.get('x_field', 'time')
        if saved_x in x_options:
            self.x_combo.set(saved_x)
        else:
            self.x_combo.set('time')

        # Load Y selection
        saved_y = saved_config.get('y_fields', None)
        if saved_y is not None:
            for y_val in saved_y:
                if y_val in y_items_all:
                    idx = y_items_all.index(y_val)
                    self.y_listbox.selection_set(idx)
        else:
            # Select smart defaults (the 3 groups)
            if len(y_items_all) >= 3:
                self.y_listbox.selection_set(0)
                self.y_listbox.selection_set(1)
                self.y_listbox.selection_set(2)

        # Start/end default to the full range
        start_idx = 0
        end_idx = n_events - 1
        
        self.start_combo.current(start_idx)
        self.end_combo.current(end_idx)

        self.start_combo.bind("<<ComboboxSelected>>", self.update_plot)
        self.end_combo.bind("<<ComboboxSelected>>", self.update_plot)

        # Figure placeholder
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.canvas_widget = None
        self.toolbar_frame = None

        # Initial plot
        self.update_plot()

    def _find_plottable_fields(self, data, prefix=""):
        """Recursively find all arrays with the same length as 'time'."""
        fields = []
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}/{key}" if prefix else key
                if isinstance(value, (np.ndarray, list)):
                    if len(value) == self.time_length:
                        fields.append(full_key)
                elif isinstance(value, dict):
                    # Skip known large sub-groups
                    if key.lower() in ['high_rate_sliprates']:
                        continue
                    fields.extend(self._find_plottable_fields(value, full_key))
        return sorted(fields)

    def _get_data_by_path(self, path):
        """Access nested data using slash-separated path."""
        parts = path.split('/')
        current = self.time_history
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _add_trigger_lines(self, ax, t_start, t_end):
        """Draw vertical trigger lines within the visible time range."""
        visible = self.trigger_times[(self.trigger_times >= t_start) & (self.trigger_times <= t_end)]
        for tr in visible:
            ax.axvline(x=tr, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    def update_plot(self, event=None):
        selected_indices = self.y_listbox.curselection()
        if not selected_indices:
            return

        y_items = [self.y_listbox.get(i) for i in selected_indices]
        x_field = self.x_combo.get()

        try:
            start_idx = int(self.start_combo.get())
            end_idx = int(self.end_combo.get())
        except ValueError:
            return

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
            self.start_combo.current(start_idx)
            self.end_combo.current(end_idx)

        try:
            t_start = self.events[start_idx]['event_time']
            t_end = self.events[end_idx]['event_time']
        except (IndexError, KeyError):
            return

        t_all = self.time_history['time']
        if t_start == t_end:
            t_start -= 1.0
            t_end += 1.0

        mask = (t_all >= t_start) & (t_all <= t_end)
        
        # Get X data
        if x_field == 'time':
            x_data = t_all[mask]
        else:
            x_data_full = self._get_data_by_path(x_field)
            if x_data_full is None:
                return
            x_data = np.asarray(x_data_full)[mask]

        if len(x_data) == 0:
            return

        n_plots = len(y_items)

        # Clean up old widgets
        if self.canvas_widget:
            self.canvas_widget.destroy()
        if self.toolbar_frame:
            self.toolbar_frame.destroy()
        if self.figure:
            plt.close(self.figure)

        if self.pressure_twin_ax is not None:
            self.pressure_twin_ax = None

        # Create figure
        share_x = (x_field == 'time')
        self.figure = Figure(figsize=(10, max(3, 2.5 * n_plots)), dpi=100)

        if n_plots == 1:
            axes = [self.figure.add_subplot(1, 1, 1)]
        else:
            axes_array = self.figure.subplots(n_plots, 1, sharex=share_x, squeeze=False)
            axes = [axes_array[i, 0] for i in range(n_plots)]

        for i, y_item in enumerate(y_items):
            ax = axes[i]
            
            if y_item in GROUPS:
                group_key = GROUPS[y_item]
                if group_key == 'eddy':
                    eddy_keys = sorted([k for k in self.time_history.keys() if 'eddy' in k.lower()])
                    for i, k in enumerate(eddy_keys):
                        y_val = moving_average(self.time_history[k][mask], 50)
                        # Pad edges if moving_average shortened the array
                        if len(y_val) < len(x_data):
                            y_val = np.pad(y_val, (0, len(x_data) - len(y_val)), 'edge')
                        y_val = y_val - y_val[0]
                        ax.plot(x_data, y_val, label=f'E{i+1}')
                    if eddy_keys:
                        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small', handletextpad=1.5, borderaxespad=1.0)
                        ax.set_ylabel('slip [μm]')
                        ax.set_title('Eddy Current Sensors')
                elif group_key == 'pressure':
                    w_p = 50
                    if 'mu' in self.time_history:
                        y_val = moving_average(self.time_history['mu'][mask], w_p)
                        if len(y_val) < len(x_data): y_val = np.pad(y_val, (0, len(x_data) - len(y_val)), 'edge')
                        ax.plot(x_data, y_val, 'r-', label='Mu')
                    ax.set_ylabel('Friction (Mu)', color='r')
                    ax.set_title(f'Pressure & Friction [Smoothed w={w_p}]')

                    if 'normal_pressure' in self.time_history:
                        self.pressure_twin_ax = ax.twinx()
                        y_val_norm = moving_average(self.time_history['normal_pressure'][mask], w_p)
                        if len(y_val_norm) < len(x_data): y_val_norm = np.pad(y_val_norm, (0, len(x_data) - len(y_val_norm)), 'edge')
                        self.pressure_twin_ax.plot(x_data, y_val_norm, 'b--', alpha=0.5, label='Normal')
                        if 'shear_pressure' in self.time_history:
                            y_val_shear = moving_average(self.time_history['shear_pressure'][mask], w_p)
                            if len(y_val_shear) < len(x_data): y_val_shear = np.pad(y_val_shear, (0, len(x_data) - len(y_val_shear)), 'edge')
                            self.pressure_twin_ax.plot(x_data, y_val_shear, 'g--', alpha=0.5, label='Shear')
                        self.pressure_twin_ax.set_ylabel('MPa')
                        self.pressure_twin_ax.legend(loc='upper right', fontsize='small')
                        self.pressure_twin_ax.grid(False)
                elif group_key == 'pzt':
                    pzt_keys = sorted([k for k in self.time_history.keys() if 'pzt' in k])
                    for k in pzt_keys:
                        y_val = moving_average(self.time_history[k][mask], 50)
                        if len(y_val) < len(x_data): y_val = np.pad(y_val, (0, len(x_data) - len(y_val)), 'edge')
                        ax.plot(x_data, y_val, label=k, alpha=0.7)
                    if pzt_keys:
                        ax.legend(loc='upper right', fontsize='small')
                        ax.set_ylabel('PZT')
                        ax.set_title('PZT Sensors [Smoothed w=50]')
            else:
                y_data_full = self._get_data_by_path(y_item)
                if y_data_full is None:
                    ax.text(0.5, 0.5, f'{y_item} not found',
                            ha='center', va='center', transform=ax.transAxes, color='gray')
                else:
                    w_i = 50
                    y_data = np.asarray(y_data_full)[mask]
                    y_val = moving_average(y_data, w_i)
                    if len(y_val) < len(x_data): y_val = np.pad(y_val, (0, len(x_data) - len(y_val)), 'edge')
                    ax.plot(x_data, y_val, linewidth=0.8)
                    ax.set_ylabel(y_item.split('/')[-1])
                    ax.set_title(f"{y_item.split('/')[-1]} [Smoothed w={w_i}]")

            ax.grid(True, linestyle='-', alpha=0.3)

            if x_field == 'time':
                self._add_trigger_lines(ax, t_start, t_end)

        axes[-1].set_xlabel(x_field if x_field != 'time' else 'Time (s)')

        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.canvas.draw()

    def _save_config(self):
        try:
            selected_indices = self.y_listbox.curselection()
            y_fields = [self.y_listbox.get(i) for i in selected_indices]
            config = {
                'x_field': self.x_combo.get(),
                'y_fields': y_fields
            }
            UserPrefs.set('TimeHistoryView', 'config', config)
        except Exception as e:
            print(f"Warning: failed to save TimeHistoryView config: {e}")

    def on_close(self):
        self._save_config()
        if self.figure:
            plt.close(self.figure)
        self.destroy()
