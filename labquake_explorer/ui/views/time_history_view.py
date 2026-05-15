import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


class TimeHistoryView(tk.Toplevel):
    def __init__(self, parent, run_idx, path):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"Time History Preview - Run {run_idx}")
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

        # Group preset buttons
        preset_frame = ttk.Frame(ctrl_frame)
        preset_frame.grid(row=1, column=6, columnspan=2, padx=5, pady=5, sticky="w")

        # Define groups: (button_label, keyword_match_function)
        self._preset_groups = {
            'LVDT': lambda f: 'lp_displacement' in f.lower() or 'lvdt' in f.lower(),
            'Eddy': lambda f: 'eddy' in f.lower(),
            'Pressure': lambda f: any(k in f.lower() for k in ['pressure', 'mu', 'tau']),
            'PZT': lambda f: 'pzt' in f.lower(),
        }
        for label, match_fn in self._preset_groups.items():
            ttk.Button(preset_frame, text=label, width=8,
                       command=lambda fn=match_fn: self._select_group(fn)).pack(side=tk.LEFT, padx=2)

        ttk.Button(ctrl_frame, text="Plot", command=self.update_plot).grid(row=2, column=0, columnspan=2, padx=15, pady=5)

        # Populate X combobox
        x_options = ['time'] + [f for f in self.plottable_fields if f != 'time']
        self.x_combo.config(values=x_options)
        self.x_combo.set('time')

        # Populate Y listbox
        for f in self.plottable_fields:
            if f != 'time':
                self.y_listbox.insert(tk.END, f)

        # Select smart defaults
        self._select_defaults()

        # Init combobox values for events
        n_events = len(self.events)
        options = [str(i) for i in range(n_events)]
        self.start_combo.config(values=options)
        self.end_combo.config(values=options)
        self.start_combo.current(0)
        self.end_combo.current(n_events - 1)

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

    def _select_defaults(self):
        """Select sensible default Y fields."""
        defaults = ['LP_displacement', 'mu', 'shear_pressure', 'normal_pressure']
        all_items = [self.y_listbox.get(i) for i in range(self.y_listbox.size())]
        for i, item in enumerate(all_items):
            base = item.split('/')[-1]
            if base in defaults:
                self.y_listbox.selection_set(i)

        # If nothing matched, select the first few eddy channels
        if not self.y_listbox.curselection():
            eddy_indices = [i for i, item in enumerate(all_items) if 'eddy' in item.lower()]
            for idx in eddy_indices[:3]:
                self.y_listbox.selection_set(idx)

    def _select_group(self, match_fn):
        """Clear selection and select all fields matching the given filter, then plot."""
        self.y_listbox.selection_clear(0, tk.END)
        all_items = [self.y_listbox.get(i) for i in range(self.y_listbox.size())]
        for i, item in enumerate(all_items):
            if match_fn(item):
                self.y_listbox.selection_set(i)
        self.update_plot()

    def _add_trigger_lines(self, ax, t_start, t_end):
        """Draw vertical trigger lines within the visible time range."""
        visible = self.trigger_times[(self.trigger_times >= t_start) & (self.trigger_times <= t_end)]
        for tr in visible:
            ax.axvline(x=tr, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    def update_plot(self, event=None):
        # Get selected Y fields
        selected_indices = self.y_listbox.curselection()
        if not selected_indices:
            return

        y_fields = [self.y_listbox.get(i) for i in selected_indices]
        x_field = self.x_combo.get()

        # Get event range
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
        x_data_full = self._get_data_by_path(x_field)
        if x_data_full is None:
            return
        x_data = np.asarray(x_data_full)[mask]

        if len(x_data) == 0:
            return

        n_plots = len(y_fields)

        # Clean up old widgets
        if self.canvas_widget:
            self.canvas_widget.destroy()
        if self.toolbar_frame:
            self.toolbar_frame.destroy()
        if self.figure:
            plt.close(self.figure)

        # Create figure
        share_x = (x_field == 'time')
        self.figure = Figure(figsize=(10, max(3, 2.5 * n_plots)), dpi=100)

        if n_plots == 1:
            axes = [self.figure.add_subplot(1, 1, 1)]
        else:
            axes = self.figure.subplots(n_plots, 1, sharex=share_x, squeeze=False)
            axes = [axes[i, 0] for i in range(n_plots)]

        for i, y_field in enumerate(y_fields):
            ax = axes[i]
            y_data_full = self._get_data_by_path(y_field)
            if y_data_full is None:
                ax.text(0.5, 0.5, f'{y_field} not found',
                        ha='center', va='center', transform=ax.transAxes, color='gray')
                continue

            y_data = np.asarray(y_data_full)[mask]
            ax.plot(x_data, y_data, linewidth=0.8)
            ax.set_ylabel(y_field.split('/')[-1])
            ax.grid(True, linestyle='-', alpha=0.3)

            if x_field == 'time':
                self._add_trigger_lines(ax, t_start, t_end)

        # X label on the last axis
        axes[-1].set_xlabel(x_field)

        self.figure.subplots_adjust(hspace=0.3)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.canvas.draw()

    def on_close(self):
        if self.figure:
            plt.close(self.figure)
        self.destroy()
