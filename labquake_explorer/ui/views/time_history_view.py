import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

# Available subplot definitions: (key, label)
SUBPLOT_DEFS = [
    ('displacement', 'Load Point Displacement'),
    ('eddy', 'Eddy Current Sensors'),
    ('pressure', 'Pressure & Friction'),
    ('pzt', 'PZT Sensors'),
]

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

        # Collect all trigger times for drawing trigger lines
        self.trigger_times = []
        for ev in self.events:
            if isinstance(ev, dict) and 'event_time' in ev:
                self.trigger_times.append(ev['event_time'])
        self.trigger_times = np.array(self.trigger_times)

        # Twin axis reference (to avoid ghosting on redraw)
        self.pressure_twin_ax = None

        # Configure layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Control Frame ---
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Comboboxes for start/end events
        ttk.Label(ctrl_frame, text="Start Event:").grid(row=0, column=0, padx=5, pady=5)
        self.start_combo = ttk.Combobox(ctrl_frame, state="readonly", width=10)
        self.start_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(ctrl_frame, text="End Event:").grid(row=0, column=2, padx=5, pady=5)
        self.end_combo = ttk.Combobox(ctrl_frame, state="readonly", width=10)
        self.end_combo.grid(row=0, column=3, padx=5, pady=5)

        # Subplot visibility selector
        ttk.Label(ctrl_frame, text="Show:").grid(row=0, column=4, padx=(15, 5), pady=5)
        self.subplot_vars = {}
        col = 5
        for key, label in SUBPLOT_DEFS:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(ctrl_frame, text=label, variable=var, command=self.rebuild_figure)
            cb.grid(row=0, column=col, padx=3, pady=5)
            self.subplot_vars[key] = var
            col += 1

        # Init combobox values
        n_events = len(self.events)
        options = [str(i) for i in range(n_events)]
        self.start_combo.config(values=options)
        self.end_combo.config(values=options)
        
        # Set defaults
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

        # Build figure and initial plot
        self.rebuild_figure()

    def _get_active_subplots(self):
        """Return list of active subplot keys in order."""
        return [key for key, _ in SUBPLOT_DEFS if self.subplot_vars[key].get()]

    def rebuild_figure(self):
        """Recreate the figure with only the selected subplots."""
        active = self._get_active_subplots()
        n = len(active)

        # Clean up old widgets
        if self.canvas_widget:
            self.canvas_widget.destroy()
        if self.toolbar_frame:
            self.toolbar_frame.destroy()
        if self.figure:
            plt.close(self.figure)

        self.pressure_twin_ax = None

        if n == 0:
            # Nothing selected, just clear
            self.figure = None
            self.canvas = None
            return

        self.figure = Figure(figsize=(10, max(3, 2.5 * n)), dpi=100)
        self.axs_map = {}
        axes_list = self.figure.subplots(n, 1, sharex=True, squeeze=False)
        for i, key in enumerate(active):
            self.axs_map[key] = axes_list[i, 0]

        self.figure.subplots_adjust(hspace=0.3)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.update_plot()

    def _add_trigger_lines(self, ax, t_start, t_end):
        """Draw vertical trigger lines within the visible time range."""
        visible = self.trigger_times[(self.trigger_times >= t_start) & (self.trigger_times <= t_end)]
        for tr in visible:
            ax.axvline(x=tr, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

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
            self.start_combo.current(start_idx)
            self.end_combo.current(end_idx)

        # Get trigger times from events
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
        t_mask = t_all[mask]

        if len(t_mask) == 0:
            return

        # Clear all axes (including removing old twin axis to prevent ghosting)
        if self.pressure_twin_ax is not None:
            self.pressure_twin_ax.remove()
            self.pressure_twin_ax = None

        for ax in self.axs_map.values():
            ax.clear()
            ax.grid(True, linestyle='-', alpha=0.3)

        active = self._get_active_subplots()

        # 1. Displacement
        if 'displacement' in self.axs_map and 'LP_displacement' in self.time_history:
            ax = self.axs_map['displacement']
            ax.plot(t_mask, self.time_history['LP_displacement'][mask], 'k-')
            ax.set_ylabel('μm')
            ax.set_title('Load Point Displacement (μm)')
            self._add_trigger_lines(ax, t_start, t_end)

        # 2. Eddy Current
        if 'eddy' in self.axs_map:
            ax = self.axs_map['eddy']
            eddy_keys = sorted([k for k in self.time_history.keys() if 'eddy' in k])
            for k in eddy_keys:
                ax.plot(t_mask, self.time_history[k][mask], label=k)
            if eddy_keys:
                ax.legend(loc='upper right', fontsize='small')
                ax.set_ylabel('μm')
                ax.set_title('Eddy Current Sensors (μm)')
            self._add_trigger_lines(ax, t_start, t_end)

        # 3. Pressure & Friction
        if 'pressure' in self.axs_map:
            ax = self.axs_map['pressure']
            if 'mu' in self.time_history:
                ax.plot(t_mask, self.time_history['mu'][mask], 'r-', label='Mu')
            ax.set_ylabel('Friction (Mu)', color='r')
            ax.set_title('Pressure & Friction')

            if 'normal_pressure' in self.time_history:
                self.pressure_twin_ax = ax.twinx()
                self.pressure_twin_ax.plot(t_mask, self.time_history['normal_pressure'][mask], 'b--', alpha=0.5, label='Normal')
                if 'shear_pressure' in self.time_history:
                    self.pressure_twin_ax.plot(t_mask, self.time_history['shear_pressure'][mask], 'g--', alpha=0.5, label='Shear')
                self.pressure_twin_ax.set_ylabel('MPa')
                self.pressure_twin_ax.legend(loc='upper right', fontsize='small')
                self.pressure_twin_ax.grid(False)

            self._add_trigger_lines(ax, t_start, t_end)

        # 4. PZT
        if 'pzt' in self.axs_map:
            ax = self.axs_map['pzt']
            pzt_keys = sorted([k for k in self.time_history.keys() if 'pzt' in k])
            for k in pzt_keys:
                ax.plot(t_mask, self.time_history[k][mask], label=k, alpha=0.7)
            if pzt_keys:
                ax.legend(loc='upper right', fontsize='small')
                ax.set_ylabel('PZT')
                ax.set_title('PZT Sensors')
            self._add_trigger_lines(ax, t_start, t_end)

        # Set x label on the bottom-most visible subplot
        if active:
            last_key = active[-1]
            self.axs_map[last_key].set_xlabel('Time (s)')
        
        self.figure.tight_layout()
        self.canvas.draw()

    def on_close(self):
        if self.figure:
            plt.close(self.figure)
        self.destroy()
