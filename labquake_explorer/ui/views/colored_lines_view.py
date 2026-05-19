import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


class ColoredLinesView(tk.Toplevel):
    """
    Colored Lines visualisation – mirrors Notebook Figure 8.

    For each sampled time step in the selected event range, all eddy-current
    sensor readings (relative to their baseline at t_start) are plotted as a
    line coloured by time using the viridis_r colourmap.
    X-axis: sensor position along the fault (mm).
    Y-axis: slip / displacement relative to event-range start (μm).
    """

    # Sensor positions in mm (same order as sorted eddy_* keys)
    DEFAULT_POSITIONS = [50, 150, 250, 350, 450]

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

        # Assign sensor positions (use defaults, trimmed/padded to match key count)
        n_keys = len(self.eddy_keys)
        if n_keys <= len(self.DEFAULT_POSITIONS):
            self.positions = np.array(self.DEFAULT_POSITIONS[:n_keys])
        else:
            step = self.DEFAULT_POSITIONS[1] - self.DEFAULT_POSITIONS[0]
            start = self.DEFAULT_POSITIONS[0]
            self.positions = np.array([start + i * step for i in range(n_keys)])

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
        self.sample_var = tk.IntVar(value=500)
        
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
        self.sample_entry.insert(0, "500")
        self.sample_entry.bind("<Return>", self._on_entry_change)
        self.sample_entry.bind("<FocusOut>", self._on_entry_change)

        # Populate event dropdowns
        n_events = len(self.events)
        options = [str(i + 1) for i in range(n_events)]
        self.start_combo.config(values=options)
        self.end_combo.config(values=options)
        self.start_combo.current(0)
        self.end_combo.current(n_events - 1)

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
            left=0.12, right=0.90,
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
            t_start = self.events[start_idx]['event_time']
            t_end   = self.events[end_idx]['event_time']
        except (IndexError, KeyError):
            return

        t_all = self.time_history['time']

        if t_start == t_end:
            t_start -= 1.0
            t_end   += 1.0

        mask   = (t_all >= t_start) & (t_all <= t_end)
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
        t_samp = t_mask[idxs]
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
            cax=self.cax,   # reuse fixed axes – no space stolen from self.ax
            label='Time [s]'
        )

        # Axes labels / formatting
        self.ax.set_xlabel('Distance [mm]')
        self.ax.set_ylabel('Slip [\u03bcm]')
        self.ax.grid(True, linestyle='-', alpha=0.6) # Standard alpha
        self.ax.set_xticks(self.positions)

        self.canvas.draw()

    def on_close(self):
        if self.figure:
            plt.close(self.figure)
        self.destroy()
