"""Main UI class for Labquake Explorer"""
import sys
import tkinter as tk
import numpy as np
import os
from tkinter import ttk, filedialog, simpledialog, messagebox
from pathlib import Path
from typing import Optional, List, Dict, Any

from labquake_explorer.data.data_manager import DataManager
from labquake_explorer.utils.config import LabquakeExplorerConfig
from labquake_explorer.utils.user_prefs import UserPrefs
from labquake_explorer.ui.views import (
    SimplePlotView, PointsSelectorView, IndexPickerView,
    SlopeAnalyzerView, DynamicStrainArrivalPickerView, CZMFitterView,
    EventAnalyzerView, TimeHistoryView, ColoredLinesView,
    SummaryAnalysisView, EventDropEditorView, EventKEditorView
)

class LabquakeExplorer:
    def __init__(self, root: tk.Tk):
        self.config = LabquakeExplorerConfig()
        self.root = root
        self.root.title(self.config.WINDOW_TITLE)
        
        self.data_manager = DataManager()
        self.child_windows: List[tk.Toplevel] = []
        self.data_tree: Optional[ttk.Treeview] = None
        self.active_context_menu: Optional[tk.Menu] = None
        self.current_file_path: Optional[Path] = None
        
        self.setup_window()
        self.create_widgets()
        self.setup_bindings()

        # debug
        # file_path = Path("/Users/hueyke/Library/CloudStorage/SynologyDrive-KeResearch-data/PSU/Gc-dataset/p5993ec.npz")
        # file_path = Path("smb://KeResearchNAS._smb._tcp.local/data/PSU/Gc-dataset/p5993ec.npz")
        # self.data_manager.load_file(file_path)
        # self.save_button.configure(state="normal")
        # self.refresh_tree()
        # print(f"File loaded: {file_path}")

    def setup_window(self) -> None:
        screen_height = self.root.winfo_screenheight()
        window_height = screen_height - (self.config.WINDOW_GAP * 3)

        self.set_window_icon(self.root)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.geometry(
            f"{self.config.WINDOW_WIDTH}x{window_height}+"
            f"{self.config.WINDOW_GAP}+{self.config.WINDOW_GAP}"
        )
        self.root.lift()
        self.root.focus_force()

    def create_widgets(self) -> None:
        self.create_context_menus()
        self.create_buttons()
        self.init_data_tree()

    def create_context_menus(self) -> None:
        self.run_menu = tk.Menu(self.root, tearoff=0)
        self.run_menu.add_command(label="Pick Events", command=self.pick_events)

        self.time_history_menu = tk.Menu(self.root, tearoff=0)
        self.time_history_menu.add_command(label="Preview Time History", command=self.preview_time_history)
        self.time_history_menu.add_command(label="Colored Lines", command=self.preview_colored_lines)
        self.time_history_menu.add_command(label="Run Drop Analysis", command=self.run_drop_analysis)
        self.time_history_menu.add_command(label="Run K Analysis", command=self.run_k_analysis)
        self.time_history_menu.add_command(label="Summary Analysis", command=self.preview_summary_analysis)

        self.event_menu = tk.Menu(self.root, tearoff=0)
        self.event_menu.add_command(label="Analyze Event", command=self.analyze_event)
        self.event_menu.add_command(label="Edit Event Drop", command=self.edit_event_drop)
        self.event_menu.add_command(label="Edit Event K", command=self.edit_event_k)
        self.event_menu.add_command(label="Pick Arrivals", command=self.pick_strain_array_arrivals)
        self.event_menu.add_command(label="Fit Cohesive Zone Model", command=self.fit_cohesive_zone_model)

        self.array_menu = tk.Menu(self.root, tearoff=0)
        self.array_menu.add_command(label="Pick Indices", command=self.pick_indices)
        self.array_menu.add_command(label="Extract Slopes", command=self.extract_slope)
        self.array_menu.add_command(label="Extract Run", command=self.pick_run)

        self.event_indices_menu = tk.Menu(self.root, tearoff=0)
        self.event_indices_menu.add_command(label="Extract Events", command=self.extract_events)

        self.event_array_menu = tk.Menu(self.root, tearoff=0)
        self.event_array_menu.add_command(label="Pick Indices", command=self.pick_indices)
        self.event_array_menu.add_command(label="Extract Slopes", command=self.extract_slope)
        self.event_array_menu.add_command(label="Min/Max", command=self.min_max)

        self.string_menu = tk.Menu(self.root, tearoff=0)
        self.string_menu.add_command(label="Edit String", command=self.edit_string)

    def create_buttons(self) -> None:
        buttons = [
            ("Load", self.load_file, "normal", 0),
            ("Refresh", self.refresh_tree, "normal", 1),
            ("Save As", self.save_file, "disabled", 3)
        ]
        
        for text, command, state, col in buttons:
            btn = tk.Button(self.root, text=text, command=command, state=state)
            btn.grid(row=1, column=col, padx=2, pady=2, sticky="w" if col < 2 else "e")
            if text == "Save As":
                self.save_button = btn

    def init_data_tree(self) -> None:
        if self.data_tree:
            self.data_tree.destroy()
            
        self.data_tree = ttk.Treeview(self.root)
        self.data_tree.grid(row=0, column=0, columnspan=4, padx=2, pady=2, sticky="nsew")
        header_text = self.current_file_path.name if self.current_file_path else "[Data File]"
        self.data_tree.heading("#0", text=header_text, anchor="w")
        
        self.data_tree.bind("<Double-1>", self.on_double_click)
        self.data_tree.bind("<Button-1>", self.on_left_click)
        self.data_tree.bind("<Button-2>", self.on_right_click)
        self.data_tree.bind("<Button-3>", self.on_right_click)
        self.data_tree.bind("<<TreeviewOpen>>", self._on_tree_expand)
        
        # Map of iid -> data for deferred (lazy) loading
        self._deferred_nodes = {}

    def _on_tree_expand(self, event):
        """Lazy-load children for collapsed groups when the user expands them."""
        iid = self.data_tree.focus()
        if iid in self._deferred_nodes:
            # Remove the dummy "(loading...)" child
            for child in self.data_tree.get_children(iid):
                self.data_tree.delete(child)
            # Build the real children
            self.build_tree(self._deferred_nodes.pop(iid), iid)

    def load_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=self.config.FILE_TYPES
        )
        if not file_path:
            return

        try:
            path = Path(file_path)
            self.data_manager.load_file(path)
            self.current_file_path = path
            self.save_button.configure(state="normal")
            self.refresh_tree()
            print(f"File loaded: {file_path}")

            # Expand the runs node
            for item in self.data_tree.get_children(""):
                if self.data_tree.item(item)["text"].startswith("runs"):
                    self.data_tree.item(item, open=True)
                    break
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def save_file(self) -> None:
        initial_file = self.current_file_path if self.current_file_path else None
        initial_dir = self.current_file_path.parent if self.current_file_path else None

        file_path = filedialog.asksaveasfilename(
                title="Save data file",
                initialfile=initial_file.name if initial_file else None,
                initialdir=str(initial_dir) if initial_dir else None,
                filetypes=self.config.FILE_TYPES,
                defaultextension=".h5"
            )
        if not file_path:
            return

        try:
            self.data_manager.save_file(Path(file_path))
            print(f"File saved: {file_path}")
            messagebox.showinfo("Success", "File saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

    def _get_open_paths(self, parent="", prefix=""):
        """Collect the text-path of every expanded node so we can restore state."""
        opened = set()
        for child in self.data_tree.get_children(parent):
            label = self.data_tree.item(child, "text").split(":")[0].strip()
            path = f"{prefix}/{label}" if prefix else label
            if self.data_tree.item(child, "open"):
                opened.add(path)
                opened |= self._get_open_paths(child, path)
        return opened

    def _restore_open_paths(self, opened_paths, parent="", prefix=""):
        """Re-expand nodes whose text-path matches the previously opened set."""
        for child in self.data_tree.get_children(parent):
            label = self.data_tree.item(child, "text").split(":")[0].strip()
            path = f"{prefix}/{label}" if prefix else label
            if path in opened_paths:
                self.data_tree.item(child, open=True)
                self._restore_open_paths(opened_paths, child, path)

    def refresh_tree(self) -> None:
        if not self.data_manager.data:
            return

        # Remember which nodes were expanded and the current scroll position
        opened_paths = self._get_open_paths()
        selected_path = None
        if self.data_tree.selection():
            try:
                selected_path = self.get_full_path(self.data_tree.selection()[0])[0]
            except Exception:
                pass
        y_scroll = self.data_tree.yview()

        self.init_data_tree()
        self.build_tree(self.data_manager.data, "")

        # Restore expanded nodes
        self._restore_open_paths(opened_paths)

        # Restore scroll position
        self.data_tree.yview_moveto(y_scroll[0] if y_scroll else 0)

        # Try to re-select the previously selected item by matching its full path
        if selected_path:
            for item in self.data_tree.get_children(""):
                if self._find_and_focus_item(item, selected_path):
                    break

    def _select_by_text(self, target_text, parent=""):
        """Find and select a tree item by its label text."""
        for child in self.data_tree.get_children(parent):
            if self.data_tree.item(child, "text") == target_text:
                self.data_tree.focus(child)
                self.data_tree.selection_set(child)
                self.data_tree.see(child)
                return True
            if self._select_by_text(target_text, child):
                return True
        return False

    # Groups with many children that should stay collapsed by default
    _COLLAPSE_KEYWORDS = ['high_rate_sliprates', 'sliprate']
    _COLLAPSE_THRESHOLD = 20  # auto-collapse if children exceed this count

    def build_tree(self, data: Dict[str, Any], parent_iid: str) -> None:
        """Recursively build tree view from data"""
        parent_label = ""
        if parent_iid:
            parent_item = self.data_tree.item(parent_iid)
            parent_label = parent_item["text"].split(":")[0].strip()
        if isinstance(data, dict):
            for key, value in data.items():
                label = self.format_tree_label(key, value)
                iid = self.data_tree.insert(parent_iid, "end", text=label)
                if isinstance(value, (dict, list)):
                    # Skip recursion for groups that are known to be huge
                    child_count = len(value) if isinstance(value, (dict, list)) else 0
                    should_collapse = (
                        any(kw in key.lower() for kw in self._COLLAPSE_KEYWORDS)
                        or child_count > self._COLLAPSE_THRESHOLD
                    )
                    if should_collapse:
                        # Insert a dummy child so the expand arrow still appears
                        self.data_tree.insert(iid, "end", text="(loading...)")
                        # Store data for lazy loading when user expands
                        self._deferred_nodes[iid] = value
                    else:
                        self.build_tree(value, iid)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                try:
                    label = f"[{i}]: {value['name']}"
                except:
                    label = self.format_tree_label(f"[{i}]", value)
                iid = self.data_tree.insert(parent_iid, "end", text=label)
                if isinstance(value, (dict, list)):
                    self.build_tree(value, iid)

    def format_tree_label(self, key: str, value: Any) -> str:
        """Format label for tree view items based on data type and content.

        Args:
            key: The key or index name
            value: The value to format

        Returns:
            Formatted string label
        """
        if isinstance(value, str):
            return f"{key}: {value}"
        elif isinstance(value, (int, float, np.floating, np.integer)):
            return f"{key}: {value}"
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return f"{key}: {value.flatten()[0]}"
            else:
                shape_str = str(list(value.shape)).replace(" ", "")  # Remove spaces
                return f"{key}: array{shape_str}"
        elif isinstance(value, list):
            if len(value) == 1 and key not in ['events', 'runs']:
                return f"{key}: {value[0]}"
            else:
                return f"{key}: array[{len(value)}]"
        return f"{key}: {type(value).__name__}"
    
    def get_full_path(self, item=None):
        def clean_up_text(s):
            return s.split(':')[0].strip()
        if item is None:
            item = self.data_tree.selection()[0]
        parent_iid = self.data_tree.parent(item)
        node = []
        # go backward until reaching root
        while parent_iid != '':
            node.insert(0, clean_up_text(self.data_tree.item(parent_iid)['text']))
            parent_iid = self.data_tree.parent(parent_iid)
        i = clean_up_text(self.data_tree.item(item, "text"))
        return "/".join(node + [i]), i
    

    def pick_events(self) -> None:
        path, item = self.get_full_path()
        y = self.data_manager.get_data(path)
        x = np.arange(len(y))
        save_path = path[:path.rfind('/')+1] + "event_indices"
        parent_id = self.data_tree.parent(self.data_tree.selection()[0])
        if self.has_child_named(parent_id, "event_indices"):
            picked_idx = self.data_manager.get_data(self.get_full_path(parent_id)[0] + "/event_indices")
        else:
            picked_idx = []
        def save_and_refresh(data):
            self.data_manager.set_data(save_path, data, add_key=True)
            self.refresh_tree()
        view = PointsSelectorView(self, x, y, picked_idx, add_remove_enabled=True, 
                                 callback=save_and_refresh,
                                 xlabel='index', ylabel=item, title=path)
        self.set_window_icon(view)
        self.child_windows.append(view)

    def min_max(self):
        path, item = self.get_full_path()
        y = self.data_manager.get_data(path)
        x = np.arange(len(y))
        idx_min = np.argmin(y)
        idx_max = np.argmax(y)
        picked_idx = [idx_max, idx_min]
        view = PointsSelectorView(self, x, y, picked_idx, add_remove_enabled=False,
                                 xlabel='index', ylabel=item, title=path)
        self.set_window_icon(view)
        self.child_windows.append(view)


    def pick_strain_array_arrivals(self):
        path, item = self.get_full_path()
        # Extract index between 'runs/[' and the next ']'
        run_start = path.find('runs/[') + 6
        run_end = path.find(']', run_start)
        run_idx = int(path[run_start:run_end])

        # Extract index between 'events/[' and the next ']'
        event_start = path.find('events/[') + 8
        event_end = path.find(']', event_start)
        event_idx = int(path[event_start:event_end])

        view = DynamicStrainArrivalPickerView(self, run_idx, event_idx)
        self.set_window_icon(view)
        self.child_windows.append(view)

    def fit_cohesive_zone_model(self):
        path, item = self.get_full_path()
        # Extract index between 'runs/[' and the next ']'
        run_start = path.find('runs/[') + 6
        run_end = path.find(']', run_start)
        run_idx = int(path[run_start:run_end])

        # Extract index between 'events/[' and the next ']'
        event_start = path.find('events/[') + 8
        event_end = path.find(']', event_start)
        event_idx = int(path[event_start:event_end])

        view = CZMFitterView(self, run_idx, event_idx)
        self.set_window_icon(view)
        self.child_windows.append(view)

    def analyze_event(self):
        path, item = self.get_full_path()
        # Extract index between 'runs/[' and the next ']'
        run_start = path.find('runs/[') + 6
        run_end = path.find(']', run_start)
        run_idx = int(path[run_start:run_end])

        # Extract index between 'events/[' and the next ']'
        event_start = path.find('events/[') + 8
        event_end = path.find(']', event_start)
        event_idx = int(path[event_start:event_end])

        view = EventAnalyzerView(self, run_idx, event_idx)
        self.set_window_icon(view)
        self.child_windows.append(view)

    def preview_time_history(self):
        path, item = self.get_full_path()
        # Ensure it's called on 'time history'
        if item != "time history":
            return
            
        # Extract index between 'runs/[' and the next ']'
        run_start = path.find('runs/[') + 6
        run_end = path.find(']', run_start)
        run_idx = int(path[run_start:run_end])

        view = TimeHistoryView(self, run_idx, path)
        self.set_window_icon(view)
        self.child_windows.append(view)

    def preview_colored_lines(self):
        path, item = self.get_full_path()
        if item != "time history":
            return
        run_start = path.find('runs/[') + 6
        run_end = path.find(']', run_start)
        run_idx = int(path[run_start:run_end])

        view = ColoredLinesView(self, run_idx, path)
        self.set_window_icon(view)
        self.child_windows.append(view)

    def run_drop_analysis(self):
        """Run batch drop analysis on all events."""
        from labquake_explorer.analysis.batch_runner import run_batch_analysis
        from labquake_explorer.analysis.event_drop_analyzer import DEFAULT_CONFIG
        import os

        path, item = self.get_full_path()
        if item != "time history":
            return
        run_start = path.find('runs/[') + 6
        run_end = path.find(']', run_start)
        run_idx = int(path[run_start:run_end])

        time_history = self.data_manager.get_data(path)
        events = self.data_manager.get_data(f"runs/[{run_idx}]/events")

        if not time_history or not events:
            messagebox.showerror("Error", "Cannot load data")
            return
            
        # Ensure events is a list (HDF5 groups might load as dicts if keys aren't 0-indexed)
        if isinstance(events, dict):
            events = [events[k] for k in sorted(events.keys())]

        # Configuration dialog
        config_win = tk.Toplevel(self.root)
        config_win.title("Drop Analysis Configuration")
        config_win.grab_set()

        cfg = dict(DEFAULT_CONFIG)
        
        # Load globally saved drop config
        saved_drop_cfg = UserPrefs.get('DropAnalysis', 'config', {})
        if saved_drop_cfg:
            try:
                for k in ['tau_smooth_w', 'lvdt_smooth_w']:
                    if k in saved_drop_cfg: cfg[k] = int(saved_drop_cfg[k])
                for k in ['push_speed', 'delay_sec', 'window_sec']:
                    if k in saved_drop_cfg: cfg[k] = float(saved_drop_cfg[k])
                if 'pre_win' in saved_drop_cfg:
                    cfg['pre_win'] = tuple(float(x) for x in saved_drop_cfg['pre_win'])
                if 'post_win' in saved_drop_cfg:
                    cfg['post_win'] = tuple(float(x) for x in saved_drop_cfg['post_win'])
            except Exception:
                pass
                
        # Load shared skip_events from run-level shared list
        cfg['skip_events'] = self.data_manager.get_run_skip_events(run_idx)

        # Try to load existing config from this run to persist settings
        try:
            existing_analysis = self.data_manager.get_data(f"runs/[{run_idx}]/analysis")
            if isinstance(existing_analysis, dict) and 'config' in existing_analysis:
                old_cfg = existing_analysis['config']
                if isinstance(old_cfg, dict):
                    # Restore pre_win / post_win (may be ndarray from HDF5)
                    if 'pre_win' in old_cfg:
                        pw = old_cfg['pre_win']
                        if hasattr(pw, 'tolist'): pw = pw.tolist()
                        cfg['pre_win'] = tuple(float(x) for x in pw)
                    if 'post_win' in old_cfg:
                        pw = old_cfg['post_win']
                        if hasattr(pw, 'tolist'): pw = pw.tolist()
                        cfg['post_win'] = tuple(float(x) for x in pw)
                    # Restore scalar settings
                    for k in ['tau_smooth_w', 'lvdt_smooth_w']:
                        if k in old_cfg:
                            cfg[k] = int(old_cfg[k])
                    for k in ['push_speed', 'delay_sec', 'window_sec']:
                        if k in old_cfg:
                            cfg[k] = float(old_cfg[k])
        except Exception:
            pass

        row = 0
        entries = {}
        for label, key, default in [
            ("Pre window start", 'pre_start', cfg['pre_win'][0]),
            ("Pre window end", 'pre_end', cfg['pre_win'][1]),
            ("Post window start", 'post_start', cfg['post_win'][0]),
            ("Post window end", 'post_end', cfg['post_win'][1]),
            ("Tau smooth window", 'tau_smooth_w', cfg['tau_smooth_w']),
            ("LVDT smooth window", 'lvdt_smooth_w', cfg['lvdt_smooth_w']),
        ]:
            ttk.Label(config_win, text=label).grid(row=row, column=0, padx=5, pady=3, sticky='w')
            var = tk.StringVar(value=str(default))
            e = ttk.Entry(config_win, textvariable=var, width=10)
            e.grid(row=row, column=1, padx=5, pady=3)
            entries[key] = var
            row += 1

        ttk.Label(config_win, text="Skip events (comma-sep):").grid(row=row, column=0, padx=5, pady=3, sticky='w')
        initial_skip = ",".join(map(str, cfg.get("skip_events", [])))
        skip_var = tk.StringVar(value=initial_skip)
        ttk.Entry(config_win, textvariable=skip_var, width=60).grid(row=row, column=1, padx=5, pady=3)
        row += 1

        # Auto-detect output folder from HDF5 path + run name
        default_output = ""
        if self.data_manager.data_path:
            h5_path = self.data_manager.data_path
            h5_dir = str(h5_path.parent)
            h5_stem = h5_path.stem  # e.g. "t0145"
            try:
                run_data = self.data_manager.get_data(f"runs/[{run_idx}]")
                run_name_raw = run_data.get('name', f'run{run_idx+1}')
                # Extract run number: "run4_8MPa" -> "run4"
                run_part = run_name_raw.split('_')[0] if '_' in run_name_raw else run_name_raw
                default_output = os.path.join(h5_dir, f"{h5_stem}_{run_part}_drop")
            except Exception:
                default_output = os.path.join(h5_dir, f"{h5_stem}_run{run_idx+1}")

        ttk.Label(config_win, text="Output folder:").grid(row=row, column=0, padx=5, pady=3, sticky='w')
        out_var = tk.StringVar(value=default_output)
        ttk.Entry(config_win, textvariable=out_var, width=50).grid(row=row, column=1, padx=5, pady=3)
        row += 1

        def on_run():
            cfg['pre_win'] = (float(entries['pre_start'].get()), float(entries['pre_end'].get()))
            cfg['post_win'] = (float(entries['post_start'].get()), float(entries['post_end'].get()))
            cfg['tau_smooth_w'] = int(entries['tau_smooth_w'].get())
            cfg['lvdt_smooth_w'] = int(entries['lvdt_smooth_w'].get())

            skip_str = skip_var.get().strip()
            if skip_str:
                cfg['skip_events'] = [int(x.strip()) for x in skip_str.split(',') if x.strip()]
            else:
                cfg['skip_events'] = []

            output_dir = out_var.get().strip() or None

            UserPrefs.set('DropAnalysis', 'config', {
                'pre_win': list(cfg['pre_win']),
                'post_win': list(cfg['post_win']),
                'tau_smooth_w': cfg['tau_smooth_w'],
                'lvdt_smooth_w': cfg['lvdt_smooth_w']
            })
            # Persist skip_events to the shared run-level list
            self.data_manager.save_run_skip_events(run_idx, cfg['skip_events'])

            config_win.destroy()

            # Run analysis
            self.root.config(cursor="wait")
            self.root.update()
            try:
                arrays = run_batch_analysis(
                    time_history, events, cfg,
                    output_dir=output_dir
                )

                # Store results in memory
                run_data = self.data_manager.get_data(f"runs/[{run_idx}]")
                run_data['analysis'] = {
                    'config': {
                        'pre_win': list(cfg['pre_win']),
                        'post_win': list(cfg['post_win']),
                        'tau_smooth_w': cfg['tau_smooth_w'],
                        'lvdt_smooth_w': cfg['lvdt_smooth_w'],
                        'skip_events': cfg['skip_events'],
                    },
                    'per_event_windows': {},
                    'results': {k: v for k, v in arrays.items()},
                }

                self.refresh_tree()
                messagebox.showinfo(
                    "Done",
                    f"Drop analysis complete for {len(events)} events.\n"
                    f"Use 'Save As' to persist results to HDF5."
                )
            except Exception as e:
                messagebox.showerror("Error", f"Analysis failed: {e}")
            finally:
                self.root.config(cursor="")

        ttk.Button(config_win, text="Run", command=on_run).grid(
            row=row, column=0, columnspan=2, pady=10
        )

    def preview_summary_analysis(self):
        path, item = self.get_full_path()
        if item != "time history":
            return
        run_start = path.find('runs/[') + 6
        run_end = path.find(']', run_start)
        run_idx = int(path[run_start:run_end])

        view = SummaryAnalysisView(self, run_idx, path)
        self.set_window_icon(view)
        self.child_windows.append(view)

    def edit_event_drop(self):
        path, item = self.get_full_path()
        run_start = path.find('runs/[') + 6
        run_end = path.find(']', run_start)
        run_idx = int(path[run_start:run_end])

        event_start = path.find('events/[') + 8
        event_end = path.find(']', event_start)
        event_idx = int(path[event_start:event_end])

        view = EventDropEditorView(self, run_idx, event_idx)
        self.set_window_icon(view)
        self.child_windows.append(view)

    def edit_event_k(self):
        path, item = self.get_full_path()
        run_start = path.find('runs/[') + 6
        run_end = path.find(']', run_start)
        run_idx = int(path[run_start:run_end])

        event_start = path.find('events/[') + 8
        event_end = path.find(']', event_start)
        event_idx = int(path[event_start:event_end])

        view = EventKEditorView(self, run_idx, event_idx)
        self.set_window_icon(view)
        self.child_windows.append(view)

    def run_k_analysis(self):
        """Run batch K stiffness analysis on all events."""
        from labquake_explorer.analysis.batch_runner import run_batch_k_analysis
        from labquake_explorer.analysis.k_stiffness_analyzer import DEFAULT_K_CONFIG
        import os

        path, item = self.get_full_path()
        if item != "time history":
            return
        run_start = path.find('runs/[') + 6
        run_end = path.find(']', run_start)
        run_idx = int(path[run_start:run_end])

        time_history = self.data_manager.get_data(path)
        events = self.data_manager.get_data(f"runs/[{run_idx}]/events")

        if not time_history or not events:
            messagebox.showerror("Error", "Cannot load data")
            return

        if isinstance(events, dict):
            events = [events[k] for k in sorted(events.keys())]

        # Configuration dialog
        config_win = tk.Toplevel(self.root)
        config_win.title("K Stiffness Analysis Configuration")
        config_win.grab_set()

        cfg = dict(DEFAULT_K_CONFIG)
        
        # Load globally saved k config
        saved_k_cfg = UserPrefs.get('KAnalysis', 'config', {})
        if saved_k_cfg:
            try:
                for k in ['k_pre_start', 'k_pre_end', 'k_highpass_freq', 'k_lowpass_freq']:
                    if k in saved_k_cfg: cfg[k] = float(saved_k_cfg[k])
                if 'k_smooth_w' in saved_k_cfg:
                    cfg['k_smooth_w'] = int(saved_k_cfg['k_smooth_w'])
                if 'k_use_ransac' in saved_k_cfg:
                    cfg['k_use_ransac'] = bool(saved_k_cfg['k_use_ransac'])
            except Exception:
                pass

        # Load shared skip_events from run-level shared list
        cfg['skip_events'] = self.data_manager.get_run_skip_events(run_idx)

        # Load existing k_analysis config
        try:
            existing = self.data_manager.get_data(f"runs/[{run_idx}]/k_analysis")
            if isinstance(existing, dict) and 'config' in existing:
                old_cfg = existing['config']
                if isinstance(old_cfg, dict):
                    for k in ['k_pre_start', 'k_pre_end', 'k_window_sec', 'k_highpass_freq', 'k_lowpass_freq']:
                        if k in old_cfg:
                            cfg[k] = float(old_cfg[k])
                    if 'k_smooth_w' in old_cfg:
                        cfg['k_smooth_w'] = int(old_cfg['k_smooth_w'])
                    if 'k_use_ransac' in old_cfg:
                        cfg['k_use_ransac'] = bool(old_cfg['k_use_ransac'])
        except Exception:
            pass

        row = 0
        entries = {}
        for label, key, default in [
            ("Pre start (s)", 'k_pre_start', cfg['k_pre_start']),
            ("Pre end (s)", 'k_pre_end', cfg['k_pre_end']),
            ("Smooth window", 'k_smooth_w', cfg['k_smooth_w']),
            ("Highpass freq (Hz, 0=off)", 'k_highpass_freq', cfg['k_highpass_freq']),
            ("Lowpass freq (Hz, 0=off)", 'k_lowpass_freq', cfg.get('k_lowpass_freq', 0.0)),
        ]:
            ttk.Label(config_win, text=label).grid(row=row, column=0, padx=5, pady=3, sticky='w')
            var = tk.StringVar(value=str(default))
            ttk.Entry(config_win, textvariable=var, width=10).grid(row=row, column=1, padx=5, pady=3)
            entries[key] = var
            row += 1

        ttk.Label(config_win, text="Skip events (comma-sep):").grid(row=row, column=0, padx=5, pady=3, sticky='w')
        initial_skip = ",".join(map(str, cfg.get('skip_events', [])))
        skip_var = tk.StringVar(value=initial_skip)
        ttk.Entry(config_win, textvariable=skip_var, width=60).grid(row=row, column=1, padx=5, pady=3)
        row += 1

        # Output folder
        default_output = ""
        if self.data_manager.data_path:
            h5_path = self.data_manager.data_path
            h5_dir = str(h5_path.parent)
            h5_stem = h5_path.stem
            try:
                run_data = self.data_manager.get_data(f"runs/[{run_idx}]")
                run_name_raw = run_data.get('name', f'run{run_idx+1}')
                run_part = run_name_raw.split('_')[0] if '_' in run_name_raw else run_name_raw
                default_output = os.path.join(h5_dir, f"{h5_stem}_{run_part}_k")
            except Exception:
                default_output = os.path.join(h5_dir, f"{h5_stem}_run{run_idx+1}_k")

        ttk.Label(config_win, text="Output folder:").grid(row=row, column=0, padx=5, pady=3, sticky='w')
        out_var = tk.StringVar(value=default_output)
        ttk.Entry(config_win, textvariable=out_var, width=50).grid(row=row, column=1, padx=5, pady=3)
        row += 1

        ttk.Label(config_win, text="Use RANSAC:").grid(row=row, column=0, padx=5, pady=3, sticky='w')
        ransac_var = tk.BooleanVar(value=cfg.get('k_use_ransac', False))
        ttk.Checkbutton(config_win, variable=ransac_var).grid(row=row, column=1, padx=5, pady=3, sticky='w')
        row += 1

        def on_run():
            cfg['k_pre_start'] = float(entries['k_pre_start'].get())
            cfg['k_pre_end'] = float(entries['k_pre_end'].get())
            cfg['k_smooth_w'] = int(entries['k_smooth_w'].get())
            cfg['k_highpass_freq'] = float(entries['k_highpass_freq'].get())
            cfg['k_lowpass_freq'] = float(entries['k_lowpass_freq'].get())
            cfg['k_use_ransac'] = ransac_var.get()

            # Auto-set window_sec to cover the full pre range
            cfg['k_window_sec'] = abs(cfg['k_pre_start']) + 0.5

            skip_str = skip_var.get().strip()
            if skip_str:
                cfg['skip_events'] = [int(x.strip()) for x in skip_str.split(',') if x.strip()]
            else:
                cfg['skip_events'] = []

            output_dir = out_var.get().strip() or None
            
            UserPrefs.set('KAnalysis', 'config', {
                'k_pre_start': cfg['k_pre_start'],
                'k_pre_end': cfg['k_pre_end'],
                'k_smooth_w': cfg['k_smooth_w'],
                'k_highpass_freq': cfg['k_highpass_freq'],
                'k_lowpass_freq': cfg.get('k_lowpass_freq', 0.0),
                'k_use_ransac': cfg['k_use_ransac']
            })
            # Persist skip_events to the shared run-level list
            self.data_manager.save_run_skip_events(run_idx, cfg['skip_events'])
            
            config_win.destroy()

            self.root.config(cursor="wait")
            self.root.update()
            try:
                arrays = run_batch_k_analysis(
                    time_history, events, cfg,
                    output_dir=output_dir
                )

                run_data = self.data_manager.get_data(f"runs/[{run_idx}]")
                run_data['k_analysis'] = {
                    'config': {
                        'k_pre_start': cfg['k_pre_start'],
                        'k_pre_end': cfg['k_pre_end'],
                        'k_smooth_w': cfg['k_smooth_w'],
                        'k_highpass_freq': cfg['k_highpass_freq'],
                        'k_lowpass_freq': cfg.get('k_lowpass_freq', 0.0),
                        'k_use_ransac': cfg['k_use_ransac'],
                        'skip_events': cfg['skip_events'],
                    },
                    'per_event_config': {},
                    'results': {k: v for k, v in arrays.items()},
                }

                self.refresh_tree()
                messagebox.showinfo(
                    "Done",
                    f"K analysis complete for {len(events)} events.\n"
                    f"Use 'Save As' to persist results to HDF5."
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", f"K analysis failed: {e}")
            finally:
                self.root.config(cursor="")

        ttk.Button(config_win, text="Run", command=on_run).grid(
            row=row, column=0, columnspan=2, pady=10
        )

    def pick_indices(self):
        item = self.data_tree.selection()[0]
        path = self.get_full_path(item)[0]#.replace('/', os.sep)# 為這個視窗把斜線換回 Windows 原生的格式
        view = IndexPickerView(self, item_y=path) # 這裡改成傳入 path
        self.set_window_icon(view)
        self.child_windows.append(view)

    def extract_slope(self):
        item = self.data_tree.selection()[0]
        path = self.get_full_path(item)[0]#.replace('/', os.sep)# 為這個視窗把斜線換回 Windows 原生的格式
        view = SlopeAnalyzerView(self, item_y=path) # 這裡改成傳入 path
        self.set_window_icon(view)
        self.child_windows.append(view)

    def pick_run(self):
        item_id = self.data_tree.selection()[0]
        item_path, item_name = self.get_full_path(item_id)

        y = self.get_data(self.data, item_path)
        x = np.arange(len(y))
        picked_idx = [int(len(y)/3), int(len(y)/3*2)]
        view = PointsSelectorView(self, x, y, picked_idx, add_remove_enabled=False, 
                                 callback=lambda idx: self.extract_run(idx),
                                 xlabel='index', ylabel=item_name, title=item_path)
        self.set_window_icon(view)
        self.child_windows.append(view)

    def extract_events(self):
        """Handle UI for event extraction and delegate to EventProcessor"""
        # Get selected item and paths
        item_id = self.data_tree.selection()[0]
        parent = self.data_tree.parent(item_id)
        event_indices_path = self.get_full_path()[0]
        parent_path = self.get_full_path(parent)[0]
        events_path = f"{parent_path}/events"

        # Check for existing events
        if self.has_child_named(parent, "events"):
            ans = messagebox.askokcancel(
                title="Confirmation", 
                message=f'This procedure will replace all data in "{events_path}".', 
                icon=messagebox.WARNING
            )
            if not ans:
                return

        # Get window size from user
        window = simpledialog.askfloat(
            'Set event time window length', 
            'Please set the duration before and after the event to be extracted.',
            initialvalue=5
        )
        if window is None:
            print('Event extraction aborted.')
            return
        print(f'Window set to (-{window}, {window})')

        try:
            # Get run data and indices
            run_data = self.data_manager.get_data(parent_path)
            event_indices = self.data_manager.get_data(event_indices_path)

            # Extract events using EventProcessor
            events = self.data_manager.event_processor.extract_events(
                run_data,
                event_indices,
                window
            )

            # Save results
            self.data_manager.set_data(events_path, events, add_key=True)
            self.refresh_tree()

            self.root.after(100, lambda: messagebox.showinfo(title="Success", message="Events extracted."))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract events: {str(e)}")

    def edit_string(self):
        """Edit a string value in the data structure"""
        path, item = self.get_full_path()
        data = self.data_manager.get_data(path)

        new_string = simpledialog.askstring('Edit String', f'{path}', initialvalue=data)
        if new_string is None:
            print('Edit String aborted.')
            return
        
        self.data_manager.set_data(path, new_string)
        self.refresh_tree()
            
    def on_double_click(self, event):
        path, item = self.get_full_path()
        print(f"Double-clicked on item: {path}")
        data = self.data_manager.get_data(path)
        
        if type(data) is np.ndarray:
            print(f"plotting {item}")
            view = SimplePlotView(self)
            self.set_window_icon(view)
            
            # --- 新增的 X 軸判斷邏輯 ---
            # 取得上一層資料夾的路徑
            parent_path = path.rsplit('/', 1)[0]
            parent_data = self.data_manager.get_data(parent_path)
            
            # 如果上一層資料夾裡有 'time' 或 'time_array'，就把它當作 X 軸
            if isinstance(parent_data, dict) and 'time' in parent_data:
                time_data = parent_data['time']
                view.ax.plot(time_data, data)
                view.ax.set_xlabel('Time (s)')
            elif isinstance(parent_data, dict) and 'time_array' in parent_data:
                time_data = parent_data['time_array']
                view.ax.plot(time_data, data)
                view.ax.set_xlabel('Time (s)')
            else:
                # 找不到時間，退回預設的 index
                view.ax.plot(data)
                view.ax.set_xlabel('index')
            # ------------------------
            
            view.ax.set_ylabel(item)
            view.ax.set_title(path.replace('/[', '['))
            view.canvas.draw() # 更新畫布
            self.child_windows.append(view)
        elif type(data) is dict:
            print('dict')
        elif type(data) is list:
            print('list')
        else:
            print(data)
        
    def on_left_click(self, event):
        if self.active_context_menu:
            self.active_context_menu.unpost()

    def on_right_click(self, event):
        # Clear previous menu
        if self.active_context_menu:
            self.active_context_menu.unpost()
        self.active_context_menu = None

        try:
            item = self.data_tree.selection()[0]
        except:
            item = None
        if not item:
            return

        # data-structure-specific context menus
        item_label = self.data_tree.item(item)['text'].split(':')
        parent = self.data_tree.parent(item)
        parent_name = self.data_tree.item(parent)['text'].split(':')[0] if parent else ""
        grandparent = self.data_tree.parent(parent)
        grandparent_name = self.data_tree.item(grandparent)['text'].split(':')[0] if grandparent else ""

        if grandparent_name == "runs":
            if item_label[0] == "event_indices":
                self.active_context_menu = self.event_indices_menu
            elif item_label[0] == "time history":
                self.active_context_menu = self.time_history_menu
            elif len(item_label) > 1 and "array" in item_label[1]:
                self.active_context_menu = self.run_menu
        elif grandparent_name == "events":
            if len(item_label) > 1 and "array" in item_label[1]:
                self.active_context_menu = self.event_array_menu
            else:
                self.active_context_menu = self.event_menu
        elif parent_name == "events":
            self.active_context_menu = self.event_menu

        # general purpose context menus
        if not self.active_context_menu:
            path, _ = self.get_full_path()
            data = self.data_manager.get_data(path)
            if isinstance(data, str):
                self.active_context_menu = self.string_menu
            elif len(item_label) > 1 and "array" in item_label[1] and parent_name == "":
                self.active_context_menu = self.array_menu

        # post context menu
        if self.active_context_menu:
            self.active_context_menu.post(event.x_root, event.y_root)
    
    def has_child_name_contains(self, item_id, keyword):
        if not item_id:
            return False
        children = self.data_tree.get_children(item_id)
        return any(keyword in self.data_tree.item(child_id, "text") for child_id in children)
    
    def has_child_named(self, item_id, name):
        if not item_id:
            return False
        children = self.data_tree.get_children(item_id)
        return any(name == self.data_tree.item(child_id, "text").split(":")[0] for child_id in children)

    def on_delete(self, event) -> None:
        """Handle deletion of items from the tree view"""
        try:
            item_id = self.data_tree.selection()[0]

            # Get the item above the selected item
            prev_item = self.data_tree.prev(item_id)
            if not prev_item:
                # If no previous item, get the parent
                prev_item = self.data_tree.parent(item_id)

            # Store the path of the item to focus
            focus_path = self.get_full_path(prev_item)[0] if prev_item else ""

            item_path = self.get_full_path(item_id)[0]

            # Confirm deletion with user
            ans = messagebox.askokcancel(
                title="Confirmation", 
                message=f'This procedure will delete "{item_path}".', 
                icon=messagebox.WARNING
            )
            if not ans:
                return

            # Attempt deletion
            try:
                self.data_manager.delete_data(item_path)
                self.refresh_tree()

                # After refresh, find and focus the previous item
                if focus_path:
                    for item in self.data_tree.get_children(""):
                        if self._find_and_focus_item(item, focus_path):
                            break

            except (ValueError, KeyError, IndexError) as e:
                messagebox.showerror("Error", f"Failed to delete item: {str(e)}")

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def _find_and_focus_item(self, item: str, target_path: str) -> bool:
        """Helper method to find and focus an item by its path

        Args:
            item: The current tree item ID to check
            target_path: The path to find

        Returns:
            bool: True if item was found and focused
        """
        current_path = self.get_full_path(item)[0]
        if current_path == target_path:
            self.data_tree.focus(item)
            self.data_tree.selection_set(item)
            self.data_tree.see(item)
            return True

        # Recursively check children
        for child in self.data_tree.get_children(item):
            if self._find_and_focus_item(child, target_path):
                return True

        return False

    def on_closing(self) -> None:
        try:
            # First withdraw (hide) all windows
            for window in self.child_windows[:]:
                if window.winfo_exists():
                    window.withdraw()
            self.root.withdraw()
            
            # Then destroy them
            for window in self.child_windows[:]:
                if window.winfo_exists():
                    window.destroy()
                self.child_windows.remove(window)
                
            self.root.destroy()
            sys.exit(0)
        except Exception as e:
            print(f"Error during cleanup: {e}")
            sys.exit(1)

    def setup_bindings(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<Delete>", self.on_delete)

    def set_window_icon(self, window: tk.Tk | tk.Toplevel) -> None:
        """Set the application icon for any window.

        Args:
            window: The window (root or Toplevel) to set the icon for
        """
        # Go up three levels from the current file to reach project root
        project_root = Path(__file__).parent.parent.parent
        icon_path = project_root / "assets" / "icons" / "labquake_explorer"

        try:
            if sys.platform == "darwin":  # macOS
                img = tk.Image("photo", file=str(icon_path.with_suffix(".png")))
                window.tk.call('wm', 'iconphoto', window._w, img)
            elif sys.platform == "win32":  # Windows
                window.iconbitmap(str(icon_path.with_suffix(".ico")))
            elif sys.platform.startswith("linux"):  # Linux
                img = tk.PhotoImage(file=str(icon_path.with_suffix(".png")))
                window.tk.call('wm', 'iconphoto', window._w, img)
        except Exception as e:
            print(f"Error setting icon for window: {e}")