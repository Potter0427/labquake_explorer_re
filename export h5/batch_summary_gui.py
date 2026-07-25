import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(r"C:\experiment\labquake_explorer_re")
from labquake_explorer.data.data_manager import DataManager

def _get_t_trig(ev):
    if isinstance(ev, dict):
        if 'event_time' in ev: return ev['event_time']
        if 'time' in ev and len(ev['time']) > 0: return ev['time'][0]
    return np.nan

def moving_average(a, n=3):
    if n <= 1: return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def extract_analysis_results(events):
    results = {}
    keys_to_extract = ['delta_tau', 'delta_mu', 'delta_lvdt', 'D_Push', 'D_max', 'D_E3', 'D_E4', 'skipped', 'k']
    
    for ev in events:
        if isinstance(ev, dict) and 'delta' in ev:
            keys_to_extract.extend([k for k in ev['delta'].keys() if k.endswith('_value')])
            break
            
    for k in keys_to_extract:
        arr = []
        for ev in events:
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
            elif k.endswith('_value') and k.startswith('E'): 
                arr.append(ev.get('delta', {}).get(k, np.nan))
            else:
                arr.append(ev.get(k, np.nan))
        
        final_key = f'delta_{k.split("_")[0]}' if k.endswith('_value') else k
        results[final_key] = np.array(arr)
        
    if 'skipped' in results:
        results['skipped'] = np.array([bool(x) if not np.isnan(x) else False for x in results['skipped']])
        
    return results

def plot_run_summary(run_data, run_name, plots_to_show, t_plot_start_event, t_plot_end_event, settings):
    time_history = run_data.get('time history', {})
    events = run_data.get('events', [])
    
    if not time_history or 'time' not in time_history or not events:
        return None
        
    results = extract_analysis_results(events)
    
    trigger_times = []
    for ev in events:
        t = _get_t_trig(ev)
        trigger_times.append(t if t is not None else np.nan)
    trigger_times = np.array(trigger_times)
    
    valid_trigs = trigger_times[~np.isnan(trigger_times)]
    if len(valid_trigs) == 0:
        return None
        
    try:
        start_idx = max(0, int(t_plot_start_event) - 1)
        end_idx = min(len(valid_trigs) - 1, int(t_plot_end_event) - 1)
    except:
        start_idx = 0
        end_idx = len(valid_trigs) - 1
        
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
        
    t_start = valid_trigs[start_idx]
    t_end = valid_trigs[end_idx]
    
    t_all = time_history['time']
    t_plot_start = t_start - 1.0
    t_plot_end = t_end + 5.0
    mask = (t_all >= t_plot_start) & (t_all <= t_plot_end)
    t_mask = t_all[mask]
    if len(t_mask) == 0: return None
    
    t_offset = t_mask[0]
    t_plot = t_mask - t_offset
    
    plt.rcParams.update({
        'font.size': settings['FONT_SIZE_TICK'],
        'axes.labelsize': settings['FONT_SIZE_LABEL'],
        'lines.linewidth': settings['LINE_WIDTH'],
    })
    
    n = len(plots_to_show)
    fig_w = settings['FIG_WIDTH']
    fig_h_per = settings['FIG_HEIGHT_PER_PLOT']
    fig, axs = plt.subplots(n, 1, figsize=(fig_w, max(4, fig_h_per * n)), sharex=True, dpi=150)
    if n == 1: axs = [axs]
    
    axs_map = dict(zip(plots_to_show, axs))
    
    def _get_analysis_in_range(key):
        if key not in results: return None, None
        arr = results[key]
        trigs = trigger_times
        valid_mask = np.ones(len(trigs), dtype=bool)
        valid_mask[0] = False 
        range_mask = valid_mask & ~np.isnan(trigs) & (trigs >= t_plot_start - 1) & (trigs <= t_plot_end + 1)
        if 'skipped' in results:
            range_mask &= ~results['skipped']
        return trigs[range_mask] - t_offset, arr[range_mask]

    def _add_trigger_lines(ax):
        t_end_ext = t_end + (t_end - t_start) * 0.01
        for i, tr_time in enumerate(trigger_times):
            if t_start <= tr_time <= t_end_ext:
                tr = tr_time - t_offset
                ax.axvline(x=tr, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

    for key, ax in axs_map.items():
        ax.grid(True, linestyle='-', alpha=0.3)
        _add_trigger_lines(ax)
        
        has_legend = False
        
        if key == 'mu':
            if 'mu' in time_history:
                mu_data = time_history['mu'][mask]
                mu_sm = moving_average(mu_data, 500)
                if len(mu_sm) < len(t_plot):
                    mu_sm = np.pad(mu_sm, (0, len(t_plot) - len(mu_sm)), 'edge')
                ax.plot(t_plot, mu_sm, 'k')
            ax.set_ylabel(r'$\mu$')
            
        elif key == 'slip' or key == 'slip_no_lvdt':
            if key == 'slip' and 'LP_displacement' in time_history:
                lvdt_raw = time_history['LP_displacement'][mask]
                lvdt_sm = moving_average(lvdt_raw, 100)
                if len(lvdt_sm) < len(t_plot):
                    lvdt_sm = np.pad(lvdt_sm, (0, len(t_plot) - len(lvdt_sm)), 'edge')
                lvdt_0 = lvdt_sm - lvdt_sm[0]
                ax.plot(t_plot, lvdt_0, 'gray', label='LVDT', alpha=0.5)
            eddy_keys = sorted([k for k in time_history.keys() if 'eddy' in k.lower()])
            for i, k in enumerate(eddy_keys):
                e_0 = time_history[k][mask] - time_history[k][mask][0]
                ax.plot(t_plot, e_0, label=f'E{i+1}', alpha=0.7)
            ax.set_ylabel('slip [μm]')
            has_legend = True

        elif key == 'delta_tau':
            t_r, vals = _get_analysis_in_range('delta_tau')
            if t_r is not None:
                valid = ~np.isnan(vals)
                ax.plot(t_r[valid], vals[valid], 'o-', markersize=4)
            ax.set_ylabel(r'$\Delta\tau$ [MPa]')

        elif key == 'delta_mu':
            t_r, vals = _get_analysis_in_range('delta_mu')
            if t_r is not None:
                valid = ~np.isnan(vals)
                ax.plot(t_r[valid], vals[valid], 'o-', markersize=4, color='darkorange')
            ax.set_ylabel(r'$\Delta\mu$')
            
        elif key == 'delta_slip':
            eddy_keys = sorted([k for k in time_history.keys() if 'eddy' in k.lower()])
            for i in range(len(eddy_keys)):
                label = f'delta_E{i+1}'
                t_r, vals = _get_analysis_in_range(label)
                if t_r is not None:
                    valid = ~np.isnan(vals)
                    ax.plot(t_r[valid], vals[valid], 'o-', alpha=0.7, markersize=4, label=f'E{i+1}')
            ax.set_ylabel(r'$\Delta$ Slip [μm]')
            has_legend = True

        elif key == 'delta_lvdt':
            t_r, vals = _get_analysis_in_range('delta_lvdt')
            if t_r is not None:
                valid = ~np.isnan(vals)
                ax.plot(t_r[valid], vals[valid], 'o-', color='slategrey', markersize=4)
            ax.set_ylabel(r'$\Delta$ LVDT [μm]')

        elif key == 'd_values':
            d_keys_to_plot = [('D_Push', 'teal', r'$D_{Push}$'), ('D_max', 'coral', r'$D_{max}$')]
            t_r_e4, vals_e4 = _get_analysis_in_range('D_E4')
            if t_r_e4 is not None and len(t_r_e4) > 0 and np.any(~np.isnan(vals_e4)):
                d_keys_to_plot.append(('D_E4', None, r'$D_{E4}$'))
            else:
                d_keys_to_plot.append(('D_E3', None, r'$D_{E3}$'))

            for dkey, color, label in d_keys_to_plot:
                t_r, vals = _get_analysis_in_range(dkey)
                if t_r is not None:
                    valid = ~np.isnan(vals)
                    kwargs = {'markersize': 4, 'alpha': 0.8, 'label': label}
                    if color: kwargs['color'] = color
                    ax.plot(t_r[valid], vals[valid], 'o-', **kwargs)
            ax.set_ylabel(r'D [μm]')
            has_legend = True
            
        elif key == 'stiffness':
            t_r, vals = _get_analysis_in_range('k')
            if t_r is not None:
                valid = ~np.isnan(vals)
                ax.plot(t_r[valid], vals[valid], 'o-', color='teal', markersize=4)
            ax.set_ylabel('k [MPa/μm]')
            
        elif key == 'slip_rate':
            eddy_keys = sorted([k for k in time_history.keys() if k.startswith('eddy_ch')])
            hr_group = time_history.get('high_rate_sliprates', {})
            for i, ek in enumerate(eddy_keys):
                ch_num = ek.replace('eddy_ch', '')
                t_rate_key = f't_sliprate_ch{ch_num}'
                rate_key = f'sliprate_ch{ch_num}'
                all_t_parts, all_r_parts = [], []
                
                if t_rate_key in time_history and rate_key in time_history:
                    t_sr = time_history[t_rate_key]
                    r_sr = time_history[rate_key]
                    sr_mask = (t_sr >= t_plot_start) & (t_sr <= t_plot_end)
                    if np.sum(sr_mask) > 0:
                        all_t_parts.append(t_sr[sr_mask] - t_offset)
                        all_r_parts.append(r_sr[sr_mask])
                        
                blk_idx = 2
                while True:
                    hr_t_name = f't_high_sliprate_ch{ch_num}_blk{blk_idx}'
                    hr_r_name = f'high_sliprate_ch{ch_num}_blk{blk_idx}'
                    t_hr = hr_group.get(hr_t_name) if isinstance(hr_group, dict) else hr_group.get(hr_t_name)
                    r_hr = hr_group.get(hr_r_name) if isinstance(hr_group, dict) else hr_group.get(hr_r_name)
                    if t_hr is None:
                        t_hr = time_history.get(hr_t_name)
                        r_hr = time_history.get(hr_r_name)
                    if t_hr is None: break
                    hr_mask = (t_hr >= t_plot_start) & (t_hr <= t_plot_end)
                    if np.sum(hr_mask) > 0:
                        all_t_parts.append(t_hr[hr_mask] - t_offset)
                        all_r_parts.append(r_hr[hr_mask])
                    blk_idx += 1
                
                if all_t_parts:
                    combined_t = np.concatenate(all_t_parts)
                    combined_r = np.concatenate(all_r_parts)
                    idx_sort = np.argsort(combined_t)
                    ax.plot(combined_t[idx_sort], combined_r[idx_sort], color=f'C{i}', alpha=0.7, label=f'E{i+1}')
            ax.set_yscale('log')
            ax.set_ylim(1e-2, 3e5)
            ax.set_ylabel('Rate [μm/s]')
            has_legend = True

        # 將 Legend 移到圖表右側外面
        if has_legend and settings.get('SHOW_LEGEND', True):
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')

    axs[-1].set_xlabel('Time [s]')
    axs[-1].set_xlim([t_plot[0], t_plot[-1]])
    
    # 根據需求，移除主標題 (fig.suptitle)
    # fig.suptitle(run_name, fontweight='bold', fontsize=settings['FONT_SIZE_TITLE'])
    
    fig.tight_layout()
    return fig

class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, current_settings):
        super().__init__(parent)
        self.title("圖片設定")
        self.geometry("300x250")
        self.transient(parent)
        self.grab_set()
        
        self.result = current_settings.copy()
        
        self.entries = {}
        row = 0
        for key, val in current_settings.items():
            ttk.Label(self, text=key).grid(row=row, column=0, padx=10, pady=5, sticky=tk.W)
            if isinstance(val, bool):
                var = tk.BooleanVar(value=val)
                c = ttk.Checkbutton(self, variable=var)
                c.grid(row=row, column=1, padx=10, pady=5, sticky=tk.W)
                self.entries[key] = var
            else:
                e = ttk.Entry(self, width=15)
                e.insert(0, str(val))
                e.grid(row=row, column=1, padx=10, pady=5)
                self.entries[key] = e
            row += 1
            
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=15)
        
        ttk.Button(btn_frame, text="確定", command=self.on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
    def on_ok(self):
        try:
            for k, w in self.entries.items():
                if isinstance(w, tk.BooleanVar):
                    self.result[k] = w.get()
                else:
                    self.result[k] = float(w.get())
            self.destroy()
        except ValueError:
            messagebox.showerror("錯誤", "數值設定必須為數字", parent=self)

class BatchPlotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Batch Summary Plotter")
        self.geometry("800x600")
        
        self.config_path = Path(__file__).parent / "batch_summary_config.json"
        
        self.settings = {
            'FIG_WIDTH': 10.0,
            'FIG_HEIGHT_PER_PLOT': 2.5,
            'FONT_SIZE_LABEL': 14.0,
            'FONT_SIZE_TICK': 12.0,
            'LINE_WIDTH': 1.5,
            'SHOW_LEGEND': True
        }
        
        # 保存樹狀列中的 Run 資訊: item_id -> { 'file_path': str, 'run_idx': int, 'run_name': str, 'start': int, 'end': int }
        self.runs_data = {}
        self.saved_runs_config = {} # 用來暫存讀取進來的 event 設定
        
        self.create_widgets()
        self.after(100, self.load_config)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左側：檔案與 Run 列表
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        btn_add = ttk.Button(left_frame, text="1. 加入 HDF5 檔案 (自動讀取 Runs)", command=self.add_files)
        btn_add.pack(fill=tk.X, pady=(0, 5))
        
        columns = ("file", "run", "start", "end")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings")
        self.tree.heading("file", text="HDF5 檔案")
        self.tree.heading("run", text="Run")
        self.tree.heading("start", text="Start Event")
        self.tree.heading("end", text="End Event")
        self.tree.column("file", width=120)
        self.tree.column("run", width=80)
        self.tree.column("start", width=80)
        self.tree.column("end", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        
        tree_btns = ttk.Frame(left_frame)
        tree_btns.pack(fill=tk.X, pady=5)
        ttk.Button(tree_btns, text="移除選定 Run", command=self.remove_run).pack(side=tk.LEFT, padx=2)
        ttk.Button(tree_btns, text="清空列表", command=self.clear_runs).pack(side=tk.LEFT, padx=2)
        
        # 右側：設定與操作
        right_frame = ttk.Frame(main_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # 編輯選定的 Run 範圍
        edit_frame = ttk.LabelFrame(right_frame, text="編輯選定 Run 的 Event 範圍", padding="5")
        edit_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(edit_frame, text="Start Event:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ent_start = ttk.Entry(edit_frame, width=10)
        self.ent_start.grid(row=0, column=1, pady=2, padx=5)
        
        ttk.Label(edit_frame, text="End Event:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ent_end = ttk.Entry(edit_frame, width=10)
        self.ent_end.grid(row=1, column=1, pady=2, padx=5)
        
        ttk.Button(edit_frame, text="更新範圍", command=self.update_run_range).grid(row=2, column=0, columnspan=2, pady=5)
        
        # 物理量選取
        plot_frame = ttk.LabelFrame(right_frame, text="2. 選取物理量", padding="5")
        plot_frame.pack(fill=tk.X, pady=5)
        
        self.plot_vars = {}
        plot_options = ['mu', 'slip', 'slip_no_lvdt', 'delta_tau', 'delta_mu', 'delta_slip', 'delta_lvdt', 'd_values', 'stiffness', 'slip_rate']
        for i, opt in enumerate(plot_options):
            var = tk.BooleanVar(value=(opt in ['mu', 'delta_tau', 'delta_mu']))
            self.plot_vars[opt] = var
            ttk.Checkbutton(plot_frame, text=opt, variable=var).grid(row=i//2, column=i%2, sticky=tk.W, padx=5, pady=2)
            
        # 圖片設定按鈕
        ttk.Button(right_frame, text="圖片設定 (大小、比例)", command=self.open_settings).pack(fill=tk.X, pady=10)
        
        # 執行按鈕
        ttk.Button(right_frame, text="單獨預覽選定 Run 的圖", command=self.preview_plot).pack(fill=tk.X, pady=5)
        ttk.Button(right_frame, text="全部批次存檔", command=self.run_batch_save).pack(fill=tk.X, pady=5)
        
        self.lbl_status = ttk.Label(right_frame, text="等待中...")
        self.lbl_status.pack(side=tk.BOTTOM, anchor=tk.W, pady=10)
        
    def add_files(self, file_paths=None):
        if file_paths is None:
            files = filedialog.askopenfilenames(title="選擇 HDF5 檔案", filetypes=[("HDF5 Files", "*.h5 *.hdf5")])
        else:
            files = file_paths
            
        if not files: return
        
        manager = DataManager()
        self.lbl_status.config(text="讀取 HDF5 中...")
        self.update()
        
        for fpath in files:
            path_obj = Path(fpath)
            str_path = str(path_obj)
            if not path_obj.exists(): continue
            try:
                manager.load_file(path_obj)
                data = manager.data
                if not data or 'runs' not in data: continue
                
                runs = data['runs']
                runs_items = enumerate(runs) if isinstance(runs, list) else runs.items()
                
                for run_idx, run_data in runs_items:
                    if not isinstance(run_data, dict): continue
                    run_name = run_data.get('name', f"Run_{run_idx}")
                    
                    # 預設範圍
                    events = run_data.get('events', [])
                    start_ev, end_ev = 1, len(events)
                    if end_ev == 0: end_ev = 9999
                    
                    # 嘗試從存檔載入
                    if str_path in self.saved_runs_config and run_name in self.saved_runs_config[str_path]:
                        start_ev = self.saved_runs_config[str_path][run_name].get('start', start_ev)
                        end_ev = self.saved_runs_config[str_path][run_name].get('end', end_ev)
                    
                    display_name = data.get('name', path_obj.name)
                    item_id = self.tree.insert("", tk.END, values=(display_name, run_name, start_ev, end_ev))
                    self.runs_data[item_id] = {
                        'file_path': str_path,
                        'run_idx': run_idx,
                        'run_name': run_name,
                        'run_data': run_data,
                        'start': start_ev,
                        'end': end_ev
                    }
            except Exception as e:
                print(f"Failed to load {fpath}: {e}")
                
        self.lbl_status.config(text="讀取完成。")
        
    def on_tree_select(self, event):
        selected = self.tree.selection()
        if not selected: return
        
        item_id = selected[0]
        data = self.runs_data.get(item_id)
        if data:
            self.ent_start.delete(0, tk.END)
            self.ent_start.insert(0, str(data['start']))
            self.ent_end.delete(0, tk.END)
            self.ent_end.insert(0, str(data['end']))
            
    def update_run_range(self):
        selected = self.tree.selection()
        if not selected: return
        
        try:
            s_val = int(self.ent_start.get())
            e_val = int(self.ent_end.get())
        except ValueError:
            messagebox.showerror("錯誤", "範圍必須為數字")
            return
            
        for item_id in selected:
            self.runs_data[item_id]['start'] = s_val
            self.runs_data[item_id]['end'] = e_val
            
            # 更新顯示
            vals = self.tree.item(item_id, 'values')
            self.tree.item(item_id, values=(vals[0], vals[1], s_val, e_val))
            
    def remove_run(self):
        for item_id in self.tree.selection():
            self.tree.delete(item_id)
            if item_id in self.runs_data:
                del self.runs_data[item_id]
                
    def clear_runs(self):
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        self.runs_data.clear()
        
    def open_settings(self):
        dlg = SettingsDialog(self, self.settings)
        self.wait_window(dlg)
        self.settings = dlg.result

    def _get_plots_to_show(self, warn=True):
        plots = [k for k, v in self.plot_vars.items() if v.get()]
        if not plots and warn:
            messagebox.showwarning("警告", "請至少勾選一個物理量。")
        return plots

    def save_config(self):
        runs_config = {}
        for data in self.runs_data.values():
            p = data['file_path']
            r = data['run_name']
            if p not in runs_config: runs_config[p] = {}
            runs_config[p][r] = {'start': data['start'], 'end': data['end']}
            
        config = {
            'settings': self.settings,
            'plots_to_show': self._get_plots_to_show(warn=False),
            'runs_config': runs_config
        }
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")
            
    def load_config(self):
        if not self.config_path.exists(): return
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            if 'settings' in config:
                self.settings.update(config['settings'])
                
            if 'plots_to_show' in config:
                for k, var in self.plot_vars.items():
                    var.set(k in config['plots_to_show'])
                    
            if 'runs_config' in config:
                self.saved_runs_config = config['runs_config']
                # 自動載入之前的檔案
                self.add_files(list(self.saved_runs_config.keys()))
                
        except Exception as e:
            print(f"Failed to load config: {e}")

    def on_close(self):
        self.save_config()
        self.destroy()
        
    def preview_plot(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "請先在列表中選擇一個 Run 來預覽。")
            return
            
        plots = self._get_plots_to_show()
        if not plots: return
        
        item_id = selected[0]
        data = self.runs_data[item_id]
        
        fig = plot_run_summary(
            data['run_data'], data['run_name'], plots,
            t_plot_start_event=data['start'], t_plot_end_event=data['end'],
            settings=self.settings
        )
        if fig is not None:
            plt.show()
        else:
            messagebox.showinfo("提示", "該 Run 沒有資料或範圍錯誤。")
            
    def run_batch_save(self):
        items = self.tree.get_children()
        if not items:
            messagebox.showwarning("警告", "列表為空，請先加入 HDF5 檔案。")
            return
            
        plots = self._get_plots_to_show()
        if not plots: return
        
        out_dir = Path(r"C:\experiment\labquake_explorer_re\export h5\batch_summary_plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        self.lbl_status.config(text="處理中，請稍後...")
        self.update()
        
        success_count = 0
        for item_id in items:
            data = self.runs_data[item_id]
            fig = plot_run_summary(
                data['run_data'], data['run_name'], plots,
                t_plot_start_event=data['start'], t_plot_end_event=data['end'],
                settings=self.settings
            )
            if fig is not None:
                file_stem = Path(data['file_path']).stem
                save_name = f"{file_stem}_{data['run_name']}.png"
                save_path = out_dir / save_name
                fig.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
                success_count += 1
                
        self.lbl_status.config(text="處理完成！")
        messagebox.showinfo("完成", f"已成功儲存 {success_count} 張圖表至：\n{out_dir}")

if __name__ == "__main__":
    app = BatchPlotGUI()
    app.mainloop()
