"""Data management and processing for Labquake Explorer"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import h5py
from labquake_explorer.data.event_processor import EventProcessor


class DataManager:
    def __init__(self):
        self.data: Optional[Dict[str, Any]] = None
        self.data_path: Optional[Path] = None
        self.data: Optional[Dict[str, Any]] = None
        self.event_processor = EventProcessor()

    def load_file(self, path: Path) -> None:
        """Load data from a file"""
        self.data_path = path
        self.event_processor.set_data_path(path)  # Set the data path in EventProcessor

        if path.suffix.lower() == '.npz':
            self._load_npz(path)
        elif path.suffix.lower() in ['.h5', '.hdf5']:
            self._load_hdf5(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
            
        self._migrate_legacy_data()

    def _load_npz(self, path: Path) -> None:
        """Load data from NPZ file"""
        with np.load(path, allow_pickle=True) as data:
            self.data = data["experiment"][()]

    def _load_hdf5(self, path: Path) -> None:
        with h5py.File(path, 'r') as h5data:
            def load_dataset(item):
                try:
                    data = np.array(item)
                    if data.dtype.kind == 'S' or data.dtype.kind == 'O':
                        if isinstance(data.flat[0], bytes):
                            if data.size == 1:
                                return data.flat[0].decode('utf-8')
                            return [x.decode('utf-8') for x in data.flat]
                    if data.size == 1:  # Convert length-1 arrays to numbers
                        return data.item()
                    return data
                except Exception as exc:
                    print(f"Dataset loading error: {str(exc)}")
                    return None
                
            def load_group(group, name=""):
                result = {}
                keys = list(group.keys())
                
                # Check if this group should be treated as a list (only for sequential numeric items)
                # We EXCLUDE groups like 'per_event_windows' or 'config' to avoid the "List Bug"
                if all(k.isdigit() for k in keys) and name not in ['per_event_windows', 'config', 'analysis']:
                    try:
                        num_keys = max(int(k) for k in keys) + 1
                        
                        # 建立一個陣列來裝資料
                        array_data = []
                        for i in range(num_keys):
                            # Find if any key matches the integer value i
                            matching_keys = [k for k in keys if int(k) == i]
                            if matching_keys:
                                # 若同時存在 '1' 與 '001' 等重複 index，選擇包含最多 key 的那一個
                                best_key = matching_keys[0]
                                if len(matching_keys) > 1:
                                    best_key = max(matching_keys, key=lambda k: len(group[k].keys()) if isinstance(group[k], h5py.Group) else 0)
                                    
                                item = group[best_key]
                                if isinstance(item, h5py.Group):
                                    array_data.append(load_group(item, best_key))
                                else:
                                    array_data.append(load_dataset(item))
                            else:
                                # 如果不存在 (例如你先轉了 run4(3)，但 0,1,2 還沒跑)
                                # 給它一個空的佔位符，防止崩潰
                                array_data.append({"name": "(No Data)"})
                                
                        return array_data
                    except ValueError:
                        pass
                    
                for key in keys:
                    try:
                        item = group[key]
                        if isinstance(item, h5py.Group):
                            result[key] = load_group(item, key)
                        else:
                            result[key] = load_dataset(item)
                    except Exception as exc:
                        print(f"Error loading {key}: {str(exc)}")
                
                return result
            self.data = load_group(h5data)

    def _migrate_legacy_data(self) -> None:
        """Migrates legacy 'analysis' and 'k_analysis' data structures to the new event-centric 'config' + 'drop' / 'k' structure."""
        if not self.data or 'runs' not in self.data:
            return

        migrated = False
        runs = self.data['runs']
        run_items = [runs[k] for k in sorted(runs.keys())] if isinstance(runs, dict) else runs

        for run_idx, run in enumerate(run_items):
            if not isinstance(run, dict):
                continue
                
            config = run.setdefault('config', {})
            
            # --- 1. Migrate skip_events ---
            legacy_skip = []
            for k in ['skip_events', 'analysis_config', 'k_analysis_config']:
                if k in run:
                    if k == 'skip_events':
                        legacy_skip.extend(run[k])
                    elif isinstance(run[k], dict) and 'skip_events' in run[k]:
                        legacy_skip.extend(run[k]['skip_events'])
            
            for grp in ['analysis', 'k_analysis']:
                if grp in run and isinstance(run[grp], dict) and 'config' in run[grp]:
                    if 'skip_events' in run[grp]['config']:
                        se = run[grp]['config']['skip_events']
                        if hasattr(se, 'tolist'): se = se.tolist()
                        if isinstance(se, (list, tuple, np.ndarray)):
                            legacy_skip.extend(se)

            if legacy_skip:
                current_skip = config.get('skip_events', [])
                merged_skip = sorted(set([int(x) for x in current_skip] + [int(x) for x in legacy_skip]))
                config['skip_events'] = merged_skip
                
            # --- 2. Migrate global configs ---
            for k in ['analysis_config', 'k_analysis_config']:
                if k in run and isinstance(run[k], dict):
                    for key, val in run[k].items():
                        if key != 'skip_events':
                            config[key] = val
                    run.pop(k)
                    migrated = True

            for grp in ['analysis', 'k_analysis']:
                if grp in run and isinstance(run[grp], dict) and 'config' in run[grp]:
                    grp_cfg = run[grp]['config']
                    if isinstance(grp_cfg, dict):
                        for key, val in grp_cfg.items():
                            if key not in ['skip_events', 'per_event_windows'] and not key.endswith('_pts'):
                                if key not in config:
                                    config[key] = val
            
            # --- 3. Migrate results into events (Nested Structure) ---
            events = run.get('events', [])
            if not isinstance(events, list):
                continue

            # (A) Legacy 'analysis' group to events
            if 'analysis' in run and isinstance(run['analysis'], dict):
                results = run['analysis'].get('results', {})
                config_pew = run['analysis'].get('config', {}).get('per_event_windows', {})
                
                if isinstance(results, dict) and 'delta_tau' in results and isinstance(results['delta_tau'], np.ndarray):
                    n_results = len(results['delta_tau'])
                    for i in range(min(len(events), n_results)):
                        if not isinstance(events[i], dict): continue
                        ev = events[i]
                        
                        tau_pts = None
                        slip_pts = None
                        lvdt_pts = None
                        
                        if i in config_pew or str(i) in config_pew:
                            pew = config_pew.get(i) or config_pew.get(str(i))
                            if isinstance(pew, dict):
                                if 'pre_win' in pew and 'post_win' in pew:
                                    pts = [pew['pre_win'][0], pew['pre_win'][1], pew['post_win'][0], pew['post_win'][1]]
                                    tau_pts = slip_pts = lvdt_pts = pts
                                else:
                                    tau_pts = pew.get('tau_pts')
                                    slip_pts = pew.get('slip_pts')
                                    lvdt_pts = pew.get('lvdt_pts')
                                    
                        tau_pts = tau_pts or [np.nan]*4
                        slip_pts = slip_pts or [np.nan]*4
                        lvdt_pts = lvdt_pts or [np.nan]*4
                        
                        ev.setdefault('tau', {})
                        ev['tau']['value'] = results.get('delta_tau', [np.nan]*n_results)[i]
                        ev['tau']['pre_start'], ev['tau']['pre_end'], ev['tau']['post_start'], ev['tau']['post_end'] = tau_pts[0:4]
                        
                        ev.setdefault('delta', {})
                        ev['delta']['pre_start'], ev['delta']['pre_end'], ev['delta']['post_start'], ev['delta']['post_end'] = slip_pts[0:4]
                        for ch in range(1, 10):
                            k_name = f'delta_E{ch}'
                            if k_name in results:
                                ev['delta'][f'E{ch}_value'] = results[k_name][i]
                        
                        ev.setdefault('lvdt', {})
                        ev['lvdt']['value'] = results.get('delta_lvdt', [np.nan]*n_results)[i]
                        ev['lvdt']['pre_start'], ev['lvdt']['pre_end'], ev['lvdt']['post_start'], ev['lvdt']['post_end'] = lvdt_pts[0:4]
                        
                        for d_val in ['D_Push', 'D_max', 'D_E3', 'skipped']:
                            if d_val in results:
                                try:
                                    ev[d_val] = results[d_val][i]
                                except (IndexError, TypeError):
                                    pass
                                
                run.pop('analysis')
                migrated = True

            # (B) Legacy 'k_analysis' group to events
            if 'k_analysis' in run and isinstance(run['k_analysis'], dict):
                results = run['k_analysis'].get('results', {})
                if isinstance(results, dict) and 'k' in results and isinstance(results['k'], np.ndarray):
                    n_results = len(results['k'])
                    for i in range(min(len(events), n_results)):
                        if not isinstance(events[i], dict): continue
                        ev = events[i]
                        ev.setdefault('k', {})
                        if 'k' in results and i < len(results['k']):
                            ev['k']['value'] = results['k'][i]
                        if 'k_pre_start' in results and i < len(results['k_pre_start']):
                            ev['k']['start'] = results['k_pre_start'][i]
                        if 'k_pre_end' in results and i < len(results['k_pre_end']):
                            ev['k']['end'] = results['k_pre_end'][i]
                run.pop('k_analysis')
                migrated = True
                
            # (C) Intermediate 'drop' / 'k' dictionaries to new nested structure
            for i, ev in enumerate(events):
                if not isinstance(ev, dict): continue
                if 'drop' in ev:
                    d = ev.pop('drop')
                    
                    tau_pts = d.get('tau_pts', [np.nan]*4)
                    slip_pts = d.get('slip_pts', [np.nan]*4)
                    lvdt_pts = d.get('lvdt_pts', [np.nan]*4)
                    
                    ev.setdefault('tau', {})
                    ev['tau']['value'] = d.get('delta_tau', np.nan)
                    ev['tau']['pre_start'], ev['tau']['pre_end'], ev['tau']['post_start'], ev['tau']['post_end'] = tau_pts[0:4]
                    
                    ev.setdefault('delta', {})
                    ev['delta']['pre_start'], ev['delta']['pre_end'], ev['delta']['post_start'], ev['delta']['post_end'] = slip_pts[0:4]
                    for ch in range(1, 10):
                        if f'delta_E{ch}' in d:
                            ev['delta'][f'E{ch}_value'] = d[f'delta_E{ch}']
                    
                    ev.setdefault('lvdt', {})
                    ev['lvdt']['value'] = d.get('delta_lvdt', np.nan)
                    ev['lvdt']['pre_start'], ev['lvdt']['pre_end'], ev['lvdt']['post_start'], ev['lvdt']['post_end'] = lvdt_pts[0:4]
                    
                    for k_val in ['D_Push', 'D_max', 'D_E3', 'skipped']:
                        if k_val in d:
                            ev[k_val] = d[k_val]
                    if 'trigger_time' in d:
                        ev['event_time'] = d['trigger_time']
                    migrated = True
                    
                if 'k' in ev and isinstance(ev['k'], dict):
                    k_d = ev['k']
                    # migrate old k format if it hasn't been migrated yet (doesn't have 'value' but has 'k')
                    if 'value' not in k_d and 'k' in k_d:
                        new_k = {'value': k_d.get('k', np.nan)}
                        if 'k_pre_start' in k_d: new_k['start'] = k_d['k_pre_start']
                        if 'k_pre_end' in k_d: new_k['end'] = k_d['k_pre_end']
                        ev['k'] = new_k
                        migrated = True

            # (D) Rename 'trigger_time' to 'event_time' at the root of events
            for i, ev in enumerate(events):
                if not isinstance(ev, dict): continue
                if 'trigger_time' in ev:
                    if 'event_time' not in ev:
                        ev['event_time'] = ev.pop('trigger_time')
                    else:
                        ev.pop('trigger_time')
                    migrated = True

            # (E) Nest configs into tau, lvdt, k
            for i, ev in enumerate(events):
                if not isinstance(ev, dict): continue
                # tau config
                if 'tau_smooth_w' in ev:
                    ev.setdefault('tau', {})['smooth_w'] = ev.pop('tau_smooth_w')
                    migrated = True
                # lvdt config
                if 'lvdt_smooth_w' in ev:
                    ev.setdefault('lvdt', {})['smooth_w'] = ev.pop('lvdt_smooth_w')
                    migrated = True
                # k config
                k_keys = ['k_smooth_w', 'k_highpass_freq', 'k_lowpass_freq', 'k_use_ransac']
                for kk in k_keys:
                    if kk in ev:
                        ev.setdefault('k', {})[kk.replace('k_', '')] = ev.pop(kk)
                        migrated = True

        if migrated:
            print("Legacy data structure migrated to new nested timeline structure.")
            try:
                self.save_file(self.data_path)
                if self.data_path.suffix.lower() in ['.h5', '.hdf5']:
                    import h5py
                    with h5py.File(self.data_path, 'r+') as f:
                        run_keys = list(runs.keys()) if isinstance(runs, dict) else [str(i) for i in range(len(runs))]
                        for run_key in run_keys:
                            for old_grp in ['analysis', 'k_analysis', 'analysis_config', 'k_analysis_config', 'skip_events']:
                                if old_grp in f[f'runs/{run_key}']:
                                    del f[f'runs/{run_key}/{old_grp}']
                            
                            # Clean up old root-level keys in events
                            if f'runs/{run_key}/events' in f:
                                ev_grp = f[f'runs/{run_key}/events']
                                for ev_idx in ev_grp.keys():
                                    for old_key in ['tau_smooth_w', 'lvdt_smooth_w', 'k_smooth_w', 'k_highpass_freq', 'k_lowpass_freq', 'k_use_ransac', 'trigger_time']:
                                        if old_key in ev_grp[ev_idx]:
                                            del ev_grp[ev_idx][old_key]
                                    # also remove intermediate drop
                                    if 'drop' in ev_grp[ev_idx]:
                                        del ev_grp[ev_idx]['drop']
                                    # old k datasets inside ev_grp['k']
                                    if ev_grp and 'k' in ev_grp and isinstance(ev_grp['k'], h5py.Group):
                                        if 'k_pre_start' in ev_grp['k']:
                                            del ev_grp['k']['k_pre_start']
                                        if 'k_pre_end' in ev_grp['k']:
                                            del ev_grp['k']['k_pre_end']
            except Exception as e:
                print(f"Failed to persist migrated data: {e}")

    def fast_save_analysis(self, run_idx: int, analysis_data: dict) -> bool:
        """
        [NEW] Fast-save feature: Directly updates the analysis node in the HDF5 file 
        using 'r+' mode, avoiding a full file rewrite.
        """
        if not self.data_path or not self.data_path.exists():
            return False
            
        try:
            import h5py
            with h5py.File(self.data_path, 'r+') as f:
                # Target group: e.g., /runs/0/analysis
                analysis_path = f"runs/{run_idx}/analysis"
                
                if analysis_path in f:
                    del f[analysis_path]
                
                analysis_group = f.create_group(analysis_path)
                
                def recursive_save(group, d):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            sub = group.create_group(k)
                            recursive_save(sub, v)
                        elif isinstance(v, np.ndarray):
                            group.create_dataset(k, data=v, compression="gzip")
                        elif isinstance(v, (list, tuple)):
                            group.create_dataset(k, data=np.array(v))
                        else:
                            # Scalar values
                            group.create_dataset(k, data=v)
                
                recursive_save(analysis_group, analysis_data)
                return True
        except Exception as e:
            print(f"Fast save failed: {e}")
            return False

    def save_file(self, path: Path) -> None:
        if not self.data:
            raise ValueError("No data to save")
    
        if path.suffix.lower() == '.npz':
            np.savez(path, experiment=self.data)
        elif path.suffix.lower() in ['.h5', '.hdf5']:
            # Use 'a' (append) mode: create file if missing, otherwise open for read/write.
            # This avoids rewriting the entire multi-GB file from scratch.
            mode = 'a' if path.exists() else 'w'
            with h5py.File(path, mode) as f:
                def _needs_update(group, key, value):
                    """Check if a dataset needs to be rewritten."""
                    if key not in group:
                        return True
                    existing = group[key]
                    if isinstance(existing, h5py.Group):
                        # Groups are always traversed recursively, not compared here
                        return False
                    # Compare dataset contents
                    try:
                        if isinstance(value, np.ndarray):
                            old = existing[()]
                            if old.shape == value.shape and old.dtype == value.dtype:
                                if np.array_equal(old, value):
                                    return False
                        elif isinstance(value, (int, float, bool, np.number)):
                            if existing[()] == value:
                                return False
                        elif isinstance(value, str):
                            old = existing[()]
                            if isinstance(old, bytes):
                                old = old.decode()
                            if old == value:
                                return False
                    except Exception:
                        pass
                    return True

                def save_item(group, key, value):
                    if isinstance(value, dict):
                        subgroup = group[key] if key in group and isinstance(group[key], h5py.Group) else None
                        if subgroup is None:
                            if key in group:
                                del group[key]
                            subgroup = group.create_group(key)
                        for k, v in value.items():
                            save_item(subgroup, k, v)
                    elif isinstance(value, np.ndarray):
                        if not _needs_update(group, key, value):
                            return
                        if key in group:
                            del group[key]
                        if value.ndim == 2:
                            group.create_dataset(key, data=value, compression="gzip")
                        else:
                            arr = np.array(value)
                            if arr.dtype == object:
                                if all(isinstance(x, (int, np.integer)) for x in arr.flat):
                                    arr = arr.astype(np.int64)
                                elif all(isinstance(x, (float, np.floating)) for x in arr.flat):
                                    arr = arr.astype(np.float64)
                                elif all(isinstance(x, bool) for x in arr.flat):
                                    arr = arr.astype(np.int8)
                                else:
                                    arr = np.array([str(x).encode() for x in arr.flat]).reshape(arr.shape)
                            elif arr.dtype.kind == 'U':
                                arr = np.array([x.encode() for x in arr.flat]).reshape(arr.shape)
                            group.create_dataset(key, data=arr, compression="gzip")
                    elif isinstance(value, (list, tuple)):
                        arr = np.array(value)
                        if arr.ndim == 2:
                            if not _needs_update(group, key, arr):
                                return
                            if key in group:
                                del group[key]
                            group.create_dataset(key, data=arr, compression="gzip")
                        else:
                            subgroup = group[key] if key in group and isinstance(group[key], h5py.Group) else None
                            if subgroup is None:
                                if key in group:
                                    del group[key]
                                subgroup = group.create_group(key)
                            for i, item in enumerate(value):
                                save_item(subgroup, str(i), item)
                    elif isinstance(value, str):
                        if not _needs_update(group, key, value):
                            return
                        if key in group:
                            del group[key]
                        group.create_dataset(key, data=value.encode())
                    elif isinstance(value, (int, float, bool, np.number, np.bool_)):
                        if not _needs_update(group, key, value):
                            return
                        if key in group:
                            del group[key]
                        group.create_dataset(key, data=value)
                    else:
                        try:
                            if key in group:
                                del group[key]
                            
                            val_arr = np.array(value)
                            if val_arr.ndim == 0:  # Scalar array fallback
                                group.create_dataset(key, data=val_arr)
                            else:
                                group.create_dataset(key, data=val_arr, compression="gzip")
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Could not save {key}: {e}")

                for k, v in self.data.items():
                    save_item(f, k, v)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

    def fast_save_event_analysis(self, run_idx: int, event_idx: int, category: str, data: dict) -> None:
        """
        Directly update a single event's category (e.g., 'drop' or 'k') in the HDF5 file 
        without rewriting the whole file. Also updates the in-memory data structure.
        """
        # 1. Update in-memory
        if not self.data or 'runs' not in self.data:
            return
        
        try:
            run = self.data['runs'][run_idx]
            if 'events' not in run:
                run['events'] = [{"name": "(No Data)"}]
            
            # Ensure the list is long enough (1-based indexing)
            while len(run['events']) <= event_idx:
                run['events'].append({})
            
            # Save to memory
            if category:
                run['events'][event_idx][category] = data
            else:
                run['events'][event_idx].update(data)
                
        except (IndexError, KeyError):
            return

        # 2. Update HDF5 file directly if it exists
        if not self.data_path or not self.data_path.exists():
            return
        
        if self.data_path.suffix.lower() not in ['.h5', '.hdf5']:
            return

        try:
            with h5py.File(self.data_path, 'r+') as f:
                events_path = f"runs/{run_idx}/events"
                if events_path not in f:
                    events_group = f.create_group(events_path)
                    # Create 0th placeholder
                    idx_0 = events_group.create_group('0')
                    idx_0.create_dataset('name', data=b'(No Data)')
                else:
                    events_group = f[events_path]
                
                event_path = f"{events_path}/{event_idx}"
                if event_path not in f:
                    event_group = events_group.create_group(str(event_idx))
                else:
                    event_group = f[event_path]
                
                if category:
                    target_path = f"{event_path}/{category}"
                    if target_path in f:
                        del f[target_path]
                    target_group = event_group.create_group(category)
                else:
                    # if category is empty, we are writing directly into the event_group (careful!)
                    target_group = event_group
                    # clear old contents except groups maybe? Or assume it's clean if category is used.
                    for k in data.keys():
                        if k in target_group:
                            del target_group[k]
                def _save_recursive(group, key, value):
                    if isinstance(value, dict):
                        sub = group.create_group(key)
                        for k, v in value.items():
                            _save_recursive(sub, k, v)
                    elif isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value, compression="gzip")
                    elif isinstance(value, (int, float, bool, np.number)):
                        group.create_dataset(key, data=value)
                    elif isinstance(value, str):
                        group.create_dataset(key, data=value.encode())
                    elif isinstance(value, (list, tuple)):
                        arr = np.array(value)
                        if arr.ndim >= 1 and arr.dtype.kind in 'fiu': # numbers
                             group.create_dataset(key, data=arr, compression="gzip")
                        else:
                            # Save as numbered subgroup (matching save_file logic)
                            sub = group.create_group(key)
                            for i, item in enumerate(value):
                                _save_recursive(sub, str(i), item)
                    else:
                        try:
                            group.create_dataset(key, data=np.array(value), compression="gzip")
                        except:
                            pass

                # The analysis is a dict, so we want its contents inside target_group
                for k, v in data.items():
                    _save_recursive(target_group, k, v)
                    
                print(f"Fast saved event {event_idx} {category} for run {run_idx} to {self.data_path}")
        except Exception as e:
            print(f"Warning: Fast save failed: {e}")
            # Non-critical, as in-memory data is still updated

    # ------------------------------------------------------------------
    # Shared skip-events list  (runs/{run_idx}/config/skip_events)
    # ------------------------------------------------------------------

    def get_run_skip_events(self, run_idx: int) -> list:
        """Return the shared skip-events list for a run (list of int)."""
        try:
            run = self.data['runs'][run_idx]
            config = run.get('config', {})
            se = config.get('skip_events', None)
            if se is not None:
                if hasattr(se, 'tolist'):
                    se = se.tolist()
                return [int(x) for x in se]
            return []
        except (IndexError, KeyError, TypeError):
            return []

    def save_run_skip_events(self, run_idx: int, skip_list: list) -> None:
        """Persist the shared skip-events list for a run.

        Updates both the in-memory dict and the HDF5 dataset at
        runs/{run_idx}/config/skip_events so the list survives an app restart.
        """
        clean = sorted({int(x) for x in skip_list})

        # 1. Update in-memory
        try:
            config = self.data['runs'][run_idx].setdefault('config', {})
            config['skip_events'] = clean
        except (IndexError, KeyError, TypeError):
            return

        # 2. Update HDF5
        if not self.data_path or not self.data_path.exists():
            return
        if self.data_path.suffix.lower() not in ['.h5', '.hdf5']:
            return

        try:
            import h5py
            with h5py.File(self.data_path, 'r+') as f:
                config_path = f"runs/{run_idx}/config"
                if config_path not in f:
                    f.create_group(config_path)
                dataset_path = f"{config_path}/skip_events"
                if dataset_path in f:
                    del f[dataset_path]
                arr = np.array(clean, dtype=np.int64)
                f.create_dataset(dataset_path, data=arr)
            print(f"Saved shared skip_events for run {run_idx}: {clean}")
        except Exception as e:
            print(f"Warning: could not persist skip_events to HDF5: {e}")


    def extract_events(self, indices: List[int], window_size: float) -> List[Dict]:
        """Extract events using provided indices"""
        if not self.data:
            raise ValueError("No data loaded")
            
        events = []
        for idx in indices:
            event = self._extract_single_event(idx, window_size)
            events.append(event)
        return events

    def _extract_single_event(self, idx: int, window: float) -> Dict:
        """Extract single event data"""
        event_time = self.data["time"][idx]
        
        idx_beg = np.argmin(np.abs(event_time - window - self.data["time"]))
        idx_end = np.argmin(np.abs(event_time + window - self.data["time"]))
        idx_event = range(idx_beg, idx_end + 1)
        
        event = {
            'event_time': event_time,
            'time': self.data['time'][idx_event]
        }
        
        for key, value in self.data.items():
            if key != "events" and isinstance(value, (np.ndarray, list)):
                try:
                    event[key] = value[idx_event]
                except IndexError:
                    event[key] = value[idx]
                    
        if 'strain' in self.data:
            event['strain'] = self._process_strain_data(event_time, window)
            
        return event

    def get_data(self, path: str) -> Any:
        """Get data at specified path"""
        if not path:  # Handle empty path
            return current
        if not self.data:
            raise ValueError("No data loaded")
        
        path = path.replace('\\', '/') #windows反斜線問題

        parts = [p for p in path.split('/') if p]  # Split and filter out empty parts
        current = self.data
        for key in parts:
            if key.startswith('[') and key.endswith(']'):
                key = int(key[1:-1])  # Convert list index to integer
            current = current[key]
        return current

    def set_data(self, path: str, value: Any, add_key: bool = False) -> None:
        """Set data at specified path"""
        if not self.data:
            raise ValueError("No data loaded")
            
        path = path.replace('\\', '/')
        parts = path.split('/')
        current = self.data
        
        for i, part in enumerate(parts[:-1]):
            if part[0] == '[' and part[-1] == ']':
                part = int(part[1:-1])
            current = current[part]
            
        last_key = parts[-1]
        if last_key[0] == '[' and last_key[-1] == ']':
            last_key = int(last_key[1:-1])
        current[last_key] = value

    def delete_data(self, path: str) -> None:
        """Delete data at specified path
        
        Args:
            path: Path to the data to delete (e.g. 'runs/[0]/events')
            
        Raises:
            ValueError: If no data is loaded or path is invalid
            KeyError: If path does not exist
        """
        if not self.data:
            raise ValueError("No data loaded")
            
        # Handle root deletion
        if path == "":
            self.data = None
            return
            
        path = path.replace('\\', '/')
        parts = path.split('/')
        current = self.data
        
        # Navigate to parent of item to delete
        for part in parts[:-1]:
            if part[0] == '[' and part[-1] == ']':
                # Handle array index
                idx = int(part[1:-1])
                if not isinstance(current, (list, tuple)):
                    raise ValueError(f"Cannot index non-sequence with {part}")
                if idx >= len(current):
                    raise IndexError(f"Index {idx} out of range for sequence of length {len(current)}")
                current = current[idx]
            else:
                # Handle dictionary key
                if not isinstance(current, dict):
                    raise ValueError(f"Cannot get key '{part}' from non-dictionary")
                if part not in current:
                    raise KeyError(f"Key '{part}' not found")
                current = current[part]
        
        # Delete the item
        last_part = parts[-1]
        if last_part[0] == '[' and last_part[-1] == ']':
            # Handle array index deletion
            idx = int(last_part[1:-1])
            if not isinstance(current, (list, tuple)):
                raise ValueError(f"Cannot delete index from non-sequence")
            if idx >= len(current):
                raise IndexError(f"Index {idx} out of range")
            current.pop(idx)
        else:
            # Handle dictionary key deletion
            if not isinstance(current, dict):
                raise ValueError(f"Cannot delete key from non-dictionary")
            if last_part not in current:
                raise KeyError(f"Key '{last_part}' not found")
            current.pop(last_part)