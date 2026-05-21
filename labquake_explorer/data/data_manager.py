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
                                # 如果這個編號的資料夾存在，就讀取它
                                item = group[matching_keys[0]]
                                if isinstance(item, h5py.Group):
                                    array_data.append(load_group(item, matching_keys[0]))
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
                    elif isinstance(value, (int, float, bool, np.number)):
                        if not _needs_update(group, key, value):
                            return
                        if key in group:
                            del group[key]
                        group.create_dataset(key, data=value)
                    else:
                        try:
                            if key in group:
                                del group[key]
                            group.create_dataset(key, data=np.array(value), compression="gzip")
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Could not save {key}: {e}")

                for k, v in self.data.items():
                    save_item(f, k, v)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

    def fast_save_analysis(self, run_idx: int, analysis: dict, group_name: str = 'analysis') -> None:
        """
        Directly update the analysis group in the HDF5 file without rewriting the whole file.
        Also updates the in-memory data structure.
        """
        # 1. Update in-memory
        if not self.data or 'runs' not in self.data:
            return
        
        try:
            self.data['runs'][run_idx][group_name] = analysis
        except (IndexError, KeyError):
            return

        # 2. Update HDF5 file directly if it exists
        if not self.data_path or not self.data_path.exists():
            return
        
        if self.data_path.suffix.lower() not in ['.h5', '.hdf5']:
            return

        try:
            with h5py.File(self.data_path, 'r+') as f:
                # Resolve the path to the analysis group
                # Structure: runs/0/analysis, runs/1/analysis, ...
                target_path = f"runs/{run_idx}/{group_name}"
                
                # Delete old group if exists
                if target_path in f:
                    del f[target_path]
                
                # Create and save new analysis group
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

                # The analysis is a dict, so we want its contents inside target_path
                analysis_group = f.create_group(target_path)
                for k, v in analysis.items():
                    _save_recursive(analysis_group, k, v)
                    
                print(f"Fast saved {group_name} for run {run_idx} to {self.data_path}")
        except Exception as e:
            print(f"Warning: Fast save failed: {e}")
            # Non-critical, as in-memory data is still updated

    # ------------------------------------------------------------------
    # Shared skip-events list  (runs/{run_idx}/skip_events)
    # ------------------------------------------------------------------

    def get_run_skip_events(self, run_idx: int) -> list:
        """Return the shared skip-events list for a run (list of int).

        Reads from the run-level 'skip_events' key first.  If that key has
        never been set, falls back to merging legacy per-analysis lists from
        analysis/config/skip_events and k_analysis/config/skip_events so that
        existing HDF5 files work seamlessly without a manual migration step.
        """
        try:
            run = self.data['runs'][run_idx]
            se = run.get('skip_events', None)
            if se is not None:
                if hasattr(se, 'tolist'):
                    se = se.tolist()
                return [int(x) for x in se]

            # --- Legacy fallback: merge from per-analysis configs ---
            merged = set()
            for group_key in ('analysis', 'k_analysis'):
                grp = run.get(group_key)
                if isinstance(grp, dict):
                    cfg = grp.get('config')
                    if isinstance(cfg, dict):
                        legacy = cfg.get('skip_events', [])
                        if legacy is not None:
                            if hasattr(legacy, 'tolist'):
                                legacy = legacy.tolist()
                            merged.update(int(x) for x in legacy)
            result = sorted(merged)
            if result:
                # Persist the merged list so next read hits the fast path
                run['skip_events'] = result
            return result
        except (IndexError, KeyError, TypeError):
            return []


    def save_run_skip_events(self, run_idx: int, skip_list: list) -> None:
        """Persist the shared skip-events list for a run.

        Updates both the in-memory dict and the HDF5 dataset at
        runs/{run_idx}/skip_events so the list survives an app restart.
        """
        clean = sorted({int(x) for x in skip_list})

        # 1. Update in-memory
        try:
            self.data['runs'][run_idx]['skip_events'] = clean
        except (IndexError, KeyError, TypeError):
            return

        # 2. Update HDF5
        if not self.data_path or not self.data_path.exists():
            return
        if self.data_path.suffix.lower() not in ['.h5', '.hdf5']:
            return

        try:
            with h5py.File(self.data_path, 'r+') as f:
                dataset_path = f"runs/{run_idx}/skip_events"
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