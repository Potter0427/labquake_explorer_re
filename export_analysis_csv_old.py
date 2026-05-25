"""
從 HDF5 檔案中匯出 Drop Analysis 結果為 CSV。
自動偵測當前目錄下的所有 .h5 檔案，每個實驗匯出一個 CSV（所有 Run 合併）。

用法：
    python export_analysis_csv.py
"""
import glob
import h5py
import numpy as np
import csv
from pathlib import Path


def export_h5_to_csv(h5_path: str):
    h5_path = Path(h5_path)

    with h5py.File(h5_path, 'r') as f:
        runs_group = f['runs']
        run_keys = sorted(runs_group.keys(), key=lambda x: int(x))

        # 定義匯出欄位順序（加入 k）
        ordered_keys = ['trigger_time', 'k', 'delta_tau',
                        'delta_E1', 'delta_E2', 'delta_E3', 'delta_E4', 'delta_E5',
                        'delta_lvdt', 'D_Push', 'D_max', 'D_E3']

        all_rows = []

        for run_key in run_keys:
            run = runs_group[run_key]
            run_name = run['name'][()].decode() if 'name' in run else f"run{run_key}"

            if 'events' not in run:
                print(f"  [{run_name}] 沒有 events 資料，跳過", flush=True)
                continue

            events_grp = run['events']
            event_keys = sorted([k for k in events_grp.keys() if k.isdigit() and int(k) > 0], key=int)
            n_events = len(event_keys)

            if n_events == 0:
                print(f"  [{run_name}] 沒有實體事件結果，跳過", flush=True)
                continue

            export_keys = ordered_keys

            # 讀取共用 skip_events
            skipped_list = []
            if 'skip_events' in run:
                se_obj = run['skip_events']
                if isinstance(se_obj, h5py.Group):
                    skipped_list = [int(se_obj[k][()]) for k in sorted(se_obj.keys(), key=int)]
                else:
                    skipped_list = se_obj[()].tolist() if hasattr(se_obj[()], 'tolist') else list(se_obj[()])

            for ev_key in event_keys:
                ev_idx = int(ev_key)
                ev_grp = events_grp[ev_key]
                row = [run_name, ev_idx]
                
                for k in export_keys:
                    val = np.nan
                    try:
                        if k == 'trigger_time':
                            if 'event_time' in ev_grp: val = ev_grp['event_time'][()]
                            elif 'trigger_time' in ev_grp: val = ev_grp['trigger_time'][()]
                        elif k == 'k':
                            if 'k' in ev_grp and 'value' in ev_grp['k']: val = ev_grp['k']['value'][()]
                            elif 'k' in ev_grp and 'k' in ev_grp['k']: val = ev_grp['k']['k'][()]
                        elif k == 'delta_tau':
                            if 'tau' in ev_grp and 'value' in ev_grp['tau']: val = ev_grp['tau']['value'][()]
                        elif k.startswith('delta_E'):
                            if 'delta' in ev_grp:
                                ch_val = f"{k.split('_')[1]}_value"
                                if ch_val in ev_grp['delta']: val = ev_grp['delta'][ch_val][()]
                        elif k == 'delta_lvdt':
                            if 'lvdt' in ev_grp and 'value' in ev_grp['lvdt']: val = ev_grp['lvdt']['value'][()]
                        elif k in ['D_Push', 'D_max', 'D_E3']:
                            if k in ev_grp: val = ev_grp[k][()]
                    except Exception:
                        pass
                        
                    row.append(f"{val:.6g}" if not np.isnan(val) else "")
                    
                is_skipped = (ev_idx in skipped_list)
                row.append("YES" if is_skipped else "")
                all_rows.append((row, export_keys))

        if not all_rows:
            print(f"  {h5_path.name} 沒有任何分析結果", flush=True)
            return

        # 用第一筆資料的欄位作為 header
        final_keys = all_rows[0][1]
        out_path = h5_path.parent / f"{h5_path.stem}.csv"

        with open(out_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Run', 'Event'] + final_keys + ['skipped'])
            for row_data, _ in all_rows:
                writer.writerow(row_data)

        print(f"  匯出完成 -> {out_path}", flush=True)


def batch_export():
    h5_files = sorted(glob.glob("*.h5"))

    if not h5_files:
        print("當前目錄下找不到任何 .h5 檔案", flush=True)
        return

    print(f"找到 {len(h5_files)} 個 .h5 檔案：{h5_files}", flush=True)

    for h5_file in h5_files:
        print(f"\n>>> 處理: {h5_file}", flush=True)
        try:
            export_h5_to_csv(h5_file)
        except Exception as e:
            print(f"  [失敗] {h5_file}: {e}", flush=True)


if __name__ == '__main__':
    batch_export()
