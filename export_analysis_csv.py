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

        # 定義匯出欄位順序
        ordered_keys = ['trigger_time', 'delta_tau',
                        'delta_E1', 'delta_E2', 'delta_E3', 'delta_E4', 'delta_E5',
                        'delta_lvdt', 'D_Push', 'D_max', 'D_E3']

        all_rows = []

        for run_key in run_keys:
            run = runs_group[run_key]
            run_name = run['name'][()].decode() if 'name' in run else f"run{run_key}"

            if 'analysis' not in run or 'results' not in run['analysis']:
                print(f"  [{run_name}] 沒有 analysis/results，跳過", flush=True)
                continue

            results = run['analysis']['results']
            available_keys = list(results.keys())
            export_keys = [k for k in ordered_keys if k in available_keys]

            # 讀取資料
            data = {}
            n_events = 0
            for k in export_keys:
                arr = results[k][()]
                data[k] = arr
                n_events = max(n_events, len(arr))

            skipped = results['skipped'][()].astype(bool) if 'skipped' in results else np.zeros(n_events, dtype=bool)

            for i in range(n_events):
                row = [run_name, i + 1]
                for k in export_keys:
                    val = data[k][i] if i < len(data[k]) else np.nan
                    row.append(f"{val:.6g}" if not np.isnan(val) else "")
                row.append("YES" if skipped[i] else "")
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
