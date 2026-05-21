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

            drop_results = run['analysis']['results'] if ('analysis' in run and 'results' in run['analysis']) else {}
            k_results = run['k_analysis']['results'] if ('k_analysis' in run and 'results' in run['k_analysis']) else {}

            if not drop_results and not k_results:
                print(f"  [{run_name}] 沒有 drop 或 k 的分析結果，跳過", flush=True)
                continue

            results = {}
            for d in (drop_results, k_results):
                for k in d.keys():
                    results[k] = d[k]

            # 固定匯出所有的欄位，避免各個 run 欄位不同導致 CSV 錯位 (ParserError)
            export_keys = ordered_keys

            # 讀取資料與總筆數
            data = {}
            n_events = 0
            for k in export_keys:
                if k in results:
                    arr = results[k][()]
                    data[k] = arr
                    n_events = max(n_events, len(arr))
                else:
                    data[k] = []

            # 讀取共用 skip_events (或從舊的 results['skipped'] fallback)
            skipped = np.zeros(n_events, dtype=bool)
            if 'skip_events' in run:
                se_obj = run['skip_events']
                if isinstance(se_obj, h5py.Group):
                    # 舊版 HDF5 結構中 skip_events 可能被存成 Group (包含 '0', '1', '2' ... 等 dataset)
                    se = [int(se_obj[k][()]) for k in sorted(se_obj.keys(), key=lambda x: int(x))]
                else:
                    se = se_obj[()]
                for idx in se:
                    if idx < n_events:
                        skipped[idx] = True
            elif 'skipped' in results:
                skipped = results['skipped'][()].astype(bool)

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
