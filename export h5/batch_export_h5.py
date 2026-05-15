import os
import glob
import sys
from labquake_2Danalysis import LabQuakeAnalyzer

def batch_export():
    # 尋找當前目錄下所有的 .tpc5 檔案
    tpc5_files = glob.glob("*.tpc5")
    
    if not tpc5_files:
        print("當前目錄下找不到任何 .tpc5 檔案", flush=True)
        return

    # 參數設定
    Diameter = 250
    Radius = Diameter / 2
    FAULT_AREA = 50 * 50 / 10000  # m^2
    SLIDING_AREA = Radius**2 * 3.14159 / 1000000  # m^2

    print(f"找到 {len(tpc5_files)} 個 .tpc5 檔案，準備開始批次轉換...", flush=True)
    print("-" * 50, flush=True)

    for filename in sorted(tpc5_files):
        print(f"\n>>> 正在處理檔案: {filename}", flush=True)
        try:
            # 解析檔名 (假設格式如 t0145_04_8MPa_run4.tpc5)
            filename_parts = filename.replace(".tpc5", "").split('_')
            
            if len(filename_parts) < 4:
                print(f"  [略過] 檔名格式不符合預期: {filename}", flush=True)
                continue

            experiment_name = filename_parts[0]
            normal_stress = filename_parts[2]
            run_name = filename_parts[3]

            # 建立分析器並匯出
            analyzer = LabQuakeAnalyzer(filename, fault_area=FAULT_AREA)
            analyzer.export_to_explorer_hdf5(
                experiment_name=experiment_name, 
                normal_stress=normal_stress, 
                run_name=run_name, 
                window_sec=1.0, 
                baseline_block=2, 
                sliding_area=SLIDING_AREA, 
                threshold=3, 
                dt_max=0.2, 
                lowpass_fc=500
            )
            print(f"  [成功] {filename} 匯出完成", flush=True)
            
        except Exception as e:
            print(f"  [失敗] 處理 {filename} 時發生錯誤: {e}", flush=True)

    print("\n" + "-" * 50, flush=True)
    print("所有檔案批次轉換完成", flush=True)

if __name__ == "__main__":
    batch_export()
