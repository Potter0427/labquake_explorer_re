import os
import glob
import sys
from labquake_1Danalysis import LabQuakeAnalyzer

def batch_export(fault_area_cm2, sliding_area_cm2, 
                 window_sec=1.0, baseline_block=2, 
                 threshold=3, dt_max=0.2, lowpass_fc=500,
                 vw_length_cm=None, vw_diameter_cm=None):
    # 尋找當前目錄下所有的 .tpc5 檔案
    tpc5_files = glob.glob("*.tpc5")
    
    if not tpc5_files:
        print("當前目錄下找不到任何 .tpc5 檔案", flush=True)
        return

    # 參數設定 (將 cm^2 轉為 m^2)
    FAULT_AREA = fault_area_cm2 / 10000.0
    SLIDING_AREA = sliding_area_cm2 / 10000.0

    print(f"找到 {len(tpc5_files)} 個 .tpc5 檔案，準備開始批次轉換...", flush=True)
    print("-" * 50, flush=True)
    
    # 判斷 1D / 2D 與提取尺寸
    if vw_diameter_cm is not None:
        from labquake_2Danalysis import LabQuakeAnalyzer
        vw_val = int(vw_diameter_cm * 10) # 轉換為 mm
        print(f" 開始分析 2D {vw_val}PC 實驗")
    elif vw_length_cm is not None:
        from labquake_1Danalysis import LabQuakeAnalyzer
        vw_val = int(vw_length_cm * 10) # 轉換為 mm
        print(f" 開始分析 1D {vw_val}P 實驗")
    else:
        print("錯誤：必須提供 vw_length_cm (1D) 或 vw_diameter_cm (2D) 其中之一", flush=True)
        return

    for filename in sorted(tpc5_files):
        print(f"\n>>> 正在處理檔案: {filename}", flush=True)
        try:
            # 解析檔名 (假設格式如 t0145_04_8MPa_run4.tpc5)
            filename_parts = filename.replace(".tpc5", "").split('_')
            
            if len(filename_parts) < 3:
                print(f"  [略過] 檔名格式不符合預期: {filename}", flush=True)
                continue

            # 允許檔名沒有夾帶標籤的情況，所以使用倒數計算來取得 stress 與 run
            experiment_name = filename_parts[0]
            normal_stress = filename_parts[-2]
            run_name = filename_parts[-1]

            # 建立分析器並匯出
            analyzer = LabQuakeAnalyzer(filename, fault_area=FAULT_AREA)
            analyzer.export_to_explorer_hdf5(
                experiment_name=experiment_name, 
                normal_stress=normal_stress, 
                run_name=run_name, 
                window_sec=window_sec, 
                baseline_block=baseline_block, 
                diameter=vw_val,
                sliding_area=SLIDING_AREA, 
                threshold=threshold, 
                dt_max=dt_max, 
                lowpass_fc=lowpass_fc
            )
            print(f"  [成功] {filename} 匯出完成", flush=True)
            
        except Exception as e:
            print(f"  [失敗] 處理 {filename} 時發生錯誤: {e}", flush=True)

    print("\n" + "-" * 50, flush=True)
    print("所有檔案批次轉換完成", flush=True)

if __name__ == "__main__":
    batch_export()
