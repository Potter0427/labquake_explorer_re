import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from tpc5_data_processor import TPC5DataProcessor
import h5py

class SignalUtils:
    """訊號處理工具箱"""
    
    @staticmethod
    def moving_average(x, w, pad_mode='reflect'):
        if w <= 1: return x
        x = np.ravel(x)
        r = w // 2
        xpad = np.pad(x, (r, w - 1 - r), mode=pad_mode)
        k = np.ones(w, dtype=float) / w
        return np.convolve(xpad, k, mode='valid')

    @staticmethod
    def lowpass(data, t, fc, order=4):
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt
        nyq = 0.5 * fs
        wn = fc / nyq
        b, a = butter(order, wn, btype='low', analog=False)
        return filtfilt(b, a, data), fs

    @staticmethod
    def threshold_sampling(u, t, threshold, dt_max):
        u_keep = [u[0]]
        t_keep = [t[0]]
        last_u = u[0]
        last_t = t[0]

        for i in range(1, len(u)):
            du = abs(u[i] - last_u)
            dt = t[i] - last_t
            if du >= threshold or dt >= dt_max:
                u_keep.append(u[i])
                t_keep.append(t[i])
                last_u = u[i]
                last_t = t[i]
        return np.array(u_keep), np.array(t_keep)

    @staticmethod
    def calculate_trend_drop(t_rel, y, pre_win, post_win):
        mask_pre = (t_rel >= pre_win[0]) & (t_rel <= pre_win[1])
        mask_post = (t_rel >= post_win[0]) & (t_rel <= post_win[1])
        result = {'valid': False}
        if np.sum(mask_pre) > 5 and np.sum(mask_post) > 5:
            try:
                coeff_pre = np.polyfit(t_rel[mask_pre], y[mask_pre], 1)
                coeff_post = np.polyfit(t_rel[mask_post], y[mask_post], 1)
                val_pre_0 = np.polyval(coeff_pre, 0)
                val_post_0 = np.polyval(coeff_post, 0)
                result.update({
                    'valid': True,
                    'coeff_pre': coeff_pre, 'coeff_post': coeff_post,
                    'delta': val_pre_0 - val_post_0,
                    'val_pre_0': val_pre_0, 'val_post_0': val_post_0
                })
            except: pass
        return result
    
    @staticmethod
    def calculate_slip_rate(disp, t, fc=500, threshold=3, dt_max=0.2):
        """
        標準化滑移率計算流程：低通 -> 降採樣 -> 微分
        回傳: (t_dec, rate) tuple
        """
        # 1. Lowpass
        d_lp, _ = SignalUtils.lowpass(disp, t, fc)
        # 2. Decimation
        d_dec, t_dec = SignalUtils.threshold_sampling(d_lp, t, threshold, dt_max)
        # 3. Differentiation
        if len(d_dec) > 1:
            rate = np.abs(np.gradient(d_dec, t_dec))
            return t_dec, rate
        return np.array([]), np.array([])

    @staticmethod
    def generate_heatmap_matrix(sensor_data_list, positions, num_steps=10000):
        """
        生成熱圖矩陣 (固定解析度版)
        num_steps: 時間軸的總點數，預設 10,000
        """
        valid_items = [item for item in sensor_data_list if item is not None and len(item[0]) > 0]
        if not valid_items:
            return None, None, None

        t_min = min([item[0][0] for item in valid_items])
        t_max = max([item[0][-1] for item in valid_items])
        
        # 使用 linspace 確保點數精確固定為 num_steps
        common_time = np.linspace(t_min, t_max, num_steps)
        
        heatmap_mat = np.zeros((len(positions), len(common_time)))
        
        for i, item in enumerate(sensor_data_list):
            if item is not None and len(item[0]) > 0:
                t_src, r_src = item
                # 進行線性插值
                heatmap_mat[i, :] = np.interp(common_time, t_src, r_src, left=0, right=0)
                
        return common_time, heatmap_mat

    @staticmethod
    def collect_event_drops(triggers, t_all, data_signal, pre_win, post_win, window_sec=3.0):
        """
        收集所有事件的 Drop 值 (Delta Mu / Delta Stress)
        """
        events_t = []
        events_v = []
        
        for tr in triggers:
            # 簡單裁切
            mask = (t_all >= tr - window_sec) & (t_all <= tr + window_sec)
            t_rel = t_all[mask] - tr
            
            if len(t_rel) > 50:
                res = SignalUtils.calculate_trend_drop(t_rel, data_signal[mask], pre_win, post_win)
                if res['valid']:
                    events_t.append(tr)
                    events_v.append(res['delta'])
                    
        return events_t, events_v
    

class LabQuakeAnalyzer:
    """數據讀取與處理核心"""
    def __init__(self, filename, fault_area=0.05):
        self.filename = filename
        self.fault_area = fault_area

        # 1. 油壓缸面積 (m^2)
        rsm500 = 6.21e-3
        rsm1000 = 12.67e-3
        
        # 2. 感測器校準 (1V = 60.482 bar)
        self.p_slope = 60.482 
        self.p_intercept = -5.497
        
        # --- 自動計算轉換係數 (V -> MPa) ---
        # 公式：MPa = V * (bar/V) * 0.1 * (油壓缸總面積 / 斷層面積)
        def get_cal_params(cylinder_area, num_cylinders):
            slope = self.p_slope * 0.1 * (num_cylinders * cylinder_area / self.fault_area)
            intercept = self.p_intercept * 0.1 * (num_cylinders * cylinder_area / self.fault_area)
            return (slope, intercept)

        # 動態檢查 tpc5 檔案內實際可用的通道，同時自適應 5 通道舊檔與 8 通道新檔
        avail_chs = []
        try:
            with TPC5DataProcessor(self.filename) as proc:
                avail_chs = proc.get_available_channels()
        except Exception:
            pass

        if avail_chs:
            eddy_chs = [c for c in range(11, 19) if c in avail_chs]
            if not eddy_chs:
                eddy_chs = range(11, 16)
        else:
            eddy_chs = range(11, 16)

        self.params = {
            'pzt_chs': range(1, 5),
            'sigma_ch': 8, 'tau_ch': 9,
            'lvdt_ch': 10, 'eddy_chs': eddy_chs,
            'sigma_cal': get_cal_params(cylinder_area=rsm500, num_cylinders=3),
            'tau_cal': get_cal_params(cylinder_area=rsm1000, num_cylinders=1),
            'eddy_cal': -95.0,
            'lvdt_cal': -5040.0,
            'rsm1000_area': rsm1000
        }

    def _read_raw(self, ch, block):
        with TPC5DataProcessor(self.filename) as proc:
            avail_blocks = proc.get_available_blocks(ch)
            if block not in avail_blocks:
                if 1 in avail_blocks:
                    block = 1
                elif avail_blocks:
                    block = avail_blocks[0]
                else:
                    raise KeyError(f"Channel {ch} has no available blocks in {self.filename}")
            data = proc.get_voltage_data(ch, block)
            info = proc.get_block_info(ch, block)
            fs = info["sample_rate_hz"]
            t_trig = info["trigger_time_seconds"]
            t_sample = info["trigger_sample"]
            start_t = -t_sample / fs
            t = np.arange(len(data)) / fs + start_t + t_trig
            return data, t, fs

    def get_physical_data(self, ch, block=1, baseline_block=7):
        raw_data, t, _ = self._read_raw(ch, block)

        if ch == self.params['sigma_ch']:
            normal = raw_data * self.params['sigma_cal'][0] + self.params['sigma_cal'][1]
            normal_bar = raw_data * self.p_slope + self.p_intercept
            normal_psi = normal_bar * 14.5038
            val = (normal, normal_bar, normal_psi)
        elif ch == self.params['tau_ch']:
            tau = raw_data * self.params['tau_cal'][0] + self.params['tau_cal'][1]
            shear_bar = raw_data * self.p_slope + self.p_intercept
            shear_psi = shear_bar * 14.5038
            val = (tau, shear_bar, shear_psi)
        elif ch == self.params['lvdt_ch'] or ch in self.params['eddy_chs']:
            try:
                base_raw, _, _ = self._read_raw(ch, baseline_block)
                base_volt = np.mean(base_raw) if len(base_raw) > 0 else raw_data[0]
            except Exception:
                base_volt = np.mean(raw_data[:min(1000, len(raw_data))]) if len(raw_data) > 0 else 0.0
            factor = self.params['lvdt_cal'] if ch == self.params['lvdt_ch'] else self.params['eddy_cal']
            val = (raw_data - base_volt) * factor
        else:
            val = raw_data
        return t, val
    
    def calculate_local_shear(self, block=1, sliding_area=None):
        _, tau = self.get_physical_data(self.params['tau_ch'], block)
        local_shear = tau[0] * (self.fault_area / sliding_area)
        
        return local_shear

    def get_trigger_times(self, ch):
        triggers = []
        with TPC5DataProcessor(self.filename) as proc:
            blocks = proc.get_available_blocks(ch)
            for b in blocks:
                info = proc.get_block_info(ch, b)
                triggers.append(info["trigger_time_seconds"])
        return np.array(triggers)

    def export_to_explorer_hdf5(self, experiment_name, normal_stress, run_name, window_sec=2.0, baseline_block=2, diameter=100, sliding_area=None, threshold=3, dt_max=0.2, lowpass_fc=500):
        """
        將資料以「追加模式 (append)」打包進以實驗命名的 HDF5 檔案中。
        threshold: threshold_sampling 的位移閾值
        dt_max: threshold_sampling 的最大時間間隔
        lowpass_fc: 低通濾波截止頻率 (Hz)
        通道配置：sigma_ch=8, tau_ch=9, lvdt_ch=10, eddy_chs=11~18
        油壓缸：RSM500×3 (正向力), RSM1000×1 (剪力)
        """
        output_h5 = f"{experiment_name}.h5"
        print(f"開始匯出資料至 {output_h5} (Run: {run_name}) ...")
        print(f"  Slip Rate 參數: threshold={threshold}, dt_max={dt_max}, lowpass_fc={lowpass_fc}")
        
        # 1. 取得 Block 1 的連續全域資料
        t_cont, sigma = self.get_physical_data(self.params['sigma_ch'], block=1)
        _, tau = self.get_physical_data(self.params['tau_ch'], block=1)
        _, lvdt = self.get_physical_data(self.params['lvdt_ch'], block=1, baseline_block=baseline_block)
        _, eddy = self.get_physical_data(list(self.params['eddy_chs'])[0], block=1, baseline_block=baseline_block)

        # 2. 取得所有事件的觸發時間 (剔除代表連續全域資料的 Block 1)
        triggers = self.get_trigger_times(self.params['sigma_ch'])[1:]
        
        # 偵測所有可用的 Block (用於高取樣事件)
        from tpc5_data_processor import TPC5DataProcessor
        with TPC5DataProcessor(self.filename) as proc:
            all_blocks = proc.get_available_blocks(self.params['sigma_ch'])

        # 3. 解析 run_name 轉換成陣列編號 (例如 run1 -> 0, run2 -> 1)
        run_number_str = ''.join(filter(str.isdigit, run_name))
        run_index = str(int(run_number_str) - 1) if run_number_str else "0"

        # 4. 以追加模式 ('a') 打開 HDF5 檔案
        with h5py.File(output_h5, 'a') as f:
            
            # --- 設定根目錄屬性 ---
            if 'name' not in f.keys():
                f.create_dataset('name', data=f"{experiment_name}_{diameter}P")
            
            # --- 建立或獲取 runs 資料夾 ---
            if "runs" not in f:
                runs_group = f.create_group("runs")
            else:
                runs_group = f["runs"]
                
            # --- 建立本次的 Run 資料夾 (如果已存在則刪除重建) ---
            if run_index in runs_group:
                print(f"發現已存在的 Run {run_index}，將會覆蓋此 Run 的資料。")
                del runs_group[run_index]
            
            current_run = runs_group.create_group(run_index)
            current_run.create_dataset('name', data=f"{run_name}_{normal_stress}")
            
            # --- 建立 time history 資料夾 ---
            th_group = current_run.create_group("time history")
            th_group.create_dataset('time', data=t_cont)
            th_group.create_dataset('LP_displacement', data=lvdt)
            th_group.create_dataset('is_1d', data=True)
            
            # eddy (5 channels) + 預算背景 Slip Rate (稀疏格式)
            for ch in self.params['eddy_chs']:
                _, eddy_data = self.get_physical_data(ch, block=1, baseline_block=baseline_block)
                th_group.create_dataset(f'eddy_ch{ch}', data=eddy_data)
                
                s_lp, _ = SignalUtils.lowpass(eddy_data, t_cont, fc=lowpass_fc)
                u_dec, t_dec = SignalUtils.threshold_sampling(s_lp, t_cont, threshold=threshold, dt_max=dt_max)
                if len(u_dec) > 1:
                    rate = np.abs(np.gradient(u_dec, t_dec))
                    mask_v = rate > 1e-2
                    th_group.create_dataset(f'sliprate_ch{ch}', data=rate[mask_v])
                    th_group.create_dataset(f't_sliprate_ch{ch}', data=np.array(t_dec)[mask_v])
                
            # 高取樣 Block 的事件速率 (存入子目錄以防 UI 樹狀圖太亂)
            hr_group = th_group.create_group("high_rate_sliprates")
            with TPC5DataProcessor(self.filename) as proc:
                for block in all_blocks:
                    if block == 1:
                        continue
                    try:
                        t_b, _ = self.get_physical_data(self.params['sigma_ch'], block)
                    except Exception:
                        continue
                    for ch in self.params['eddy_chs']:
                        if block not in proc.get_available_blocks(ch):
                            continue
                        try:
                            _, slip = self.get_physical_data(ch, block, baseline_block=baseline_block)
                        except Exception:
                            continue
                        s_lp, _ = SignalUtils.lowpass(slip, t_b, fc=lowpass_fc)
                        u_dec, t_dec = SignalUtils.threshold_sampling(s_lp, t_b, threshold=threshold, dt_max=dt_max)
                        if len(u_dec) > 1:
                            rate_b = np.abs(np.gradient(u_dec, t_dec))
                            mask_v = rate_b > 1e-2
                            hr_group.create_dataset(f'high_sliprate_ch{ch}_blk{block}', data=rate_b[mask_v])
                            hr_group.create_dataset(f't_high_sliprate_ch{ch}_blk{block}', data=np.array(t_dec)[mask_v])
                
            # pressure
            th_group.create_dataset('normal_pressure', data=sigma[0])
            th_group.create_dataset('normal_bar', data=sigma[1])
            th_group.create_dataset('normal_psi', data=sigma[2])
            th_group.create_dataset('shear_pressure', data=tau[0])
            th_group.create_dataset('shear_bar', data=tau[1])
            th_group.create_dataset('shear_psi', data=tau[2])
            with np.errstate(divide='ignore', invalid='ignore'):
                mu = np.where(sigma[0] != 0, tau[0] / sigma[0], 0)
            th_group.create_dataset('mu', data=mu)

            # tau_local (requires sliding_area)
            if sliding_area is not None:
                tau_local = self.calculate_local_shear(block=1, sliding_area=sliding_area)
                th_group.create_dataset('tau_local', data=tau_local)
            
            # pzt
            for ch in self.params['pzt_chs']:
                _, pzt_data = self.get_physical_data(ch, block=1)
                th_group.create_dataset(f'pzt_ch{ch}', data=pzt_data)

            # --- 建立 events 資料夾 ---
            events_group = current_run.create_group("events")
            
            valid_events_count = 0
            for i, trig in enumerate(triggers):
                event_idx = events_group.create_group(f"{i+1:03d}")
                mask = (t_cont >= trig - window_sec) & (t_cont <= trig + window_sec)
                
                if np.sum(mask) > 0:
                    event_idx.create_dataset('time', data=t_cont[mask])
                    event_idx.create_dataset('event_time', data=trig)
                    event_idx.create_dataset('pressure', data=sigma[0][mask])
                    event_idx.create_dataset('displacement', data=lvdt[mask])
                    event_idx.create_dataset('eddy', data=eddy[mask])
                    valid_events_count += 1

        print(f" 成功將 {run_name} 匯出至 {output_h5}，共包含 {valid_events_count} 個事件。")

class LabQuakePlotter:
    @staticmethod
    def add_trigger_lines(ax, triggers, label_every=3, show_labels=True, y_frac=1.1):
        """
        畫出觸發線
        y_frac: 標籤文字的垂直位置 (1.0=圖頂端, 1.1=圖上方10%處) -> 改回 1.1 就不會撞標題了
        """
        tx = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        
        # 只畫視窗範圍內的 trigger (避免畫太多效能變差)
        x_min, x_max = ax.get_xlim()
        valid_trigs = triggers[(triggers >= x_min) & (triggers <= x_max)]
        
        for tr in valid_trigs:
            idx = np.argmin(np.abs(triggers - tr))
            block_id = idx + 1 
            
            ax.axvline(x=tr, color='gray', linestyle=':', alpha=0.3)
            
            if show_labels and label_every and (block_id % label_every == 0):
                # 使用 y_frac 參數
                ax.text(tr, y_frac, str(block_id), transform=tx, ha='center', va='bottom', 
                        rotation=90, fontsize=8, clip_on=False)

    @staticmethod
    def plot_trend_arrow(ax, x_pos, y_start, y_end, color, label_text=None, ha='left'):
        """
        通用型趨勢箭頭繪製
        x_pos: 箭頭的 x 座標
        y_start, y_end: 箭頭的起點與終點 y 座標
        color: 箭頭與文字顏色
        label_text: 顯示文字
        ha: 文字水平對齊方式 ('left' 或 'right')
        """
        # 畫雙頭箭頭
        ax.annotate('', xy=(x_pos, y_end), xytext=(x_pos, y_start),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=2.5))
        
        if label_text:
            # 根據對齊方式決定文字的偏移方向
            offset = 0.1 if ha == 'left' else -0.1
            ax.text(x_pos + offset, (y_start + y_end) / 2, label_text, 
                    ha=ha, va='center', color=color, fontweight='bold', fontsize=10)