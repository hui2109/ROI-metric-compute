import numpy as np
from scipy import signal
from scipy.fft import fft
from scipy.integrate import simps
from scipy.interpolate import interp1d
from ..interfaces.MetricParam import MetricParam


def mtf(metricParam: MetricParam):
    croped_roi_origianl = metricParam.croped_roi_all.croped_roi_origianl
    PixelSpacing = metricParam.PixelSpacing
    smoothing_window = metricParam.SmoothingWindow

    rows, cols = croped_roi_origianl.shape
    Fs = 1.0 / PixelSpacing[0]
    mtf_areas_row, cutoff_freqs_row = [], []
    for r in range(rows):
        profile = croped_roi_origianl[r, :]
        area, cutoff = compute_single_profile_mtf(profile, Fs, smoothing_window)
        if area is not None:
            mtf_areas_row.append(area)
            cutoff_freqs_row.append(cutoff)
    mtf_areas_col, cutoff_freqs_col = [], []
    for c in range(cols):
        profile = croped_roi_origianl[:, c]
        area, cutoff = compute_single_profile_mtf(profile, Fs, smoothing_window)
        if area is not None:
            mtf_areas_col.append(area)
            cutoff_freqs_col.append(cutoff)
    return {'mtf': [np.mean(mtf_areas_row), np.mean(cutoff_freqs_row), np.mean(mtf_areas_col), np.mean(cutoff_freqs_col)]}


def compute_single_profile_mtf(profile, Fs, smoothing_window=None, interp_points=100):
    """
    计算单条剖面的 MTF 曲线、面积和 10% 截止频率（支持剖面插值）
    参数:
        profile: 原始 1D 边缘剖面 (numpy array)
        Nfft: FFT 点数
        Fs: 采样频率 (pixel_size_mm^-1)
        smoothing_window: 平滑窗口（可选）
        interp_factor: 插值100数据点  统一
    """
    # ---------- 0. 健壮性检查 ----------
    if profile.size < 2 or np.allclose(profile, profile[0]):
        # print(profile)
        # raise ValueError(f"{roi,ID,date,z,}profile must have at least 2 points and not be constant.")
        return 0, 0  # 或 (None, None) 看你后续如何处理
    # 1. 插值 profile 到 interp_points 长度
    x_old = np.linspace(0, 1, len(profile))
    x_new = np.linspace(0, 1, interp_points)
    # bounds_error=False 可以避免超界直接报错
    profile = interp1d(
        x_old, profile,
        kind='linear',
        bounds_error=False,
        fill_value=(profile[0], profile[-1])
    )(x_new)
    esf = profile.astype(np.float32)
    esf -= esf.min()
    peak = esf.max()
    if peak <= 1e-9:
        return 0, 0
    esf /= peak
    lsf = np.gradient(esf)
    Nfft = 1 << (interp_points - 1).bit_length()  # 自动确定下一个2的幂
    LSF_fft = fft(lsf, n=Nfft)
    mag = np.abs(LSF_fft[:Nfft // 2])
    if mag.max() == 0:
        return 0, 0
    mag /= mag.max()
    freqs = np.linspace(0, Fs / 2, Nfft // 2, endpoint=False)
    if smoothing_window is not None:
        mag = signal.savgol_filter(mag, smoothing_window, 3)
    mtf_area = simps(mag, freqs)
    try:
        cutoff = np.interp(0.1, mag[::-1], freqs[::-1])
    except Exception:
        cutoff = np.nan
    return mtf_area, cutoff
