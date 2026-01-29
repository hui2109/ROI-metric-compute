import itertools

import numpy as np
from scipy.stats import skew, kurtosis
from ..interfaces.MetricParam import MetricParam


def static(metricParam: MetricParam):
    croped_roi_origianl, croped_roi_mask = metricParam.croped_roi_all.croped_roi_origianl, metricParam.croped_roi_all.croped_roi_mask
    pixels = croped_roi_origianl[croped_roi_mask > 0]
    result = {}

    if pixels.size < 2 or np.all(pixels == pixels[0]):
        result.update(dict.fromkeys([
            'std', 'range', 'mean', 'median',
            'gra_num', 'grastd_num', 'blur_num', 'edge_num'
        ], -999.0))
        result.update(manual_texture_features(croped_roi_origianl, croped_roi_mask))
        return result

    # === 添加指标 ===
    result['std'] = float(np.std(pixels))  # 标准差（可选）
    result['range'] = float(np.max(pixels) - np.min(pixels))  # 范围
    result['mean'] = float(np.mean(pixels))
    result['median'] = float(np.median(pixels))

    # 求阈值分割（比如0HU） 正负像素数量比值  以及  其他百分位数指标
    r1 = add_threshold_ratio_features(pixels,
                                      up_percentiles=[50, 60, 70, 80, 85, 95],
                                      low_percentiles=[5, 10, 15, 25, 35, 45],
                                      percentiles=[95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60])
    result.update(r1)

    return {'static': result}


def manual_texture_features(img, mask):
    """
        手动计算 texture features（支持任意小尺寸 mask）
        支持维度：[H,W] 或 [D,H,W]
    """

    values = img[mask > 0].flatten()
    values = values.astype(np.float32)

    if values.size == 0:
        return {
            'Uniformity': 0,
            'Entropy': 0,
            'Skewness': 0,
            'Kurtosis': 0
        }

    # 计算概率直方图（可调 bins）
    hist, _ = np.histogram(values, bins=64, density=True)
    prob = hist[hist > 0]
    entropys_num = -np.sum(prob * np.log2(prob))
    uniformity_num = np.sum(prob ** 2)
    skewness_num = skew(values)
    kurtosis_num = kurtosis(values)
    return {
        'Uniformity': uniformity_num,
        'Entropy': entropys_num,
        'Skewness': skewness_num,
        'Kurtosis': kurtosis_num
    }


def add_threshold_ratio_features(pixels, *, up_percentiles, low_percentiles, percentiles):
    result = {}

    # 先计算百分位数指标
    for p in percentiles:
        key2 = f'percentile_{p}'
        result[key2] = float(np.percentile(pixels, p))

    # === Step 2: 将百分位数转为实际像素值阈值
    up_thresholds = {p: np.percentile(pixels, p) for p in up_percentiles}
    low_thresholds = {p: np.percentile(pixels, p) for p in low_percentiles}

    for t1, t2 in itertools.product(up_thresholds, low_thresholds):
        pos_count = np.sum(pixels > t1)
        neg_count = np.sum(pixels < t2)
        pos_vals = pixels[pixels > t1]
        neg_vals = pixels[pixels < t2]

        ratio_key = f'count_up{t1}low{t2}_ratio'
        diff_key = f'value_up{t1}low{t2}_diff'
        mean_ratio_key = f'value_up{t1}low{t2}_ratio'

        neg_mean = np.mean(neg_vals) if neg_vals.size > 0 else np.nan
        pos_mean = np.mean(pos_vals) if pos_vals.size > 0 else np.nan

        if neg_count == 0:
            result[ratio_key] = -999
        else:
            result[ratio_key] = pos_count / neg_count
        result[diff_key] = abs(pos_mean - neg_mean) if neg_vals.size > 0 else -999

        # 安全设置：避免除以 0 或 nan
        if np.isnan(neg_mean) or neg_mean == 0:
            result[mean_ratio_key] = -999
        else:
            result[mean_ratio_key] = abs(pos_mean / neg_mean)

    return result
