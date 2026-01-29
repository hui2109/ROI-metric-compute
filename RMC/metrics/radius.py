import numpy as np
from ..interfaces.MetricParam import MetricParam


def radius(metricParam: MetricParam):
    croped_roi_origianl = metricParam.croped_roi_all.croped_roi_origianl
    PixelSpacing = metricParam.PixelSpacing
    threshold_list = metricParam.ThresholdsForRadius
    res = {}

    for th in threshold_list:
        res[f"radius_{th}"] = cal_radius(croped_roi_origianl, PixelSpacing, th)
        
    return {'radius': res}


def cal_radius(croped_roi_origianl, PixelSpacing, th=3000):
    # 1. 提取图像中 >3000 HU 的区域
    binary_mask = (croped_roi_origianl > th).astype(np.uint8)

    # 2. 计算区域内的像素总数
    num_pixels = np.sum(binary_mask)

    # 3. 计算面积（mm²）
    pixel_area_mm2 = PixelSpacing[0] * PixelSpacing[1]  # 注意 spacing 是 [row_spacing, col_spacing]
    area_mm2 = num_pixels * pixel_area_mm2

    # 4. 计算等效圆的半径 r = sqrt(A / π)
    if area_mm2 > 0:
        radius_mm = np.sqrt(area_mm2 / np.pi)
    else:
        radius_mm = 0

    return radius_mm
