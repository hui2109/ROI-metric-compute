from .static import static
import numpy as np
from ..interfaces.MetricParam import MetricParam


def zstatic(metricParam: MetricParam):
    """
        计算 z-score 标准化图像的统计指标
    """

    croped_roi_origianl = metricParam.croped_roi_all.croped_roi_origianl
    std = croped_roi_origianl.std()

    if std < 1e-8:
        std_img = np.zeros_like(croped_roi_origianl)
    else:
        std_img = (croped_roi_origianl - croped_roi_origianl.mean()) / (std + 1e-8)

    metricParam.croped_roi_all.croped_roi_origianl = std_img

    return {"zstatic": static(metricParam)['static']}
