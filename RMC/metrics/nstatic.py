from .static import static
import numpy as np
from ..interfaces.MetricParam import MetricParam


def nstatic(metricParam: MetricParam):
    """
        计算 min-max 归一化图像的统计指标
    """

    croped_roi_origianl = metricParam.croped_roi_all.croped_roi_origianl
    ptp = croped_roi_origianl.ptp()

    if ptp < 1e-8:
        norm_img = croped_roi_origianl
    else:
        norm_img = (croped_roi_origianl - croped_roi_origianl.min()) / (ptp + 1e-8)

    metricParam.croped_roi_all.croped_roi_origianl = norm_img

    return {"nstatic": static(metricParam)['static']}
