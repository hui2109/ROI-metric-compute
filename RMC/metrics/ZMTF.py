import numpy as np
from .MTF import mtf
from ..interfaces.MetricParam import MetricParam


def zmtf(metricParam: MetricParam):
    croped_roi_origianl = metricParam.croped_roi_all.croped_roi_origianl

    if croped_roi_origianl.size == 0 or np.all(np.isnan(croped_roi_origianl)) or np.nanstd(croped_roi_origianl) < 1e-8:
        zimg = croped_roi_origianl
    else:
        zimg = (croped_roi_origianl - np.nanmean(croped_roi_origianl)) / (np.nanstd(croped_roi_origianl) + 1e-8)

    metricParam.croped_roi_all.croped_roi_origianl = zimg

    return {'zmtf': mtf(metricParam)['mtf']}
