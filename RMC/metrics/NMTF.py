import numpy as np
from .MTF import mtf
from ..interfaces.MetricParam import MetricParam


def nmtf(metricParam: MetricParam):
    croped_roi_origianl = metricParam.croped_roi_all.croped_roi_origianl

    ptp = np.nanmax(croped_roi_origianl) - np.nanmin(croped_roi_origianl)
    if croped_roi_origianl.size == 0 or np.all(np.isnan(croped_roi_origianl)) or ptp < 1e-8:
        nimg = croped_roi_origianl
    else:
        nimg = (croped_roi_origianl - np.nanmin(croped_roi_origianl)) / (ptp + 1e-8)

    metricParam.croped_roi_all.croped_roi_origianl = nimg

    return {'nmtf': mtf(metricParam)['mtf']}
