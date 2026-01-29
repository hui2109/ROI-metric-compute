from tkinter import Image

import numpy as np
import cv2
from ..interfaces.OneExam import OneCTSeries
from ..interfaces.RTStruct import ROIRegion


def gen_roi_volume(one_CT_series: OneCTSeries, roi_region: ROIRegion) -> np.ndarray:
    ImagePositionPatient = __get_ImagePositionPatient(one_CT_series)
    PixelSpacing = __get_PixelSpacing(one_CT_series)
    mask = np.zeros_like(one_CT_series.Volume, dtype=np.uint8)

    for contour in roi_region.Contours:
        counter_data = np.array(contour.ContourData).reshape((-1, 3))
        mask = gen_mask(mask, counter_data, ImagePositionPatient, PixelSpacing)

    return mask


def gen_mask(mask: np.ndarray, counter_data: np.ndarray, ImagePositionPatient: np.ndarray, PixelSpacing: np.ndarray) -> np.ndarray:
    counter_points = (counter_data - ImagePositionPatient) / PixelSpacing
    counter_points = np.round(counter_points).astype(np.int32)

    for k in np.unique(counter_points[:, 2]):
        # 防止一些奇怪的点报错
        if k > mask.shape[0]:
            continue

        layer_pts = counter_points[counter_points[:, 2] == k, :2]
        layer_pts = np.ascontiguousarray(layer_pts, dtype=np.int32)  # OpenCV 强制拷贝成连续内存

        pts_cv = layer_pts.reshape((-1, 1, 2))  # OpenCV 需要 int32，形状为 (N,1,2)
        cv2.fillPoly(mask[k], [pts_cv], color=1)

    return mask


def __get_ImagePositionPatient(one_CT_series: OneCTSeries) -> np.ndarray:
    return np.array(one_CT_series.Images[0].ImagePositionPatient)


def __get_PixelSpacing(one_CT_series: OneCTSeries) -> np.ndarray:
    x, y = one_CT_series.Images[0].PixelSpacing
    z = one_CT_series.Images[1].ImagePositionPatient[2] - one_CT_series.Images[0].ImagePositionPatient[2]

    return np.array([x, y, z])
