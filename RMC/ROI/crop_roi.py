import numpy as np

from ..interfaces.RTStruct import ROIRegion


def get_rectangle_roi_dict(image_Volume, mask_Volume, *, margin_rmin=0, margin_rmax=0, margin_cmin=0, margin_cmax=0, margin=0) -> dict[int, list[np.ndarray]]:
    if margin != 0:
        if margin_rmin == 0:
            margin_rmin = margin
        if margin_rmax == 0:
            margin_rmax = margin
        if margin_cmin == 0:
            margin_cmin = margin
        if margin_cmax == 0:
            margin_cmax = margin

    target_Volume = image_Volume * mask_Volume
    rectangle_roi_dict = {}

    for s in range(target_Volume.shape[0]):
        bbox = get_rectangle_roi(mask_Volume[s])
        if bbox is not None:
            rmin, rmax, cmin, cmax = bbox

            rmin = max(0, rmin - margin_rmin)
            rmax = min(target_Volume.shape[1], rmax + margin_rmax)
            cmin = max(0, cmin - margin_cmin)
            cmax = min(target_Volume.shape[2], cmax + margin_cmax)

            rectangle_roi_dict[s] = [image_Volume[s][rmin:rmax + 1, cmin:cmax + 1], mask_Volume[s][rmin:rmax + 1, cmin:cmax + 1], target_Volume[s][rmin:rmax + 1, cmin:cmax + 1]]

    return rectangle_roi_dict


def get_rectangle_roi(mask_slice):
    if not mask_slice.any():
        bbox = None
    else:
        row_any = mask_slice.any(axis=1)
        col_any = mask_slice.any(axis=0)

        rmin, rmax = np.where(row_any)[0][[0, -1]]
        cmin, cmax = np.where(col_any)[0][[0, -1]]
        bbox = (rmin, rmax, cmin, cmax)

    return bbox
