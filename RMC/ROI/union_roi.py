import numpy as np


def get_union_roi(mask_Volume_1: np.ndarray, mask_Volume_2: np.ndarray):
    return mask_Volume_1 | mask_Volume_2
