import numpy as np
from pydantic import BaseModel, ConfigDict, conlist


class CropedROI(BaseModel):
    croped_roi_origianl: np.ndarray
    croped_roi_mask: np.ndarray
    croped_roi_target: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MetricParam(BaseModel):
    croped_roi_all: CropedROI
    PixelSpacing: conlist(float, min_length=2, max_length=2)
    SmoothingWindow: int = 11
    ThresholdsForRadius: tuple[int] = (3500, 3000, 2500, 2000, 1500, 1000, 500, 300)
    ThresholdsForDiameter: tuple[int] = (5000, 4000, 3000, 2000, 1500, 1000, 500, 300, 100, 50, 0)
