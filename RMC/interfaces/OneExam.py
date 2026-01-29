from datetime import date
from typing import Union, Optional

import numpy as np
import pydicom
from pydantic import BaseModel, conlist, ConfigDict


class OneImage(BaseModel):
    SOPInstanceUID: str
    InstanceNumber: int
    Rows: int
    Columns: int
    ImagePositionPatient: conlist(float, min_length=3, max_length=3)
    PixelSpacing: conlist(float, min_length=2, max_length=2)
    BitsAllocated: int
    pixel_array: np.ndarray
    RescaleSlope: float
    RescaleIntercept: float
    WindowWidth: float
    WindowCenter: float
    ds_object: Optional[pydicom.dataset.FileDataset] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OneCTSeries(BaseModel):
    Modality: str
    SeriesInstanceUID: str
    SeriesNumber: str
    SeriesDate: date
    ImageOrientationPatient: conlist(float, min_length=6, max_length=6)
    SliceThickness: float
    SliceLocation: float
    KVP: float
    XRayTubeCurrent: float
    ConvolutionKernel: str
    Manufacturer: str
    Images: list[OneImage]
    Volume: Optional[np.ndarray] = None  # 3D volume

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OneMRISeries(BaseModel):
    pass


class OneExam(BaseModel):
    PatientID: str
    given_name: str
    PatientSex: str
    StudyDate: date
    StudyInstanceUID: str

    CT: Union[None, list[OneCTSeries]]
    MRI: Union[None, list[OneMRISeries]]
