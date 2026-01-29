from typing import Union, Optional

import pydicom
import numpy as np
from pydantic import BaseModel, ConfigDict


class Contour(BaseModel):
    ContourData: list[float]
    ContourGeometricType: str
    NumberOfContourPoints: int

    ReferencedSOPClassUIDs: list[Optional[str]]
    ReferencedSOPInstanceUIDs: list[Optional[str]]


class ROIRegion(BaseModel):
    ROIName: str
    ROINumber: int
    ROIGenerationAlgorithm: str
    ReferencedFrameOfReferenceUID: str

    ROIDisplayColor: list[float]
    RTROIInterpretedType: str
    Contours: list[Contour]
    Volume: Optional[np.ndarray] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RTStruct(BaseModel):
    Modality: str
    SOPClassUID: str
    SOPInstanceUID: str
    FrameOfReferenceUID: str
    ImageSeriesInstanceUID: str
    StructureSetLabel: str
    StructureSetROISequence: list[ROIRegion]
    ds_object: Optional[pydicom.dataset.FileDataset] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
