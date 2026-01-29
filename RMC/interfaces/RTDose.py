from typing import Optional

import pydicom
from pydantic import BaseModel, ConfigDict


class RTDose(BaseModel):
    Modality: str
    DoseUnits: Optional[str]
    ds_object: Optional[pydicom.dataset.FileDataset] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
