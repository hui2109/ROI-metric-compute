from typing import Optional

import pydicom
from pydantic import BaseModel, ConfigDict


class RTPlan(BaseModel):
    Modality: str
    RTPlanLabel: Optional[str]
    ds_object: Optional[pydicom.dataset.FileDataset] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
