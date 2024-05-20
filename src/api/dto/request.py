from __future__ import annotations

from typing import List
from pydantic import BaseModel

class ModelTraining(BaseModel):
    colour_model_id: str
    epochs: int


class ModelInference(BaseModel):
    colour_model_id: str
    colour_name: List[str]
