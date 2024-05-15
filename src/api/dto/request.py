from __future__ import annotations

from typing import List
from pydantic import BaseModel

class ModelTraining(BaseModel):
    model_id: str
    epochs: int


class ModelInference(BaseModel):
    model_id: str
    colour_name: List[str]
