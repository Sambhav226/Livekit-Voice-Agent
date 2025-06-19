from enum import Enum
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel

class Document(BaseModel):
    id: str
    query: str
    source: str = ""
    chunk_index: int
    chunk_no: int
    start_index: float
    end_index: float
    context: str
    text: str
    relevance_score: float = 0.0