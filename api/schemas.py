from typing import List, Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    k: Optional[int] = 5


class SourceInfo(BaseModel):
    source_file: str
    page: Optional[str] = None
    content_preview: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    num_sources: int
