from typing import List
from pydantic import BaseModel

class QueryRequest(BaseModel):
    repo_url: str
    query: str

class DocumentMetadata(BaseModel):
    file_path: str
    type: str
    is_code: bool = False
    is_implementation: bool = False
    title: str = ""

class Document(BaseModel):
    text: str
    meta_data: DocumentMetadata

class QueryResponse(BaseModel):
    rationale: str
    answer: str
    contexts: List[Document]