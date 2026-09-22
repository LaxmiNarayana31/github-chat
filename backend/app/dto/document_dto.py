from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata describing a retrieved code or document chunk."""
    file_path: str = Field(default="", description="Relative path of the source file")
    type: str = Field(default="", description="File extension or language type (e.g. py, ts, md)")
    is_code: bool = Field(default=False, description="Whether this file contains executable source code")
    is_implementation: bool = Field(default=False, description="Whether this file represents core implementation")
    title: str = Field(default="", description="Document title or relative path reference")


class Document(BaseModel):
    """Container for document chunk text and associated metadata."""
    text: str = Field(..., description="Chunk content text")
    meta_data: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Associated chunk metadata")
