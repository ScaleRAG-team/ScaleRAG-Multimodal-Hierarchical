# data_schema.py

# Defines Pydantic models for structured representation of parsed scientific papers.
# Each paper stores extracted text chunks, tables, and figures with metadata (page, source, captions).
# Used as the unified output schema before indexing in the RAG pipeline.


from typing import List, Optional
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class DocumentChunk(BaseModel):

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(description="The text content of the chunk.")
    section_title: Optional[str] = Field(None, description="Associated section title.")
    page_number: int = Field(description="Page number where text starts.")
    source_pdf: str = Field(description="Source PDF ID or path.")
    

class TableData(BaseModel):

    id: UUID = Field(default_factory=uuid4)
    caption: Optional[str] = Field(None)
    table_text: str = Field(description="Extracted table text.")
    page_number: int = Field(description="Page number where text starts.")
    source_pdf: str = Field(description="Source PDF ID or path.")
    

class FigureData(BaseModel):

    id: UUID = Field(default_factory=uuid4)
    caption: Optional[str] = Field(None)
    image_path: Optional[str] = Field(None)
    page_number: int = Field(description="Page number where text starts.")
    source_pdf: str = Field(description="Source PDF ID or path.")




class ScientificPaperParsed(BaseModel):
    """
    Standalone Pydantic model representing a parsed multi-modal scientific paper.
    This structure holds the extracted data before it goes into any database or index.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique ID for the parsed paper.")
    # Metadata
    title: Optional[str] = Field(None, description="Title of the paper.")
    authors: List[str] = Field(default_factory=list, description="List of authors.")
    source_path: str = Field(description="Original file path or URL of the PDF.")

    # Extracted Multi-Modal Content
    text_chunks: List[DocumentChunk] = Field(default_factory=list, description="List of extracted text chunks.")
    tables: List[TableData] = Field(default_factory=list, description="List of extracted tables.")
    figures: List[FigureData] = Field(default_factory=list, description="List of extracted figures.")

    