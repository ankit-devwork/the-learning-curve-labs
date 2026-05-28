from sqlalchemy import Column, String, Integer, Text, ForeignKey
from pgvector.sqlalchemy import Vector
from app.core.database import Base

class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Cascade deletes: If a document is deleted from Postgres, clean up all its chunks automatically
    document_id = Column(
        String(36), 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    
    # Tracks the sequence of text blocks (e.g., Chunk 0, Chunk 1, Chunk 2)
    chunk_index = Column(Integer, nullable=False)
    
    # The raw text content of this specific block
    content = Column(Text, nullable=False)
    
    # 🧬 pgvector column: Configured for OpenAI / LiteLLM 'text-embedding-3-small' dimension (1536)
    embedding = Column(Vector(768), nullable=False)