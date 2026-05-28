import enum
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Enum, JSON,func
from app.core.database import Base


class DocumentStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PROCESSING, nullable=False)
    
    # This will hold our final JSON structural extraction (summary, risks, tasks)
    extracted_data = Column(JSON, nullable=True)
    
    # Decoupling tracker: handles if Neo4j falls behind but Postgres succeeds
    graph_sync_status = Column(String(50), default="NOT_STARTED", nullable=False)
    error_message = Column(String(1000), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)