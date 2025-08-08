"""SQLAlchemy database models."""

from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, 
    JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON as JSONB  # Use JSON for SQLite compatibility
import uuid
from enum import Enum

from .connection import Base


class DocumentSource(str, Enum):
    """Document source enumeration."""
    UNHCR = "unhcr"
    WHO = "who" 
    UNICEF = "unicef"
    WFP = "wfp"
    OCHA = "ocha"
    UNDP = "undp"


class ImageType(str, Enum):
    """Image type enumeration."""
    CHART = "chart"
    INFOGRAPHIC = "infographic"
    MAP = "map"
    DIAGRAM = "diagram"
    PHOTO = "photo"


class DatasetStatus(str, Enum):
    """Dataset status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingStatus(str, Enum):
    """Training status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Document(Base):
    """Document metadata and content."""
    
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(String(2048), nullable=False, unique=True)
    title = Column(String(1024), nullable=False)
    source = Column(String(50), nullable=False)
    language = Column(String(10), nullable=False)
    content = Column(Text)
    content_type = Column(String(50), default="html")
    word_count = Column(Integer, default=0)
    
    # Metadata
    publication_date = Column(DateTime)
    scraped_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Quality metrics
    quality_score = Column(Float, default=0.0)
    processing_status = Column(String(50), default="pending")
    
    # Additional metadata as JSON
    document_metadata = Column(JSONB, default=dict)
    
    # Relationships
    images = relationship("Image", back_populates="document", cascade="all, delete-orphan")
    dataset_items = relationship("DatasetItem", back_populates="document")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_source_language', 'source', 'language'),
        Index('idx_documents_scraped_at', 'scraped_at'),
        Index('idx_documents_quality_score', 'quality_score'),
        Index('idx_documents_processing_status', 'processing_status'),
        CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='quality_score_range'),
        CheckConstraint('word_count >= 0', name='word_count_positive'),
    )
    
    @validates('source')
    def validate_source(self, key, source):
        if source not in [s.value for s in DocumentSource]:
            raise ValueError(f"Invalid source: {source}")
        return source
    
    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title[:50]}...', source={self.source})>"


class Image(Base):
    """Image metadata and OCR results."""
    
    __tablename__ = "images"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Image properties
    src = Column(String(2048))
    alt_text = Column(Text)
    width = Column(Integer)
    height = Column(Integer)
    page_number = Column(Integer)
    
    # Classification
    image_type = Column(String(50), default="infographic")
    confidence_score = Column(Float, default=0.0)
    
    # OCR results
    ocr_text = Column(Text)
    ocr_confidence = Column(Float, default=0.0)
    ocr_engines_used = Column(JSONB, default=list)
    ocr_results = Column(JSONB, default=dict)
    
    # Processing metadata
    processed_at = Column(DateTime)
    processing_duration = Column(Float)  # seconds
    
    # File storage
    storage_path = Column(String(1024))
    file_size = Column(Integer)
    file_hash = Column(String(64))  # SHA-256 hash
    
    # Relationships
    document = relationship("Document", back_populates="images")
    dataset_items = relationship("DatasetItem", back_populates="image")
    
    # Indexes
    __table_args__ = (
        Index('idx_images_document_id', 'document_id'),
        Index('idx_images_type', 'image_type'),
        Index('idx_images_confidence', 'confidence_score'),
        Index('idx_images_ocr_confidence', 'ocr_confidence'),
        Index('idx_images_processed_at', 'processed_at'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='confidence_score_range'),
        CheckConstraint('ocr_confidence >= 0 AND ocr_confidence <= 1', name='ocr_confidence_range'),
        CheckConstraint('width >= 0 AND height >= 0', name='dimensions_positive'),
    )
    
    @validates('image_type')
    def validate_image_type(self, key, image_type):
        if image_type not in [t.value for t in ImageType]:
            raise ValueError(f"Invalid image type: {image_type}")
        return image_type
    
    def __repr__(self):
        return f"<Image(id={self.id}, type={self.image_type}, document_id={self.document_id})>"


class DatasetItem(Base):
    """Vision-language dataset items."""
    
    __tablename__ = "dataset_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=False)
    
    # Instruction-response pair
    instruction = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    instruction_type = Column(String(50), default="description")
    
    # Quality metrics
    quality_score = Column(Float, nullable=False)
    human_validated = Column(Boolean, default=False)
    validation_score = Column(Float)
    
    # Dataset assignment
    split = Column(String(20), default="train")  # train, validation, test
    dataset_version = Column(String(50), default="v1.0")
    
    # Language information
    source_language = Column(String(10), default="en")
    target_language = Column(String(10), nullable=False)
    
    # Generation metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    generation_model = Column(String(100))
    generation_params = Column(JSONB, default=dict)
    
    # Context information
    context = Column(JSONB, default=dict)
    
    # Relationships
    document = relationship("Document", back_populates="dataset_items")
    image = relationship("Image", back_populates="dataset_items")
    
    # Indexes
    __table_args__ = (
        Index('idx_dataset_items_split', 'split'),
        Index('idx_dataset_items_quality', 'quality_score'),
        Index('idx_dataset_items_language', 'target_language'),
        Index('idx_dataset_items_version', 'dataset_version'),
        Index('idx_dataset_items_validated', 'human_validated'),
        Index('idx_dataset_items_created_at', 'created_at'),
        UniqueConstraint('document_id', 'image_id', 'instruction', 'target_language', 
                        name='unique_dataset_item'),
        CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='dataset_quality_range'),
        CheckConstraint('validation_score IS NULL OR (validation_score >= 0 AND validation_score <= 1)', 
                       name='validation_score_range'),
    )
    
    @validates('split')
    def validate_split(self, key, split):
        valid_splits = ['train', 'validation', 'test']
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {valid_splits}")
        return split
    
    def __repr__(self):
        return f"<DatasetItem(id={self.id}, split={self.split}, language={self.target_language})>"


class TrainingRun(Base):
    """Model training run metadata."""
    
    __tablename__ = "training_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Model information
    base_model = Column(String(200), nullable=False)
    model_path = Column(String(1024))
    processor_path = Column(String(1024))
    
    # Training configuration
    languages = Column(JSONB, nullable=False)
    dataset_version = Column(String(50), nullable=False)
    training_config = Column(JSONB, default=dict)
    
    # Training metrics
    status = Column(String(50), default="pending")
    total_steps = Column(Integer, default=0)
    current_step = Column(Integer, default=0)
    
    # Performance metrics
    train_loss = Column(Float)
    eval_loss = Column(Float)
    eval_bleu = Column(Float)
    eval_rouge = Column(Float)
    eval_bertscore = Column(Float)
    
    # Training history
    metrics_history = Column(JSONB, default=dict)
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Resource usage
    gpu_hours = Column(Float)
    peak_memory_gb = Column(Float)
    
    # Experiment tracking
    wandb_run_id = Column(String(100))
    wandb_project = Column(String(100))
    
    # Additional metadata
    training_metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_training_runs_status', 'status'),
        Index('idx_training_runs_dataset_version', 'dataset_version'),
        Index('idx_training_runs_started_at', 'started_at'),
        Index('idx_training_runs_eval_loss', 'eval_loss'),
        CheckConstraint('current_step >= 0', name='current_step_positive'),
        CheckConstraint('total_steps >= 0', name='total_steps_positive'),
        CheckConstraint('current_step <= total_steps', name='current_step_valid'),
    )
    
    @validates('status')
    def validate_status(self, key, status):
        if status not in [s.value for s in TrainingStatus]:
            raise ValueError(f"Invalid status: {status}")
        return status
    
    def __repr__(self):
        return f"<TrainingRun(id={self.id}, name='{self.name}', status={self.status})>"
    
    @property
    def is_running(self) -> bool:
        """Check if training is currently running."""
        return self.status == TrainingStatus.RUNNING.value
    
    @property 
    def is_completed(self) -> bool:
        """Check if training completed successfully."""
        return self.status == TrainingStatus.COMPLETED.value
    
    @property
    def progress_percentage(self) -> float:
        """Get training progress as percentage."""
        if self.total_steps == 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100.0)