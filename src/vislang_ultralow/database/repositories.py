"""Data access layer with repository pattern."""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.exc import IntegrityError

from .models import Document, Image, DatasetItem, TrainingRun, DocumentSource, ImageType

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common operations."""
    
    def __init__(self, session: Session, model_class):
        self.session = session
        self.model_class = model_class
    
    def create(self, **kwargs) -> Any:
        """Create new instance."""
        try:
            instance = self.model_class(**kwargs)
            self.session.add(instance)
            self.session.flush()
            return instance
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Failed to create {self.model_class.__name__}: {e}")
            raise
    
    def get_by_id(self, id: Any) -> Optional[Any]:
        """Get instance by ID."""
        return self.session.query(self.model_class).filter(
            self.model_class.id == id
        ).first()
    
    def get_all(self, limit: int = 1000, offset: int = 0) -> List[Any]:
        """Get all instances with pagination."""
        return self.session.query(self.model_class).offset(offset).limit(limit).all()
    
    def update(self, id: Any, **kwargs) -> Optional[Any]:
        """Update instance by ID."""
        instance = self.get_by_id(id)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            self.session.flush()
        return instance
    
    def delete(self, id: Any) -> bool:
        """Delete instance by ID."""
        instance = self.get_by_id(id)
        if instance:
            self.session.delete(instance)
            self.session.flush()
            return True
        return False
    
    def count(self) -> int:
        """Count total instances."""
        return self.session.query(self.model_class).count()


class DocumentRepository(BaseRepository):
    """Repository for Document operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, Document)
    
    def get_by_url(self, url: str) -> Optional[Document]:
        """Get document by URL."""
        return self.session.query(Document).filter(Document.url == url).first()
    
    def get_by_source(self, source: str, limit: int = 100) -> List[Document]:
        """Get documents by source."""
        return self.session.query(Document).filter(
            Document.source == source
        ).limit(limit).all()
    
    def get_by_language(self, language: str, limit: int = 100) -> List[Document]:
        """Get documents by language."""
        return self.session.query(Document).filter(
            Document.language == language
        ).limit(limit).all()
    
    def get_by_quality_threshold(self, min_quality: float, limit: int = 100) -> List[Document]:
        """Get documents above quality threshold."""
        return self.session.query(Document).filter(
            Document.quality_score >= min_quality
        ).order_by(desc(Document.quality_score)).limit(limit).all()
    
    def get_recent(self, days: int = 7, limit: int = 100) -> List[Document]:
        """Get recently scraped documents."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return self.session.query(Document).filter(
            Document.scraped_at >= cutoff_date
        ).order_by(desc(Document.scraped_at)).limit(limit).all()
    
    def search_by_content(self, query: str, limit: int = 50) -> List[Document]:
        """Search documents by content using full-text search."""
        # Simple text search - in production, use PostgreSQL full-text search
        return self.session.query(Document).filter(
            or_(
                Document.title.ilike(f'%{query}%'),
                Document.content.ilike(f'%{query}%')
            )
        ).limit(limit).all()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get document statistics."""
        stats = {
            'total_documents': self.count(),
            'by_source': {},
            'by_language': {},
            'quality_distribution': {
                'high': 0,    # >= 0.8
                'medium': 0,  # 0.5 - 0.8
                'low': 0      # < 0.5
            },
            'recent_activity': {
                'last_24h': 0,
                'last_7d': 0,
                'last_30d': 0
            }
        }
        
        # Source distribution
        source_counts = self.session.query(
            Document.source, func.count(Document.id)
        ).group_by(Document.source).all()
        stats['by_source'] = {source: count for source, count in source_counts}
        
        # Language distribution
        lang_counts = self.session.query(
            Document.language, func.count(Document.id)
        ).group_by(Document.language).all()
        stats['by_language'] = {lang: count for lang, count in lang_counts}
        
        # Quality distribution
        high_quality = self.session.query(Document).filter(
            Document.quality_score >= 0.8
        ).count()
        medium_quality = self.session.query(Document).filter(
            and_(Document.quality_score >= 0.5, Document.quality_score < 0.8)
        ).count()
        low_quality = self.session.query(Document).filter(
            Document.quality_score < 0.5
        ).count()
        
        stats['quality_distribution'] = {
            'high': high_quality,
            'medium': medium_quality,
            'low': low_quality
        }
        
        # Recent activity
        now = datetime.utcnow()
        stats['recent_activity']['last_24h'] = self.session.query(Document).filter(
            Document.scraped_at >= now - timedelta(hours=24)
        ).count()
        stats['recent_activity']['last_7d'] = self.session.query(Document).filter(
            Document.scraped_at >= now - timedelta(days=7)
        ).count()
        stats['recent_activity']['last_30d'] = self.session.query(Document).filter(
            Document.scraped_at >= now - timedelta(days=30)
        ).count()
        
        return stats
    
    def bulk_create(self, documents_data: List[Dict[str, Any]]) -> List[Document]:
        """Bulk create documents."""
        documents = []
        for doc_data in documents_data:
            try:
                doc = Document(**doc_data)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to create document: {e}")
                continue
        
        if documents:
            self.session.add_all(documents)
            self.session.flush()
        
        return documents


class ImageRepository(BaseRepository):
    """Repository for Image operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, Image)
    
    def get_by_document(self, document_id: str) -> List[Image]:
        """Get images for a document."""
        return self.session.query(Image).filter(
            Image.document_id == document_id
        ).all()
    
    def get_by_type(self, image_type: str, limit: int = 100) -> List[Image]:
        """Get images by type."""
        return self.session.query(Image).filter(
            Image.image_type == image_type
        ).limit(limit).all()
    
    def get_processed(self, limit: int = 100) -> List[Image]:
        """Get processed images with OCR results."""
        return self.session.query(Image).filter(
            Image.processed_at.isnot(None)
        ).order_by(desc(Image.processed_at)).limit(limit).all()
    
    def get_unprocessed(self, limit: int = 100) -> List[Image]:
        """Get unprocessed images."""
        return self.session.query(Image).filter(
            Image.processed_at.is_(None)
        ).limit(limit).all()
    
    def get_by_confidence_threshold(self, min_confidence: float, limit: int = 100) -> List[Image]:
        """Get images above confidence threshold."""
        return self.session.query(Image).filter(
            Image.ocr_confidence >= min_confidence
        ).order_by(desc(Image.ocr_confidence)).limit(limit).all()
    
    def get_by_file_hash(self, file_hash: str) -> Optional[Image]:
        """Get image by file hash."""
        return self.session.query(Image).filter(Image.file_hash == file_hash).first()
    
    def update_ocr_results(
        self, 
        image_id: str, 
        ocr_text: str, 
        confidence: float, 
        engines_used: List[str],
        detailed_results: Dict[str, Any]
    ) -> Optional[Image]:
        """Update OCR results for an image."""
        image = self.get_by_id(image_id)
        if image:
            image.ocr_text = ocr_text
            image.ocr_confidence = confidence
            image.ocr_engines_used = engines_used
            image.ocr_results = detailed_results
            image.processed_at = datetime.utcnow()
            self.session.flush()
        return image
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get image statistics."""
        stats = {
            'total_images': self.count(),
            'by_type': {},
            'processing_status': {
                'processed': 0,
                'unprocessed': 0
            },
            'confidence_distribution': {
                'high': 0,    # >= 0.8
                'medium': 0,  # 0.5 - 0.8  
                'low': 0      # < 0.5
            },
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }
        
        # Type distribution
        type_counts = self.session.query(
            Image.image_type, func.count(Image.id)
        ).group_by(Image.image_type).all()
        stats['by_type'] = {img_type: count for img_type, count in type_counts}
        
        # Processing status
        stats['processing_status']['processed'] = self.session.query(Image).filter(
            Image.processed_at.isnot(None)
        ).count()
        stats['processing_status']['unprocessed'] = self.session.query(Image).filter(
            Image.processed_at.is_(None)
        ).count()
        
        # Confidence distribution (for processed images only)
        high_conf = self.session.query(Image).filter(
            and_(Image.ocr_confidence >= 0.8, Image.processed_at.isnot(None))
        ).count()
        medium_conf = self.session.query(Image).filter(
            and_(Image.ocr_confidence >= 0.5, Image.ocr_confidence < 0.8, Image.processed_at.isnot(None))
        ).count()
        low_conf = self.session.query(Image).filter(
            and_(Image.ocr_confidence < 0.5, Image.processed_at.isnot(None))
        ).count()
        
        stats['confidence_distribution'] = {
            'high': high_conf,
            'medium': medium_conf,
            'low': low_conf
        }
        
        # Average confidence
        avg_conf = self.session.query(func.avg(Image.ocr_confidence)).filter(
            Image.processed_at.isnot(None)
        ).scalar()
        stats['avg_confidence'] = float(avg_conf) if avg_conf else 0.0
        
        # Average processing time
        avg_time = self.session.query(func.avg(Image.processing_duration)).filter(
            Image.processing_duration.isnot(None)
        ).scalar()
        stats['avg_processing_time'] = float(avg_time) if avg_time else 0.0
        
        return stats


class DatasetRepository(BaseRepository):
    """Repository for DatasetItem operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, DatasetItem)
    
    def get_by_split(self, split: str, limit: int = 1000) -> List[DatasetItem]:
        """Get dataset items by split."""
        return self.session.query(DatasetItem).filter(
            DatasetItem.split == split
        ).limit(limit).all()
    
    def get_by_language(self, language: str, limit: int = 1000) -> List[DatasetItem]:
        """Get dataset items by target language."""
        return self.session.query(DatasetItem).filter(
            DatasetItem.target_language == language
        ).limit(limit).all()
    
    def get_by_version(self, version: str, limit: int = 1000) -> List[DatasetItem]:
        """Get dataset items by version."""
        return self.session.query(DatasetItem).filter(
            DatasetItem.dataset_version == version
        ).limit(limit).all()
    
    def get_high_quality(self, min_quality: float = 0.8, limit: int = 1000) -> List[DatasetItem]:
        """Get high-quality dataset items."""
        return self.session.query(DatasetItem).filter(
            DatasetItem.quality_score >= min_quality
        ).order_by(desc(DatasetItem.quality_score)).limit(limit).all()
    
    def get_validated(self, limit: int = 1000) -> List[DatasetItem]:
        """Get human-validated dataset items."""
        return self.session.query(DatasetItem).filter(
            DatasetItem.human_validated == True
        ).limit(limit).all()
    
    def get_for_training(
        self, 
        languages: List[str], 
        min_quality: float = 0.7,
        version: str = "v1.0"
    ) -> Dict[str, List[DatasetItem]]:
        """Get dataset items formatted for training."""
        base_query = self.session.query(DatasetItem).filter(
            and_(
                DatasetItem.target_language.in_(languages),
                DatasetItem.quality_score >= min_quality,
                DatasetItem.dataset_version == version
            )
        )
        
        result = {}
        for split in ['train', 'validation', 'test']:
            items = base_query.filter(DatasetItem.split == split).all()
            result[split] = items
        
        return result
    
    def create_dataset_split(
        self, 
        items: List[DatasetItem], 
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> None:
        """Assign split labels to dataset items."""
        import random
        random.shuffle(items)
        
        total = len(items)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        for i, item in enumerate(items):
            if i < train_end:
                item.split = 'train'
            elif i < val_end:
                item.split = 'validation'
            else:
                item.split = 'test'
        
        self.session.flush()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_items': self.count(),
            'by_split': {},
            'by_language': {},
            'by_instruction_type': {},
            'quality_distribution': {
                'high': 0,    # >= 0.8
                'medium': 0,  # 0.5 - 0.8
                'low': 0      # < 0.5
            },
            'validation_status': {
                'validated': 0,
                'unvalidated': 0
            },
            'avg_quality': 0.0,
            'by_version': {}
        }
        
        # Split distribution
        split_counts = self.session.query(
            DatasetItem.split, func.count(DatasetItem.id)
        ).group_by(DatasetItem.split).all()
        stats['by_split'] = {split: count for split, count in split_counts}
        
        # Language distribution
        lang_counts = self.session.query(
            DatasetItem.target_language, func.count(DatasetItem.id)
        ).group_by(DatasetItem.target_language).all()
        stats['by_language'] = {lang: count for lang, count in lang_counts}
        
        # Instruction type distribution
        type_counts = self.session.query(
            DatasetItem.instruction_type, func.count(DatasetItem.id)
        ).group_by(DatasetItem.instruction_type).all()
        stats['by_instruction_type'] = {inst_type: count for inst_type, count in type_counts}
        
        # Quality distribution
        high_quality = self.session.query(DatasetItem).filter(
            DatasetItem.quality_score >= 0.8
        ).count()
        medium_quality = self.session.query(DatasetItem).filter(
            and_(DatasetItem.quality_score >= 0.5, DatasetItem.quality_score < 0.8)
        ).count()
        low_quality = self.session.query(DatasetItem).filter(
            DatasetItem.quality_score < 0.5
        ).count()
        
        stats['quality_distribution'] = {
            'high': high_quality,
            'medium': medium_quality,
            'low': low_quality
        }
        
        # Validation status
        validated = self.session.query(DatasetItem).filter(
            DatasetItem.human_validated == True
        ).count()
        stats['validation_status'] = {
            'validated': validated,
            'unvalidated': self.count() - validated
        }
        
        # Average quality
        avg_quality = self.session.query(func.avg(DatasetItem.quality_score)).scalar()
        stats['avg_quality'] = float(avg_quality) if avg_quality else 0.0
        
        # Version distribution
        version_counts = self.session.query(
            DatasetItem.dataset_version, func.count(DatasetItem.id)
        ).group_by(DatasetItem.dataset_version).all()
        stats['by_version'] = {version: count for version, count in version_counts}
        
        return stats


class TrainingRepository(BaseRepository):
    """Repository for TrainingRun operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, TrainingRun)
    
    def get_by_status(self, status: str) -> List[TrainingRun]:
        """Get training runs by status."""
        return self.session.query(TrainingRun).filter(
            TrainingRun.status == status
        ).order_by(desc(TrainingRun.started_at)).all()
    
    def get_running(self) -> List[TrainingRun]:
        """Get currently running training runs."""
        return self.get_by_status("running")
    
    def get_completed(self, limit: int = 50) -> List[TrainingRun]:
        """Get completed training runs."""
        return self.session.query(TrainingRun).filter(
            TrainingRun.status == "completed"
        ).order_by(desc(TrainingRun.completed_at)).limit(limit).all()
    
    def get_best_models(self, metric: str = "eval_loss", limit: int = 10) -> List[TrainingRun]:
        """Get best performing models by metric."""
        if metric == "eval_loss":
            order_by = asc(getattr(TrainingRun, metric))
        else:
            order_by = desc(getattr(TrainingRun, metric))
        
        return self.session.query(TrainingRun).filter(
            and_(
                TrainingRun.status == "completed",
                getattr(TrainingRun, metric).isnot(None)
            )
        ).order_by(order_by).limit(limit).all()
    
    def update_progress(self, run_id: str, current_step: int, metrics: Dict[str, float]) -> Optional[TrainingRun]:
        """Update training progress."""
        run = self.get_by_id(run_id)
        if run:
            run.current_step = current_step
            if metrics:
                # Update latest metrics
                for metric, value in metrics.items():
                    if hasattr(run, metric):
                        setattr(run, metric, value)
                
                # Update history
                if not run.metrics_history:
                    run.metrics_history = {}
                
                for metric, value in metrics.items():
                    if metric not in run.metrics_history:
                        run.metrics_history[metric] = []
                    run.metrics_history[metric].append({
                        'step': current_step,
                        'value': value,
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            self.session.flush()
        return run
    
    def complete_training(
        self, 
        run_id: str, 
        final_metrics: Dict[str, float],
        model_path: str,
        processor_path: Optional[str] = None
    ) -> Optional[TrainingRun]:
        """Mark training as completed."""
        run = self.get_by_id(run_id)
        if run:
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            run.model_path = model_path
            if processor_path:
                run.processor_path = processor_path
            
            # Calculate duration
            if run.started_at:
                duration = run.completed_at - run.started_at
                run.duration_seconds = int(duration.total_seconds())
            
            # Update final metrics
            for metric, value in final_metrics.items():
                if hasattr(run, metric):
                    setattr(run, metric, value)
            
            self.session.flush()
        return run
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'total_runs': self.count(),
            'by_status': {},
            'avg_duration_hours': 0.0,
            'total_gpu_hours': 0.0,
            'best_performances': {},
            'recent_activity': {
                'last_24h': 0,
                'last_7d': 0,
                'last_30d': 0
            }
        }
        
        # Status distribution
        status_counts = self.session.query(
            TrainingRun.status, func.count(TrainingRun.id)
        ).group_by(TrainingRun.status).all()
        stats['by_status'] = {status: count for status, count in status_counts}
        
        # Average duration
        avg_duration = self.session.query(func.avg(TrainingRun.duration_seconds)).filter(
            TrainingRun.duration_seconds.isnot(None)
        ).scalar()
        if avg_duration:
            stats['avg_duration_hours'] = float(avg_duration) / 3600.0
        
        # Total GPU hours
        total_gpu = self.session.query(func.sum(TrainingRun.gpu_hours)).scalar()
        stats['total_gpu_hours'] = float(total_gpu) if total_gpu else 0.0
        
        # Best performances
        metrics = ['eval_loss', 'eval_bleu', 'eval_rouge', 'eval_bertscore']
        for metric in metrics:
            if metric == 'eval_loss':
                best = self.session.query(func.min(getattr(TrainingRun, metric))).filter(
                    TrainingRun.status == "completed"
                ).scalar()
            else:
                best = self.session.query(func.max(getattr(TrainingRun, metric))).filter(
                    TrainingRun.status == "completed"
                ).scalar()
            stats['best_performances'][metric] = float(best) if best else None
        
        # Recent activity
        now = datetime.utcnow()
        stats['recent_activity']['last_24h'] = self.session.query(TrainingRun).filter(
            TrainingRun.started_at >= now - timedelta(hours=24)
        ).count()
        stats['recent_activity']['last_7d'] = self.session.query(TrainingRun).filter(
            TrainingRun.started_at >= now - timedelta(days=7)
        ).count()
        stats['recent_activity']['last_30d'] = self.session.query(TrainingRun).filter(
            TrainingRun.started_at >= now - timedelta(days=30)
        ).count()
        
        return stats