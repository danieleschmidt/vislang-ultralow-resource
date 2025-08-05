"""FastAPI application for VisLang inference."""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from PIL import Image
import io
import torch
from pathlib import Path

from .trainer import VisionLanguageTrainer
from .database import get_session, get_database_manager
from .database.repositories import DocumentRepository, TrainingRepository

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VisLang-UltraLow-Resource API",
    description="API for vision-language inference on humanitarian content",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
_model_instance: Optional[VisionLanguageTrainer] = None


class InferenceRequest(BaseModel):
    """Request model for inference."""
    instruction: str
    max_length: int = 256
    num_beams: int = 4
    temperature: float = 1.0


class InferenceResponse(BaseModel):
    """Response model for inference."""
    response: str
    instruction: str
    processing_time_ms: float
    model_info: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: bool
    model_loaded: bool
    version: str


def get_model() -> VisionLanguageTrainer:
    """Get loaded model instance."""
    global _model_instance
    if _model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model_instance


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting VisLang API server...")
    
    # Try to load a default model if available
    try:
        model_dir = Path("./models/best")
        if model_dir.exists():
            await load_model(str(model_dir))
            logger.info("Default model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Check database connection
    try:
        db_manager = get_database_manager()
        health = db_manager.health_check()
        db_healthy = health.get("database", False)
    except Exception:
        db_healthy = False
    
    return HealthResponse(
        status="healthy" if db_healthy else "degraded",
        database=db_healthy,
        model_loaded=_model_instance is not None,
        version="0.1.0"
    )


@app.post("/load_model")
async def load_model(model_path: str):
    """Load a trained model."""
    global _model_instance
    
    try:
        # Import here to avoid startup issues
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        # Load model and processor
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(model_path)
        
        # Create trainer instance
        _model_instance = VisionLanguageTrainer(
            model=model,
            processor=processor,
            languages=["en"]  # Will be updated based on model config
        )
        
        logger.info(f"Model loaded from {model_path}")
        return {"status": "success", "message": f"Model loaded from {model_path}"}
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/predict", response_model=InferenceResponse)
async def predict(
    file: UploadFile = File(...),
    request: InferenceRequest = Depends(),
    model: VisionLanguageTrainer = Depends(get_model)
):
    """Generate response for uploaded image and instruction."""
    import time
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Load image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate response
        response = model.generate_response(
            image=image,
            instruction=request.instruction,
            max_length=request.max_length,
            num_beams=request.num_beams,
            temperature=request.temperature
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            response=response,
            instruction=request.instruction,
            processing_time_ms=processing_time,
            model_info=model.get_model_info()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List available trained models."""
    try:
        with get_session() as session:
            training_repo = TrainingRepository(session)
            completed_runs = training_repo.get_completed(limit=20)
            
            models = []
            for run in completed_runs:
                models.append({
                    "id": str(run.id),
                    "name": run.name,
                    "languages": run.languages,
                    "eval_loss": run.eval_loss,
                    "eval_bleu": run.eval_bleu,
                    "model_path": run.model_path,
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None
                })
            
            return {"models": models}
            
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        with get_session() as session:
            doc_repo = DocumentRepository(session)
            doc_stats = doc_repo.get_statistics()
            
            training_repo = TrainingRepository(session)
            training_stats = training_repo.get_statistics()
            
            return {
                "documents": doc_stats,
                "training": training_stats,
                "model_loaded": _model_instance is not None
            }
            
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "VisLang-UltraLow-Resource API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


# Export app for use with ASGI servers
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)