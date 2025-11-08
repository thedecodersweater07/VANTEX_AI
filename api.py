"""
VANTEX_AI API

REST API for interacting with VANTEX_AI\'s NLP capabilities.
Provides endpoints for text processing, intent recognition, and context management.
"""

import os
import json
import logging
import uvicorn
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import NLP components
from nlp.text_processor import TextProcessor
from nlp.intent_recognizer import IntentRecognizer, Intent, IntentType
from nlp.context_manager import ContextManager, ContextType
from nlp.intent_classifier import IntentClassifier, TrainingExample

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VANTEX_AI API",
    description="API for VANTEX_AI's NLP capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP components
text_processor = TextProcessor()
intent_recognizer = IntentRecognizer(use_ml=True)
context_manager = ContextManager(
    user_id="default_user",
    session_id=f"session_{int(datetime.now().timestamp())}",
    max_history=50
)

# Try to load intent classifier if model exists
intent_classifier = None
try:
    intent_classifier = IntentClassifier("models/intent/intent_classifier.joblib")
    logger.info("Loaded pre-trained intent classifier")
except Exception as e:
    logger.warning(f"Could not load intent classifier: {e}")
    intent_classifier = None

# Request/Response Models
class ProcessTextRequest(BaseModel):
    text: str
    language: str = "en"
    return_tokens: bool = False
    return_entities: bool = True
    return_sentiment: bool = True

class ProcessTextResponse(BaseModel):
    text: str
    language: str
    sentences: List[Dict[str, Any]]
    tokens: Optional[List[Dict[str, Any]]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    sentiment: Optional[Dict[str, float]] = None

class RecognizeIntentRequest(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None
    language: str = "en"

class IntentResponse(BaseModel):
    intent: str
    confidence: float
    text: str
    slots: List[Dict[str, Any]] = []

class ContextUpdate(BaseModel):
    key: str
    value: Any
    context_type: str = "conversation"
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = {}

class ConversationTurn(BaseModel):
    role: str
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "VANTEX_AI API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/process",
            "/intent",
            "/context",
            "/conversation"
        ]
    }

@app.post("/process", response_model=ProcessTextResponse)
async def process_text(request: ProcessTextRequest):
    """Process text with the NLP pipeline."""
    try:
        # Process the text
        sentences = text_processor.process_text(
            request.text,
            language=request.language
        )
        
        # Prepare response
        response = {
            "text": request.text,
            "language": request.language,
            "sentences": [
                {
                    "text": sent.text,
                    "tokens": [
                        {
                            "text": token.text,
                            "lemma": token.lemma_,
                            "pos": token.pos_,
                            "tag": token.tag_,
                            "dep": token.dep_,
                            "is_alpha": token.is_alpha,
                            "is_stop": token.is_stop
                        }
                        for token in sent.tokens
                    ] if request.return_tokens else None,
                    "entities": [
                        {
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char
                        }
                        for ent in sent.entities
                    ] if request.return_entities and hasattr(sent, 'entities') else None
                }
                for sent in sentences
            ]
        }
        
        # Add sentiment if requested
        if request.return_sentiment:
            sentiment = text_processor.analyze_sentiment(request.text)
            response["sentiment"] = {
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intent", response_model=IntentResponse)
async def recognize_intent(request: RecognizeIntentRequest):
    """Recognize intent from text."""
    try:
        # Use the intent recognizer
        intent = await intent_recognizer.recognize_intent(
            text=request.text,
            context=request.context
        )
        
        # If we have a classifier, use it to get more accurate results
        if intent_classifier and intent.confidence < 0.8:  # Only use ML if confidence is low
            ml_result = intent_classifier.predict(request.text)[0]
            if ml_result["confidence"] > 0.7:  # Only use if ML is confident
                intent.type = IntentType[ml_result["intent"].upper()]
                intent.confidence = ml_result["confidence"]
        
        # Update context with the recognized intent
        context_manager.add_to_history(
            role="user",
            content=request.text,
            intent={
                "type": intent.type.name,
                "confidence": intent.confidence,
                "slots": [{"name": s.name, "value": s.value, "confidence": s.confidence} 
                         for s in intent.slots]
            }
        )
        
        return {
            "intent": intent.type.name,
            "confidence": intent.confidence,
            "text": request.text,
            "slots": [
                {
                    "name": slot.name,
                    "value": slot.value,
                    "confidence": slot.confidence
                }
                for slot in intent.slots
            ]
        }
        
    except Exception as e:
        logger.error(f"Error recognizing intent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context/{key}")
async def get_context(key: str, context_type: str = "conversation"):
    """Get a context value."""
    try:
        ctx_type = ContextType[context_type.upper()]
        value = context_manager.get_context(key, context_type=ctx_type)
        
        if value is None:
            raise HTTPException(status_code=404, detail=f"Context '{key}' not found")
            
        return {"key": key, "value": value}
        
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid context type: {context_type}")
    except Exception as e:
        logger.error(f"Error getting context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context")
async def set_context(update: ContextUpdate):
    """Set a context value."""
    try:
        ctx_type = ContextType[update.context_type.upper()]
        context_manager.set_context(
            key=update.key,
            value=update.value,
            context_type=ctx_type,
            ttl=update.ttl,
            **update.metadata
        )
        
        return {"status": "success", "message": f"Context '{update.key}' updated"}
        
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid context type: {update.context_type}")
    except Exception as e:
        logger.error(f"Error setting context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation")
async def get_conversation(limit: int = 10):
    """Get conversation history."""
    try:
        history = list(context_manager.history)[:limit]
        return {"conversation": history}
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def start_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    logger.info(f"Starting VANTEX_AI API on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        reload=True
    )

if __name__ == "__main__":
    start_api()
