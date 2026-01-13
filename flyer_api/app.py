"""
Flyer Offer Detection and OCR API
Uses YOLOv8 for offer detection and DeepSeek-OCR for text extraction
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from pathlib import Path
import uuid
from datetime import datetime
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from detection_module import OfferDetector
from ocr_module import OCRProcessor
from database import save_flyer_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Flyer Offer Detection API",
    description="API for detecting offers in flyers and extracting text using DeepSeek-OCR",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OUTPUT_DIR = Path(__file__).parent / "outputs" / "flyer_crops"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize detector and OCR processor
detector = OfferDetector()
ocr_processor = OCRProcessor()


# Pydantic models for response
class OfferPosition(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ExtractedText(BaseModel):
    full_text: str
    product_title: Optional[str] = None
    original_price: Optional[str] = None
    discounted_price: Optional[str] = None
    discount_percentage: Optional[str] = None
    promotional_text: Optional[str] = None


class OfferResult(BaseModel):
    offer_id: str
    image_path: str
    image_url: str
    position: OfferPosition
    extracted_text: ExtractedText
    confidence: float


class APIResponse(BaseModel):
    success: bool
    flyer_id: str
    total_offers: int
    offers: List[OfferResult]
    processing_time_ms: float
    message: Optional[str] = None


@app.get("/")
async def root():
    """API health check endpoint"""
    return {
        "status": "running",
        "api": "Flyer Offer Detection API with DeepSeek-OCR",
        "version": "1.0.0",
        "endpoints": {
            "POST /detect": "Upload flyer for offer detection and OCR",
            "GET /health": "Detailed health check",
            "GET /": "This endpoint"
        }
    }


@app.post("/detect", response_model=APIResponse)
async def detect_offers(
    file: UploadFile = File(..., description="Flyer image (JPG/PNG)"),
    flyer_id: Optional[str] = Form(None, description="Optional flyer ID for reference")
):
    """
    Detect offers in a flyer image and extract text using DeepSeek-OCR
    """
    start_time = datetime.now()
    
    # Validate file type - check extension
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Only JPG and PNG are supported. Got: {file_ext}"
        )
    
    # Generate flyer ID if not provided
    if not flyer_id:
        flyer_id = f"flyer_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"Processing flyer: {flyer_id}")
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        logger.info(f"Image shape: {image.shape}")
        
        # Step 1: Detect offers
        logger.info("Starting offer detection...")
        detections = detector.detect_offers(image)
        
        if not detections:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return APIResponse(
                success=True,
                flyer_id=flyer_id,
                total_offers=0,
                offers=[],
                processing_time_ms=processing_time,
                message="No offers detected in the flyer"
            )
        
        logger.info(f"Detected {len(detections)} offers")
        
        # Step 2: Process each detected offer
        offers = []
        
        for idx, detection in enumerate(detections):
            offer_id = f"{flyer_id}_offer_{idx+1}"
            
            # Extract bounding box
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Crop the offer region
            crop = image[y:y+h, x:x+w]
            
            # Save cropped image
            crop_filename = f"{offer_id}.jpg"
            crop_path = OUTPUT_DIR / crop_filename
            cv2.imwrite(str(crop_path), crop)
            
            logger.info(f"Processing offer {idx+1}/{len(detections)}: {offer_id}")
            
            # Step 3: Perform OCR
            ocr_result = ocr_processor.extract_text(crop)
            
            # Create offer result
            offer = OfferResult(
                offer_id=offer_id,
                image_path=str(crop_path),
                image_url=f"/outputs/flyer_crops/{crop_filename}",
                position=OfferPosition(x=x, y=y, width=w, height=h),
                extracted_text=ExtractedText(
                    full_text=ocr_result.get('full_text', ''),
                    product_title=ocr_result.get('product_title'),
                    original_price=ocr_result.get('original_price'),
                    discounted_price=ocr_result.get('discounted_price'),
                    discount_percentage=ocr_result.get('discount_percentage'),
                    promotional_text=ocr_result.get('promotional_text')
                ),
                confidence=confidence
            )
            
            offers.append(offer)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Processing completed in {processing_time:.2f}ms")
        
        # Save to database
        try:
            save_flyer_results(
                flyer_data={
                    'flyer_id': flyer_id,
                    'filename': file.filename,
                    'total_offers': len(offers),
                    'processing_time_ms': processing_time
                },
                offers_data=[offer.dict() for offer in offers]
            )
            logger.info("Results saved to database")
        except Exception as e:
            logger.error(f"Database save failed: {e}")
        
        return APIResponse(
            success=True,
            flyer_id=flyer_id,
            total_offers=len(offers),
            offers=offers,
            processing_time_ms=processing_time,
            message=f"Successfully detected and processed {len(offers)} offers"
        )
    
    except Exception as e:
        logger.error(f"Error processing flyer: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "detector": detector.get_status(),
        "ocr_processor": ocr_processor.get_status(),
        "output_directory": str(OUTPUT_DIR),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
