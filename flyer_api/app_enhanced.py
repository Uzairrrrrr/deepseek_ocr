"""
Enhanced Flyer Offer Detection and OCR API
Supports both unified LLM processing and traditional two-stage approach
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
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
from unified_processor import UnifiedFlyerProcessor
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
    description="API for detecting offers in flyers and extracting text using DeepSeek-OCR LLM",
    version="2.0.0"
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

# Initialize processors
USE_UNIFIED = False  # Set to True to use unified LLM processing

try:
    if USE_UNIFIED:
        logger.info("Initializing UNIFIED LLM processor (recommended)")
        unified_processor = UnifiedFlyerProcessor()
        detector = None
        ocr_processor = None
    else:
        logger.info("Initializing TWO-STAGE processing (detection + OCR)")
        logger.info("Stage 1: Using DeepSeek-OCR to detect offers")
        logger.info("Stage 2: Using DeepSeek-OCR to extract text from crops")
        unified_processor = None
        detector = OfferDetector(use_llm=True)
        ocr_processor = OCRProcessor()
except Exception as e:
    logger.error(f"Failed to initialize processors: {e}")
    raise


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
    ocr_confidence: Optional[float] = None  # OCR confidence score


class OfferResult(BaseModel):
    offer_id: str
    image_path: str
    image_url: str
    position: OfferPosition
    extracted_text: ExtractedText
    confidence: float  # Detection confidence
    has_text: Optional[bool] = None  # Whether text is present
    has_image: Optional[bool] = None  # Whether product image is present


class APIResponse(BaseModel):
    success: bool
    flyer_id: str
    total_offers: int
    offers: List[OfferResult]
    processing_time_ms: float
    processing_mode: str
    message: Optional[str] = None


@app.get("/")
async def root():
    """API health check endpoint"""
    return {
        "status": "running",
        "api": "Flyer Offer Detection API with DeepSeek-OCR",
        "version": "2.0.0",
        "processing_mode": "unified_llm" if USE_UNIFIED else "two_stage",
        "endpoints": {
            "POST /detect": "Upload flyer for offer detection and OCR",
            "GET /health": "Detailed health check",
            "GET /": "This endpoint"
        }
    }


@app.post("/detect", response_model=APIResponse)
async def detect_offers(
    file: UploadFile = File(..., description="Flyer image (JPG/PNG)"),
    flyer_id: Optional[str] = Form(None, description="Optional flyer ID for reference"),
    force_mode: Optional[str] = Query(None, description="Force 'unified' or 'two_stage' mode")
):
    """
    Detect offers in a flyer image and extract text using DeepSeek-OCR
    """
    start_time = datetime.now()
    
    # Validate file type
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
        
        # Determine processing mode
        use_unified_mode = USE_UNIFIED
        if force_mode == "two_stage":
            use_unified_mode = False
            logger.info("Forced two-stage mode via parameter")
        elif force_mode == "unified":
            use_unified_mode = True
            logger.info("Forced unified mode via parameter")
        
        # Process based on mode
        if use_unified_mode and unified_processor:
            processing_mode = "unified_llm"
            offers = await process_unified(image, flyer_id)
        else:
            processing_mode = "two_stage"
            offers = await process_two_stage(image, flyer_id)
        
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
            processing_mode=processing_mode,
            message=f"Successfully detected and processed {len(offers)} offers using {processing_mode} mode"
        )
    
    except Exception as e:
        logger.error(f"Error processing flyer: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def process_unified(image: np.ndarray, flyer_id: str) -> List[OfferResult]:
    """Process flyer using unified LLM approach"""
    logger.info("Using UNIFIED LLM processing")
    
    # Process in one pass
    detections = unified_processor.process_flyer(image)
    
    if not detections:
        logger.warning("Unified processing returned no offers")
        return []
    
    logger.info(f"Unified processing found {len(detections)} offers")
    
    offers = []
    
    for idx, detection in enumerate(detections):
        offer_id = f"{flyer_id}_offer_{idx+1}"
        
        # Extract data
        x, y, w, h = detection['bbox']
        confidence = detection['confidence']
        extracted_text = detection.get('extracted_text', {})
        
        # Crop the offer region
        crop = image[y:y+h, x:x+w]
        
        # Save cropped image
        crop_filename = f"{offer_id}.jpg"
        crop_path = OUTPUT_DIR / crop_filename
        cv2.imwrite(str(crop_path), crop)
        
        # Create offer result
        offer = OfferResult(
            offer_id=offer_id,
            image_path=str(crop_path),
            image_url=f"/outputs/flyer_crops/{crop_filename}",
            position=OfferPosition(x=x, y=y, width=w, height=h),
            extracted_text=ExtractedText(
                full_text=extracted_text.get('full_text', ''),
                product_title=extracted_text.get('product_title'),
                original_price=extracted_text.get('original_price'),
                discounted_price=extracted_text.get('discounted_price'),
                discount_percentage=extracted_text.get('discount_percentage'),
                promotional_text=extracted_text.get('promotional_text'),
                ocr_confidence=extracted_text.get('confidence', 0.0)
            ),
            confidence=confidence,
            has_text=detection.get('has_text', None),
            has_image=detection.get('has_image', None)
        )
        
        offers.append(offer)
    
    return offers


async def process_two_stage(image: np.ndarray, flyer_id: str) -> List[OfferResult]:
    """Process flyer using traditional two-stage approach"""
    logger.info("Using TWO-STAGE processing")
    
    # Step 1: Detect offers
    logger.info("Stage 1: Detecting offers...")
    detections = detector.detect_offers(image)
    
    if not detections:
        logger.warning("No offers detected")
        return []
    
    logger.info(f"Detected {len(detections)} offers")
    
    # Step 2: Process each detected offer
    offers = []
    
    for idx, detection in enumerate(detections):
        offer_id = f"{flyer_id}_offer_{idx+1}"
        
        # Extract bounding box
        x, y, w, h = detection['bbox']
        confidence = detection['confidence']
        
        # Crop the offer region with safety checks
        img_h, img_w = image.shape[:2]
        y_end = min(y + h, img_h)
        x_end = min(x + w, img_w)
        crop = image[y:y_end, x:x_end]
        
        # Save cropped image with high quality
        crop_filename = f"{offer_id}.jpg"
        crop_path = OUTPUT_DIR / crop_filename
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        logger.info(f"Stage 2: OCR for offer {idx+1}/{len(detections)}: {offer_id}")
        
        # Perform OCR on cropped image
        ocr_result = ocr_processor.extract_text(crop) if ocr_processor else {'full_text': '', 'confidence': 0.0}
        
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
                promotional_text=ocr_result.get('promotional_text'),
                ocr_confidence=ocr_result.get('confidence', 0.0)
            ),
            confidence=confidence,
            has_text=detection.get('has_text', None),
            has_image=detection.get('has_image', None)
        )
        
        offers.append(offer)
    
    return offers


@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    health_info = {
        "status": "healthy",
        "mode": "unified_llm" if USE_UNIFIED else "two_stage",
        "output_directory": str(OUTPUT_DIR),
        "timestamp": datetime.now().isoformat()
    }
    
    if USE_UNIFIED:
        health_info["processor"] = unified_processor.get_status()
    else:
        health_info["detector"] = detector.get_status()
        health_info["ocr_processor"] = ocr_processor.get_status()
    
    return health_info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
