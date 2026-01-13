"""
Offer Detection Module
Uses DeepSeek-OCR (Vision-Language Model) for intelligent offer detection
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import json
import re
import tempfile
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch

logger = logging.getLogger(__name__)


class OfferDetector:
    """
    Detects offer containers in flyer images using DeepSeek-OCR VLM
    """
    
    def __init__(self, use_llm: bool = True, model_path: str = 'deepseek-ai/DeepSeek-OCR-Small'):
        """Initialize the offer detector"""
        self.use_llm = use_llm
        self.model = None
        self.tokenizer = None
        
        if use_llm:
            try:
                logger.info(f"Loading DeepSeek-OCR for offer detection: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                self.model = self.model.eval()
                logger.info("DeepSeek-OCR loaded successfully for detection")
            except Exception as e:
                logger.error(f"Failed to load DeepSeek-OCR, falling back to traditional CV: {e}")
                self.use_llm = False
        else:
            logger.info("OfferDetector initialized with traditional CV method")
    
    
    def detect_offers(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Dict]:
        """
        Detect offer boxes in the flyer image
        
        Args:
            image: Input flyer image (BGR format)
            conf_threshold: Confidence threshold for detections
        
        Returns:
            List of dictionaries containing bbox coordinates and confidence
        """
        if self.use_llm and self.model is not None:
            return self._detect_with_llm(image)
        else:
            return self._detect_traditional(image)
    
    
    def _detect_with_llm(self, image: np.ndarray) -> List[Dict]:
        """
        Detect offers using DeepSeek-OCR as a Vision-Language Model
        This method asks the LLM to analyze the flyer and identify offer locations
        """
        logger.info("Using DeepSeek-OCR VLM for offer detection")
        
        try:
            # Convert to RGB PIL image
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            pil_image = Image.fromarray(image_rgb)
            img_h, img_w = image.shape[:2]
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                pil_image.save(tmp.name, 'JPEG', quality=95)
                tmp_path = tmp.name
            
            try:
                # Create a temporary output directory for the model
                output_dir = tempfile.mkdtemp(prefix='deepseek_ocr_')
                
                # Structured prompt for offer detection
                prompt = f"""<image>
Analyze this retail flyer image ({img_w}x{img_h} pixels) and identify all individual offer/product sections.

For each offer section, provide:
1. Bounding box coordinates as [x, y, width, height] in pixels
2. A brief description

IMPORTANT:
- Ignore logos, headers, footers, and decorative elements
- Only detect actual product offers with prices
- Each offer should be a distinct product or promotion
- Provide coordinates as precise integers

Respond ONLY with a JSON array:
[
  {{"bbox": [x, y, width, height], "description": "product name"}},
  ...
]"""
                
                # Perform inference
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=tmp_path,
                    output_path=output_dir,
                    base_size=1024,
                    image_size=1024,
                    crop_mode=False,
                    save_results=False,
                    test_compress=False
                )
                
                # Parse result
                if isinstance(result, dict):
                    text_result = result.get('text', result.get('content', result.get('output', str(result))))
                elif isinstance(result, str):
                    text_result = result
                else:
                    text_result = str(result)
                
                logger.info(f"LLM detection result: {text_result[:200]}...")
                
                # Extract JSON from response
                detections = self._parse_llm_detections(text_result, img_w, img_h)
                
                if detections:
                    logger.info(f"LLM detected {len(detections)} offers")
                    return detections
                else:
                    logger.warning("LLM detection returned no results, falling back to traditional CV")
                    return self._detect_traditional(image)
                    
            finally:
                try:
                    import os
                    import shutil
                    os.unlink(tmp_path)
                    if 'output_dir' in locals():
                        shutil.rmtree(output_dir, ignore_errors=True)
                except:
                    pass
        
        except Exception as e:
            logger.error(f"LLM detection failed: {e}", exc_info=True)
            logger.info("Falling back to traditional CV")
            return self._detect_traditional(image)
    
    
    def _parse_llm_detections(self, text: str, img_w: int, img_h: int) -> List[Dict]:
        """Parse LLM output to extract bounding boxes"""
        detections = []
        
        try:
            # Try to find JSON array in response
            json_match = re.search(r'\[\s*\{[^]]+\}\s*\]', text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                for item in parsed:
                    if 'bbox' in item and isinstance(item['bbox'], list) and len(item['bbox']) == 4:
                        x, y, w, h = item['bbox']
                        
                        # Validate and clip coordinates
                        x = max(0, min(int(x), img_w - 1))
                        y = max(0, min(int(y), img_h - 1))
                        w = max(10, min(int(w), img_w - x))
                        h = max(10, min(int(h), img_h - y))
                        
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': 0.95,  # High confidence for LLM
                            'description': item.get('description', '')
                        })
            
            else:
                # Try alternative parsing - look for coordinate patterns
                # Pattern: [x, y, width, height] or (x, y, width, height)
                coord_pattern = r'[\[(]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\])]'
                matches = re.finditer(coord_pattern, text)
                
                for match in matches:
                    x = int(match.group(1))
                    y = int(match.group(2))
                    w = int(match.group(3))
                    h = int(match.group(4))
                    
                    # Validate
                    if 0 <= x < img_w and 0 <= y < img_h and 10 < w <= img_w and 10 < h <= img_h:
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': 0.90
                        })
        
        except Exception as e:
            logger.error(f"Failed to parse LLM detections: {e}")
        
        return detections
    
    
    def _detect_traditional(self, image: np.ndarray) -> List[Dict]:
        """
        Detect offers using traditional computer vision techniques
        """
        logger.info("Using traditional CV for offer detection")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            morph,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Get image dimensions
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w
        
        detections = []
        
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area and aspect ratio
            area = w * h
            area_ratio = area / img_area
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter criteria for offer boxes
            if not (0.02 < area_ratio < 0.25):
                continue
            
            if not (0.4 < aspect_ratio < 2.5):
                continue
            
            # Check solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.6:
                continue
            
            # Check if contour is rectangular enough
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            
            if len(approx) >= 4:
                # Add some padding
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img_w - x, w + 2 * padding)
                h = min(img_h - y, h + 2 * padding)
                
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.85
                })
        
        # Remove overlapping detections
        detections = self._non_max_suppression(detections)
        
        logger.info(f"Traditional CV detected {len(detections)} offers")
        return detections
    
    
    def _non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if len(detections) == 0:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        confidences = np.array([d['confidence'] for d in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = boxes[:, 2] * boxes[:, 3]
        order = confidences.argsort()[::-1]
        
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        
        return [detections[i] for i in keep]
    
    
    def get_status(self) -> Dict:
        """Get detector status"""
        return {
            "method": "llm" if self.use_llm else "traditional_cv",
            "model_loaded": self.model is not None,
            "model_type": "deepseek-ocr-vlm" if self.use_llm else "none"
        }
