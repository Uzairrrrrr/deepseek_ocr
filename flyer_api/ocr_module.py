"""
Enhanced OCR Processing Module with DeepSeek-OCR
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any
import re
import logging
import json
from pathlib import Path
from PIL import Image
import torch
import tempfile
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processor using DeepSeek-OCR"""
    
    def __init__(self, engine: str = 'deepseek', languages: List[str] = ['en', 'ar'], 
                 model_path: str = 'deepseek-ai/DeepSeek-OCR-Small'):
        self.engine_name = engine
        self.languages = languages
        self.model_path = model_path
        self.ocr_engine = None
        self.tokenizer = None
        self.device = 'cpu'  # Force CPU due to limited VRAM
        
        logger.info(f"Initializing OCR processor with engine: {engine}")
        logger.info(f"Device: {self.device}")
        
        try:
            self._init_deepseek_ocr()
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek-OCR: {e}")
            self.engine_name = 'none'
    
    def _init_deepseek_ocr(self):
        """Initialize DeepSeek-OCR"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading DeepSeek-OCR from: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            self.ocr_engine = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            
            self.ocr_engine = self.ocr_engine.eval()
            logger.info("Model loaded on CPU")
            logger.info("DeepSeek-OCR initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek-OCR: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text from image with structured prompt"""
        if self.engine_name == 'deepseek':
            raw_text = self._extract_deepseek_ocr(image)
        else:
            raw_text = "OCR not available"
        
        # Try to parse as JSON first (structured response)
        parsed_data = self._parse_structured_response(raw_text)
        
        # If JSON parsing fails, fall back to regex parsing
        if not parsed_data['full_text']:
            parsed_data = self._parse_offer_text(raw_text)
        
        return {
            'full_text': parsed_data.get('full_text', raw_text),
            **parsed_data
        }
    
    def _parse_structured_response(self, text: str) -> Dict[str, Optional[str]]:
        """Parse structured JSON response from LLM"""
        result = {
            'product_title': None,
            'original_price': None,
            'discounted_price': None,
            'discount_percentage': None,
            'promotional_text': None,
            'full_text': ''
        }
        
        if not text:
            return result
        
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Extract fields from JSON
                result['product_title'] = parsed.get('product_title')
                result['original_price'] = parsed.get('original_price')
                result['discounted_price'] = parsed.get('discounted_price')
                result['discount_percentage'] = parsed.get('discount_percentage')
                result['promotional_text'] = parsed.get('promotional_text')
                result['full_text'] = parsed.get('all_text', text)
                
                logger.info("Successfully parsed structured JSON response")
                return result
        
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing failed, will use fallback: {e}")
        except Exception as e:
            logger.error(f"Structured parsing error: {e}")
        
        # If JSON parsing fails, return empty dict (will trigger fallback)
        return result
    
    def _extract_deepseek_ocr(self, image: np.ndarray) -> str:
        """Extract text using DeepSeek-OCR with structured prompt"""
        try:
            # Convert to RGB PIL image
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            pil_image = Image.fromarray(image_rgb)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                pil_image.save(tmp.name, 'JPEG', quality=95)
                tmp_path = tmp.name
            
            try:
                # Structured prompt for retail offer extraction
                prompt = """<image>
Extract ALL text from this retail product offer image.

Provide the following information in JSON format:
{
  "product_title": "Full product name",
  "original_price": "XX.XX" (if shown),
  "discounted_price": "XX.XX",
  "discount_percentage": "XX%" (if shown),
  "promotional_text": "Any special offers or conditions",
  "all_text": "Complete text from the image"
}

IMPORTANT:
- Extract text in both English and Arabic
- Preserve price formats exactly (e.g., 21.95, ر.س 21.95)
- Include ALL visible text in the "all_text" field
- If a field is not present, use null

Respond ONLY with valid JSON."""
                
                # Perform inference with optimized parameters for cropped offers
                result = self.ocr_engine.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=tmp_path,
                    base_size=512,  # Smaller for cropped offers
                    image_size=384,  # Adjusted for offer images
                    crop_mode=False,
                    save_results=False,
                    test_compress=False
                )
                
                # Extract text - DeepSeek returns different formats
                if isinstance(result, dict):
                    # Try different keys
                    text = result.get('text', 
                           result.get('content', 
                           result.get('output',
                           result.get('response', str(result)))))
                elif isinstance(result, str):
                    text = result
                else:
                    text = str(result)
                
                # Clean up the text
                text = text.strip()
                
                # Remove markdown code blocks if present
                text = re.sub(r'```json\s*', '', text)
                text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
                text = re.sub(r'`', '', text)
                
                logger.info(f"Extracted {len(text)} characters of text")
                logger.debug(f"Raw OCR result: {text[:300]}...")
                return text
                
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"DeepSeek-OCR extraction failed: {e}", exc_info=True)
            return ""
    
    def _parse_offer_text(self, text: str) -> Dict[str, Optional[str]]:
        """Parse extracted text with better pattern matching"""
        if not text:
            return {
                'product_title': None,
                'original_price': None,
                'discounted_price': None,
                'discount_percentage': None,
                'promotional_text': None
            }
        
        result = {
            'product_title': None,
            'original_price': None,
            'discounted_price': None,
            'discount_percentage': None,
            'promotional_text': None
        }
        
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # Find prices with better patterns
        # Supports: 21.95, ر.س 21.95, 21-95, etc.
        price_pattern = r'(?:ر\.س|SAR|SR)?\s*(\d+)[.,\-](\d{1,2})\s*(?:ر\.س|SAR|SR)?'
        prices_found = []
        
        for line in lines:
            matches = re.finditer(price_pattern, line)
            for match in matches:
                price_str = f"{match.group(1)}.{match.group(2)}"
                prices_found.append((price_str, line))
        
        # Find discount percentages
        discount_pattern = r'(\d+)\s*%'
        for line in lines:
            match = re.search(discount_pattern, line)
            if match:
                result['discount_percentage'] = f"{match.group(1)}%"
                break
        
        # Assign prices - last price is usually discounted
        if len(prices_found) >= 2:
            result['original_price'] = prices_found[0][0]
            result['discounted_price'] = prices_found[-1][0]
        elif len(prices_found) == 1:
            result['discounted_price'] = prices_found[0][0]
        
        # Extract product title (first substantial line)
        for line in lines[:5]:
            # Skip price-only lines
            if re.search(r'^\s*[\d\s.,\-ر\.سSAR]+\s*$', line):
                continue
            # Skip "Per Pack", "Each", etc.
            if re.search(r'^(per|each|للحبة|للعلبة)', line, re.IGNORECASE):
                continue
            if len(line) > 5:
                result['product_title'] = line
                break
        
        # Promotional text
        promo_keywords = ['off', 'sale', 'offer', 'عرض', 'خصم', 'تخفيض', 'or', 'أو']
        for line in lines:
            if any(keyword in line.lower() for keyword in promo_keywords):
                if result['promotional_text']:
                    result['promotional_text'] += " | " + line
                else:
                    result['promotional_text'] = line
        
        return result
    
    def get_status(self) -> Dict:
        """Get OCR processor status"""
        return {
            "engine": self.engine_name,
            "languages": self.languages,
            "initialized": self.ocr_engine is not None,
            "device": self.device
        }
