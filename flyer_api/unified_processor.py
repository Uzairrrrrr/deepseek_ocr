"""
CPU-Only Unified Processor - For systems with limited GPU memory
Forces CPU usage to avoid CUDA OOM errors
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import json
import re
import tempfile
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
import os

logger = logging.getLogger(__name__)


class UnifiedFlyerProcessor:
    """
    CPU-only unified processor for DeepSeek-OCR
    Optimized for systems with limited GPU memory
    """
    
    def __init__(self, model_path: str = 'deepseek-ai/DeepSeek-OCR'):
        """
        Initialize unified processor in CPU-only mode
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
        # FORCE CPU MODE
        self.device = torch.device('cpu')
        logger.info("🖥️  CPU-ONLY MODE (GPU has insufficient memory)")
        
        # Disable CUDA completely
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        try:
            logger.info(f"Loading DeepSeek-OCR for CPU processing: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # Load model in CPU mode with float32
            logger.info("Loading model with float32 for CPU (this may take a minute)...")
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # CPU needs float32
                device_map=None,
                low_cpu_mem_usage=True  # Optimize CPU memory
            )
            
            # Ensure on CPU
            self.model = self.model.cpu()
            self.model.eval()
            
            # Disable gradients
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.initialized = True
            logger.info(f"✓ CPU-only processor initialized successfully")
            logger.info(f"⚠️  Note: CPU inference will be slower (~30-60 seconds per image)")
            
        except Exception as e:
            logger.error(f"Failed to initialize CPU processor: {e}", exc_info=True)
            raise
    
    
    def process_flyer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process flyer on CPU
        """
        if not self.initialized:
            raise RuntimeError("Processor not initialized")
        
        logger.info("🔄 Processing flyer with CPU-only LLM")
        
        tmp_path = None
        output_dir = None
        
        try:
            # Convert to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            pil_image = Image.fromarray(image_rgb)
            img_h, img_w = image.shape[:2]
            
            logger.info(f"Processing {img_w}x{img_h} image on CPU (please wait ~60 seconds)...")
            
            # Save temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                pil_image.save(tmp.name, 'JPEG', quality=95)
                tmp_path = tmp.name
            
            output_dir = tempfile.mkdtemp(prefix='deepseek_ocr_')
            
            # Optimized prompt for CPU processing
            prompt = f"""<image>
Analyze this retail flyer ({img_w}x{img_h} pixels) and find ALL product offers with prices.

For EACH offer, return:
[
  {{
    "bbox": [x, y, width, height],
    "product_title": "Product name",
    "original_price": "XX.XX" or null,
    "discounted_price": "XX.XX",
    "discount_percentage": "XX%" or null,
    "promotional_text": "Promo text" or null,
    "all_text": "Full text"
  }}
]

Rules:
- Only product offers with prices
- Ignore headers/footers/logos
- Extract English and Arabic text
- Return ONLY valid JSON array"""
            
            logger.info("⏳ Running CPU inference (this will take 30-60 seconds)...")
            
            # CPU inference with no_grad
            with torch.no_grad():
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=tmp_path,
                    output_path=output_dir,
                    base_size=640,  # SMALL model config
                    image_size=640,  # SMALL model config
                    crop_mode=False,
                    save_results=False,
                    test_compress=False
                )
            
            # Extract text
            if isinstance(result, dict):
                text_result = result.get('text', result.get('content', str(result)))
            elif isinstance(result, str):
                text_result = result
            else:
                text_result = str(result)
            
            logger.info(f"✓ CPU inference complete! Result: {len(text_result)} chars")
            
            # Parse response
            offers = self._parse_unified_response(text_result, img_w, img_h)
            
            if offers:
                logger.info(f"✓ Detected {len(offers)} offers")
            else:
                logger.warning("⚠️  No offers detected")
                logger.debug(f"Response: {text_result[:300]}")
            
            return offers
            
        except Exception as e:
            logger.error(f"CPU processing failed: {e}", exc_info=True)
            return []
        
        finally:
            try:
                import shutil
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                if output_dir and os.path.exists(output_dir):
                    shutil.rmtree(output_dir, ignore_errors=True)
            except:
                pass
    
    
    def _parse_unified_response(self, text: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
        """Parse LLM response"""
        offers = []
        
        try:
            # Clean response
            text = text.strip()
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*$', '', text)
            
            logger.debug(f"Parsing response: {text[:200]}...")
            
            # Find JSON
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                if not isinstance(parsed, list):
                    parsed = [parsed]
                
                logger.info(f"Found {len(parsed)} items in JSON")
                
                for idx, item in enumerate(parsed):
                    if not isinstance(item, dict) or 'bbox' not in item:
                        logger.warning(f"Item {idx} invalid, skipping")
                        continue
                    
                    bbox = item['bbox']
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        logger.warning(f"Item {idx} bbox invalid, skipping")
                        continue
                    
                    try:
                        x, y, w, h = [int(v) for v in bbox]
                    except:
                        continue
                    
                    # Clamp to bounds
                    x = max(0, min(x, img_w - 1))
                    y = max(0, min(y, img_h - 1))
                    w = max(10, min(w, img_w - x))
                    h = max(10, min(h, img_h - y))
                    
                    offer = {
                        'bbox': (x, y, w, h),
                        'confidence': 0.9,
                        'extracted_text': {
                            'full_text': item.get('all_text', ''),
                            'product_title': item.get('product_title'),
                            'original_price': item.get('original_price'),
                            'discounted_price': item.get('discounted_price'),
                            'discount_percentage': item.get('discount_percentage'),
                            'promotional_text': item.get('promotional_text')
                        }
                    }
                    
                    offers.append(offer)
                    logger.info(f"  ✓ Offer {idx+1}: {offer['extracted_text']['product_title']}")
                
            else:
                logger.warning("No JSON array in response")
        
        except Exception as e:
            logger.error(f"Parse error: {e}", exc_info=True)
        
        return offers
    
    
    def get_status(self) -> Dict:
        """Get status"""
        return {
            "type": "unified_vlm",
            "model": "deepseek-ocr",
            "device": "cpu",
            "mode": "cpu_only",
            "cuda_available": False,
            "initialized": self.initialized,
            "note": "CPU-only mode for low-memory GPUs",
            "capabilities": ["detection", "ocr", "structured_extraction"]
        }