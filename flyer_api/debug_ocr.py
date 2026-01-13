"""
Debug script to test DeepSeek-OCR directly
"""

import cv2
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image

print("\n" + "="*70)
print("DEEPSEEK-OCR DEBUG TEST")
print("="*70)

# Load model
print("\n1. Loading DeepSeek-OCR model...")
try:
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
    model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
    model = model.eval()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Find a cropped image
print("\n2. Finding test image...")
crop_images = list(Path("outputs/flyer_crops").glob("*.jpg"))

if not crop_images:
    print("✗ No cropped images found!")
    print("  Run the API first to generate cropped images")
    sys.exit(1)

test_image = crop_images[0]
print(f"✓ Using image: {test_image}")

# Test different prompts
prompts = [
    "<image>\nFree OCR.",
    "<image>\n<|grounding|>OCR this image.",
    "<image>\nExtract all text from this image.",
    "<image>\n<|grounding|>Convert the document to markdown.",
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n{'='*70}")
    print(f"TEST #{i}: {prompt[:50]}...")
    print(f"{'='*70}")
    
    try:
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(test_image),
            base_size=512,
            image_size=384,
            crop_mode=False,
            save_results=False,
            test_compress=False
        )
        
        print(f"\nResult Type: {type(result)}")
        print(f"Result: {result}")
        
        # Try to extract text
        if isinstance(result, dict):
            print(f"\nDictionary keys: {result.keys()}")
            for key in ['text', 'content', 'output', 'response']:
                if key in result:
                    print(f"\n{key}: {result[key][:200] if result[key] else 'EMPTY'}")
        elif isinstance(result, str):
            print(f"\nString result: {result[:500]}")
        else:
            print(f"\nUnexpected type: {type(result)}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("Debug test complete")
print(f"{'='*70}\n")
