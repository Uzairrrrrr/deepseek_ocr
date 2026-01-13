"""
Simple Demo Script - Test the Flyer API with a single image
"""

import requests
import json
from pathlib import Path
import sys

API_URL = "http://localhost:8000"


def demo():
    """Run a simple demo of the flyer detection API"""
    
    print("\n" + "="*70)
    print("  FLYER DETECTION API - SIMPLE DEMO")
    print("="*70)
    
    # Check if API is running
    print("\n1. Checking API status...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ API is running")
            health = response.json()
            print(f"   ✓ Mode: {health['mode']}")
        else:
            print("   ✗ API returned error")
            return
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Cannot connect to API at {API_URL}")
        print(f"\n   Please start the server first:")
        print(f"     cd flyer_api")
        print(f"     python app_enhanced.py")
        return
    
    # Find test image
    print("\n2. Looking for test images...")
    test_dir = Path("test_images")
    
    if not test_dir.exists():
        print(f"   ✗ Directory test_images/ not found")
        print(f"   Please create it and add flyer images")
        return
    
    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not images:
        print(f"   ✗ No images found in test_images/")
        print(f"   Please add JPG or PNG flyer images")
        return
    
    test_image = images[0]
    print(f"   ✓ Using: {test_image.name}")
    
    # Send request
    print("\n3. Sending image to API...")
    print(f"   (This may take 30-60 seconds on first run while model loads)")
    
    with open(test_image, 'rb') as f:
        files = {'file': (test_image.name, f, 'image/jpeg')}
        data = {'flyer_id': 'demo_001'}
        
        try:
            response = requests.post(
                f"{API_URL}/detect",
                files=files,
                data=data,
                timeout=120
            )
        except requests.exceptions.Timeout:
            print("   ✗ Request timed out")
            print("   The model may still be loading, try again")
            return
    
    # Process response
    if response.status_code != 200:
        print(f"   ✗ Error: {response.status_code}")
        print(f"   {response.text}")
        return
    
    result = response.json()
    
    print(f"   ✓ Success!")
    
    # Display results
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    
    print(f"\n  Flyer ID: {result['flyer_id']}")
    print(f"  Total Offers Detected: {result['total_offers']}")
    print(f"  Processing Mode: {result['processing_mode']}")
    print(f"  Processing Time: {result['processing_time_ms']:.1f}ms")
    
    if result['total_offers'] == 0:
        print("\n  ⚠ No offers detected in this image")
        print("  Try with a different flyer image")
        return
    
    print(f"\n  Offers:")
    for i, offer in enumerate(result['offers'], 1):
        print(f"\n  ─── Offer #{i} ───")
        print(f"    Position: ({offer['position']['x']}, {offer['position']['y']})")
        print(f"    Size: {offer['position']['width']} × {offer['position']['height']} px")
        print(f"    Confidence: {offer['confidence']:.2%}")
        
        text = offer['extracted_text']
        
        if text['product_title']:
            print(f"    Product: {text['product_title']}")
        
        if text['discounted_price']:
            price_str = text['discounted_price']
            if text['original_price']:
                price_str = f"{text['original_price']} → {text['discounted_price']}"
            print(f"    Price: {price_str}")
        
        if text['discount_percentage']:
            print(f"    Discount: {text['discount_percentage']}")
        
        if text['full_text']:
            preview = text['full_text'].replace('\n', ' ')[:80]
            print(f"    Text: \"{preview}...\"")
        
        print(f"    Image: {offer['image_path']}")
    
    print("\n" + "="*70)
    print("  SUCCESS!")
    print("="*70)
    print(f"\n  Cropped images saved to: outputs/flyer_crops/")
    print(f"  Total offers found: {result['total_offers']}")
    
    print("\n  Next steps:")
    print("    • Review cropped images for quality")
    print("    • Run full tests: python test_enhanced.py")
    print("    • Read docs: IMPLEMENTATION_GUIDE.md")
    print("\n")


if __name__ == "__main__":
    demo()
