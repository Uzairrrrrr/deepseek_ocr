"""
Enhanced Testing Script for Flyer Processing System
Tests both unified and two-stage modes with detailed analysis
"""

import requests
import json
import time
from pathlib import Path
import sqlite3
from typing import Dict, List

API_URL = "http://localhost:8000"


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    """Print section divider"""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


def check_api_health():
    """Check if API is running and get status"""
    print_header("API HEALTH CHECK")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        
        if response.status_code == 200:
            health = response.json()
            print(f"\n✓ API Status: {health['status'].upper()}")
            print(f"✓ Processing Mode: {health['mode']}")
            print(f"✓ Timestamp: {health['timestamp']}")
            
            if 'processor' in health:
                proc = health['processor']
                print(f"\n  Unified Processor:")
                print(f"    - Type: {proc['type']}")
                print(f"    - Model: {proc['model']}")
                print(f"    - Initialized: {proc['initialized']}")
                print(f"    - Capabilities: {', '.join(proc['capabilities'])}")
            
            if 'detector' in health:
                det = health['detector']
                print(f"\n  Detector:")
                print(f"    - Method: {det['method']}")
                print(f"    - Model Loaded: {det['model_loaded']}")
            
            if 'ocr_processor' in health:
                ocr = health['ocr_processor']
                print(f"\n  OCR Processor:")
                print(f"    - Engine: {ocr['engine']}")
                print(f"    - Languages: {', '.join(ocr['languages'])}")
                print(f"    - Initialized: {ocr['initialized']}")
            
            return True
        else:
            print(f"\n✗ API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to API at {API_URL}")
        print(f"  Make sure the server is running:")
        print(f"    cd flyer_api && python app_enhanced.py")
        return False
    except Exception as e:
        print(f"\n✗ Error checking API: {e}")
        return False


def test_flyer_detection(image_path: Path, mode: str = None) -> Dict:
    """Test flyer detection with specified mode"""
    
    mode_str = f" (force {mode})" if mode else ""
    print_section(f"Testing: {image_path.name}{mode_str}")
    
    start_time = time.time()
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            data = {'flyer_id': f'test_{image_path.stem}_{int(time.time())}'}
            
            # Add mode parameter if specified
            params = {'force_mode': mode} if mode else {}
            
            print(f"  📤 Uploading flyer...")
            response = requests.post(
                f"{API_URL}/detect",
                files=files,
                data=data,
                params=params,
                timeout=60
            )
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n  ✓ SUCCESS")
            print(f"    Flyer ID: {result['flyer_id']}")
            print(f"    Total Offers: {result['total_offers']}")
            print(f"    Processing Mode: {result['processing_mode']}")
            print(f"    Server Time: {result['processing_time_ms']:.1f}ms")
            print(f"    Total Time: {request_time * 1000:.1f}ms")
            
            return result
        else:
            print(f"\n  ✗ ERROR: {response.status_code}")
            print(f"    {response.text}")
            return None
            
    except Exception as e:
        print(f"\n  ✗ EXCEPTION: {e}")
        return None


def analyze_offers(result: Dict):
    """Analyze and display offer details"""
    
    if not result or result['total_offers'] == 0:
        print("\n  ⚠ No offers detected")
        return
    
    print_section("OFFER ANALYSIS")
    
    offers = result['offers']
    
    # Statistics
    with_text = sum(1 for o in offers if o['extracted_text']['full_text'])
    with_product = sum(1 for o in offers if o['extracted_text']['product_title'])
    with_price = sum(1 for o in offers if o['extracted_text']['discounted_price'])
    with_discount = sum(1 for o in offers if o['extracted_text']['discount_percentage'])
    
    print(f"\n  📊 Statistics:")
    print(f"    Total Offers: {len(offers)}")
    print(f"    With Text: {with_text}/{len(offers)} ({with_text/len(offers)*100:.0f}%)")
    print(f"    With Product: {with_product}/{len(offers)} ({with_product/len(offers)*100:.0f}%)")
    print(f"    With Price: {with_price}/{len(offers)} ({with_price/len(offers)*100:.0f}%)")
    print(f"    With Discount: {with_discount}/{len(offers)} ({with_discount/len(offers)*100:.0f}%)")
    
    # Individual offers
    print(f"\n  📋 Individual Offers:")
    
    for idx, offer in enumerate(offers, 1):
        print(f"\n    Offer #{idx}: {offer['offer_id']}")
        print(f"      Confidence: {offer['confidence']:.2f}")
        print(f"      Position: ({offer['position']['x']}, {offer['position']['y']})")
        print(f"      Size: {offer['position']['width']}×{offer['position']['height']}px")
        
        text = offer['extracted_text']
        
        if text['product_title']:
            print(f"      ✓ Product: {text['product_title']}")
        else:
            print(f"      ✗ Product: Not detected")
        
        if text['discounted_price']:
            price_str = text['discounted_price']
            if text['original_price']:
                price_str = f"{text['original_price']} → {text['discounted_price']}"
            print(f"      ✓ Price: {price_str}")
        else:
            print(f"      ✗ Price: Not detected")
        
        if text['discount_percentage']:
            print(f"      ✓ Discount: {text['discount_percentage']}")
        
        if text['promotional_text']:
            promo = text['promotional_text'][:60]
            print(f"      ✓ Promo: {promo}...")
        
        if text['full_text']:
            preview = text['full_text'].replace('\n', ' ')[:100]
            print(f"      📄 Text ({len(text['full_text'])} chars): \"{preview}...\"")
        else:
            print(f"      ⚠ Text: EMPTY")


def compare_modes(image_path: Path):
    """Compare unified and two-stage modes"""
    
    print_header(f"MODE COMPARISON: {image_path.name}")
    
    results = {}
    
    # Test unified mode
    print("\n🔵 Testing UNIFIED mode...")
    results['unified'] = test_flyer_detection(image_path, mode='unified')
    
    # Test two-stage mode
    print("\n🟢 Testing TWO-STAGE mode...")
    results['two_stage'] = test_flyer_detection(image_path, mode='two_stage')
    
    # Compare results
    print_section("COMPARISON")
    
    if results['unified'] and results['two_stage']:
        u = results['unified']
        t = results['two_stage']
        
        print(f"\n  📊 Detection:")
        print(f"    Unified:   {u['total_offers']} offers")
        print(f"    Two-Stage: {t['total_offers']} offers")
        
        print(f"\n  ⏱️  Performance:")
        print(f"    Unified:   {u['processing_time_ms']:.1f}ms")
        print(f"    Two-Stage: {t['processing_time_ms']:.1f}ms")
        speedup = t['processing_time_ms'] / u['processing_time_ms']
        print(f"    Speedup:   {speedup:.2f}x")
        
        # Compare text extraction
        u_with_text = sum(1 for o in u['offers'] if o['extracted_text']['full_text'])
        t_with_text = sum(1 for o in t['offers'] if o['extracted_text']['full_text'])
        
        print(f"\n  📝 Text Extraction:")
        print(f"    Unified:   {u_with_text}/{u['total_offers']} offers with text")
        print(f"    Two-Stage: {t_with_text}/{t['total_offers']} offers with text")
        
        # Recommendation
        print(f"\n  💡 Recommendation:")
        if speedup > 1.2 and u['total_offers'] >= t['total_offers']:
            print(f"    ✓ Use UNIFIED mode (faster and equal/better detection)")
        elif u['total_offers'] > t['total_offers']:
            print(f"    ✓ Use UNIFIED mode (detects more offers)")
        elif t_with_text > u_with_text:
            print(f"    ⚠ Use TWO-STAGE mode (better text extraction)")
        else:
            print(f"    ✓ Use UNIFIED mode (default)")
    
    return results


def view_database():
    """View database contents"""
    
    print_header("DATABASE CONTENTS")
    
    try:
        conn = sqlite3.connect('flyer_detection.db')
        cursor = conn.cursor()
        
        # Get recent flyers
        cursor.execute("""
            SELECT * FROM flyers 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        flyers = cursor.fetchall()
        
        if not flyers:
            print("\n  No flyers in database")
            return
        
        print(f"\n  📁 Recent Flyers: {len(flyers)}")
        
        for flyer in flyers:
            print(f"\n    Flyer: {flyer[1]}")
            print(f"      File: {flyer[2]}")
            print(f"      Offers: {flyer[3]}")
            print(f"      Time: {flyer[4]:.1f}ms")
            print(f"      Created: {flyer[5]}")
        
        conn.close()
        
    except Exception as e:
        print(f"\n  ✗ Error reading database: {e}")


def main():
    """Main test execution"""
    
    print_header("FLYER PROCESSING SYSTEM - ENHANCED TESTS")
    
    # 1. Health check
    if not check_api_health():
        return
    
    # 2. Find test images
    test_dir = Path("test_images")
    if not test_dir.exists():
        print(f"\n⚠ No test_images/ directory found")
        return
    
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not test_images:
        print(f"\n⚠ No test images found in test_images/")
        return
    
    print(f"\n📁 Found {len(test_images)} test images")
    
    # 3. Test first image with both modes
    test_image = test_images[0]
    results = compare_modes(test_image)
    
    # 4. Show detailed analysis
    if results.get('unified'):
        print_header("UNIFIED MODE - DETAILED ANALYSIS")
        analyze_offers(results['unified'])
    
    # 5. View database
    view_database()
    
    # 6. Summary
    print_header("TEST SUMMARY")
    print("\n  ✓ All tests completed")
    print(f"  📊 Results saved to database")
    print(f"  📁 Cropped images in: outputs/flyer_crops/")
    
    print(f"\n  💡 Next Steps:")
    print(f"    1. Review cropped images for quality")
    print(f"    2. Check text extraction accuracy")
    print(f"    3. Adjust prompts if needed")
    print(f"    4. Test with more diverse flyers")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
