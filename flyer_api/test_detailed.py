import requests
import json
from pathlib import Path
import sqlite3

API_URL = "http://localhost:8000"

def view_database():
    """View what's saved in the database"""
    print("\n" + "="*70)
    print("DATABASE CONTENTS")
    print("="*70)
    
    try:
        conn = sqlite3.connect('flyer_detection.db')
        cursor = conn.cursor()
        
        # Get flyers
        cursor.execute("SELECT * FROM flyers ORDER BY created_at DESC LIMIT 5")
        flyers = cursor.fetchall()
        
        print(f"\nTotal Flyers in DB: {len(flyers)}")
        
        for flyer in flyers:
            print(f"\n--- Flyer: {flyer[1]} ---")
            print(f"  Filename: {flyer[2]}")
            print(f"  Total Offers: {flyer[3]}")
            print(f"  Processing Time: {flyer[4]:.2f}ms")
            print(f"  Created: {flyer[5]}")
            
            # Get offers for this flyer
            cursor.execute("SELECT * FROM offers WHERE flyer_id = ?", (flyer[0],))
            offers = cursor.fetchall()
            
            print(f"\n  Offers ({len(offers)}):")
            for idx, offer in enumerate(offers[:5], 1):  # Show first 5
                print(f"\n    Offer {idx}: {offer[2]}")
                print(f"      Position: ({offer[4]}, {offer[5]}) - {offer[6]}x{offer[7]}")
                if offer[9]:  # product_title
                    print(f"      Product: {offer[9]}")
                if offer[11]:  # discounted_price
                    print(f"      Price: {offer[11]}")
                if offer[8]:  # full_text (first 100 chars)
                    text_preview = offer[8][:100]
                    print(f"      Text: {text_preview}...")
        
        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")


def test_api_detailed():
    """Test API with detailed output"""
    test_images = list(Path("test_images").glob("*.jpg"))
    
    if not test_images:
        print("No images found in test_images/")
        return
    
    print(f"\n{'='*70}")
    print(f"TESTING FLYER DETECTION API")
    print(f"{'='*70}")
    print(f"\nFound {len(test_images)} test images")
    print(f"Testing with: {test_images[0].name}")
    
    with open(test_images[0], 'rb') as f:
        files = {'file': (test_images[0].name, f, 'image/jpeg')}
        data = {'flyer_id': 'test_detailed_001'}
        
        print("\nSending request to API...")
        response = requests.post(f"{API_URL}/detect", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n{'='*70}")
        print(f"✓ API RESPONSE - SUCCESS")
        print(f"{'='*70}")
        print(f"\nFlyer ID: {result['flyer_id']}")
        print(f"Total Offers: {result['total_offers']}")
        print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
        print(f"Message: {result['message']}")
        
        print(f"\n{'='*70}")
        print(f"DETECTED OFFERS")
        print(f"{'='*70}")
        
        for i, offer in enumerate(result['offers'], 1):
            print(f"\n--- Offer #{i}: {offer['offer_id']} ---")
            print(f"  Confidence: {offer['confidence']:.2f}")
            print(f"  Position: x={offer['position']['x']}, y={offer['position']['y']}")
            print(f"  Size: {offer['position']['width']}x{offer['position']['height']}")
            print(f"  Image: {offer['image_path']}")
            
            text = offer['extracted_text']
            
            # Show extracted fields
            if text['product_title']:
                print(f"  ✓ Product: {text['product_title']}")
            else:
                print(f"  ✗ Product: Not detected")
            
            if text['original_price']:
                print(f"  ✓ Original Price: {text['original_price']}")
            
            if text['discounted_price']:
                print(f"  ✓ Discounted Price: {text['discounted_price']}")
            else:
                print(f"  ✗ Price: Not detected")
            
            if text['discount_percentage']:
                print(f"  ✓ Discount: {text['discount_percentage']}")
            
            if text['promotional_text']:
                print(f"  ✓ Promo: {text['promotional_text']}")
            
            # Show OCR text (first 200 chars)
            if text['full_text']:
                preview = text['full_text'][:200].replace('\n', ' ')
                print(f"\n  OCR Text ({len(text['full_text'])} chars):")
                print(f"  \"{preview}...\"")
            else:
                print(f"\n  ⚠ OCR Text: EMPTY - DeepSeek-OCR may not be extracting text properly")
        
        # Show summary
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        
        total_with_text = sum(1 for o in result['offers'] if o['extracted_text']['full_text'])
        total_with_price = sum(1 for o in result['offers'] if o['extracted_text']['discounted_price'])
        total_with_product = sum(1 for o in result['offers'] if o['extracted_text']['product_title'])
        
        print(f"Total Offers: {result['total_offers']}")
        print(f"Offers with OCR Text: {total_with_text}/{result['total_offers']}")
        print(f"Offers with Price: {total_with_price}/{result['total_offers']}")
        print(f"Offers with Product Name: {total_with_product}/{result['total_offers']}")
        
        if total_with_text == 0:
            print(f"\n⚠ WARNING: No OCR text extracted!")
            print(f"  DeepSeek-OCR may need troubleshooting.")
            print(f"  Check the server logs for OCR errors.")
        
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    # Test API
    test_api_detailed()
    
    # View database
    view_database()
    
    print(f"\n{'='*70}")
    print("Test complete!")
    print(f"{'='*70}\n")
