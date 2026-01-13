import requests
import json
from pathlib import Path
import mimetypes

API_URL = "http://localhost:8000"

# Find test images
test_images = list(Path("test_images").glob("*.jpg"))

if not test_images:
    print("No images found in test_images/")
    exit(1)

print(f"\nFound {len(test_images)} test images")
print(f"Testing with: {test_images[0].name}")

# Test the API with explicit content type
with open(test_images[0], 'rb') as f:
    files = {
        'file': (test_images[0].name, f, 'image/jpeg')  # Explicitly set MIME type
    }
    data = {'flyer_id': 'test_001'}
    
    print("\nSending request...")
    response = requests.post(f"{API_URL}/detect", files=files, data=data)

if response.status_code == 200:
    result = response.json()
    print(f"\n✓ Success!")
    print(f"Total Offers Detected: {result['total_offers']}")
    print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
    
    for i, offer in enumerate(result['offers'][:3], 1):
        print(f"\n--- Offer {i} ---")
        text = offer['extracted_text']
        if text['product_title']:
            print(f"Product: {text['product_title']}")
        if text['discounted_price']:
            print(f"Price: {text['discounted_price']}")
else:
    print(f"\n✗ Error: {response.status_code}")
    print(response.text)