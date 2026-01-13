# Flyer Processing System - Implementation Guide

## Overview

This system uses **DeepSeek-OCR**, a powerful Vision-Language Model (VLM), to detect and extract text from retail flyer offers. It provides two processing modes:

### 1. **Unified LLM Mode (RECOMMENDED)**
- Single-pass processing: detection + extraction in one API call
- Higher accuracy through contextual understanding
- Faster processing (one model inference instead of N+1)
- Structured JSON output from LLM

### 2. **Two-Stage Mode (Fallback)**
- Separate detection and OCR stages
- More granular control
- Better for debugging

---

## Architecture

### Key Components

```
flyer_api/
├── app_enhanced.py          # Enhanced API with both modes
├── unified_processor.py      # Unified LLM processor (RECOMMENDED)
├── detection_module.py       # LLM-based detection
├── ocr_module.py            # Structured OCR extraction
├── database.py              # SQLite storage
└── test_enhanced.py         # Comprehensive testing
```

---

## Installation

### 1. Install Dependencies

```bash
cd flyer_api
pip install -r requirements.txt
```

Required packages:
- `transformers>=4.46.3`
- `torch`
- `fastapi`
- `uvicorn`
- `opencv-python`
- `pillow`
- `sqlalchemy`
- `numpy`

### 2. Download Model (Automatic)

The system will automatically download `deepseek-ai/DeepSeek-OCR` from HuggingFace on first run.

**Note**: The model is ~10GB. Ensure you have sufficient disk space.

---

## Usage

### Starting the API Server

#### Option 1: Enhanced API (Unified Mode - Recommended)

```bash
cd flyer_api
python app_enhanced.py
```

This starts the server on `http://localhost:8000` in **unified mode**.

#### Option 2: Original API (Two-Stage Mode)

```bash
cd flyer_api
python app.py
```

### API Endpoints

#### POST `/detect`

Upload a flyer image for processing.

**Request:**
- `file`: JPG or PNG image (multipart/form-data)
- `flyer_id`: Optional identifier (string)
- `force_mode`: Optional - "unified" or "two_stage"

**Response:**
```json
{
  "success": true,
  "flyer_id": "flyer_abc123",
  "total_offers": 4,
  "processing_time_ms": 3452.5,
  "processing_mode": "unified_llm",
  "message": "Successfully detected and processed 4 offers",
  "offers": [
    {
      "offer_id": "flyer_abc123_offer_1",
      "image_path": "/path/to/crop.jpg",
      "image_url": "/outputs/flyer_crops/flyer_abc123_offer_1.jpg",
      "position": {
        "x": 50,
        "y": 100,
        "width": 300,
        "height": 250
      },
      "extracted_text": {
        "full_text": "Fresh Strawberries 250g...",
        "product_title": "Fresh Strawberries",
        "original_price": "25.95",
        "discounted_price": "21.95",
        "discount_percentage": "15%",
        "promotional_text": "Limited time offer"
      },
      "confidence": 0.95
    }
  ]
}
```

#### GET `/health`

Check API status and configuration.

---

## Key Features

### ✅ Accurate Detection
- LLM understands what an "offer" is semantically
- Ignores decorative elements, headers, footers
- Handles diverse layouts and designs

### ✅ Multi-Language Support
- Extracts both English and Arabic text
- Preserves original formatting

### ✅ Structured Extraction
- Product titles
- Original and discounted prices
- Discount percentages
- Promotional text
- Complete raw text

### ✅ Precise Coordinates
- Returns exact pixel coordinates (x, y, width, height)
- Suitable for frontend hover/click interactions

### ✅ Error Handling
- Graceful fallback to traditional CV if LLM fails
- Clear error messages
- Empty text handling

### ✅ Performance
- Targets < 5 seconds per flyer
- CPU-optimized (GPU optional)
- Single-pass processing in unified mode

---

## Prompt Engineering

The system uses carefully crafted prompts for optimal results:

### Detection Prompt (Unified Mode)

```
Analyze this retail flyer image (WxH pixels) and identify all product offers.

For EACH offer section, provide:
1. Bounding box: [x, y, width, height] in pixels
2. Extracted text data

IMPORTANT RULES:
- Ignore headers, footers, logos, and decorative elements
- Only detect actual product offers with prices
- Each offer must be a distinct product
- Extract ALL text (English and Arabic)

Respond with a JSON array of offers...
```

### OCR Prompt (Two-Stage Mode)

```
Extract ALL text from this retail product offer image.

Provide the following information in JSON format:
{
  "product_title": "Full product name",
  "original_price": "XX.XX" (if shown),
  ...
}

Respond ONLY with valid JSON.
```

---

## Configuration

### Mode Selection

Edit `app_enhanced.py`:

```python
USE_UNIFIED = True  # True for unified, False for two-stage
```

### Model Parameters

For better quality (slower):
```python
base_size=1280
image_size=1280
```

For faster processing (lower quality):
```python
base_size=512
image_size=512
```

---

## Testing

### Run Comprehensive Tests

```bash
cd flyer_api
python test_enhanced.py
```

This will:
1. Test both unified and two-stage modes
2. Compare results and performance
3. Show detailed extraction for each offer
4. Display database contents

### Test Single Image

```python
import requests

with open('test_flyer.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f},
        data={'flyer_id': 'test_001'}
    )
    
result = response.json()
print(f"Detected {result['total_offers']} offers")
```

---

## Performance Optimization

### 1. Model Loading (One-Time Cost)
- First request: ~30-60 seconds (model download + load)
- Subsequent requests: Fast (model cached in memory)

### 2. Unified vs Two-Stage

**Unified Mode:**
- 1 model inference for entire flyer
- ~2-4 seconds for typical flyer

**Two-Stage Mode:**
- 1 inference for detection + N inferences for OCR
- ~3-6 seconds for typical flyer with 4-6 offers

### 3. GPU Acceleration (Optional)

Edit model initialization:
```python
self.model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # Use bfloat16 for GPU
).cuda()  # Move to GPU
```

**Requirements:**
- NVIDIA GPU with CUDA
- `torch` with CUDA support
- At least 12GB VRAM

---

## Troubleshooting

### Issue: No offers detected

**Solutions:**
1. Check image quality (min 800x600 recommended)
2. Ensure offers have clear boundaries
3. Try adjusting `base_size` parameter
4. Check logs for LLM output

### Issue: Empty OCR text

**Solutions:**
1. Verify DeepSeek-OCR is loaded correctly
2. Check cropped image quality
3. Review prompt effectiveness
4. Try unified mode for better context

### Issue: Slow processing

**Solutions:**
1. Use unified mode (faster)
2. Reduce `base_size` and `image_size`
3. Enable GPU acceleration
4. Check system resources

### Issue: Wrong coordinates

**Solutions:**
1. Verify image dimensions in prompt
2. Check coordinate validation logic
3. Use unified mode for better accuracy
4. Review LLM output in logs

---

## Database Schema

### Flyers Table
```sql
CREATE TABLE flyers (
    id INTEGER PRIMARY KEY,
    flyer_id VARCHAR(100) UNIQUE,
    filename VARCHAR(255),
    total_offers INTEGER,
    processing_time_ms FLOAT,
    created_at DATETIME
)
```

### Offers Table
```sql
CREATE TABLE offers (
    id INTEGER PRIMARY KEY,
    flyer_id INTEGER REFERENCES flyers(id),
    offer_id VARCHAR(100) UNIQUE,
    image_path VARCHAR(500),
    x INTEGER,
    y INTEGER,
    width INTEGER,
    height INTEGER,
    full_text TEXT,
    product_title VARCHAR(500),
    original_price VARCHAR(50),
    discounted_price VARCHAR(50),
    discount_percentage VARCHAR(20),
    promotional_text TEXT,
    confidence FLOAT,
    created_at DATETIME
)
```

---

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY flyer_api/ ./flyer_api/

# Expose port
EXPOSE 8000

# Run API
CMD ["python", "flyer_api/app_enhanced.py"]
```

### Security Considerations

1. **File Size Limits**: Add max file size validation
2. **Rate Limiting**: Implement request throttling
3. **Authentication**: Add API key authentication
4. **CORS**: Restrict origins in production

---

## Future Enhancements

1. **Batch Processing**: Process multiple flyers in parallel
2. **Caching**: Cache results for duplicate flyers
3. **Fine-tuning**: Train on domain-specific data
4. **Real-time Processing**: WebSocket support
5. **Analytics**: Track accuracy metrics

---

## Support

For issues or questions:
1. Check logs: `logging.basicConfig(level=logging.DEBUG)`
2. Review cropped images in `outputs/flyer_crops/`
3. Test with debug scripts: `debug_ocr.py`
4. Verify model loading: Check `/health` endpoint

---

## License

This implementation uses:
- **DeepSeek-OCR**: [License from HuggingFace]
- **FastAPI**: MIT License
- **PyTorch**: BSD License
