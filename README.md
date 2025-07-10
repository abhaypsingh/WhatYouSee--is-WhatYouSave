# Screen Region OCR Capture

Advanced screen region capture application with real-time OCR processing, DPI-aware precision, and high-quality image capture.

## Features

- **Interactive Region Selection**: Click and drag with visual feedback and coordinate display
- **DPI-Aware Capture**: Handles high-DPI displays with precise coordinate mapping
- **Test Capture**: Preview captured region before starting continuous capture
- **High-Quality Screenshots**: Lossless PNG capture with BMP backup for quality reference
- **Real-time OCR**: Processes captured images with Tesseract OCR in background
- **Optimized Performance**: Non-blocking OCR processing with advanced image preprocessing
- **Automatic File Management**: Timestamped files with organized directory structure
- **Visual Feedback**: Screenshot background during selection, coordinate display, size validation

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`

## Usage

1. Run the application:
```bash
python screen_ocr_app.py
```

2. **Step 1**: Click "Select Region" and drag to select screen area
   - Real-time coordinate display
   - Visual feedback with corner indicators
   - Size validation (minimum 10x10 pixels)

3. **Step 2**: Click "Test Capture" to verify region selection
   - Preview captured image
   - Verify dimensions and quality
   - Check DPI scaling effects

4. **Step 3**: Adjust capture interval if needed (default: 1 second)

5. **Step 4**: Click "Start Capture" to begin continuous capturing

6. Screenshots and OCR text files will be saved in `captures/` directory

## Output Directory Structure

The application automatically creates organized directories:

```
captures/
├── screenshots/           # Original captured images
│   ├── capture_YYYYMMDD_HHMMSS_mmm.png
│   ├── test_capture_YYYYMMDD_HHMMSS.png
│   └── capture_YYYYMMDD_HHMMSS_mmm.bmp (first capture reference)
├── ocr_results/          # OCR text files
│   └── capture_YYYYMMDD_HHMMSS_mmm.txt
├── processed_images/     # Preprocessed images for debugging
│   └── capture_YYYYMMDD_HHMMSS_mmm_processed.png
└── logs/                 # Application logs
```

## Technical Details

- Uses **PIL ImageGrab** for DPI-aware, high-quality screenshot capture
- **DPI Detection**: Automatically detects and handles display scaling
- **Coordinate Precision**: Ensures exact region capture with validation
- **Image Quality**: Lossless PNG compression with BMP backup for reference
- **Advanced OCR**: 4 preprocessing methods with multiple Tesseract configurations
- **Batch Processing**: Continuous OCR processing every 30 seconds
- **Organized Storage**: Automatic directory creation with structured file organization
- **Real-time Status**: Live OCR processing status updates
- **Cross-Platform**: Windows DPI awareness with Linux/macOS support

## Quality Improvements

- **Precise Region Capture**: Eliminated coordinate mismatches
- **DPI Scaling Handling**: Proper scaling factor detection and application
- **Visual Validation**: Test capture with size verification
- **Lossless Quality**: No image compression during capture
- **Enhanced OCR Processing**: Multiple preprocessing methods with confidence scoring
- **Batch OCR Processing**: Continuous background processing every 30 seconds
- **Automatic Directory Management**: Organized file structure with automatic creation
- **Real-time Monitoring**: Live status updates for OCR processing queue

## Requirements

- Python 3.7+
- Tesseract OCR engine
- Display server (X11, Wayland, or Windows)
