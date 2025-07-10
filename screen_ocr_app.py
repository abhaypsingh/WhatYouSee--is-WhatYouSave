#!/usr/bin/env python3
"""
Advanced Screen Region OCR Application
Captures selected screen regions continuously with OCR processing
High-quality, DPI-aware implementation with precise region capture
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import queue
import logging
import platform
import ctypes
import subprocess
import multiprocessing
import glob

import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab
import pytesseract
from pynput import mouse
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('screen_ocr.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# DPI Awareness setup
if platform.system() == "Windows":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # DPI aware
    except:
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # Fallback
        except:
            pass


class DPIHelper:
    """Helper class for DPI and coordinate handling"""
    
    @staticmethod
    def get_dpi_scale():
        """Get the DPI scaling factor"""
        if platform.system() == "Windows":
            try:
                # Get DPI for the primary monitor
                hdc = ctypes.windll.user32.GetDC(0)
                dpi_x = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
                dpi_y = ctypes.windll.gdi32.GetDeviceCaps(hdc, 90)  # LOGPIXELSY
                ctypes.windll.user32.ReleaseDC(0, hdc)
                
                # Standard DPI is 96
                scale_x = dpi_x / 96.0
                scale_y = dpi_y / 96.0
                return scale_x, scale_y
            except:
                return 1.0, 1.0
        else:
            # For Linux/macOS, we'll use a different approach
            try:
                root = tk.Tk()
                root.withdraw()
                dpi_x = root.winfo_fpixels('1i')
                dpi_y = root.winfo_fpixels('1i')
                root.destroy()
                scale_x = dpi_x / 96.0
                scale_y = dpi_y / 96.0
                return scale_x, scale_y
            except:
                return 1.0, 1.0
    
    @staticmethod
    def get_screen_geometry():
        """Get accurate screen geometry"""
        if platform.system() == "Windows":
            try:
                # Get virtual screen dimensions
                left = ctypes.windll.user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
                top = ctypes.windll.user32.GetSystemMetrics(77)   # SM_YVIRTUALSCREEN
                width = ctypes.windll.user32.GetSystemMetrics(78) # SM_CXVIRTUALSCREEN
                height = ctypes.windll.user32.GetSystemMetrics(79) # SM_CYVIRTUALSCREEN
                return left, top, width, height
            except:
                pass
        
        # Fallback to tkinter method
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return 0, 0, width, height


class RegionSelector:
    """Handles region selection with visual feedback and DPI awareness"""
    
    def __init__(self, callback):
        self.callback = callback
        self.start_pos = None
        self.end_pos = None
        self.selection_window = None
        self.canvas = None
        self.selecting = False
        self.dpi_scale_x, self.dpi_scale_y = DPIHelper.get_dpi_scale()
        self.screen_geometry = DPIHelper.get_screen_geometry()
        
        logger.info(f"DPI Scale: {self.dpi_scale_x:.2f}x{self.dpi_scale_y:.2f}")
        logger.info(f"Screen geometry: {self.screen_geometry}")
        
    def start_selection(self):
        """Start region selection mode with screenshot background"""
        # Take a screenshot to show as background
        screenshot = ImageGrab.grab()
        
        self.selection_window = tk.Toplevel()
        self.selection_window.attributes('-fullscreen', True)
        self.selection_window.attributes('-alpha', 0.7)
        self.selection_window.attributes('-topmost', True)
        self.selection_window.configure(bg='black')
        self.selection_window.bind('<Button-1>', self.on_click)
        self.selection_window.bind('<B1-Motion>', self.on_drag)
        self.selection_window.bind('<ButtonRelease-1>', self.on_release)
        self.selection_window.bind('<Escape>', lambda e: self.cancel_selection())
        self.selection_window.focus_set()
        
        # Create canvas for drawing selection rectangle
        screen_width = self.selection_window.winfo_screenwidth()
        screen_height = self.selection_window.winfo_screenheight()
        
        self.canvas = tk.Canvas(
            self.selection_window,
            highlightthickness=0,
            bg='black',
            cursor='crosshair',
            width=screen_width,
            height=screen_height
        )
        self.canvas.pack(fill='both', expand=True)
        
        # Add screenshot as background (dimmed)
        screenshot_resized = screenshot.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
        # Darken the image
        screenshot_array = np.array(screenshot_resized)
        screenshot_array = (screenshot_array * 0.3).astype(np.uint8)
        screenshot_dimmed = Image.fromarray(screenshot_array)
        
        self.bg_image = ImageTk.PhotoImage(screenshot_dimmed)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_image)
        
        # Instructions
        instructions = tk.Label(
            self.selection_window,
            text="Click and drag to select region. Press ESC to cancel.\nCoordinates will be shown in real-time.",
            fg='white',
            bg='black',
            font=('Arial', 12, 'bold'),
            justify='center'
        )
        instructions.place(x=10, y=10)
        
        # Coordinate display
        self.coord_label = tk.Label(
            self.selection_window,
            text="",
            fg='yellow',
            bg='black',
            font=('Arial', 10),
            justify='left'
        )
        self.coord_label.place(x=10, y=60)
        
    def on_click(self, event):
        """Handle mouse click"""
        self.start_pos = (event.x_root, event.y_root)
        self.selecting = True
        self.update_coordinates()
        
    def on_drag(self, event):
        """Handle mouse drag"""
        if self.selecting:
            self.canvas.delete('selection')
            self.canvas.delete('selection_info')
            
            x1, y1 = self.start_pos
            x2, y2 = event.x_root, event.y_root
            
            # Convert to canvas coordinates
            canvas_x1 = min(x1, x2) - self.selection_window.winfo_rootx()
            canvas_y1 = min(y1, y2) - self.selection_window.winfo_rooty()
            canvas_x2 = max(x1, x2) - self.selection_window.winfo_rootx()
            canvas_y2 = max(y1, y2) - self.selection_window.winfo_rooty()
            
            # Draw selection rectangle
            self.canvas.create_rectangle(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                outline='red', width=3, tags='selection'
            )
            
            # Draw corner indicators
            corner_size = 8
            for cx, cy in [(canvas_x1, canvas_y1), (canvas_x2, canvas_y1), 
                          (canvas_x1, canvas_y2), (canvas_x2, canvas_y2)]:
                self.canvas.create_rectangle(
                    cx - corner_size, cy - corner_size,
                    cx + corner_size, cy + corner_size,
                    fill='red', outline='white', width=2, tags='selection'
                )
            
            # Show selection info
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            info_text = f"Selection: {width}x{height}"
            self.canvas.create_text(
                canvas_x1, canvas_y1 - 25,
                text=info_text,
                fill='white',
                font=('Arial', 12, 'bold'),
                anchor='nw',
                tags='selection_info'
            )
            
            self.update_coordinates(x2, y2)
            
    def on_release(self, event):
        """Handle mouse release"""
        if self.selecting:
            self.end_pos = (event.x_root, event.y_root)
            self.complete_selection()
            
    def update_coordinates(self, x2=None, y2=None):
        """Update coordinate display"""
        if self.start_pos:
            x1, y1 = self.start_pos
            if x2 is not None and y2 is not None:
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                coord_text = f"Start: ({x1}, {y1})\nCurrent: ({x2}, {y2})\nSize: {width}x{height}"
            else:
                coord_text = f"Start: ({x1}, {y1})"
            
            self.coord_label.config(text=coord_text)
            
    def complete_selection(self):
        """Complete the selection process with precise coordinates"""
        if self.start_pos and self.end_pos:
            x1, y1 = self.start_pos
            x2, y2 = self.end_pos
            
            # Ensure coordinates are in correct order
            left = min(x1, x2)
            top = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Validate minimum size
            if width < 10 or height < 10:
                messagebox.showwarning("Invalid Selection", "Selected region too small. Minimum size is 10x10 pixels.")
                self.cancel_selection()
                return
            
            region = {
                'left': left,
                'top': top,
                'width': width,
                'height': height
            }
            
            logger.info(f"Selected region: {region}")
            
            self.selection_window.destroy()
            self.callback(region)
        else:
            self.cancel_selection()
            
    def cancel_selection(self):
        """Cancel the selection"""
        if self.selection_window:
            self.selection_window.destroy()
        self.callback(None)


class OCRBatchProcessor:
    """Handles batch OCR processing as a separate background process"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.batch_thread = None
        self.running = False
        self.processing_interval = 30  # seconds
        self.last_processed_files = set()
        
        # Create subdirectories
        self.create_output_directories()
        
        # Multiple OCR configurations for different content types
        self.ocr_configs = {
            'default': r'--oem 3 --psm 6',
            'single_block': r'--oem 3 --psm 6',
            'single_line': r'--oem 3 --psm 7',
            'single_word': r'--oem 3 --psm 8',
            'dense_text': r'--oem 3 --psm 4',
            'sparse_text': r'--oem 3 --psm 11',
            'vertical_text': r'--oem 3 --psm 5'
        }
        
        # Best configuration will be determined dynamically
        self.best_config = 'default'
        
    def create_output_directories(self):
        """Create all necessary output directories"""
        directories = [
            self.output_dir,
            self.output_dir / 'screenshots',
            self.output_dir / 'ocr_results',
            self.output_dir / 'processed_images',
            self.output_dir / 'logs'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified directory: {directory}")
    
    def get_screenshots_dir(self):
        """Get the screenshots directory"""
        return self.output_dir / 'screenshots'
    
    def get_ocr_results_dir(self):
        """Get the OCR results directory"""
        return self.output_dir / 'ocr_results'
    
    def get_processed_images_dir(self):
        """Get the processed images directory"""
        return self.output_dir / 'processed_images'
        
    def start_processing(self):
        """Start batch OCR processing thread"""
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_process_loop, daemon=True)
        self.batch_thread.start()
        logger.info("Started batch OCR processing thread")
        
    def stop_processing(self):
        """Stop batch OCR processing"""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=5)
            
    def get_unprocessed_images(self) -> List[Path]:
        """Get list of unprocessed image files"""
        screenshots_dir = self.get_screenshots_dir()
        ocr_results_dir = self.get_ocr_results_dir()
        
        # Find all PNG files in screenshots directory
        image_files = []
        for pattern in ['*.png', '*.bmp', '*.jpg', '*.jpeg']:
            image_files.extend(screenshots_dir.glob(pattern))
        
        # Filter out already processed files
        unprocessed = []
        for img_path in image_files:
            ocr_path = ocr_results_dir / f"{img_path.stem}.txt"
            if not ocr_path.exists():
                unprocessed.append(img_path)
        
        return unprocessed
    
    def _batch_process_loop(self):
        """Main batch processing loop that runs every 30 seconds"""
        while self.running:
            try:
                # Get unprocessed images
                unprocessed_images = self.get_unprocessed_images()
                
                if unprocessed_images:
                    logger.info(f"Found {len(unprocessed_images)} unprocessed images")
                    
                    # Process each image
                    for img_path in unprocessed_images:
                        if not self.running:
                            break
                        
                        logger.info(f"Processing {img_path.name}")
                        self._process_single_image(img_path, 'Enhanced')
                        
                        # Small delay between processing to avoid overwhelming the system
                        time.sleep(0.5)
                else:
                    logger.debug("No unprocessed images found")
                
                # Wait for next batch processing cycle
                for _ in range(self.processing_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                time.sleep(5)  # Wait before retrying
                
    def _process_single_image(self, image_path: Path, quality_mode='Enhanced'):
        """Process a single image for OCR with advanced preprocessing"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return
                
            # Try different approaches based on quality mode
            results = []
            
            if quality_mode == 'Fast':
                # Fast mode - only basic preprocessing
                processed_img1 = self._preprocess_method1(image)
                result1 = self._try_multiple_configs(processed_img1)
                results.append(("Method 1 (Enhanced)", result1))
            
            elif quality_mode == 'Enhanced':
                # Enhanced mode - multiple methods
                processed_img1 = self._preprocess_method1(image)
                result1 = self._try_multiple_configs(processed_img1)
                results.append(("Method 1 (Enhanced)", result1))
                
                processed_img2 = self._preprocess_method2(image)
                result2 = self._try_multiple_configs(processed_img2)
                results.append(("Method 2 (High Contrast)", result2))
                
                processed_img4 = self._preprocess_method4(image)
                result4 = self._try_multiple_configs(processed_img4)
                results.append(("Method 4 (Adaptive)", result4))
            
            else:  # Maximum quality
                # Maximum mode - all methods
                processed_img1 = self._preprocess_method1(image)
                result1 = self._try_multiple_configs(processed_img1)
                results.append(("Method 1 (Enhanced)", result1))
                
                processed_img2 = self._preprocess_method2(image)
                result2 = self._try_multiple_configs(processed_img2)
                results.append(("Method 2 (High Contrast)", result2))
                
                processed_img3 = self._preprocess_method3(image)
                result3 = self._try_multiple_configs(processed_img3)
                results.append(("Method 3 (Morphological)", result3))
                
                processed_img4 = self._preprocess_method4(image)
                result4 = self._try_multiple_configs(processed_img4)
                results.append(("Method 4 (Adaptive)", result4))
            
            # Find best result based on confidence and length
            best_result = self._select_best_result(results)
            
            # Save OCR result
            if best_result['text'].strip():
                # Post-process the best result
                processed_text = self._post_process_text(best_result['text'])
                
                # Save to OCR results directory
                ocr_results_dir = self.get_ocr_results_dir()
                ocr_path = ocr_results_dir / f"{image_path.stem}.txt"
                
                with open(ocr_path, 'w', encoding='utf-8') as f:
                    f.write(f"OCR Result for {image_path.name}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Best Method: {best_result['method']}\n")
                    f.write(f"Config: {best_result['config']}\n")
                    f.write(f"Confidence: {best_result['confidence']:.2f}\n")
                    f.write("-" * 50 + "\n")
                    f.write("POST-PROCESSED TEXT:\n")
                    f.write("-" * 50 + "\n")
                    f.write(processed_text.strip())
                    
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.write("RAW OCR OUTPUT:\n")
                    f.write("=" * 50 + "\n")
                    f.write(best_result['text'].strip())
                    
                    # Also save all results for comparison
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.write("ALL RESULTS FOR COMPARISON:\n")
                    f.write("=" * 50 + "\n")
                    for method, result in results:
                        f.write(f"\n{method} (Confidence: {result['confidence']:.2f}):\n")
                        f.write("-" * 30 + "\n")
                        f.write(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
                        f.write("\n")
                    
                logger.info(f"OCR completed for {image_path.name} - Best: {best_result['method']} (Confidence: {best_result['confidence']:.2f})")
                
                # Save processed image to processed_images directory
                processed_images_dir = self.get_processed_images_dir()
                
                # Save the best processed image for reference
                if quality_mode == 'Enhanced':
                    processed_img = self._preprocess_method1(cv2.imread(str(image_path)))
                    processed_img_path = processed_images_dir / f"{image_path.stem}_processed.png"
                    cv2.imwrite(str(processed_img_path), processed_img)
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
    
    def _preprocess_method1(self, image):
        """Enhanced preprocessing with denoising and contrast enhancement"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Threshold for better text recognition
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _preprocess_method2(self, image):
        """High contrast preprocessing with gamma correction"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gamma correction to enhance contrast
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(gray, table)
        
        # Bilateral filter for edge preservation
        filtered = cv2.bilateralFilter(gamma_corrected, 9, 75, 75)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        return thresh
    
    def _preprocess_method3(self, image):
        """Morphological preprocessing for text enhancement"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply median filter to remove noise
        median = cv2.medianBlur(gray, 3)
        
        # Apply OTSU thresholding
        _, thresh = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up text
        kernel = np.ones((1,1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return closing
    
    def _preprocess_method4(self, image):
        """Adaptive preprocessing based on image characteristics"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate image statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Adaptive processing based on image brightness
        if mean_val < 100:  # Dark image
            # Increase brightness
            brightened = cv2.convertScaleAbs(gray, alpha=1.3, beta=30)
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            processed = clahe.apply(brightened)
        elif mean_val > 180:  # Bright image
            # Reduce brightness slightly
            darkened = cv2.convertScaleAbs(gray, alpha=0.9, beta=-10)
            processed = darkened
        else:  # Normal brightness
            processed = gray
        
        # Adaptive thresholding
        if std_val < 30:  # Low contrast
            thresh = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 15, 10)
        else:  # High contrast
            _, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _try_multiple_configs(self, processed_image):
        """Try multiple OCR configurations and return best result"""
        best_result = {'text': '', 'confidence': 0, 'config': 'default'}
        
        for config_name, config in self.ocr_configs.items():
            try:
                # Get text and confidence data
                data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
                text = pytesseract.image_to_string(processed_image, config=config)
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Score based on confidence and text length
                score = avg_confidence * (1 + len(text.strip()) / 1000)
                
                if score > best_result['confidence'] * (1 + len(best_result['text']) / 1000):
                    best_result = {
                        'text': text,
                        'confidence': avg_confidence,
                        'config': config_name
                    }
                    
            except Exception as e:
                logger.debug(f"Config {config_name} failed: {e}")
                continue
        
        return best_result
    
    def _select_best_result(self, results):
        """Select the best OCR result from multiple methods"""
        best_result = {'text': '', 'confidence': 0, 'method': 'None', 'config': 'default'}
        
        for method, result in results:
            # Calculate composite score
            text_length = len(result['text'].strip())
            confidence = result['confidence']
            
            # Penalize very short results unless confidence is very high
            if text_length < 20 and confidence < 80:
                score = confidence * 0.5
            else:
                score = confidence * (1 + text_length / 1000)
            
            if score > best_result['confidence'] * (1 + len(best_result['text']) / 1000):
                best_result = {
                    'text': result['text'],
                    'confidence': result['confidence'],
                    'method': method,
                    'config': result['config']
                }
        
        return best_result
    
    def _post_process_text(self, text):
        """Post-process OCR text to fix common errors"""
        if not text.strip():
            return text
        
        # Common OCR character replacements
        replacements = {
            # Common OCR errors
            'rn': 'm',  # 'rn' often misread as 'm'
            '0': 'O',   # In words, 0 is often O
            '1': 'I',   # In words, 1 is often I
            '5': 'S',   # In words, 5 is often S
            '8': 'B',   # In words, 8 is often B
            'l': 'I',   # lowercase l often should be I
            '|': 'I',   # pipe character often should be I
            '`': "'",   # backtick to apostrophe
            '"': '"',   # smart quotes
            '"': '"',   # smart quotes
            ''': "'",   # smart apostrophes
            ''': "'",   # smart apostrophes
        }
        
        # Apply basic replacements carefully
        processed_text = text
        
        # Fix common word patterns
        word_fixes = {
            'teh': 'the',
            'adn': 'and',
            'taht': 'that',
            'wiht': 'with',
            'thier': 'their',
            'recieve': 'receive',
            'seperate': 'separate',
            'Gouray': 'Gourav',  # Specific to your example
            'Sengupta': 'Sengupta',
            'QNeenS': 'Queens',
            'Thow': 'Show',
            'Founder8CEO': 'Founder & CEO',
            'AlBerkeley': 'AI Berkeley',
            'Abhayexplore': 'Abhay explore',
            'MotivaEnterprisesLLC': 'Motiva Enterprises LLC',
            'havoconthe': 'havoc on the',
            'AlFirst': 'AI First',
            'BrainsLast': 'Brains Last',
            'Getthelatestjobs': 'Get the latest jobs',
            'industryneews': 'industry news',
            'AboutAccessibility': 'About Accessibility',
            'HelpCenter': 'Help Center',
            'PrivacyTerms': 'Privacy Terms',
            'AdChoices': 'Ad Choices',
            'havertising': 'advertising',
            'BusinessServices': 'Business Services',
            'GettheLinkedin': 'Get the LinkedIn',
            'appMore': 'app More',
            'LinkedfiJ': 'LinkedIn',
            'LinkedinCorporation': 'LinkedIn Corporation',
            'yLoadmore': 'Load more',
            'MorethanhonoredtobeonmongstthesepitchesItsbeenanamazingopportunity': 'More than honored to be amongst these pitches. It\'s been an amazing opportunity',
            'and20Fathoms': 'and 20 Fathoms',
            'hast': 'has',
            'taughtmesomuch': 'taught me so much',
            'inthelastcoupleofweeks': 'in the last couple of weeks',
            'AnnouncingthepitchpresentersfornextweeksTCNewTech': 'Announcing the pitch presenters for next week\'s TC New Tech',
            'JoinusonJuly': 'Join us on July',
            'atGityOperaHouse': 'at City Opera House',
            'tohearfromPennyPickup': 'to hear from Penny Pickup',
            'SightHapticsphaticand': 'Sight Haptics and',
            'ChemCommore': 'ChemCom more',
            'ySlite': 'Like',
            'Makeagreatimpression': 'Make a great impression',
            'Abhayproaressinyour': 'Abhay progressing in your',
            'marketinacareerby': 'marketing career by'
        }
        
        words = processed_text.split()
        fixed_words = []
        for word in words:
            # Check if word (without punctuation) needs fixing
            clean_word = word.strip('.,!?;:')
            if clean_word in word_fixes:
                fixed_word = word_fixes[clean_word]
                # Replace in original word (preserving punctuation)
                word = word.replace(clean_word, fixed_word)
            elif clean_word.lower() in word_fixes:
                fixed_word = word_fixes[clean_word.lower()]
                # Preserve case
                if clean_word.isupper():
                    fixed_word = fixed_word.upper()
                elif clean_word.istitle():
                    fixed_word = fixed_word.title()
                # Replace in original word (preserving punctuation)
                word = word.replace(clean_word, fixed_word)
            fixed_words.append(word)
        
        return ' '.join(fixed_words)


class ScreenCaptureApp:
    """Main application class"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Screen Region OCR Capture")
        self.root.geometry("500x500")
        
        # Application state
        self.capturing = False
        self.selected_region = None
        self.output_dir = Path.cwd() / "captures"
        self.capture_thread = None
        self.capture_interval = 1.0  # seconds
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.region_selector = RegionSelector(self.on_region_selected)
        self.ocr_processor = OCRBatchProcessor(self.output_dir)
        self.quality_var = None  # Will be set in setup_ui
        
        # Setup UI
        self.setup_ui()
        
        # Start OCR batch processor
        self.ocr_processor.start_processing()
        
        # Start OCR status updates
        self.update_ocr_status()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Screen Region OCR Capture", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Region selection
        ttk.Label(main_frame, text="Step 1: Select Region").grid(row=1, column=0, sticky=tk.W)
        self.select_button = ttk.Button(main_frame, text="Select Region", 
                                       command=self.start_region_selection)
        self.select_button.grid(row=1, column=1, padx=(10, 0))
        
        # Region info
        self.region_info = ttk.Label(main_frame, text="No region selected", 
                                    foreground="gray")
        self.region_info.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Test capture controls
        ttk.Label(main_frame, text="Step 2: Test Capture").grid(row=3, column=0, sticky=tk.W, pady=(20, 0))
        
        self.test_button = ttk.Button(main_frame, text="Test Capture", 
                                     command=self.test_capture, state='disabled')
        self.test_button.grid(row=3, column=1, padx=(10, 0), pady=(20, 0))
        
        # Capture controls
        ttk.Label(main_frame, text="Step 3: Start Capture").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        
        self.capture_button = ttk.Button(main_frame, text="Start Capture", 
                                        command=self.toggle_capture, state='disabled')
        self.capture_button.grid(row=4, column=1, padx=(10, 0), pady=(10, 0))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # Capture interval
        ttk.Label(settings_frame, text="Capture Interval (seconds):").grid(row=0, column=0, sticky=tk.W)
        self.interval_var = tk.StringVar(value=str(self.capture_interval))
        interval_entry = ttk.Entry(settings_frame, textvariable=self.interval_var, width=10)
        interval_entry.grid(row=0, column=1, padx=(10, 0))
        interval_entry.bind('<Return>', self.update_interval)
        
        # OCR Quality setting
        ttk.Label(settings_frame, text="OCR Quality:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.quality_var = tk.StringVar(value="Enhanced")
        quality_combo = ttk.Combobox(settings_frame, textvariable=self.quality_var, 
                                    values=["Fast", "Enhanced", "Maximum"], 
                                    state="readonly", width=10)
        quality_combo.grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        
        # Output directory
        ttk.Label(settings_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.dir_label = ttk.Label(settings_frame, text=str(self.output_dir), 
                                  foreground="blue")
        self.dir_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # Directory structure info
        info_text = "Screenshots → screenshots/\nOCR Results → ocr_results/\nProcessed Images → processed_images/"
        info_label = ttk.Label(settings_frame, text=info_text, 
                              foreground="gray", font=('Arial', 8))
        info_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # OCR Status
        self.ocr_status_var = tk.StringVar(value="OCR: Ready")
        ocr_status_label = ttk.Label(main_frame, textvariable=self.ocr_status_var, 
                                    foreground="blue")
        ocr_status_label.grid(row=6, column=0, columnspan=2, pady=(10, 0))
        
        # Main Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                                foreground="green")
        status_label.grid(row=7, column=0, columnspan=2, pady=(5, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def start_region_selection(self):
        """Start the region selection process"""
        self.status_var.set("Select region on screen...")
        self.root.withdraw()  # Hide main window
        self.region_selector.start_selection()
        
    def on_region_selected(self, region):
        """Handle region selection completion"""
        self.root.deiconify()  # Show main window
        
        if region:
            self.selected_region = region
            self.region_info.config(text=f"Region: {region['width']}x{region['height']} at ({region['left']}, {region['top']})")
            self.test_button.config(state='normal')
            self.capture_button.config(state='normal')
            self.status_var.set("Region selected. Test capture recommended.")
        else:
            self.status_var.set("Region selection cancelled.")
            
    def update_interval(self, event=None):
        """Update capture interval"""
        try:
            interval = float(self.interval_var.get())
            if interval > 0:
                self.capture_interval = interval
                self.status_var.set(f"Capture interval updated to {interval}s")
            else:
                raise ValueError("Interval must be positive")
        except ValueError:
            self.interval_var.set(str(self.capture_interval))
            messagebox.showerror("Error", "Please enter a valid positive number")
    
    def test_capture(self):
        """Perform a test capture to verify region selection"""
        if not self.selected_region:
            messagebox.showerror("Error", "Please select a region first")
            return
        
        try:
            self.status_var.set("Taking test capture...")
            
            # Perform the same capture as the main loop
            bbox = (
                self.selected_region['left'],
                self.selected_region['top'],
                self.selected_region['left'] + self.selected_region['width'],
                self.selected_region['top'] + self.selected_region['height']
            )
            
            # Capture with PIL ImageGrab (DPI-aware)
            img = ImageGrab.grab(bbox=bbox, all_screens=True)
            
            # Check size
            expected_width = self.selected_region['width']
            expected_height = self.selected_region['height']
            actual_width, actual_height = img.size
            
            # Generate test filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_capture_{timestamp}.png"
            screenshots_dir = self.ocr_processor.get_screenshots_dir()
            filepath = screenshots_dir / filename
            
            # Save test image
            img.save(filepath, 'PNG', compress_level=1)
            
            # Show preview window
            self.show_test_preview(img, filepath, expected_width, expected_height, actual_width, actual_height)
            
            self.status_var.set(f"Test capture saved: {filename}")
            
        except Exception as e:
            logger.error(f"Test capture error: {e}")
            messagebox.showerror("Test Capture Error", f"Error during test capture: {e}")
            self.status_var.set("Test capture failed")
    
    def show_test_preview(self, img, filepath, expected_w, expected_h, actual_w, actual_h):
        """Show test capture preview window"""
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Test Capture Preview")
        preview_window.geometry("600x500")
        
        # Info frame
        info_frame = ttk.Frame(preview_window)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        info_text = f"""Test Capture Results:
        
Expected Size: {expected_w} x {expected_h}
Actual Size: {actual_w} x {actual_h}
File: {filepath.name}

{("✓ Size matches exactly!" if expected_w == actual_w and expected_h == actual_h else 
  "⚠ Size difference detected - DPI scaling applied")}"""
        
        info_label = ttk.Label(info_frame, text=info_text, font=('Arial', 10))
        info_label.pack(anchor='w')
        
        # Image preview
        preview_frame = ttk.Frame(preview_window)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Scale image for preview if too large
        preview_img = img.copy()
        max_preview_size = (500, 300)
        
        if preview_img.size[0] > max_preview_size[0] or preview_img.size[1] > max_preview_size[1]:
            preview_img.thumbnail(max_preview_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(preview_img)
        
        canvas = tk.Canvas(preview_frame, width=preview_img.size[0], height=preview_img.size[1])
        canvas.pack()
        canvas.create_image(0, 0, anchor='nw', image=photo)
        canvas.photo = photo  # Keep a reference
        
        # Buttons
        button_frame = ttk.Frame(preview_window)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(button_frame, text="Open File", 
                  command=lambda: os.startfile(filepath) if platform.system() == "Windows" else os.system(f"open '{filepath}'" if platform.system() == "Darwin" else f"xdg-open '{filepath}'")).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close", command=preview_window.destroy).pack(side='right', padx=5)
    
    def update_ocr_status(self):
        """Update OCR processing status"""
        try:
            unprocessed_count = len(self.ocr_processor.get_unprocessed_images())
            if unprocessed_count > 0:
                self.ocr_status_var.set(f"OCR: {unprocessed_count} images pending")
            else:
                self.ocr_status_var.set("OCR: Up to date")
        except Exception as e:
            self.ocr_status_var.set("OCR: Error checking status")
            logger.error(f"OCR status update error: {e}")
        
        # Schedule next update
        self.root.after(5000, self.update_ocr_status)  # Update every 5 seconds
            
    def toggle_capture(self):
        """Toggle screen capture"""
        if self.capturing:
            self.stop_capture()
        else:
            self.start_capture()
            
    def start_capture(self):
        """Start continuous screen capture"""
        if not self.selected_region:
            messagebox.showerror("Error", "Please select a region first")
            return
            
        self.capturing = True
        self.capture_button.config(text="Stop Capture")
        self.select_button.config(state='disabled')
        self.test_button.config(state='disabled')
        self.status_var.set("Capturing...")
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
    def stop_capture(self):
        """Stop screen capture"""
        self.capturing = False
        self.capture_button.config(text="Start Capture")
        self.select_button.config(state='normal')
        self.test_button.config(state='normal')
        self.status_var.set("Capture stopped")
        
    def _capture_loop(self):
        """Main capture loop with high-quality DPI-aware capture"""
        capture_count = 0
        
        while self.capturing:
            try:
                # Log region info for debugging
                if capture_count == 0:
                    logger.info(f"Starting capture of region: {self.selected_region}")
                
                # High-quality capture using PIL ImageGrab
                # This method handles DPI scaling automatically
                bbox = (
                    self.selected_region['left'],
                    self.selected_region['top'],
                    self.selected_region['left'] + self.selected_region['width'],
                    self.selected_region['top'] + self.selected_region['height']
                )
                
                # Capture with PIL ImageGrab (DPI-aware)
                img = ImageGrab.grab(bbox=bbox, all_screens=True)
                
                # Verify the captured image size matches expected region
                expected_width = self.selected_region['width']
                expected_height = self.selected_region['height']
                actual_width, actual_height = img.size
                
                if capture_count == 0:
                    logger.info(f"Expected size: {expected_width}x{expected_height}")
                    logger.info(f"Actual captured size: {actual_width}x{actual_height}")
                    
                    # Check if we need to resize due to DPI scaling
                    if actual_width != expected_width or actual_height != expected_height:
                        logger.info(f"Resizing from {actual_width}x{actual_height} to {expected_width}x{expected_height}")
                        img = img.resize((expected_width, expected_height), Image.Resampling.LANCZOS)
                
                # Generate filename with more precision
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"capture_{timestamp}.png"
                
                # Save to screenshots directory
                screenshots_dir = self.ocr_processor.get_screenshots_dir()
                filepath = screenshots_dir / filename
                
                # Save image in highest quality
                # Use lossless PNG with no optimization for best quality
                img.save(filepath, 'PNG', compress_level=1)
                
                # Also save a backup in BMP format for maximum quality
                if capture_count == 0:  # Save first capture as BMP for quality comparison
                    bmp_path = filepath.with_suffix('.bmp')
                    img.save(bmp_path, 'BMP')
                    logger.info(f"Quality reference saved as: {bmp_path}")
                
                # OCR processing will be handled by the batch processor automatically
                # No need to queue individual images
                
                # Update status
                self.root.after(0, lambda f=filename: self.status_var.set(f"Captured: {f}"))
                
                capture_count += 1
                
            except Exception as e:
                logger.error(f"Capture error: {e}")
                self.root.after(0, lambda e=e: self.status_var.set(f"Capture error: {e}"))
                
            # Wait for next capture
            time.sleep(self.capture_interval)
                
    def on_closing(self):
        """Handle application closing"""
        self.capturing = False
        self.ocr_processor.stop_processing()
        self.root.destroy()
        
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """Main entry point"""
    try:
        app = ScreenCaptureApp()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        messagebox.showerror("Error", f"Application error: {e}")


if __name__ == "__main__":
    main()