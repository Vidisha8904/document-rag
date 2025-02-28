import os
import io
import tempfile
from typing import Dict, List, Tuple, Any
import logging

# Core libraries
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import cv2
import pix2tex
from PIL import Image
import camelot
import matplotlib.pyplot as plt
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScannedPDFExtractor:
    """Extract content from scanned PDFs including text, images, and mathematical formulas."""
    
    def __init__(self, 
                 tesseract_path: str = None, 
                 dpi: int = 300,
                 math_detection_threshold: float = 0.7,
                 output_dir: str = "extracted_content"):
        """
        Initialize the PDF extractor.
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
            dpi: DPI for PDF to image conversion
            math_detection_threshold: Confidence threshold for math formula detection
            output_dir: Directory to save extracted images
        """
        # Configure tesseract
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        self.dpi = dpi
        self.math_threshold = math_detection_threshold
        self.output_dir = output_dir
        
        # Initialize math formula recognition
        try:
            self.math_recognizer = pix2tex.LatexOCR()
            self.math_recognition_available = True
        except Exception as e:
            logger.warning(f"Math formula recognition not available: {e}")
            self.math_recognition_available = False
            
        # Create output directory if needed
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract all content from a scanned PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted content
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images
        try:
            pages = convert_from_path(pdf_path, dpi=self.dpi)
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return {"error": str(e)}
        
        result = {
            "text": [],
            "images": [],
            "tables": [],
            "math_formulas": []
        }
        
        # Process each page
        for i, page in enumerate(pages):
            logger.info(f"Processing page {i+1}/{len(pages)}")
            
            # Save page as temporary image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                temp_path = temp.name
                page.save(temp_path, "PNG")
            
            # Extract content
            page_result = self._process_page(temp_path, i+1, pdf_path)
            
            # Add to results
            result["text"].append(page_result["text"])
            result["images"].extend(page_result["images"])
            result["tables"].extend(page_result["tables"])
            result["math_formulas"].extend(page_result["math_formulas"])
            
            # Clean up temp file
            os.remove(temp_path)
            
        logger.info(f"Extraction complete: {len(result['text'])} pages, {len(result['images'])} images, "
                   f"{len(result['tables'])} tables, {len(result['math_formulas'])} math formulas")
        
        return result
    
    def _process_page(self, image_path: str, page_num: int, original_pdf_path: str) -> Dict[str, Any]:
        """Process a single page image to extract all content."""
        result = {
            "text": "",
            "images": [],
            "tables": [],
            "math_formulas": []
        }
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return result
        
        # Extract text
        result["text"] = self._extract_text(image_path)
        
        # Extract images
        result["images"] = self._extract_images(image, page_num)
        
        # Extract tables
        try:
            tables = camelot.read_pdf(original_pdf_path, pages=str(page_num))
            for i, table in enumerate(tables):
                table_data = {
                    "page": page_num,
                    "table_id": i+1,
                    "data": table.df.to_dict(),
                    "accuracy": table.accuracy
                }
                result["tables"].append(table_data)
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        # Extract math formulas
        if self.math_recognition_available:
            result["math_formulas"] = self._extract_math_formulas(image, page_num)
        
        return result
    
    def _extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            # Configure advanced OCR settings
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image_path, config=custom_config)
            return text
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def _extract_images(self, image: np.ndarray, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from the page."""
        extracted_images = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip small contours (likely text)
                if w < 100 or h < 100:
                    continue
                
                # Extract image region
                img_region = image[y:y+h, x:x+w]
                
                # Check if region has enough non-text content
                if self._is_likely_image(img_region):
                    # Save image
                    img_filename = f"page_{page_num}_image_{i}.png"
                    img_path = os.path.join(self.output_dir, img_filename)
                    cv2.imwrite(img_path, img_region)
                    
                    extracted_images.append({
                        "page": page_num,
                        "image_id": i,
                        "path": img_path,
                        "position": {"x": x, "y": y, "width": w, "height": h}
                    })
            
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
        
        return extracted_images
    
    def _is_likely_image(self, image_region: np.ndarray) -> bool:
        """Determine if a region is likely an image and not text."""
        # Calculate ratio of text-like pixel patterns
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Count non-zero pixels
        total_pixels = thresh.shape[0] * thresh.shape[1]
        if total_pixels == 0:
            return False
            
        text_pixels = cv2.countNonZero(thresh)
        text_ratio = text_pixels / total_pixels
        
        # If less than 30% is text-like, it's probably an image
        return text_ratio < 0.3
    
    def _extract_math_formulas(self, image: np.ndarray, page_num: int) -> List[Dict[str, Any]]:
        """Extract mathematical formulas from the page."""
        math_formulas = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use specialized filters for math symbols detection
            # This is a simplified approach - a production system would use ML models
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip very small regions
                if w < 50 or h < 20:
                    continue
                
                # Extract potential formula region
                formula_region = image[y:y+h, x:x+w]
                
                # Check if region has math formula characteristics
                if self._is_likely_formula(formula_region):
                    # Convert to PIL Image for pix2tex
                    pil_img = Image.fromarray(cv2.cvtColor(formula_region, cv2.COLOR_BGR2RGB))
                    
                    # Recognize formula
                    try:
                        latex = self.math_recognizer(pil_img)
                        
                        # Save formula image
                        formula_filename = f"page_{page_num}_formula_{i}.png"
                        formula_path = os.path.join(self.output_dir, formula_filename)
                        cv2.imwrite(formula_path, formula_region)
                        
                        math_formulas.append({
                            "page": page_num,
                            "formula_id": i,
                            "latex": latex,
                            "image_path": formula_path,
                            "position": {"x": x, "y": y, "width": w, "height": h}
                        })
                    except Exception as e:
                        logger.warning(f"Failed to recognize formula: {e}")
            
        except Exception as e:
            logger.error(f"Math formula extraction failed: {e}")
        
        return math_formulas
    
    def _is_likely_formula(self, image_region: np.ndarray) -> bool:
        """Determine if a region likely contains a mathematical formula."""
        # This is a simplified heuristic - a real system would use ML classifiers
        
        # Convert to grayscale and apply thresholding
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate horizontal and vertical gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Look for patterns typical in math formulas (horizontal lines, symbols, etc.)
        horizontal_lines = cv2.reduce(thresh, 0, cv2.REDUCE_AVG).flatten()
        vertical_lines = cv2.reduce(thresh, 1, cv2.REDUCE_AVG).flatten()
        
        # Count transitions (a measure of complexity)
        h_transitions = np.sum(np.abs(np.diff(horizontal_lines > 50)))
        v_transitions = np.sum(np.abs(np.diff(vertical_lines > 50)))
        
        # Formulas typically have high horizontal/vertical transitions
        density = np.mean(thresh) / 255.0
        complexity = (h_transitions + v_transitions) / (image_region.shape[0] + image_region.shape[1])
        
        # Heuristic formula detection
        return complexity > 0.2 and 0.05 < density < 0.5

# Example usage
if __name__ == "__main__":
    extractor = ScannedPDFExtractor()
    results = extractor.extract_from_pdf("NIPS-2017-attention-is-all-you-need-Paper.pdf")
    
    # Print summary of results
    print(f"Extracted {len(results['text'])} pages of text")
    print(f"Found {len(results['images'])} images")
    print(f"Detected {len(results['tables'])} tables")
    print(f"Recognized {len(results['math_formulas'])} mathematical formulas")
    
    # Print first page text
    if results['text']:
        print("\nSample text from first page:")
        print(results['text'][0][:500] + "...")