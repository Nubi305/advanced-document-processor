import logging
import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union
import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class OCRExtractor:
    """
    Extracts text from images and scanned PDFs using OCR.
    
    Supports multiple OCR engines: PaddleOCR (default), Tesseract, and cloud-based OCR.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the OCR extractor with configuration.
        
        Args:
            config: OCR configuration dictionary
        """
        self.config = config
        self.default_engine = config.get("default_engine", "paddle")
        self._paddle_ocr = None
        self._tesseract_setup_done = False
        
        # Initialize OCR engines based on configuration
        if self.default_engine == "paddle":
            self._init_paddle_ocr()
        elif self.default_engine == "tesseract":
            self._init_tesseract()
    
    def _init_paddle_ocr(self):
        """Initialize PaddleOCR engine."""
        try:
            from paddleocr import PaddleOCR
            
            paddle_config = self.config["engines"]["paddle"]
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=paddle_config.get("use_angle_classifier", True),
                lang=paddle_config.get("lang", "en"),
                use_gpu=paddle_config.get("use_gpu", False),
                det_model_dir=paddle_config.get("det_model_dir"),
                rec_model_dir=paddle_config.get("rec_model_dir"),
                # Add additional parameters as needed
            )
            logger.info("PaddleOCR initialized successfully")
        except ImportError:
            logger.error("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
            if self.default_engine == "paddle":
                logger.warning("Falling back to Tesseract OCR")
                self.default_engine = "tesseract"
                self._init_tesseract()
        except Exception as e:
            logger.error(f"Error initializing PaddleOCR: {e}")
            if self.default_engine == "paddle":
                logger.warning("Falling back to Tesseract OCR")
                self.default_engine = "tesseract"
                self._init_tesseract()
    
    def _init_tesseract(self):
        """Initialize Tesseract OCR engine."""
        try:
            import pytesseract
            
            tesseract_config = self.config["engines"]["tesseract"]
            tesseract_path = tesseract_config.get("path")
            
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            # Test if Tesseract is working
            pytesseract.get_tesseract_version()
            self._tesseract_setup_done = True
            logger.info("Tesseract OCR initialized successfully")
        except ImportError:
            logger.error("Pytesseract not available. Install with: pip install pytesseract")
        except Exception as e:
            logger.error(f"Error initializing Tesseract OCR: {e}")
    
    def extract_text(self, file_path: str) -> Dict:
        """
        Extract text from an image or PDF using OCR.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing extracted text and confidence scores
        """
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"text": "", "confidence": 0.0, "error": "File not found"}
        
        # Convert PDF to images if necessary
        if file_path.lower().endswith('.pdf'):
            return self._process_pdf(file_path)
        else:
            return self._process_image(file_path)
    
    def _process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a PDF file by converting pages to images and running OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict with extracted text and confidence
        """
        try:
            doc = fitz.open(pdf_path)
            all_page_text = []
            total_confidence = 0.0
            page_count = 0
            
            # Create a temporary directory for storing page images
            with tempfile.TemporaryDirectory() as temp_dir:
                for page_num in range(doc.page_count):
                    # Convert PDF page to image
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    image_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
                    pix.save(image_path)
                    
                    # Process the page image
                    result = self._process_image(image_path)
                    if result.get("text"):
                        all_page_text.append(result["text"])
                        total_confidence += result.get("confidence", 0.0)
                        page_count += 1
            
            doc.close()
            
            # Combine results
            combined_text = "\n\n".join(all_page_text)
            avg_confidence = total_confidence / max(1, page_count)
            
            return {
                "text": combined_text,
                "confidence": avg_confidence,
                "page_count": doc.page_count
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF with OCR: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e)}
    
    def _process_image(self, image_path: str) -> Dict:
        """
        Process a single image with OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with extracted text and confidence
        """
        if self.default_engine == "paddle":
            return self._process_with_paddle(image_path)
        elif self.default_engine == "tesseract":
            return self._process_with_tesseract(image_path)
        elif self.default_engine == "cloud":
            return self._process_with_cloud(image_path)
        else:
            logger.error(f"Unknown OCR engine: {self.default_engine}")
            return {"text": "", "confidence": 0.0, "error": f"Unknown OCR engine: {self.default_engine}"}
    
    def _process_with_paddle(self, image_path: str) -> Dict:
        """
        Process an image with PaddleOCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with extracted text and confidence
        """
        if not self._paddle_ocr:
            self._init_paddle_ocr()
            if not self._paddle_ocr:
                logger.error("Failed to initialize PaddleOCR")
                return {"text": "", "confidence": 0.0, "error": "PaddleOCR initialization failed"}
        
        try:
            # Run OCR
            results = self._paddle_ocr.ocr(image_path, cls=True)
            
            if not results or not results[0]:
                return {"text": "", "confidence": 0.0}
            
            # Extract text and confidence
            text_lines = []
            total_confidence = 0.0
            boxes = []
            
            for line in results[0]:
                if isinstance(line, list) and len(line) >= 2:
                    box_coords = line[0]
                    text_prob = line[1]
                    if isinstance(text_prob, tuple) and len(text_prob) == 2:
                        text, prob = text_prob
                        text_lines.append(text)
                        total_confidence += prob
                        boxes.append({
                            "text": text,
                            "confidence": prob,
                            "bbox": box_coords
                        })
            
            # Calculate average confidence
            avg_confidence = total_confidence / max(1, len(text_lines))
            
            return {
                "text": "\n".join(text_lines),
                "confidence": avg_confidence,
                "boxes": boxes
            }
            
        except Exception as e:
            logger.error(f"Error in PaddleOCR processing: {e}")
            # Fallback to Tesseract if PaddleOCR fails
            if self.default_engine == "paddle":
                logger.warning("Falling back to Tesseract for this image")
                return self._process_with_tesseract(image_path)
            return {"text": "", "confidence": 0.0, "error": str(e)}
    
    def _process_with_tesseract(self, image_path: str) -> Dict:
        """
        Process an image with Tesseract OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with extracted text and confidence
        """
        if not self._tesseract_setup_done:
            self._init_tesseract()
            if not self._tesseract_setup_done:
                logger.error("Failed to initialize Tesseract")
                return {"text": "", "confidence": 0.0, "error": "Tesseract initialization failed"}
        
        try:
            import pytesseract
            from PIL import Image
            
            # Get tesseract parameters
            tesseract_config = self.config["engines"]["tesseract"]["params"]
            lang = tesseract_config.get("lang", "eng")
            oem = tesseract_config.get("oem", 1)
            psm = tesseract_config.get("psm", 6)
            
            # Prepare configuration string
            config = f"--oem {oem} --psm {psm}"
            
            # Open the image
            image = Image.open(image_path)
            
            # Run OCR
            text = pytesseract.image_to_string(image, lang=lang, config=config)
            
            # Get confidence data
            data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            confidences = [float(conf) / 100.0 for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / max(1, len(confidences)) if confidences else 0.0
            
            # Get bounding boxes
            boxes = []
            for i in range(len(data['text'])):
                if data['text'][i].strip() and data['conf'][i] != '-1':
                    boxes.append({
                        "text": data['text'][i],
                        "confidence": float(data['conf'][i]) / 100.0,
                        "bbox": [
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ]
                    })
            
            return {
                "text": text,
                "confidence": avg_confidence,
                "boxes": boxes
            }
            
        except ImportError:
            logger.error("Pytesseract not available")
            return {"text": "", "confidence": 0.0, "error": "Pytesseract not available"}
        except Exception as e:
            logger.error(f"Error in Tesseract processing: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e)}
    
    def _process_with_cloud(self, image_path: str) -> Dict:
        """
        Process an image with cloud-based OCR service.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with extracted text and confidence
        """
        # Get cloud provider configuration
        cloud_config = self.config["engines"]["cloud"]
        provider = cloud_config.get("provider", "google")
        
        if provider == "google":
            return self._process_with_google_cloud(image_path)
        elif provider == "aws":
            return self._process_with_aws(image_path)
        elif provider == "azure":
            return self._process_with_azure(image_path)
        else:
            logger.error(f"Unknown cloud provider: {provider}")
            return {"text": "", "confidence": 0.0, "error": f"Unknown cloud provider: {provider}"}
    
    def _process_with_google_cloud(self, image_path: str) -> Dict:
        """
        Process an image with Google Cloud Document AI.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with extracted text and confidence
        """
        try:
            # This method should be implemented when google-cloud-documentai is available
            # For now, log a warning and fallback to local OCR
            logger.warning("Google Cloud Document AI not implemented. Falling back to local OCR")
            if self.default_engine == "paddle":
                return self._process_with_paddle(image_path)
            else:
                return self._process_with_tesseract(image_path)
        except Exception as e:
            logger.error(f"Error in Google Cloud OCR processing: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e)}
    
    def _process_with_aws(self, image_path: str) -> Dict:
        """Process with AWS Textract."""
        # Similar implementation as Google Cloud
        logger.warning("AWS Textract not implemented. Falling back to local OCR")
        return self._process_with_tesseract(image_path)
    
    def _process_with_azure(self, image_path: str) -> Dict:
        """Process with Azure Form Recognizer."""
        # Similar implementation as Google Cloud
        logger.warning("Azure Form Recognizer not implemented. Falling back to local OCR")
        return self._process_with_tesseract(image_path)