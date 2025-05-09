import logging
import fitz  # PyMuPDF
import pdfplumber
from typing import Dict, List, Optional, Tuple, Any, Union
import os

logger = logging.getLogger(__name__)

class TextExtractor:
    """
    Extracts text from PDF documents using PyMuPDF and PDFPlumber.
    
    This class prioritizes direct text extraction for text-based PDFs,
    providing better extraction quality than OCR for such documents.
    """
    
    def __init__(self):
        self.min_text_length = 50  # Minimum text length to consider a PDF as text-based
    
    def is_text_based_pdf(self, file_path: str) -> bool:
        """
        Determine if a PDF is text-based (has extractable text) or scanned.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            bool: True if the PDF is text-based, False if it appears to be scanned
        """
        try:
            # First try with PyMuPDF which is faster
            doc = fitz.open(file_path)
            # Sample a few pages to check for text
            pages_to_check = min(3, doc.page_count)
            total_text = ""
            
            for i in range(pages_to_check):
                page = doc[i]
                text = page.get_text()
                total_text += text
                
                # If we find substantial text on any page, consider it text-based
                if len(text) > self.min_text_length:
                    logger.debug(f"Found text in PDF {file_path} on page {i+1}")
                    doc.close()
                    return True
            
            doc.close()
            
            # If PyMuPDF didn't find much text, try with PDFPlumber as backup
            if len(total_text) <= self.min_text_length:
                with pdfplumber.open(file_path) as pdf:
                    for i in range(min(3, len(pdf.pages))):
                        page = pdf.pages[i]
                        text = page.extract_text() or ""
                        
                        if len(text) > self.min_text_length:
                            logger.debug(f"PDFPlumber found text in PDF {file_path} on page {i+1}")
                            return True
            
            # If both methods didn't find substantial text, consider it a scanned document
            logger.info(f"PDF {file_path} appears to be scanned (insufficient extractable text)")
            return False
            
        except Exception as e:
            logger.error(f"Error determining if PDF is text-based: {e}")
            # Default to assuming it's a scanned document if we can't analyze it
            return False
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        if not file_path.lower().endswith('.pdf'):
            logger.warning(f"File {file_path} is not a PDF, text extraction may fail")
        
        # Try with PyMuPDF first (faster and often better quality)
        try:
            doc = fitz.open(file_path)
            pages_text = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                pages_text.append(text)
            
            doc.close()
            full_text = "\n\n".join(pages_text)
            
            # If PyMuPDF extraction yields too little text, try PDFPlumber
            if len(full_text.strip()) < self.min_text_length:
                logger.info(f"PyMuPDF extracted limited text, trying PDFPlumber for {file_path}")
                return self._extract_with_pdfplumber(file_path)
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {e}")
            # Fallback to PDFPlumber
            return self._extract_with_pdfplumber(file_path)
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """
        Extract text using PDFPlumber (used as a fallback).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                pages_text = []
                
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    pages_text.append(text)
                
                return "\n\n".join(pages_text)
                
        except Exception as e:
            logger.error(f"Error extracting text with PDFPlumber: {e}")
            return ""
    
    def extract_text_with_positions(self, file_path: str) -> List[Dict]:
        """
        Extract text with position information.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List[Dict]: List of text elements with position information
        """
        try:
            doc = fitz.open(file_path)
            result = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                for span in line["spans"]:
                                    text_element = {
                                        "text": span["text"],
                                        "page": page_num,
                                        "bbox": [span["bbox"][0], span["bbox"][1], 
                                                 span["bbox"][2], span["bbox"][3]],
                                        "font": span.get("font", ""),
                                        "size": span.get("size", 0),
                                        "color": span.get("color", 0)
                                    }
                                    result.append(text_element)
            
            doc.close()
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text with positions: {e}")
            return []